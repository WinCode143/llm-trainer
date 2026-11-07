"""Tests for the pieces that would break silently: dataset normalization, chunking, BM25,
GRPO rewards, document extraction and the agents' tools.

Run:  cd backend && uv run pytest -q
They load no models and need no GPU: they're fast (seconds).
"""
import io
import json

import time

import pytest

from app.services import datasets as ds
from app.services import graph, rag
from app.services import tools as T
from app.workers import rewards


# ---------------------------------------------------------------- datasets
def test_detects_and_normalizes_alpaca():
    fmt, rows = ds.normalize([{"instruction": "Capital of France?", "output": "Paris"}])
    assert fmt == "chat"
    assert rows[0]["messages"] == [{"role": "user", "content": "Capital of France?"},
                                   {"role": "assistant", "content": "Paris"}]


def test_normalizes_sharegpt_and_system_prompt():
    fmt, rows = ds.normalize([{"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello!"}]}],
                             system_prompt="You are helpful")
    assert fmt == "chat" and rows[0]["messages"][0] == {"role": "system", "content": "You are helpful"}


def test_normalizes_preferences_given_as_lists():
    fmt, rows = ds.normalize([{"prompt": "p", "chosen": [{"role": "assistant", "content": "good"}],
                               "rejected": [{"role": "assistant", "content": "bad"}]}])
    assert fmt == "preference" and rows[0]["chosen"] == "good" and rows[0]["rejected"] == "bad"


def test_drops_examples_without_an_answer():
    _, rows = ds.normalize([{"instruction": "no answer", "output": ""},
                            {"instruction": "ok", "output": "answer"}])
    assert len(rows) == 1


def test_unknown_format_fails():
    with pytest.raises(ValueError):
        ds.normalize([{"odd_field": 1}])


def test_reads_csv_and_txt():
    assert ds._read_records("a.csv", b"instruction,output\nhello,bye\n")[0]["output"] == "bye"
    assert len(ds._read_records("a.txt", b"paragraph one\n\nparagraph two")) == 2


def test_split_keeps_every_row():
    rows = [{"text": str(i)} for i in range(100)]
    train, ev = ds.split(rows, 0.2)
    assert len(train) == 80 and len(ev) == 20 and len(train) + len(ev) == len(rows)


# ---------------------------------------------------------------- chunking
def test_chunking_respects_size_and_overlaps():
    text = ". ".join(f"Sentence number {i} with some filler" for i in range(60))
    chunks = rag.chunk_text(text, 200, 40)
    assert chunks and all(len(c) <= 260 for c in chunks)
    assert len(chunks) > 1


def test_sections_nest_by_level():
    secs, _ = rag.split_sections("## A\na\n### A1\na1\n## B\nb")
    assert [s for s, _ in secs] == ["A", "A › A1", "B"]


def test_qa_template_splits_questions():
    pages = [("One?\nanswer one\nTwo?\nanswer two", {})]
    chunks = rag.make_chunks("faq.md", pages, 800, 0, "qa")
    assert len(chunks) == 2


def test_laws_template_splits_articles():
    pages = [("Article 1. Purpose\nx\nArticle 2. Hours\ny", {})]
    assert len(rag.make_chunks("law.txt", pages, 800, 0, "laws")) == 2


def test_table_template_one_row_per_chunk():
    chunks = rag.make_chunks("t.csv", [("a,b\n1,2\n3,4", {})], 800, 0, "table")
    assert [c for c, _ in chunks] == ["a: 1; b: 2", "a: 3; b: 4"]


def test_html_ignores_scripts_and_keeps_headings():
    html = "<html><head><style>x{}</style></head><body><nav><img src=x></nav><h2>Title</h2><p>Visible text</p><script>1</script></body></html>"
    md = rag._html(html)
    assert "## Title" in md and "Visible text" in md and "x{}" not in md and "1" not in md.split("Visible text")[1]


# ---------------------------------------------------------------- BM25
def test_bm25_folds_accents_and_ranks():
    idx = rag.BM25(["a", "b"], ["The reactor runs at 612 degrees", "Queen Ilsa founded Fréédonia"],
                   sources=["s1", "s2"])
    assert idx.search("who founded FREEDONIA?", 2)[0][0] == "b"


def test_bm25_filters_by_document():
    idx = rag.BM25(["a", "b"], ["text about cats", "text about cats"], sources=["one", "two"])
    assert [i for i, _ in idx.search("cats", 5, allowed={"two"})] == ["b"]


# ---------------------------------------------------------------- GRPO rewards
def _c(text):
    return [[{"role": "assistant", "content": text}]]


def test_correctness_reward():
    high = rewards.correctness(None, _c("The capital is Aurora City"), ["The capital is Aurora City"])[0]
    low = rewards.correctness(None, _c("I don't know"), ["The capital is Aurora City"])[0]
    assert high > 1.5 > low


def test_reasoning_format_reward():
    assert rewards.reasoning_format(None, _c("<think>I think about it quite a bit</think> Answer"))[0] == 1.0
    assert rewards.reasoning_format(None, _c("Direct answer"))[0] == 0.0


def test_length_reward_penalizes_excess():
    short = rewards.make_length(10)(None, _c(" ".join(["x"] * 10)))[0]
    long = rewards.make_length(10)(None, _c(" ".join(["x"] * 25)))[0]
    assert short == 1.0 and long == 0.0


# ---------------------------------------------------------------- agent tools
def test_calculator_handles_percentages_and_rejects_code():
    assert T.TOOLS["calculator"]["run"]({}, {"expression": "12% of 3,382,662"}).startswith("405")
    assert "Error" in T.TOOLS["calculator"]["run"]({}, {"expression": "__import__('os').listdir()"})


def test_sql_is_read_only():
    run = T.TOOLS["sql_query"]["run"]
    assert "Error" in run({"url": "sqlite:///:memory:"}, {"query": "delete from x"})
    assert "Error" in run({"url": "sqlite:///:memory:"}, {"query": "select 1; drop table x"})


def test_files_cant_leave_the_folder(tmp_path):
    cfg = {"root_dir": str(tmp_path)}
    (tmp_path / "inside.txt").write_text("ok")
    assert T.TOOLS["read_file"]["run"](cfg, {"path": "inside.txt"}) == "ok"
    assert "Error" in T.TOOLS["read_file"]["run"](cfg, {"path": "../../etc/passwd"})
    assert "Error" in T.TOOLS["write_file"]["run"](cfg, {"path": "../outside.txt", "content": "x"})


def test_http_requires_allowed_domains():
    assert "isn't allowed" in T.TOOLS["http_request"]["run"]({"allowed_hosts": "ok.com"}, {"url": "https://bad.com/x"})


# ---------------------------------------------------------------- graph and agents
def test_graph_recovers_truncated_json():
    truncated = '{"entities": [{"name": "Ann", "type": "person", "description": "x"}, {"name": "Louis"'
    assert graph._parse(truncated)["entities"][0]["name"] == "Ann"


def test_graph_normalizes_names():
    assert graph.norm("The Roman Empire") == "roman empire"


def test_parses_tool_calls():
    from app.services.agents import parse_calls, visible_text
    text = 'Let me check <tool_call>{"name": "search_web", "arguments": {"query": "x"}}</tool_call>'
    calls = parse_calls(text)
    assert calls == [{"name": "search_web", "arguments": {"query": "x"}}]
    assert visible_text(text) == "Let me check"


def test_parsing_accepts_a_json_block():
    from app.services.agents import parse_calls
    assert parse_calls('```json\n{"name": "calculate", "arguments": {"expression": "2+2"}}\n```')[0]["name"] == "calculate"


# ---------------------------------------------------------------- code sandbox
def test_sandbox_available_or_warns():
    from app.services import sandbox
    st = sandbox.status()
    assert st["backend"] in ("bubblewrap", "none") and isinstance(st["available"], bool)


@pytest.mark.skipif(not __import__("app.services.sandbox", fromlist=["x"]).available(),
                    reason="bubblewrap not available")
class TestSandbox:
    def test_runs_normal_code(self):
        from app.services import sandbox
        assert sandbox.run_python("print(2**10)")["output"] == "1024"

    def test_cant_see_the_project_data(self):
        from app.config import settings
        from app.services import sandbox
        r = sandbox.run_python(f"import os; print(os.path.exists({str(settings.db_path)!r}))")
        assert r["output"] == "False"

    def test_what_it_writes_doesnt_get_out(self, tmp_path):
        from app.services import sandbox
        destination = tmp_path / "leak.txt"
        sandbox.run_python(f"open({str(destination)!r}, 'w').write('x')")
        assert not destination.exists()

    def test_no_network_by_default(self):
        from app.services import sandbox
        r = sandbox.run_python("import urllib.request; urllib.request.urlopen('https://example.com', timeout=5)")
        assert "Error" in r["error"] or "error" in r["error"].lower()

    def test_read_only_data_folder(self, tmp_path):
        from app.services import sandbox
        (tmp_path / "datum.txt").write_text("hello")
        r = sandbox.run_python("print(open('/data/datum.txt').read())", ro_dir=str(tmp_path))
        assert r["output"] == "hello"
        r2 = sandbox.run_python("open('/data/new.txt','w').write('x')", ro_dir=str(tmp_path))
        assert not (tmp_path / "new.txt").exists() and r2["error"]

    def test_time_limit(self):
        from app.services import sandbox
        assert sandbox.run_python("while True: pass", timeout_s=3)["timeout"] is True


# --------------------------------------------------------------------- flows
class TestFlows:
    """The flow compiler: valid templates, safe conditions and validation warnings."""

    def test_templates_compile(self):
        """The templates have to compile. The only warnings allowed are choosing a
        collection or an MCP server: they depend on what each user has and are filled in when used."""
        from app.services import flows
        for key, tpl in flows.TEMPLATES.items():
            warnings = [a for a in flows.validate(tpl["spec"]) if "collection" not in a and "MCP server" not in a]
            assert not warnings, f"the {key} template produces unexpected warnings: {warnings}"
            flows.compile_flow(tpl["spec"])  # must not raise

    def test_export_to_code_produces_valid_python(self):
        """The exporter is the way out: if it generates broken code, it's useless."""
        import ast
        from app.services import flows
        for key, tpl in flows.TEMPLATES.items():
            code = flows.to_python(tpl["spec"], key)
            ast.parse(code)  # raises SyntaxError if it's wrong
            assert "StateGraph" in code and "def build()" in code

    def test_router_conditions_are_translated_on_export(self):
        """In the editor you write `supported == True`; in the exported code that variable lives in the state."""
        from app.services.flows import _cond_py
        assert _cond_py("supported == True") == "v.get('supported') == True"
        assert _cond_py("len(text) > 3") == "len(v.get('text')) > 3"

    def test_lists_render_readably(self):
        """A list put into a prompt can't come out as ['a', 'b'] with brackets."""
        from app.services.flows import render
        assert render("{xs}", {"xs": ["bread", "milk"]}) == "1. bread\n2. milk"

    def test_for_each_turns_text_into_a_list(self):
        from app.services.flows import _as_list
        assert _as_list("one\ntwo\n\nthree") == ["one", "two", "three"]
        assert _as_list(["a", "b"]) == ["a", "b"]

    def test_missing_variable_doesnt_break_render(self):
        from app.services.flows import render
        assert render("hello {who}, {missing}", {"who": "world"}) == "hello world, "

    def test_conditions(self):
        from app.services.flows import eval_condition
        assert eval_condition("supported == True", {"supported": True})
        assert eval_condition("len(text) > 3", {"text": "abcd"})
        assert not eval_condition("kind in options", {"kind": "x", "options": ["a"]})
        # no arbitrary calls or attribute access
        assert not eval_condition("__import__('os').system('ls')", {})
        assert not eval_condition("text.upper()", {"text": "a"})

    def test_validate_flags_loose_nodes(self):
        from app.services.flows import validate
        spec = {"nodes": [{"id": "a", "type": "input"}, {"id": "b", "type": "output"}], "edges": []}
        assert validate(spec)


# ---------------------------------------------------------- helper models
class TestProviders:
    """The single generation entry point and key handling."""

    def test_complete_without_model_or_helper_warns(self, monkeypatch):
        from app.services import providers
        from app.services.inference import manager
        monkeypatch.setattr(manager, "current", None)
        with pytest.raises(RuntimeError, match="helper"):
            providers.complete(None, "hello")

    def test_missing_helper_is_explicit(self):
        from app.services import providers
        with pytest.raises(RuntimeError, match="no longer exists"):
            providers.complete(999999, "hello")

    def test_agent_with_helper_keeps_the_question_after_a_tool_call(self, monkeypatch):
        """With an external helper, each turn has to carry the whole conversation: after a tool
        call the last message is the tool's result, and the user's question must still be there."""
        from app.services import agents, providers
        calls = []

        def fake_complete(helper, prompt, *, system=None, history=None, **kw):
            calls.append({"prompt": prompt, "system": system, "history": history or []})
            if len(calls) == 1:
                return '<tool_call>{"name": "calculate", "arguments": {"expression": "2+3"}}</tool_call>'
            return "The answer is 5."

        monkeypatch.setattr(providers, "complete", fake_complete)
        agent = {"helper": 1, "system_prompt": "Be brief.", "tools": [{"id": "calculator", "config": {}}],
                 "max_steps": 3}
        events = list(agents.run(agent, [{"role": "user", "content": "What is 2+3?"}]))
        assert events[-1]["type"] == "done" and events[-1]["final"] == "The answer is 5."
        second = calls[1]
        assert second["prompt"].startswith("Result of calculate")
        assert second["history"][0] == {"role": "user", "content": "What is 2+3?"}
        assert second["history"][1]["role"] == "assistant"
        assert "Be brief." in second["system"]

    def test_anthropic_history_alternates_roles(self):
        from app.services import providers
        merged = providers._merge_roles([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                                         {"role": "user", "content": "c"}, {"role": "user", "content": "d"}])
        assert [m["role"] for m in merged] == ["user", "assistant", "user"]
        assert merged[-1]["content"] == "c\n\nd"

    def test_complete_json_rescues_wrapped_json(self, monkeypatch):
        """Models often return the JSON wrapped in text; it has to be pulled out anyway."""
        from app.services import providers
        monkeypatch.setattr(providers, "complete", lambda *a, **k: 'Sure:\n{"score": 4}\nHope it helps!')
        assert providers.complete_json(None, "…") == {"score": 4}

    def test_complete_json_returns_none_without_json(self, monkeypatch):
        from app.services import providers
        monkeypatch.setattr(providers, "complete", lambda *a, **k: "no idea at all")
        assert providers.complete_json(None, "…") is None

    def test_keys_never_go_back_whole(self):
        """The list the interface sees may only carry the masked key."""
        from app.db import Provider, engine
        from app.services import providers
        from sqlmodel import Session
        with Session(engine) as s:
            p = Provider(name="_test_mask", kind="openai", base_url="http://x/v1", api_key="sk-supersecret1234")
            s.add(p)
            s.commit()
            pid = p.id
        try:
            row = next(r for r in providers.listing() if r["id"] == pid)
            assert row["api_key"] == "…1234"
            assert "supersecret" not in row["api_key"]
            assert row["has_key"] is True
        finally:
            with Session(engine) as s:
                s.delete(s.get(Provider, pid))
                s.commit()


# ------------------------------------------------------- job cleanup
class TestJobCleanup:
    """What must never be deleted: running jobs and the files of a registered model."""

    def _job(self, s, **kw):
        from app.db import Job
        j = Job(name=kw.pop("name", "t"), type="train", config=kw.pop("config", {}), **kw)
        s.add(j)
        s.commit()
        s.refresh(j)
        return j

    def test_doesnt_delete_running_jobs(self, tmp_path):
        from app.api.jobs import CleanupBody, cleanup
        from app.db import Job, Session, engine
        with Session(engine) as s:
            alive = self._job(s, name="_t_alive", status="running")
            dead = self._job(s, name="_t_dead", status="failed")
            try:
                ids = {j["id"] for j in cleanup(CleanupBody(dry_run=True), s)["jobs"]}
                assert dead.id in ids
                assert alive.id not in ids, "a running job can't be part of a cleanup"
            finally:
                for j in (alive, dead):
                    if (row := s.get(Job, j.id)):
                        s.delete(row)
                s.commit()

    def test_protects_the_files_of_a_registered_model(self, tmp_path):
        from app.api.jobs import _delete_one, _output_info
        from app.db import Job, Model, Session, engine
        out = tmp_path / "output"
        out.mkdir()
        (out / "adapter.bin").write_bytes(b"x" * 1000)
        with Session(engine) as s:
            j = self._job(s, name="_t_prot", status="completed", config={"output_dir": str(out)})
            m = Model(name="_t_model", kind="lora", path=str(out / "final"))
            s.add(m)
            s.commit()
            try:
                assert _output_info(s.get(Job, j.id), s)["models"] == ["_t_model"]
                r = _delete_one(s.get(Job, j.id), s, delete_output=True)
                s.commit()
                assert r["kept_output"] is True
                assert out.exists(), "files a model depends on can't be deleted"
            finally:
                with Session(engine) as s2:
                    if (row := s2.get(Model, m.id)):
                        s2.delete(row)
                    s2.commit()

    def test_deletes_files_when_nothing_depends_on_them(self, tmp_path):
        from app.api.jobs import _delete_one
        from app.db import Job, Session, engine
        out = tmp_path / "huerfano"
        out.mkdir()
        (out / "adapter.bin").write_bytes(b"x" * 2000)
        with Session(engine) as s:
            j = self._job(s, name="_t_huerf", status="failed", config={"output_dir": str(out)})
            r = _delete_one(s.get(Job, j.id), s, delete_output=True)
            s.commit()
        assert r["deleted_output"] is True
        assert not out.exists()


# ------------------------------------------- evaluation and synthesis metrics
class TestEvaluationMetrics:
    """These functions produce the numbers you use to decide which model is better.
    A bug here doesn't show: it just gives a slightly different metric and the conclusion comes out wrong."""

    def test_strip_think_removes_the_reasoning(self):
        from app.services.evaluation import strip_think
        assert strip_think("<think>I really doubt it</think>The answer is 42") == "The answer is 42"
        assert strip_think("no tags") == "no tags"

    def test_strip_think_with_an_unclosed_block(self):
        """If the model runs out of tokens inside <think>, the reasoning can't sneak in as the answer."""
        from app.services.evaluation import strip_think
        out = strip_think("<think>I was thinking and ran out of")
        assert "I was thinking" not in out

    def test_strip_think_with_a_stray_closing_tag(self):
        """Qwen3.5 in GGUF opens the block from the template, so it only emits the closing tag.
        Without this, the whole reasoning sneaked in as if it were the answer and contaminated
        every metric, which is exactly what metrics can't afford."""
        from app.services.evaluation import strip_think
        raw = "We need to answer briefly. Two sentences. Good.\n</think>\n\nLoRA trains a few small matrices."
        assert strip_think(raw) == "LoRA trains a few small matrices."

    def test_keywords(self):
        from app.services.evaluation import _contains
        assert _contains("The period is 30 calendar days", ["30", "days"]) == 1.0
        assert _contains("The period is 30 days", ["30", "months"]) == 0.5
        assert _contains("whatever", []) is None

    def test_keywords_ignore_case(self):
        from app.services.evaluation import _contains
        assert _contains("PARIS is the capital", ["paris"]) == 1.0

    def test_summary_averages_per_variant_and_skips_missing(self):
        from app.services.evaluation import summarize
        res = [
            {"variant": "m1 (RAG)", "model_name": "m1", "rag": True, "judge_score": 8, "similarity": 0.9},
            {"variant": "m1 (RAG)", "model_name": "m1", "rag": True, "judge_score": 6, "similarity": None},
            {"variant": "m1", "model_name": "m1", "rag": False, "judge_score": 4},
        ]
        out = summarize(res)
        assert out["m1 (RAG)"]["judge_score"] == 7.0
        # The second row has no similarity: it mustn't count as 0 and drag the mean down
        assert out["m1 (RAG)"]["similarity"] == 0.9
        assert out["m1 (RAG)"]["n"] == 2
        assert out["m1"]["judge_score"] == 4.0

    def test_judge_calibration(self):
        from app.services.evaluation import judge_agreement
        # judge_score goes from 1 to 10 and your ratings from 1 to 5: the function halves it before comparing
        perfect = [{"judge_score": n * 2, "rating": n} for n in (1, 2, 3, 4, 5)]
        r = judge_agreement(perfect)
        assert r["spearman"] == 1.0 and r["mae"] == 0.0 and r["judge_bias"] == 0.0

    def test_calibration_flags_a_generous_judge(self):
        from app.services.evaluation import judge_agreement
        generous = [{"judge_score": n * 2 + 2, "rating": n} for n in (1, 2, 3, 4)]
        r = judge_agreement(generous)
        assert r["judge_bias"] == 1.0, "a judge that scores too high must show a positive bias"

    def test_calibration_without_enough_data(self):
        from app.services.evaluation import judge_agreement
        assert judge_agreement([{"judge_score": 8, "rating": 4}])["spearman"] is None


class TestSynth:
    """Synthetic dataset generation: whatever slips through here ends up training a model."""

    def test_extracts_pairs_from_json_wrapped_in_text(self):
        from app.services.synth import _pairs
        out = _pairs('Sure:\n[{"question": "What is X?", "answer": "It is Y"}]\nDone!')
        assert out == [{"question": "What is X?", "answer": "It is Y"}]

    def test_drops_incomplete_pairs(self):
        from app.services.synth import _pairs
        out = _pairs('[{"question": "no answer"}, {"answer": "no question"}, {"question": "a", "answer": "b"}]')
        assert out == [{"question": "a", "answer": "b"}]

    def test_invalid_json_doesnt_blow_up(self):
        from app.services.synth import _pairs
        assert _pairs("the model went off on a tangent") == []

    def test_dedupe_removes_near_identical_questions(self, monkeypatch):
        """It runs on the CPU on purpose: on the GPU the test passed or failed depending on what was loaded at
        the time, and a test that depends on that proves nothing."""
        from app.services import rag as rag_svc
        monkeypatch.setattr(rag_svc, "_device", lambda: "cpu")
        rag_svc.unload_models()
        from app.services.synth import _dedupe
        rows = [{"question": "What is the capital of France?", "answer": "Paris"},
                {"question": "What is the capital of France?", "answer": "Paris"},
                {"question": "How many bones does the human body have?", "answer": "206"}]
        out, removed = _dedupe(rows, 0.92)
        assert removed == 1 and len(out) == 2


class TestFileDeletion:
    """The “this path is mine” check has to survive symbolic links:
    data/outputs points to another disk, and comparing it unresolved meant nothing was ever deleted."""

    def test_deletes_inside_the_managed_folders(self, tmp_path, monkeypatch):
        from app.api import models as api
        from app.config import settings
        root = tmp_path / "data"
        (root / "outputs" / "something").mkdir(parents=True)
        (root / "outputs" / "something" / "weights.bin").write_bytes(b"x" * 100)
        monkeypatch.setattr(settings, "data_dir", root)
        assert api._delete_if_ours(root / "outputs" / "something") is True
        assert not (root / "outputs" / "something").exists()

    def test_follows_the_link_to_another_disk(self, tmp_path, monkeypatch):
        from app.api import models as api
        from app.config import settings
        real = tmp_path / "other_disk" / "outputs" / "model"
        real.mkdir(parents=True)
        (real / "weights.bin").write_bytes(b"x" * 100)
        root = tmp_path / "data"
        root.mkdir()
        (root / "outputs").symlink_to(tmp_path / "other_disk" / "outputs")
        monkeypatch.setattr(settings, "data_dir", root)
        # The path goes through the link, like the ones the database stores
        assert api._delete_if_ours(root / "outputs" / "model") is True
        assert not real.exists(), "the link to another disk must be followed, not ignored"

    def test_refuses_outside_its_folders(self, tmp_path, monkeypatch):
        from app.api import models as api
        from app.config import settings
        outside = tmp_path / "foreign"
        outside.mkdir()
        (outside / "important.txt").write_text("don't touch")
        monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
        (tmp_path / "data").mkdir()
        assert api._delete_if_ours(outside) is False
        assert (outside / "important.txt").exists()


class TestPurgeOutputs:
    """Delete weights but keep the history: the gigabytes go, the metrics stay."""

    def test_deletes_the_weights_but_not_the_row(self, tmp_path):
        from app.api.jobs import PurgeBody, purge_outputs
        from app.db import Job, Session, engine
        out = tmp_path / "output"
        out.mkdir()
        (out / "adapter.bin").write_bytes(b"x" * 500_000)
        with Session(engine) as s:
            j = Job(name="_t_purge", type="train", status="completed",
                    config={"output_dir": str(out), "method": "sft"},
                    result={"train_loss": 0.42})
            s.add(j)
            s.commit()
            s.refresh(j)
            jid = j.id
            try:
                r = purge_outputs(PurgeBody(ids=[jid], dry_run=False), s)
                assert r["count"] == 1
                assert not out.exists(), "the weights had to be deleted"
                back = s.get(Job, jid)
                assert back is not None, "the job's row must NOT disappear"
                assert back.result["train_loss"] == 0.42, "the metrics have to survive"
                assert back.config["outputs_purged"] is True
            finally:
                if (row := s.get(Job, jid)):
                    s.delete(row)
                    s.commit()

    def test_dry_run_touches_nothing(self, tmp_path):
        from app.api.jobs import PurgeBody, purge_outputs
        from app.db import Job, Session, engine
        out = tmp_path / "output"
        out.mkdir()
        (out / "weights.bin").write_bytes(b"x" * 1000)
        with Session(engine) as s:
            j = Job(name="_t_dry", type="train", status="completed", config={"output_dir": str(out)})
            s.add(j)
            s.commit()
            s.refresh(j)
            try:
                r = purge_outputs(PurgeBody(ids=[j.id], dry_run=True), s)
                assert r["count"] == 1 and out.exists()
            finally:
                s.delete(s.get(Job, j.id))
                s.commit()


class TestDownloadRegistration:
    """Names when registering a downloaded model: two downloads at once used to collide."""

    def test_finds_the_first_free_name(self):
        from app.api.models import _free_name
        from app.db import Model, Session, engine
        created = []
        with Session(engine) as s:
            try:
                assert _free_name(s, "_t_free") == "_t_free"
                for expected in ("_t_free-2", "_t_free-3"):
                    m = Model(name=_free_name(s, "_t_free"), kind="hf", path=f"x{len(created)}")
                    s.add(m)
                    s.commit()
                    s.refresh(m)
                    created.append(m.id)
                    assert _free_name(s, "_t_free") == expected
            finally:
                for i in created:
                    if (row := s.get(Model, i)):
                        s.delete(row)
                s.commit()


class TestVisionGGUF:
    """A GGUF only sees images if its mmproj is loaded and the message format is translated."""

    def test_translates_the_image_format(self):
        """llama.cpp expects image_url; with our format the image travelled and the model
        answered that it couldn't see it."""
        from app.services.inference import _for_llamacpp
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "what do you see?"},
            {"type": "image", "url": "data:image/png;base64,AAA"}]}]
        out = _for_llamacpp(msgs)[0]["content"]
        assert out[0] == {"type": "text", "text": "what do you see?"}
        assert out[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}

    def test_without_vision_images_are_flagged(self):
        from app.services.inference import _text_only
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "look"}, {"type": "image", "url": "x"}]}]
        c = _text_only(msgs)[0]["content"]
        assert "look" in c and "doesn't support vision" in c

    def test_only_enables_vision_with_an_mmproj_next_to_it(self, tmp_path):
        from app.services.inference import manager
        lone = tmp_path / "gemma-4-something.gguf"
        lone.write_bytes(b"x")
        assert manager._vision_handler(str(lone)) == (None, None), "no mmproj, no vision"

    def test_doesnt_guess_the_handler_of_an_unknown_family(self, tmp_path):
        from app.services.inference import manager
        (tmp_path / "mmproj-F16.gguf").write_bytes(b"x")
        odd = tmp_path / "made-up-model-7b.gguf"
        odd.write_bytes(b"x")
        assert manager._vision_handler(str(odd)) == (None, None)


class TestTemplatesAndContext:
    """The dataset templates and the truncated-context warning."""

    def test_templates_normalize_to_the_right_format(self):
        """If a template doesn't go through the normalizer, the dataset it creates can't be trained."""
        from app.services import datasets as ds
        from app.services.dataset_templates import TEMPLATES
        for key, t in TEMPLATES.items():
            fmt, rows = ds.normalize(t["filas"], t["formato"])
            assert fmt == t["formato"], f"{key}: comes out as {fmt} but claims to be {t['formato']}"
            assert len(rows) == len(t["filas"]), f"{key}: rows were lost while normalizing"

    def test_each_template_says_how_many_examples_and_why(self):
        from app.services.dataset_templates import listing
        for t in listing():
            assert t["cuantos"] and t["consejo"], f"{t['key']} has no guidance"
            assert t["metodo"] in ("sft", "dpo", "grpo")

    def test_warns_when_the_context_doesnt_fit(self, monkeypatch):
        """Without a warning, the model answers with half the chunks and it looks like RAG failed."""
        from app.services import rag
        from app.services.inference import Loaded, manager
        monkeypatch.setattr(manager, "current",
                            Loaded(1, "x", "gguf", "p", None, "q", "llama.cpp", n_ctx=2048))
        assert rag.context_overflow("short", "?") is None
        warning = rag.context_overflow("x" * 9000, "?")
        assert warning and "2048" in warning

    def test_no_warning_without_a_loaded_model(self, monkeypatch):
        from app.services import rag
        from app.services.inference import manager
        monkeypatch.setattr(manager, "current", None)
        assert rag.context_overflow("x" * 99999) is None


class TestBackups:
    """The module that protects what can't be replaced had no tests."""

    def test_create_list_and_prune(self, tmp_path, monkeypatch):
        from app.config import settings
        from app.services import backup
        monkeypatch.setattr(settings, "data_dir", tmp_path)
        monkeypatch.setattr(settings, "backup_keep", 2)
        (tmp_path / "llmtrainer.db").write_bytes(b"fake database")
        (tmp_path / "datasets").mkdir()
        (tmp_path / "datasets" / "a.jsonl").write_text('{"x": 1}')

        for _ in range(3):
            backup.create(note="test")
        # backup_keep=2: the oldest one has to have gone by itself
        assert len(backup.listing()) == 2, "prune didn't respect the limit"
        assert all(b["note"] == "test" for b in backup.listing())

    def test_the_backup_holds_what_it_says(self, tmp_path, monkeypatch):
        import tarfile
        from app.config import settings
        from app.services import backup
        monkeypatch.setattr(settings, "data_dir", tmp_path)
        (tmp_path / "llmtrainer.db").write_bytes(b"x")
        (tmp_path / "rag").mkdir()
        (tmp_path / "rag" / "chroma.sqlite3").write_bytes(b"y")
        # Weights must NOT go in: they're huge and can be regenerated
        (tmp_path / "outputs").mkdir()
        (tmp_path / "outputs" / "model.bin").write_bytes(b"z" * 1000)

        info = backup.create()
        with tarfile.open(backup.backups_dir() / info["file"]) as t:
            inside = t.getnames()
        assert any("llmtrainer.db" in n for n in inside)
        assert any("rag" in n for n in inside)
        assert not any("outputs" in n for n in inside), "weights can't go into the backup"

    def test_verify_detects_a_broken_file(self, tmp_path, monkeypatch):
        from app.config import settings
        from app.services import backup
        monkeypatch.setattr(settings, "data_dir", tmp_path)
        broken = backup.backups_dir() / "backup-20260101-000000.tar.gz"
        broken.write_bytes(b"this isn't a tar.gz")
        r = backup.verify(broken.name)
        assert r["ok"] is False and "damaged" in r["error"]

    def test_verify_flags_missing_tables(self, tmp_path, monkeypatch):
        """An old backup doesn't have the new tables: restoring it takes you back to before they existed."""
        import sqlite3
        from app.config import settings
        from app.services import backup
        monkeypatch.setattr(settings, "data_dir", tmp_path)
        con = sqlite3.connect(tmp_path / "llmtrainer.db")
        con.execute("create table dataset (id integer)")
        con.commit()
        con.close()
        info = backup.create()
        r = backup.verify(info["file"])
        assert r["ok"] is True
        assert "questionset" in r["missing_tables"], "it should flag the missing tables"
        assert r["rows"]["dataset"] == 0


class TestTasks:
    """The background task registry: progress, result, error and cancellation."""

    def test_a_task_that_succeeds(self):
        from app.services import tasks
        t = tasks.submit("test", "sum", lambda task: 2 + 2)
        for _ in range(100):
            if t.status != "running":
                break
            time.sleep(0.05)
        assert t.status == "completed" and t.result == 4 and t.progress == 1.0

    def test_a_failing_task_keeps_the_error(self):
        from app.services import tasks

        def blows_up(task):
            raise ValueError("broken on purpose")

        t = tasks.submit("test", "fails", blows_up)
        for _ in range(100):
            if t.status != "running":
                break
            time.sleep(0.05)
        assert t.status == "failed" and "broken on purpose" in t.error

    def test_cancellation_is_visible_from_inside(self):
        """The flag is all the work has to find out: if it doesn't arrive, it can't stop."""
        from app.services import tasks
        seen = []

        def long_task(task):
            for _ in range(200):
                if task.cancel_requested:
                    seen.append(True)
                    return "cancelled"
                time.sleep(0.02)
            return "finished whole"

        t = tasks.submit("test", "long", long_task)
        time.sleep(0.1)
        tasks.cancel(t.id)
        for _ in range(100):
            if t.status != "running":
                break
            time.sleep(0.05)
        assert seen == [True] and t.result == "cancelled"


class TestVllm:
    """vLLM lives in another environment and is talked to over HTTP: what can be tested without a GPU is the contract."""

    def test_knows_whether_it_is_installed(self):
        from app.services import vllm_engine
        assert isinstance(vllm_engine.available(), bool)

    def test_the_request_body_respects_the_parameters(self):
        from app.services.inference import GenParams
        from app.services.vllm_engine import VllmServer
        s = VllmServer.__new__(VllmServer)
        s.served_name = "base"
        p = GenParams(max_new_tokens=128, temperature=0.4, top_p=0.8, top_k=20, seed=7, enable_thinking=False)
        body = VllmServer._payload(s, [{"role": "user", "content": "hi"}], p, stream=True, template_kwargs=None)
        assert body["max_tokens"] == 128 and body["temperature"] == 0.4 and body["seed"] == 7
        assert body["stream"] is True
        # enable_thinking travels as a template argument, not as a standalone parameter
        assert body["chat_template_kwargs"]["enable_thinking"] is False

    def test_tools_travel_in_openai_format(self):
        from app.services.inference import GenParams
        from app.services.vllm_engine import VllmServer
        s = VllmServer.__new__(VllmServer)
        s.served_name = "lora"
        schema = {"name": "search", "description": "…", "parameters": {}}
        body = VllmServer._payload(s, [{"role": "user", "content": "x"}], GenParams(),
                                   stream=False, template_kwargs={"tools": [schema]})
        assert body["tools"] == [{"type": "function", "function": schema}]


class TestMcpClient:
    """The MCP client was broken since the library moved to 2.x and nobody noticed."""

    def test_digs_the_real_error_out_of_the_taskgroup(self):
        from app.services.tools import error_real
        try:
            raise BaseExceptionGroup("unhandled errors in a TaskGroup", [FileNotFoundError("npx doesn't exist")])
        except BaseExceptionGroup as e:
            assert error_real(e) == "FileNotFoundError: npx doesn't exist"

    def test_a_standalone_url_is_treated_as_http(self):
        from app.services.tools import _mcp_server
        assert _mcp_server("http://x/mcp/") == {"transport": "http", "url": "http://x/mcp/"}

    def test_a_missing_server_is_explicit(self):
        from app.services.tools import _mcp_server
        with pytest.raises(RuntimeError, match="no longer exists"):
            _mcp_server(999999)

    def test_keys_never_go_back_whole(self):
        from app.api.connections import _mask
        assert _mask({"GITHUB_TOKEN": "ghp_supersecret9876"}) == {"GITHUB_TOKEN": "…9876"}


class TestTriggers:
    """What opens the app to the outside: who can run what, and when."""

    def _bot(self, allowed):
        from app.db import Trigger
        return Trigger(id=1, name="bot", kind="telegram", target_kind="flow", target_id=1,
                       config={"bot_token": "x", "allowed_chat_ids": allowed})

    def _msg(self, chat, text="hello"):
        return {"update_id": 1, "message": {"chat": {"id": chat}, "text": text}}

    def test_without_a_list_nothing_runs_and_it_tells_the_chat_id(self, monkeypatch):
        from app.services import triggers as tr
        ran = []
        monkeypatch.setattr(tr, "execute", lambda *a, **k: ran.append(1) or {"estado": "ok", "salida": "x"})
        sent = []
        tr.handle_update(self._bot([]), self._msg(555), lambda m, **d: sent.append((m, d)))
        assert not ran, "a bot without an allowlist can't run anything"
        assert "555" in sent[0][1]["text"], "it has to tell them their chat_id so it can be added"

    def test_a_stranger_is_silently_ignored(self, monkeypatch):
        from app.services import triggers as tr
        monkeypatch.setattr(tr, "execute", lambda *a, **k: pytest.fail("it shouldn't have run"))
        sent = []
        tr.handle_update(self._bot([111]), self._msg(999), lambda m, **d: sent.append(m))
        assert sent == [], "a foreign chat gets no reply: not even confirmation that the bot exists"

    def test_an_allowed_chat_runs_and_gets_the_answer(self, monkeypatch):
        from app.services import triggers as tr
        monkeypatch.setattr(tr, "execute", lambda t, text, o: {"estado": "ok", "salida": f"echo: {text}"})
        sent = []
        tr.handle_update(self._bot([111]), self._msg(111, "hello"), lambda m, **d: sent.append((m, d)))
        texts = [d["text"] for m, d in sent if m == "sendMessage"]
        assert texts == ["echo: hello"]

    def test_an_approval_arrives_with_buttons_and_resumes(self, monkeypatch):
        from app.services import triggers as tr
        monkeypatch.setattr(tr, "execute", lambda t, text, o: {
            "estado": "waiting", "salida": {"message": "Save it?", "preview": "draft"}, "thread_id": "th1", "flow_id": 7})
        sent = []
        bot = self._bot([111])
        tr.handle_update(bot, self._msg(111, "write it"), lambda m, **d: sent.append((m, d)))
        keyboard = [d for m, d in sent if m == "sendMessage" and "reply_markup" in d][0]
        buttons = keyboard["reply_markup"]["inline_keyboard"][0]
        ok = buttons[0]["callback_data"]
        assert ok.startswith("ok:") and buttons[1]["callback_data"].startswith("no:")

        resumed = {}
        monkeypatch.setattr(tr, "_run_flow", lambda fid, text, thread_id=None, resume=None:
                            resumed.update(fid=fid, thread=thread_id, resume=resume) or {"estado": "ok", "salida": "saved"})
        cq = {"update_id": 2, "callback_query": {"id": "c1", "data": ok, "message": {"chat": {"id": 111}}}}
        tr.handle_update(bot, cq, lambda m, **d: sent.append((m, d)))
        assert resumed == {"fid": 7, "thread": "th1", "resume": {"rejected": False, "comment": "from Telegram", "edited": None}}

    def test_a_foreign_chats_button_doesnt_resume(self, monkeypatch):
        from app.services import triggers as tr
        monkeypatch.setattr(tr, "_run_flow", lambda *a, **k: pytest.fail("it shouldn't have resumed"))
        tr._pending["k1"] = {"flow_id": 1, "thread_id": "t", "texto": "x"}
        cq = {"update_id": 3, "callback_query": {"id": "c", "data": "ok:k1", "message": {"chat": {"id": 999}}}}
        tr.handle_update(self._bot([111]), cq, lambda m, **d: None)
        assert "k1" in tr._pending, "the approval stays pending for its owner"
        tr._pending.pop("k1")

    def test_when_a_schedule_is_due(self):
        from datetime import datetime
        from app.services.triggers import is_due
        a = datetime(2026, 9, 24, 8, 30)
        assert is_due({"cada_min": 60}, None, a)
        assert not is_due({"cada_min": 60}, datetime(2026, 9, 24, 8, 0), a)
        assert is_due({"cada_min": 60}, datetime(2026, 9, 24, 7, 29), a)
        assert is_due({"diario": "08:00"}, datetime(2026, 9, 23, 8, 0), a), "it hadn't run today yet"
        assert not is_due({"diario": "08:00"}, datetime(2026, 9, 24, 8, 1), a), "it already ran today"
        assert not is_due({"diario": "09:00"}, None, a), "it isn't time yet"

    def test_a_wrong_token_finds_nothing(self):
        from app.services.triggers import find_webhook
        assert find_webhook("token-that-doesnt-exist") is None


class TestAgentEvaluation:
    """How it's decided whether an agent picked the right tool."""

    ALIAS = {"calculate": {"calculate", "calculator"},
             "search_documents": {"search_documents", "rag_search", "search my documents (rag)"}}

    def test_the_catalog_id_counts_even_if_the_model_sees_another_name(self):
        """The model calls “calculate” and the case says “calculator”: that's a hit.
        Without translating names, the first real evaluation gave 17% when it was 67%."""
        from app.services.agent_eval import _matches
        assert _matches("calculator", ["calculate"], self.ALIAS)
        assert _matches("rag_search", ["search_documents"], self.ALIAS)
        assert not _matches("calculator", ["search_documents"], self.ALIAS)

    def test_none_means_no_tools(self, monkeypatch):
        from app.services import agent_eval
        monkeypatch.setattr(agent_eval, "run_agent", lambda a, m: iter([{"type": "done", "final": "Hi!", "steps": 1}]))
        r = agent_eval.evaluate_case({}, {"question": "hi", "tool": "none"}, alias={})
        assert r["acierto"] and r["a_la_primera"]

    def test_getting_it_right_after_failing_isnt_first_try(self, monkeypatch):
        from app.services import agent_eval
        events = [{"type": "tool_call", "name": "search_documents"}, {"type": "tool_call", "name": "calculate"},
                   {"type": "done", "final": "69104", "steps": 3}]
        monkeypatch.setattr(agent_eval, "run_agent", lambda a, m: iter(events))
        r = agent_eval.evaluate_case({}, {"question": "1234*56", "tool": "calculator", "keywords": ["69104"]}, alias=self.ALIAS)
        assert r["acierto"] and not r["a_la_primera"] and r["claves"] == 1.0

    def test_the_summary_groups_the_failures(self):
        from app.services.agent_eval import summarize
        rows = [{"acierto": False, "a_la_primera": False, "esperada": "rag_search", "llamadas": [], "claves": None,
                  "pasos": 1, "segundos": 1.0, "error": None}] * 2 + \
                [{"acierto": True, "a_la_primera": True, "esperada": "calculator", "llamadas": ["calculate"], "claves": 1.0,
                  "pasos": 2, "segundos": 1.0, "error": None}]
        s = summarize(rows)
        assert s["herramienta_correcta"] == 0.33
        assert s["fallos_frecuentes"][0] == ("expected rag_search, used none", 2)


class TestCallFormats:
    """Each model family writes tool calls its own way."""

    def test_recognizes_gemma4_native_format(self):
        from app.services.agents import parse_calls, visible_text
        raw = ('<|channel>thought\n<channel|><|tool_call>call:mcp_list_directory'
                 '{"path": "/tmp/folder"}<tool_call|>')
        assert parse_calls(raw) == [{"name": "mcp_list_directory", "arguments": {"path": "/tmp/folder"}}]
        assert visible_text(raw) == "", "neither the thinking channel nor the call may reach the user"

    def test_gemma4_with_unquoted_keys(self):
        from app.services.agents import parse_calls
        raw = '<|channel>thought\n<channel|><|tool_call>call:mcp_open_nodes{names:["User"]}<tool_call|>'
        assert parse_calls(raw) == [{"name": "mcp_open_nodes", "arguments": {"names": ["User"]}}]
        raw = '<|tool_call>call:search{query:<|"|>price: lithium, today<|"|>,k:3}<tool_call|>'
        assert parse_calls(raw) == [{"name": "search", "arguments": {"query": "price: lithium, today", "k": 3}}]
        # Sometimes with a colon between the name and the arguments
        raw = '<|tool_call>call:mcp_arxiv_search:{"max_results":5,"query":"GRPO"}<tool_call|>'
        assert parse_calls(raw) == [{"name": "mcp_arxiv_search", "arguments": {"max_results": 5, "query": "GRPO"}}]

    def test_recognizes_the_call_with_attributes(self):
        from app.services.agents import parse_calls, visible_text
        raw = """<tool_call name="mcp_open_nodes" arguments='{"names": ["User"]}'></tool_call>"""
        assert parse_calls(raw) == [{"name": "mcp_open_nodes", "arguments": {"names": ["User"]}}]
        assert visible_text(raw) == ""

    def test_still_recognizes_the_usual_format(self):
        from app.services.agents import parse_calls
        assert parse_calls('<tool_call>{"name": "calculate", "arguments": {"expr": "2+2"}}</tool_call>') == \
            [{"name": "calculate", "arguments": {"expr": "2+2"}}]

    def test_strip_think_removes_gemmas_channel(self):
        from app.services.evaluation import strip_think
        assert strip_think("<|channel>thought\nlet me think<channel|>The answer is 4") == "The answer is 4"


class TestConnectedMcp:
    """External MCPs are useful when they're plugged into something: a flow node or an agent that saves here."""

    def test_a_nodes_arguments_arrive_with_their_type(self):
        assert T._typed({"limit": "5", "q": "hello", "types": '["dataset"]', "ok": "true", "id": "2106.09685x"}) == \
            {"limit": 5, "q": "hello", "types": ["dataset"], "ok": True, "id": "2106.09685x"}

    def test_an_mcp_node_without_a_tool_explains_whats_missing(self):
        text, _ = T.execute({"run": T.TOOLS["mcp"]["run"], "config": {"server_id": 1}}, {})
        assert "Choose the MCP server and the tool" in text

    def test_the_flow_warns_about_a_half_configured_mcp_node(self):
        from app.services.flows import validate
        spec = {"nodes": [{"id": "a", "type": "tool", "label": "HF", "config": {"tool_id": "mcp", "config": {"server_id": 1}}}],
                "edges": []}
        assert any("MCP server" in w for w in validate(spec))

    def test_recipes_only_use_servers_we_know_how_to_register(self):
        from app.api.connections import CATALOG, OWN, RECIPES
        known = {c["key"] for c in CATALOG} | {OWN}
        for r in RECIPES:
            assert set(r["usa"]) <= known, r["key"]
            assert all(r["usa"].values()), "each server with its short list of tools"

    def test_ingest_file_doesnt_leave_the_inbox_folder(self, monkeypatch):
        import anyio
        from mcp.server.mcpserver.exceptions import ToolError

        from app.api import mcp_server
        from app.db import RagCollection
        monkeypatch.setattr(mcp_server, "_collection", lambda name: RagCollection(id=1, name=name, embed_model="x"))
        with pytest.raises(ToolError, match="data/inbox"):
            anyio.run(mcp_server.ingest_file, "c", "../llmtrainer.db")


    def test_an_mcp_nodes_json_can_be_used_by_field(self):
        from app.services.flows import render
        assert render("{t_json[text]}", {"t_json": {"text": "hello"}}) == "hello"
        # If the field doesn't exist the flow doesn't blow up: the template stays as is
        assert render("{t_json[text]}", {}) == "{t_json[text]}"
