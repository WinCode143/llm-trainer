"""Pruebas de las piezas que se romperían en silencio: normalización de datasets, troceado, BM25,
recompensas de GRPO, extracción de documentos y las herramientas de los agentes.

Ejecutar:  cd backend && .venv/bin/python -m pytest -q
No cargan modelos ni GPU: son rápidas (segundos).
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
def test_detecta_y_normaliza_alpaca():
    fmt, rows = ds.normalize([{"instruction": "¿Capital de Francia?", "output": "París"}])
    assert fmt == "chat"
    assert rows[0]["messages"] == [{"role": "user", "content": "¿Capital de Francia?"},
                                   {"role": "assistant", "content": "París"}]


def test_normaliza_sharegpt_y_system_prompt():
    fmt, rows = ds.normalize([{"conversations": [{"from": "human", "value": "hola"}, {"from": "gpt", "value": "¡hola!"}]}],
                             system_prompt="Eres útil")
    assert fmt == "chat" and rows[0]["messages"][0] == {"role": "system", "content": "Eres útil"}


def test_normaliza_preferencias_con_listas():
    fmt, rows = ds.normalize([{"prompt": "p", "chosen": [{"role": "assistant", "content": "buena"}],
                               "rejected": [{"role": "assistant", "content": "mala"}]}])
    assert fmt == "preference" and rows[0]["chosen"] == "buena" and rows[0]["rejected"] == "mala"


def test_descarta_ejemplos_sin_respuesta():
    _, rows = ds.normalize([{"instruction": "sin respuesta", "output": ""},
                            {"instruction": "ok", "output": "respuesta"}])
    assert len(rows) == 1


def test_formato_desconocido_falla():
    with pytest.raises(ValueError):
        ds.normalize([{"campo_raro": 1}])


def test_lee_csv_y_txt():
    assert ds._read_records("a.csv", b"instruction,output\nhola,adios\n")[0]["output"] == "adios"
    assert len(ds._read_records("a.txt", b"parrafo uno\n\nparrafo dos")) == 2


def test_split_reparte_todo():
    rows = [{"text": str(i)} for i in range(100)]
    train, ev = ds.split(rows, 0.2)
    assert len(train) == 80 and len(ev) == 20 and len(train) + len(ev) == len(rows)


# ---------------------------------------------------------------- troceado
def test_chunking_respeta_tamano_y_solapa():
    texto = ". ".join(f"Frase número {i} con algo de relleno" for i in range(60))
    chunks = rag.chunk_text(texto, 200, 40)
    assert chunks and all(len(c) <= 260 for c in chunks)
    assert len(chunks) > 1


def test_secciones_anidan_por_nivel():
    secs, _ = rag.split_sections("## A\na\n### A1\na1\n## B\nb")
    assert [s for s, _ in secs] == ["A", "A › A1", "B"]


def test_plantilla_qa_separa_preguntas():
    pages = [("¿Uno?\nrespuesta uno\n¿Dos?\nrespuesta dos", {})]
    chunks = rag.make_chunks("faq.md", pages, 800, 0, "qa")
    assert len(chunks) == 2


def test_plantilla_laws_separa_articulos():
    pages = [("Artículo 1. Objeto\nx\nArtículo 2. Horario\ny", {})]
    assert len(rag.make_chunks("ley.txt", pages, 800, 0, "laws")) == 2


def test_plantilla_table_una_fila_por_fragmento():
    chunks = rag.make_chunks("t.csv", [("a,b\n1,2\n3,4", {})], 800, 0, "table")
    assert [c for c, _ in chunks] == ["a: 1; b: 2", "a: 3; b: 4"]


def test_html_ignora_scripts_y_conserva_titulos():
    html = "<html><head><style>x{}</style></head><body><nav><img src=x></nav><h2>Título</h2><p>Texto visible</p><script>1</script></body></html>"
    md = rag._html(html)
    assert "## Título" in md and "Texto visible" in md and "x{}" not in md and "1" not in md.split("Texto visible")[1]


# ---------------------------------------------------------------- BM25
def test_bm25_ignora_acentos_y_ordena():
    idx = rag.BM25(["a", "b"], ["El reactor opera a 612 grados", "La reina Ilsa fundó Freedonia"],
                   sources=["s1", "s2"])
    assert idx.search("¿quién fundó FREEDONIA?", 2)[0][0] == "b"


def test_bm25_filtra_por_documento():
    idx = rag.BM25(["a", "b"], ["texto sobre gatos", "texto sobre gatos"], sources=["uno", "dos"])
    assert [i for i, _ in idx.search("gatos", 5, allowed={"dos"})] == ["b"]


# ---------------------------------------------------------------- recompensas GRPO
def _c(text):
    return [[{"role": "assistant", "content": text}]]


def test_recompensa_correccion():
    alta = rewards.correctness(None, _c("La capital es Villa Aurora"), ["La capital es Villa Aurora"])[0]
    baja = rewards.correctness(None, _c("No lo sé"), ["La capital es Villa Aurora"])[0]
    assert alta > 1.5 > baja


def test_recompensa_formato_razonamiento():
    assert rewards.reasoning_format(None, _c("<think>lo pienso bastante</think> Respuesta"))[0] == 1.0
    assert rewards.reasoning_format(None, _c("Respuesta directa"))[0] == 0.0


def test_recompensa_longitud_penaliza_el_exceso():
    corto = rewards.make_length(10)(None, _c(" ".join(["x"] * 10)))[0]
    largo = rewards.make_length(10)(None, _c(" ".join(["x"] * 25)))[0]
    assert corto == 1.0 and largo == 0.0


# ---------------------------------------------------------------- herramientas de agentes
def test_calculadora_admite_porcentajes_y_rechaza_codigo():
    assert T.TOOLS["calculator"]["run"]({}, {"expression": "12% de 3.382.662"}).startswith("405")
    assert "Error" in T.TOOLS["calculator"]["run"]({}, {"expression": "__import__('os').listdir()"})


def test_sql_solo_lectura():
    run = T.TOOLS["sql_query"]["run"]
    assert "Error" in run({"url": "sqlite:///:memory:"}, {"query": "delete from x"})
    assert "Error" in run({"url": "sqlite:///:memory:"}, {"query": "select 1; drop table x"})


def test_archivos_no_salen_de_la_carpeta(tmp_path):
    cfg = {"root_dir": str(tmp_path)}
    (tmp_path / "dentro.txt").write_text("ok")
    assert T.TOOLS["read_file"]["run"](cfg, {"path": "dentro.txt"}) == "ok"
    assert "Error" in T.TOOLS["read_file"]["run"](cfg, {"path": "../../etc/passwd"})
    assert "Error" in T.TOOLS["write_file"]["run"](cfg, {"path": "../fuera.txt", "content": "x"})


def test_http_exige_dominios_autorizados():
    assert "no está autorizado" in T.TOOLS["http_request"]["run"]({"allowed_hosts": "ok.com"}, {"url": "https://malo.com/x"})


# ---------------------------------------------------------------- grafo y agentes
def test_grafo_recupera_json_cortado():
    truncado = '{"entidades": [{"nombre": "Ana", "tipo": "persona", "descripcion": "x"}, {"nombre": "Luis"'
    assert graph._parse(truncado)["entidades"][0]["nombre"] == "Ana"


def test_grafo_normaliza_nombres():
    assert graph.norm("El Imperio Romano") == "imperio romano"


def test_parseo_de_llamadas_a_herramientas():
    from app.services.agents import parse_calls, visible_text
    texto = 'Voy a mirarlo <tool_call>{"name": "buscar_web", "arguments": {"query": "x"}}</tool_call>'
    calls = parse_calls(texto)
    assert calls == [{"name": "buscar_web", "arguments": {"query": "x"}}]
    assert visible_text(texto) == "Voy a mirarlo"


def test_parseo_acepta_bloque_json():
    from app.services.agents import parse_calls
    assert parse_calls('```json\n{"name": "calcular", "arguments": {"expression": "2+2"}}\n```')[0]["name"] == "calcular"


# ---------------------------------------------------------------- aislamiento del código
def test_sandbox_disponible_o_avisa():
    from app.services import sandbox
    st = sandbox.status()
    assert st["backend"] in ("bubblewrap", "ninguno") and isinstance(st["available"], bool)


@pytest.mark.skipif(not __import__("app.services.sandbox", fromlist=["x"]).available(),
                    reason="bubblewrap no disponible")
class TestSandbox:
    def test_ejecuta_codigo_normal(self):
        from app.services import sandbox
        assert sandbox.run_python("print(2**10)")["output"] == "1024"

    def test_no_ve_los_datos_del_proyecto(self):
        from app.config import settings
        from app.services import sandbox
        r = sandbox.run_python(f"import os; print(os.path.exists({str(settings.db_path)!r}))")
        assert r["output"] == "False"

    def test_lo_que_escribe_no_sale(self, tmp_path):
        from app.services import sandbox
        destino = tmp_path / "fuga.txt"
        sandbox.run_python(f"open({str(destino)!r}, 'w').write('x')")
        assert not destino.exists()

    def test_sin_red_por_defecto(self):
        from app.services import sandbox
        r = sandbox.run_python("import urllib.request; urllib.request.urlopen('https://example.com', timeout=5)")
        assert "Error" in r["error"] or "error" in r["error"].lower()

    def test_carpeta_de_datos_en_solo_lectura(self, tmp_path):
        from app.services import sandbox
        (tmp_path / "dato.txt").write_text("hola")
        r = sandbox.run_python("print(open('/datos/dato.txt').read())", ro_dir=str(tmp_path))
        assert r["output"] == "hola"
        r2 = sandbox.run_python("open('/datos/nuevo.txt','w').write('x')", ro_dir=str(tmp_path))
        assert not (tmp_path / "nuevo.txt").exists() and r2["error"]

    def test_limite_de_tiempo(self):
        from app.services import sandbox
        assert sandbox.run_python("while True: pass", timeout_s=3)["timeout"] is True


# --------------------------------------------------------------------- flujos
class TestFlows:
    """El compilador de flujos: plantillas válidas, condiciones seguras y avisos de validación."""

    def test_plantillas_compilan(self):
        """Las plantillas tienen que compilar. Los únicos avisos admitidos son los de elegir una
        colección o un servidor MCP: dependen de lo que tenga cada uno y se rellenan al usarlas."""
        from app.services import flows
        for key, tpl in flows.TEMPLATES.items():
            avisos = [a for a in flows.validate(tpl["spec"]) if "colección" not in a and "servidor MCP" not in a]
            assert not avisos, f"la plantilla {key} genera avisos inesperados: {avisos}"
            flows.compile_flow(tpl["spec"])  # no debe lanzar

    def test_exportar_a_codigo_produce_python_valido(self):
        """El exportador es la puerta de salida: si genera código roto, no sirve de nada."""
        import ast
        from app.services import flows
        for key, tpl in flows.TEMPLATES.items():
            code = flows.to_python(tpl["spec"], key)
            ast.parse(code)  # lanza SyntaxError si está mal
            assert "StateGraph" in code and "def construir()" in code

    def test_condiciones_del_router_se_traducen_al_exportar(self):
        """En el editor se escribe `apoyada == True`; en el código exportado esa variable vive en el estado."""
        from app.services.flows import _cond_py
        assert _cond_py("apoyada == True") == "v.get('apoyada') == True"
        assert _cond_py("len(texto) > 3") == "len(v.get('texto')) > 3"

    def test_listas_se_renderizan_legibles(self):
        """Una lista puesta en un prompt no puede salir como ['a', 'b'] con corchetes."""
        from app.services.flows import render
        assert render("{xs}", {"xs": ["pan", "leche"]}) == "1. pan\n2. leche"

    def test_para_cada_convierte_texto_en_lista(self):
        from app.services.flows import _as_list
        assert _as_list("uno\ndos\n\ntres") == ["uno", "dos", "tres"]
        assert _as_list(["a", "b"]) == ["a", "b"]

    def test_render_variable_ausente_no_rompe(self):
        from app.services.flows import render
        assert render("hola {quien}, {falta}", {"quien": "mundo"}) == "hola mundo, "

    def test_condiciones(self):
        from app.services.flows import eval_condition
        assert eval_condition("apoyada == True", {"apoyada": True})
        assert eval_condition("len(texto) > 3", {"texto": "abcd"})
        assert not eval_condition("tipo in opciones", {"tipo": "x", "opciones": ["a"]})
        # nada de llamadas arbitrarias ni acceso a atributos
        assert not eval_condition("__import__('os').system('ls')", {})
        assert not eval_condition("texto.upper()", {"texto": "a"})

    def test_validate_detecta_nodos_sueltos(self):
        from app.services.flows import validate
        spec = {"nodes": [{"id": "a", "type": "input"}, {"id": "b", "type": "output"}], "edges": []}
        assert validate(spec)


# ---------------------------------------------------------- modelos ayudantes
class TestProviders:
    """El punto único de generación y el manejo de las claves."""

    def test_complete_sin_modelo_ni_ayudante_avisa(self, monkeypatch):
        from app.services import providers
        from app.services.inference import manager
        monkeypatch.setattr(manager, "current", None)
        with pytest.raises(RuntimeError, match="ayudante"):
            providers.complete(None, "hola")

    def test_ayudante_inexistente_es_explicito(self):
        from app.services import providers
        with pytest.raises(RuntimeError, match="ya no existe"):
            providers.complete(999999, "hola")

    def test_complete_json_rescata_json_envuelto(self, monkeypatch):
        """Los modelos suelen devolver el JSON rodeado de texto; hay que sacarlo igual."""
        from app.services import providers
        monkeypatch.setattr(providers, "complete", lambda *a, **k: 'Claro:\n{"score": 4}\n¡Espero que sirva!')
        assert providers.complete_json(None, "…") == {"score": 4}

    def test_complete_json_devuelve_none_si_no_hay_json(self, monkeypatch):
        from app.services import providers
        monkeypatch.setattr(providers, "complete", lambda *a, **k: "no tengo ni idea")
        assert providers.complete_json(None, "…") is None

    def test_las_claves_no_vuelven_enteras(self):
        """El listado que ve la interfaz sólo puede llevar la clave enmascarada."""
        from app.db import Provider, engine
        from app.services import providers
        from sqlmodel import Session
        with Session(engine) as s:
            p = Provider(name="_test_mask", kind="openai", base_url="http://x/v1", api_key="sk-supersecreto1234")
            s.add(p)
            s.commit()
            pid = p.id
        try:
            row = next(r for r in providers.listing() if r["id"] == pid)
            assert row["api_key"] == "…1234"
            assert "supersecreto" not in row["api_key"]
            assert row["has_key"] is True
        finally:
            with Session(engine) as s:
                s.delete(s.get(Provider, pid))
                s.commit()


# ------------------------------------------------------- limpieza de trabajos
class TestJobCleanup:
    """Lo que nunca debe borrarse: trabajos en curso y archivos de un modelo registrado."""

    def _job(self, s, **kw):
        from app.db import Job
        j = Job(name=kw.pop("name", "t"), type="train", config=kw.pop("config", {}), **kw)
        s.add(j)
        s.commit()
        s.refresh(j)
        return j

    def test_no_borra_trabajos_en_curso(self, tmp_path):
        from app.api.jobs import CleanupBody, cleanup
        from app.db import Job, Session, engine
        with Session(engine) as s:
            vivo = self._job(s, name="_t_vivo", status="running")
            muerto = self._job(s, name="_t_muerto", status="failed")
            try:
                ids = {j["id"] for j in cleanup(CleanupBody(dry_run=True), s)["jobs"]}
                assert muerto.id in ids
                assert vivo.id not in ids, "un trabajo en curso no puede entrar en la limpieza"
            finally:
                for j in (vivo, muerto):
                    if (row := s.get(Job, j.id)):
                        s.delete(row)
                s.commit()

    def test_protege_los_archivos_de_un_modelo_registrado(self, tmp_path):
        from app.api.jobs import _delete_one, _output_info
        from app.db import Job, Model, Session, engine
        out = tmp_path / "salida"
        out.mkdir()
        (out / "adapter.bin").write_bytes(b"x" * 1000)
        with Session(engine) as s:
            j = self._job(s, name="_t_prot", status="completed", config={"output_dir": str(out)})
            m = Model(name="_t_modelo", kind="lora", path=str(out / "final"))
            s.add(m)
            s.commit()
            try:
                assert _output_info(s.get(Job, j.id), s)["models"] == ["_t_modelo"]
                r = _delete_one(s.get(Job, j.id), s, delete_output=True)
                s.commit()
                assert r["kept_output"] is True
                assert out.exists(), "no se pueden borrar archivos de los que depende un modelo"
            finally:
                with Session(engine) as s2:
                    if (row := s2.get(Model, m.id)):
                        s2.delete(row)
                    s2.commit()

    def test_borra_los_archivos_cuando_nadie_depende(self, tmp_path):
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


# ------------------------------------------- métricas de evaluación y síntesis
class TestEvaluationMetrics:
    """Estas funciones producen los números con los que se decide qué modelo es mejor.
    Un fallo aquí no se ve: sólo da una métrica algo distinta y la conclusión sale mal."""

    def test_strip_think_quita_el_razonamiento(self):
        from app.services.evaluation import strip_think
        assert strip_think("<think>dudo mucho</think>La respuesta es 42") == "La respuesta es 42"
        assert strip_think("sin etiquetas") == "sin etiquetas"

    def test_strip_think_con_bloque_sin_cerrar(self):
        """Si el modelo se queda sin tokens dentro de <think>, no puede colarse el razonamiento como respuesta."""
        from app.services.evaluation import strip_think
        out = strip_think("<think>estaba pensando y me quedé sin")
        assert "estaba pensando" not in out

    def test_strip_think_con_cierre_suelto(self):
        """Qwen3.5 en GGUF abre el bloque desde la plantilla, así que sólo emite el cierre.
        Sin esto, el razonamiento entero se colaba como si fuera la respuesta y contaminaba
        todas las métricas, que es justo lo que las métricas no pueden permitirse."""
        from app.services.evaluation import strip_think
        crudo = "We need answer in Spanish. Two sentences. Good.\n</think>\n\nLoRA ajusta pocas matrices."
        assert strip_think(crudo) == "LoRA ajusta pocas matrices."

    def test_palabras_clave(self):
        from app.services.evaluation import _contains
        assert _contains("El plazo es de 30 días naturales", ["30", "días"]) == 1.0
        assert _contains("El plazo es de 30 días", ["30", "meses"]) == 0.5
        assert _contains("da igual", []) is None

    def test_palabras_clave_ignora_mayusculas(self):
        from app.services.evaluation import _contains
        assert _contains("MADRID es la capital", ["madrid"]) == 1.0

    def test_resumen_promedia_por_variante_y_omite_ausentes(self):
        from app.services.evaluation import summarize
        res = [
            {"variant": "m1 (RAG)", "model_name": "m1", "rag": True, "judge_score": 8, "similarity": 0.9},
            {"variant": "m1 (RAG)", "model_name": "m1", "rag": True, "judge_score": 6, "similarity": None},
            {"variant": "m1", "model_name": "m1", "rag": False, "judge_score": 4},
        ]
        out = summarize(res)
        assert out["m1 (RAG)"]["judge_score"] == 7.0
        # La segunda fila no tiene similitud: no debe contar como 0 y hundir la media
        assert out["m1 (RAG)"]["similarity"] == 0.9
        assert out["m1 (RAG)"]["n"] == 2
        assert out["m1"]["judge_score"] == 4.0

    def test_calibracion_del_juez(self):
        from app.services.evaluation import judge_agreement
        # judge_score va de 1 a 10 y tus notas de 1 a 5: la función divide entre 2 antes de comparar
        perfecto = [{"judge_score": n * 2, "rating": n} for n in (1, 2, 3, 4, 5)]
        r = judge_agreement(perfecto)
        assert r["spearman"] == 1.0 and r["mae"] == 0.0 and r["judge_bias"] == 0.0

    def test_calibracion_detecta_juez_generoso(self):
        from app.services.evaluation import judge_agreement
        generoso = [{"judge_score": n * 2 + 2, "rating": n} for n in (1, 2, 3, 4)]
        r = judge_agreement(generoso)
        assert r["judge_bias"] == 1.0, "un juez que puntúa de más debe salir con sesgo positivo"

    def test_calibracion_sin_datos_suficientes(self):
        from app.services.evaluation import judge_agreement
        assert judge_agreement([{"judge_score": 8, "rating": 4}])["spearman"] is None


class TestSynth:
    """Generación de datasets sintéticos: lo que se cuela aquí acaba entrenando un modelo."""

    def test_extrae_pares_de_un_json_envuelto_en_texto(self):
        from app.services.synth import _pairs
        out = _pairs('Claro:\n[{"question": "¿Qué es X?", "answer": "Es Y"}]\n¡Listo!')
        assert out == [{"question": "¿Qué es X?", "answer": "Es Y"}]

    def test_descarta_pares_incompletos(self):
        from app.services.synth import _pairs
        out = _pairs('[{"question": "sin respuesta"}, {"answer": "sin pregunta"}, {"question": "a", "answer": "b"}]')
        assert out == [{"question": "a", "answer": "b"}]

    def test_json_invalido_no_revienta(self):
        from app.services.synth import _pairs
        assert _pairs("el modelo se fue por las ramas") == []

    def test_dedupe_quita_preguntas_casi_iguales(self, monkeypatch):
        """Va por CPU a propósito: en GPU el test fallaba o no según lo que hubiera cargado en ese
        momento, y una prueba que depende de eso no prueba nada."""
        from app.services import rag as rag_svc
        monkeypatch.setattr(rag_svc, "_device", lambda: "cpu")
        rag_svc.unload_models()
        from app.services.synth import _dedupe
        rows = [{"question": "¿Cuál es la capital de Francia?", "answer": "París"},
                {"question": "¿Cuál es la capital de Francia?", "answer": "París"},
                {"question": "¿Cuántos huesos tiene el cuerpo humano?", "answer": "206"}]
        out, quitadas = _dedupe(rows, 0.92)
        assert quitadas == 1 and len(out) == 2


class TestBorradoDeArchivos:
    """La comprobación de «esta ruta es mía» tiene que sobrevivir a los enlaces simbólicos:
    data/outputs apunta a otro disco, y compararla sin resolver hacía que no borrase nunca nada."""

    def test_borra_dentro_de_las_carpetas_gestionadas(self, tmp_path, monkeypatch):
        from app.api import models as api
        from app.config import settings
        raiz = tmp_path / "data"
        (raiz / "outputs" / "algo").mkdir(parents=True)
        (raiz / "outputs" / "algo" / "peso.bin").write_bytes(b"x" * 100)
        monkeypatch.setattr(settings, "data_dir", raiz)
        assert api._borrar_si_es_nuestro(raiz / "outputs" / "algo") is True
        assert not (raiz / "outputs" / "algo").exists()

    def test_sigue_el_enlace_a_otro_disco(self, tmp_path, monkeypatch):
        from app.api import models as api
        from app.config import settings
        real = tmp_path / "otro_disco" / "outputs" / "modelo"
        real.mkdir(parents=True)
        (real / "peso.bin").write_bytes(b"x" * 100)
        raiz = tmp_path / "data"
        raiz.mkdir()
        (raiz / "outputs").symlink_to(tmp_path / "otro_disco" / "outputs")
        monkeypatch.setattr(settings, "data_dir", raiz)
        # La ruta va por el enlace, como las que guarda la base de datos
        assert api._borrar_si_es_nuestro(raiz / "outputs" / "modelo") is True
        assert not real.exists(), "el enlace a otro disco debe seguirse, no ignorarse"

    def test_se_niega_fuera_de_sus_carpetas(self, tmp_path, monkeypatch):
        from app.api import models as api
        from app.config import settings
        fuera = tmp_path / "ajeno"
        fuera.mkdir()
        (fuera / "importante.txt").write_text("no tocar")
        monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
        (tmp_path / "data").mkdir()
        assert api._borrar_si_es_nuestro(fuera) is False
        assert (fuera / "importante.txt").exists()


class TestPurgeOutputs:
    """Borrar pesos conservando el historial: los gigas se van, las métricas se quedan."""

    def test_borra_los_pesos_pero_no_la_fila(self, tmp_path):
        from app.api.jobs import PurgeBody, purge_outputs
        from app.db import Job, Session, engine
        out = tmp_path / "salida"
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
                assert not out.exists(), "los pesos tenían que borrarse"
                vuelto = s.get(Job, jid)
                assert vuelto is not None, "la fila del trabajo NO puede desaparecer"
                assert vuelto.result["train_loss"] == 0.42, "las métricas tienen que sobrevivir"
                assert vuelto.config["outputs_purged"] is True
            finally:
                if (row := s.get(Job, jid)):
                    s.delete(row)
                    s.commit()

    def test_dry_run_no_toca_nada(self, tmp_path):
        from app.api.jobs import PurgeBody, purge_outputs
        from app.db import Job, Session, engine
        out = tmp_path / "salida"
        out.mkdir()
        (out / "peso.bin").write_bytes(b"x" * 1000)
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


class TestRegistroDeDescargas:
    """Nombres al registrar un modelo descargado: dos descargas a la vez chocaban."""

    def test_busca_el_primer_nombre_libre(self):
        from app.api.models import _nombre_libre
        from app.db import Model, Session, engine
        creados = []
        with Session(engine) as s:
            try:
                assert _nombre_libre(s, "_t_libre") == "_t_libre"
                for esperado in ("_t_libre-2", "_t_libre-3"):
                    m = Model(name=_nombre_libre(s, "_t_libre"), kind="hf", path=f"x{len(creados)}")
                    s.add(m)
                    s.commit()
                    s.refresh(m)
                    creados.append(m.id)
                    assert _nombre_libre(s, "_t_libre") == esperado
            finally:
                for i in creados:
                    if (row := s.get(Model, i)):
                        s.delete(row)
                s.commit()


class TestVisionGGUF:
    """Un GGUF ve imágenes sólo si se le carga el mmproj y se traduce el formato de mensajes."""

    def test_traduce_el_formato_de_imagen(self):
        """llama.cpp espera image_url; con nuestro formato la imagen viajaba y el modelo
        contestaba que no podía verla."""
        from app.services.inference import _para_llamacpp
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "¿qué ves?"},
            {"type": "image", "url": "data:image/png;base64,AAA"}]}]
        out = _para_llamacpp(msgs)[0]["content"]
        assert out[0] == {"type": "text", "text": "¿qué ves?"}
        assert out[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}

    def test_sin_vision_las_imagenes_se_avisan(self):
        from app.services.inference import _text_only
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "mira"}, {"type": "image", "url": "x"}]}]
        c = _text_only(msgs)[0]["content"]
        assert "mira" in c and "no admite visión" in c

    def test_solo_activa_vision_con_mmproj_al_lado(self, tmp_path):
        from app.services.inference import manager
        suelto = tmp_path / "gemma-4-algo.gguf"
        suelto.write_bytes(b"x")
        assert manager._vision_handler(str(suelto)) == (None, None), "sin mmproj no hay visión"

    def test_no_adivina_el_manejador_de_una_familia_desconocida(self, tmp_path):
        from app.services.inference import manager
        (tmp_path / "mmproj-F16.gguf").write_bytes(b"x")
        raro = tmp_path / "modelo-inventado-7b.gguf"
        raro.write_bytes(b"x")
        assert manager._vision_handler(str(raro)) == (None, None)


class TestPlantillasYContexto:
    """Las plantillas de dataset y el aviso de contexto recortado."""

    def test_las_plantillas_se_normalizan_al_formato_correcto(self):
        """Si una plantilla no pasa por el normalizador, el dataset que crea no se puede entrenar."""
        from app.services import datasets as ds
        from app.services.plantillas import PLANTILLAS
        for key, t in PLANTILLAS.items():
            fmt, filas = ds.normalize(t["filas"], t["formato"])
            assert fmt == t["formato"], f"{key}: sale como {fmt} y dice ser {t['formato']}"
            assert len(filas) == len(t["filas"]), f"{key}: se perdieron filas al normalizar"

    def test_cada_plantilla_dice_cuantos_ejemplos_y_por_que(self):
        from app.services.plantillas import listado
        for t in listado():
            assert t["cuantos"] and t["consejo"], f"{t['key']} sin guía"
            assert t["metodo"] in ("sft", "dpo", "grpo")

    def test_avisa_cuando_el_contexto_no_cabe(self, monkeypatch):
        """Sin aviso, el modelo responde con la mitad de los fragmentos y parece que el RAG falló."""
        from app.services import rag
        from app.services.inference import Loaded, manager
        monkeypatch.setattr(manager, "current",
                            Loaded(1, "x", "gguf", "p", None, "q", "llama.cpp", n_ctx=2048))
        assert rag.contexto_excedido("corto", "¿?") is None
        aviso = rag.contexto_excedido("x" * 9000, "¿?")
        assert aviso and "2048" in aviso

    def test_sin_modelo_cargado_no_avisa_de_nada(self, monkeypatch):
        from app.services import rag
        from app.services.inference import manager
        monkeypatch.setattr(manager, "current", None)
        assert rag.contexto_excedido("x" * 99999) is None


class TestCopiasDeSeguridad:
    """El módulo que protege lo irreemplazable no tenía ninguna prueba."""

    def test_crear_listar_y_podar(self, tmp_path, monkeypatch):
        from app.config import settings
        from app.services import backup
        monkeypatch.setattr(settings, "data_dir", tmp_path)
        monkeypatch.setattr(settings, "backup_keep", 2)
        (tmp_path / "llmtrainer.db").write_bytes(b"base de datos falsa")
        (tmp_path / "datasets").mkdir()
        (tmp_path / "datasets" / "a.jsonl").write_text('{"x": 1}')

        for _ in range(3):
            backup.create(note="prueba")
        # backup_keep=2: la más vieja tiene que haberse ido sola
        assert len(backup.listing()) == 2, "prune no respetó el límite"
        assert all(b["note"] == "prueba" for b in backup.listing())

    def test_la_copia_lleva_lo_que_dice_llevar(self, tmp_path, monkeypatch):
        import tarfile
        from app.config import settings
        from app.services import backup
        monkeypatch.setattr(settings, "data_dir", tmp_path)
        (tmp_path / "llmtrainer.db").write_bytes(b"x")
        (tmp_path / "rag").mkdir()
        (tmp_path / "rag" / "chroma.sqlite3").write_bytes(b"y")
        # Los pesos NO deben entrar: son enormes y se pueden volver a generar
        (tmp_path / "outputs").mkdir()
        (tmp_path / "outputs" / "modelo.bin").write_bytes(b"z" * 1000)

        info = backup.create()
        with tarfile.open(backup.backups_dir() / info["file"]) as t:
            dentro = t.getnames()
        assert any("llmtrainer.db" in n for n in dentro)
        assert any("rag" in n for n in dentro)
        assert not any("outputs" in n for n in dentro), "los pesos no pueden entrar en la copia"

    def test_verificar_detecta_un_archivo_roto(self, tmp_path, monkeypatch):
        from app.config import settings
        from app.services import backup
        monkeypatch.setattr(settings, "data_dir", tmp_path)
        roto = backup.backups_dir() / "backup-20260101-000000.tar.gz"
        roto.write_bytes(b"esto no es un tar.gz")
        r = backup.verify(roto.name)
        assert r["ok"] is False and "dañado" in r["error"]

    def test_verificar_avisa_de_tablas_que_faltan(self, tmp_path, monkeypatch):
        """Una copia vieja no tiene las tablas nuevas: restaurarla te devuelve a antes de tenerlas."""
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
        assert "questionset" in r["missing_tables"], "debería avisar de las tablas ausentes"
        assert r["rows"]["dataset"] == 0


class TestTareas:
    """El registro de tareas en segundo plano: progreso, resultado, error y cancelación."""

    def test_una_tarea_que_va_bien(self):
        from app.services import tasks
        t = tasks.submit("test", "suma", lambda task: 2 + 2)
        for _ in range(100):
            if t.status != "running":
                break
            time.sleep(0.05)
        assert t.status == "completed" and t.result == 4 and t.progress == 1.0

    def test_una_tarea_que_falla_guarda_el_error(self):
        from app.services import tasks

        def revienta(task):
            raise ValueError("se rompió a propósito")

        t = tasks.submit("test", "falla", revienta)
        for _ in range(100):
            if t.status != "running":
                break
            time.sleep(0.05)
        assert t.status == "failed" and "se rompió a propósito" in t.error

    def test_la_cancelacion_se_ve_desde_dentro(self):
        """La bandera es lo único que tiene el trabajo para enterarse: si no llega, no se puede parar."""
        from app.services import tasks
        visto = []

        def largo(task):
            for _ in range(200):
                if task.cancel_requested:
                    visto.append(True)
                    return "cancelado"
                time.sleep(0.02)
            return "terminó entero"

        t = tasks.submit("test", "largo", largo)
        time.sleep(0.1)
        tasks.cancel(t.id)
        for _ in range(100):
            if t.status != "running":
                break
            time.sleep(0.05)
        assert visto == [True] and t.result == "cancelado"


class TestVllm:
    """vLLM vive en otro entorno y se habla por HTTP: lo que se puede probar sin GPU es el contrato."""

    def test_sabe_si_esta_instalado(self):
        from app.services import vllm_engine
        assert isinstance(vllm_engine.available(), bool)

    def test_el_cuerpo_de_la_peticion_respeta_los_parametros(self):
        from app.services.inference import GenParams
        from app.services.vllm_engine import VllmServer
        s = VllmServer.__new__(VllmServer)
        s.served_name = "base"
        p = GenParams(max_new_tokens=128, temperature=0.4, top_p=0.8, top_k=20, seed=7, enable_thinking=False)
        body = VllmServer._payload(s, [{"role": "user", "content": "hola"}], p, stream=True, template_kwargs=None)
        assert body["max_tokens"] == 128 and body["temperature"] == 0.4 and body["seed"] == 7
        assert body["stream"] is True
        # enable_thinking viaja como argumento de la plantilla, no como parámetro suelto
        assert body["chat_template_kwargs"]["enable_thinking"] is False

    def test_las_herramientas_viajan_en_formato_openai(self):
        from app.services.inference import GenParams
        from app.services.vllm_engine import VllmServer
        s = VllmServer.__new__(VllmServer)
        s.served_name = "lora"
        esquema = {"name": "buscar", "description": "…", "parameters": {}}
        body = VllmServer._payload(s, [{"role": "user", "content": "x"}], GenParams(),
                                   stream=False, template_kwargs={"tools": [esquema]})
        assert body["tools"] == [{"type": "function", "function": esquema}]


class TestMcpCliente:
    """El cliente MCP estaba roto desde que la librería pasó a 2.x y nadie lo vio."""

    def test_saca_el_error_real_de_dentro_del_taskgroup(self):
        from app.services.tools import error_real
        try:
            raise BaseExceptionGroup("unhandled errors in a TaskGroup", [FileNotFoundError("npx no existe")])
        except BaseExceptionGroup as e:
            assert error_real(e) == "FileNotFoundError: npx no existe"

    def test_url_suelta_se_trata_como_http(self):
        from app.services.tools import _mcp_server
        assert _mcp_server("http://x/mcp/") == {"transport": "http", "url": "http://x/mcp/"}

    def test_servidor_inexistente_es_explicito(self):
        from app.services.tools import _mcp_server
        with pytest.raises(RuntimeError, match="ya no existe"):
            _mcp_server(999999)

    def test_las_claves_no_vuelven_enteras(self):
        from app.api.connections import _mask
        assert _mask({"GITHUB_TOKEN": "ghp_supersecreto9876"}) == {"GITHUB_TOKEN": "…9876"}


class TestDisparadores:
    """Lo que abre la app al exterior: quién puede lanzar qué, y cuándo."""

    def _bot(self, permitidos):
        from app.db import Trigger
        return Trigger(id=1, name="bot", kind="telegram", target_kind="flow", target_id=1,
                       config={"bot_token": "x", "allowed_chat_ids": permitidos})

    def _msg(self, chat, texto="hola"):
        return {"update_id": 1, "message": {"chat": {"id": chat}, "text": texto}}

    def test_sin_lista_no_ejecuta_nada_y_dice_el_chat_id(self, monkeypatch):
        from app.services import triggers as tr
        ejecutado = []
        monkeypatch.setattr(tr, "ejecutar", lambda *a, **k: ejecutado.append(1) or {"estado": "ok", "salida": "x"})
        enviados = []
        tr.procesar_update(self._bot([]), self._msg(555), lambda m, **d: enviados.append((m, d)))
        assert not ejecutado, "un bot sin lista de permitidos no puede ejecutar nada"
        assert "555" in enviados[0][1]["text"], "debe decirle su chat_id para poder añadirlo"

    def test_un_desconocido_se_ignora_en_silencio(self, monkeypatch):
        from app.services import triggers as tr
        monkeypatch.setattr(tr, "ejecutar", lambda *a, **k: pytest.fail("no debía ejecutarse"))
        enviados = []
        tr.procesar_update(self._bot([111]), self._msg(999), lambda m, **d: enviados.append(m))
        assert enviados == [], "a un chat ajeno no se le contesta: ni siquiera confirma que el bot existe"

    def test_un_chat_permitido_ejecuta_y_recibe_la_respuesta(self, monkeypatch):
        from app.services import triggers as tr
        monkeypatch.setattr(tr, "ejecutar", lambda t, texto, o: {"estado": "ok", "salida": f"eco: {texto}"})
        enviados = []
        tr.procesar_update(self._bot([111]), self._msg(111, "hola"), lambda m, **d: enviados.append((m, d)))
        textos = [d["text"] for m, d in enviados if m == "sendMessage"]
        assert textos == ["eco: hola"]

    def test_una_aprobacion_llega_con_botones_y_se_reanuda(self, monkeypatch):
        from app.services import triggers as tr
        monkeypatch.setattr(tr, "ejecutar", lambda t, texto, o: {
            "estado": "espera", "salida": {"message": "¿Guardo?", "preview": "borrador"}, "thread_id": "th1", "flow_id": 7})
        enviados = []
        bot = self._bot([111])
        tr.procesar_update(bot, self._msg(111, "redacta"), lambda m, **d: enviados.append((m, d)))
        teclado = [d for m, d in enviados if m == "sendMessage" and "reply_markup" in d][0]
        botones = teclado["reply_markup"]["inline_keyboard"][0]
        ok = botones[0]["callback_data"]
        assert ok.startswith("ok:") and botones[1]["callback_data"].startswith("no:")

        reanudado = {}
        monkeypatch.setattr(tr, "_correr_flujo", lambda fid, texto, thread_id=None, resume=None:
                            reanudado.update(fid=fid, thread=thread_id, resume=resume) or {"estado": "ok", "salida": "guardado"})
        cq = {"update_id": 2, "callback_query": {"id": "c1", "data": ok, "message": {"chat": {"id": 111}}}}
        tr.procesar_update(bot, cq, lambda m, **d: enviados.append((m, d)))
        assert reanudado == {"fid": 7, "thread": "th1", "resume": {"rejected": False, "comment": "desde Telegram", "edited": None}}

    def test_el_boton_de_un_chat_ajeno_no_reanuda(self, monkeypatch):
        from app.services import triggers as tr
        monkeypatch.setattr(tr, "_correr_flujo", lambda *a, **k: pytest.fail("no debía reanudar"))
        tr._pendientes["k1"] = {"flow_id": 1, "thread_id": "t", "texto": "x"}
        cq = {"update_id": 3, "callback_query": {"id": "c", "data": "ok:k1", "message": {"chat": {"id": 999}}}}
        tr.procesar_update(self._bot([111]), cq, lambda m, **d: None)
        assert "k1" in tr._pendientes, "la aprobación sigue pendiente para su dueño"
        tr._pendientes.pop("k1")

    def test_cuando_toca_un_programado(self):
        from datetime import datetime
        from app.services.triggers import toca
        a = datetime(2026, 9, 24, 8, 30)
        assert toca({"cada_min": 60}, None, a)
        assert not toca({"cada_min": 60}, datetime(2026, 9, 24, 8, 0), a)
        assert toca({"cada_min": 60}, datetime(2026, 9, 24, 7, 29), a)
        assert toca({"diario": "08:00"}, datetime(2026, 9, 23, 8, 0), a), "hoy aún no había corrido"
        assert not toca({"diario": "08:00"}, datetime(2026, 9, 24, 8, 1), a), "ya corrió hoy"
        assert not toca({"diario": "09:00"}, None, a), "todavía no es la hora"

    def test_un_token_equivocado_no_encuentra_nada(self):
        from app.services.triggers import buscar_webhook
        assert buscar_webhook("token-que-no-existe") is None


class TestEvaluacionDeAgentes:
    """Cómo se decide si un agente acertó con la herramienta."""

    ALIAS = {"calcular": {"calcular", "calculator", "calculadora"},
             "buscar_documentos": {"buscar_documentos", "rag_search", "buscar en mis documentos (rag)"}}

    def test_el_id_del_catalogo_cuenta_aunque_el_modelo_vea_otro_nombre(self):
        """El modelo llama «calcular» y el caso dice «calculator»: eso es un acierto.
        Sin traducir nombres, la primera evaluación real dio 17% cuando era 67%."""
        from app.services.agent_eval import _coincide
        assert _coincide("calculator", ["calcular"], self.ALIAS)
        assert _coincide("rag_search", ["buscar_documentos"], self.ALIAS)
        assert not _coincide("calculator", ["buscar_documentos"], self.ALIAS)

    def test_ninguna_significa_no_usar_herramientas(self, monkeypatch):
        from app.services import agent_eval
        monkeypatch.setattr(agent_eval, "run_agent", lambda a, m: iter([{"type": "done", "final": "¡Hola!", "steps": 1}]))
        r = agent_eval.evaluar_caso({}, {"question": "hola", "tool": "ninguna"}, alias={})
        assert r["acierto"] and r["a_la_primera"]

    def test_acertar_tras_fallar_no_es_a_la_primera(self, monkeypatch):
        from app.services import agent_eval
        eventos = [{"type": "tool_call", "name": "buscar_documentos"}, {"type": "tool_call", "name": "calcular"},
                   {"type": "done", "final": "69104", "steps": 3}]
        monkeypatch.setattr(agent_eval, "run_agent", lambda a, m: iter(eventos))
        r = agent_eval.evaluar_caso({}, {"question": "1234*56", "tool": "calculator", "keywords": ["69104"]}, alias=self.ALIAS)
        assert r["acierto"] and not r["a_la_primera"] and r["claves"] == 1.0

    def test_el_resumen_agrupa_los_fallos(self):
        from app.services.agent_eval import resumir
        filas = [{"acierto": False, "a_la_primera": False, "esperada": "rag_search", "llamadas": [], "claves": None,
                  "pasos": 1, "segundos": 1.0, "error": None}] * 2 + \
                [{"acierto": True, "a_la_primera": True, "esperada": "calculator", "llamadas": ["calcular"], "claves": 1.0,
                  "pasos": 2, "segundos": 1.0, "error": None}]
        s = resumir(filas)
        assert s["herramienta_correcta"] == 0.33
        assert s["fallos_frecuentes"][0] == ("esperaba rag_search, usó ninguna", 2)


class TestFormatosDeLlamada:
    """Cada familia de modelos escribe las llamadas a herramientas a su manera."""

    def test_reconoce_el_formato_nativo_de_gemma4(self):
        from app.services.agents import parse_calls, visible_text
        crudo = ('<|channel>thought\n<channel|><|tool_call>call:mcp_list_directory'
                 '{"path": "/tmp/carpeta"}<tool_call|>')
        assert parse_calls(crudo) == [{"name": "mcp_list_directory", "arguments": {"path": "/tmp/carpeta"}}]
        assert visible_text(crudo) == "", "ni el canal de pensamiento ni la llamada pueden llegar al usuario"

    def test_gemma4_con_claves_sin_comillas(self):
        from app.services.agents import parse_calls
        crudo = '<|channel>thought\n<channel|><|tool_call>call:mcp_open_nodes{names:["Usuario"]}<tool_call|>'
        assert parse_calls(crudo) == [{"name": "mcp_open_nodes", "arguments": {"names": ["Usuario"]}}]
        crudo = '<|tool_call>call:buscar{query:<|"|>precio: litio, hoy<|"|>,k:3}<tool_call|>'
        assert parse_calls(crudo) == [{"name": "buscar", "arguments": {"query": "precio: litio, hoy", "k": 3}}]
        # A veces con dos puntos entre el nombre y los argumentos
        crudo = '<|tool_call>call:mcp_arxiv_search:{"max_results":5,"query":"GRPO"}<tool_call|>'
        assert parse_calls(crudo) == [{"name": "mcp_arxiv_search", "arguments": {"max_results": 5, "query": "GRPO"}}]

    def test_reconoce_la_llamada_con_atributos(self):
        from app.services.agents import parse_calls, visible_text
        crudo = """<tool_call name="mcp_open_nodes" arguments='{"names": ["Usuario"]}'></tool_call>"""
        assert parse_calls(crudo) == [{"name": "mcp_open_nodes", "arguments": {"names": ["Usuario"]}}]
        assert visible_text(crudo) == ""

    def test_sigue_reconociendo_el_formato_de_siempre(self):
        from app.services.agents import parse_calls
        assert parse_calls('<tool_call>{"name": "calcular", "arguments": {"expr": "2+2"}}</tool_call>') == \
            [{"name": "calcular", "arguments": {"expr": "2+2"}}]

    def test_strip_think_quita_el_canal_de_gemma(self):
        from app.services.evaluation import strip_think
        assert strip_think("<|channel>thought\nvoy a pensar<channel|>La respuesta es 4") == "La respuesta es 4"


class TestMcpConectado:
    """Los MCP externos sirven si están enchufados a algo: a un nodo de flujo o a un agente que guarda aquí."""

    def test_los_argumentos_de_un_nodo_llegan_con_su_tipo(self):
        assert T._tipar({"limit": "5", "q": "hola", "tipos": '["dataset"]', "ok": "true", "id": "2106.09685x"}) == \
            {"limit": 5, "q": "hola", "tipos": ["dataset"], "ok": True, "id": "2106.09685x"}

    def test_un_nodo_mcp_sin_herramienta_explica_que_falta(self):
        texto, _ = T.execute({"run": T.TOOLS["mcp"]["run"], "config": {"server_id": 1}}, {})
        assert "Elige el servidor MCP y la herramienta" in texto

    def test_el_flujo_avisa_del_nodo_mcp_a_medio_configurar(self):
        from app.services.flows import validate
        spec = {"nodes": [{"id": "a", "type": "tool", "label": "HF", "config": {"tool_id": "mcp", "config": {"server_id": 1}}}],
                "edges": []}
        assert any("servidor MCP" in w for w in validate(spec))

    def test_las_recetas_sólo_usan_servidores_que_sabemos_registrar(self):
        from app.api.connections import CATALOGO, PROPIO, RECETAS
        conocidos = {c["key"] for c in CATALOGO} | {PROPIO}
        for r in RECETAS:
            assert set(r["usa"]) <= conocidos, r["key"]
            assert all(r["usa"].values()), "cada servidor con su lista corta de herramientas"

    def test_ingest_file_no_sale_de_la_carpeta_de_entrada(self, monkeypatch):
        import anyio
        from mcp.server.mcpserver.exceptions import ToolError

        from app.api import mcp_server
        from app.db import RagCollection
        monkeypatch.setattr(mcp_server, "_collection", lambda name: RagCollection(id=1, name=name, embed_model="x"))
        with pytest.raises(ToolError, match="data/entrada"):
            anyio.run(mcp_server.ingest_file, "c", "../llmtrainer.db")


    def test_el_json_de_un_nodo_mcp_se_puede_usar_por_campos(self):
        from app.services.flows import render
        assert render("{t_json[text]}", {"t_json": {"text": "hola"}}) == "hola"
        # Si el campo no existe no revienta el flujo: queda la plantilla tal cual
        assert render("{t_json[text]}", {}) == "{t_json[text]}"
