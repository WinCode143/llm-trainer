"""GraphRAG local: grafo de entidades y relaciones extraído por el LLM de los fragmentos de una colección.

Construcción:  por fragmento → entidades {nombre, tipo, descripción} y relaciones {origen, destino, relación}
               → fusión por nombre normalizado → comunidades (Louvain) → resumen de las comunidades grandes.
Consulta:      entidades más parecidas a la pregunta (embeddings + coincidencia de nombre) → vecinos a 1 salto
               → contexto con relaciones + fragmentos donde aparecen + resúmenes de comunidad relevantes.
Sirve para preguntas relacionales o de varios saltos («¿qué relación hay entre X e Y?», «¿quiénes participaron…?»)
que la búsqueda por similitud responde mal porque la información está repartida en varios fragmentos.
"""
import json
import re
import threading
import unicodedata
from collections import Counter, defaultdict

import numpy as np

from ..config import settings
from . import rag

EXTRACT_PROMPT = """Extrae del TEXTO las entidades importantes (personas, organizaciones, lugares, conceptos,
objetos, eventos, fechas clave) y las relaciones explícitas entre ellas.
Devuelve SOLO JSON con esta forma:
{{"entidades": [{{"nombre": "...", "tipo": "persona|organización|lugar|concepto|objeto|evento|fecha", "descripcion": "una frase"}}],
  "relaciones": [{{"origen": "...", "destino": "...", "relacion": "verbo o frase corta"}}]}}
Usa exactamente los mismos nombres de entidad en las relaciones. Máximo {max_ent} entidades.

TEXTO:
{text}"""

COMMUNITY_PROMPT = """Estas entidades y relaciones forman un tema dentro de una base de documentos:
{items}

Escribe un título corto (máx. 8 palabras) en la primera línea y, debajo, un resumen de 3-5 frases de qué trata
y cómo se relacionan las entidades principales."""

_cache: dict[int, dict] = {}
_lock = threading.Lock()


def _path(cid: int):
    return settings.rag_dir / f"graph_{cid}.json"


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name.lower().strip())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"^(el|la|los|las|the)\s+", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _parse(out: str | None) -> dict:
    """JSON del modelo; si viene cortado (límite de tokens), se cierra en el último objeto completo."""
    m = re.search(r"\{.*", out or "", re.S)
    if not m:
        return {}
    txt = m.group(0)
    end = txt.rfind("}")
    try:
        return json.loads(txt[: end + 1])
    except json.JSONDecodeError:
        pass
    cuts = [i for i, ch in enumerate(txt) if ch == "}"][-30:]
    for cut in reversed(cuts):
        for suffix in ("]}", "]}]}", "}]}", "]}}"):
            try:
                d = json.loads(txt[: cut + 1] + suffix)
                if isinstance(d, dict):
                    return d
            except json.JSONDecodeError:
                continue
    return {}


def build(coll, task, sources: list[str] | None = None, max_chunks: int | None = None, max_entities: int = 8,
          summarize_communities: bool = True, helper: int | None = None) -> dict:
    col = rag._col(coll.id)
    got = col.get(where={"source": {"$in": sources}} if sources else None, include=["documents", "metadatas"])
    chunks = [(i, d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"]) if not m.get("level")]
    chunks.sort(key=lambda x: (x[2].get("source", ""), x[2].get("chunk", 0)))
    if max_chunks:
        chunks = chunks[:max_chunks]
    if not chunks:
        raise ValueError("La colección no tiene fragmentos")
    ents: dict[str, dict] = {}
    edges: dict[tuple, dict] = {}
    failed = 0
    for n, (cid, doc, meta) in enumerate(chunks):
        if task.cancel_requested:
            break
        task.update(0.85 * n / len(chunks), f"Fragmento {n + 1}/{len(chunks)} · {len(ents)} entidades, {len(edges)} relaciones")
        out = rag._llm(EXTRACT_PROMPT.format(text=doc[:2500], max_ent=max_entities), 700, helper)
        if out is None and n == 0:
            raise RuntimeError("Carga un modelo (Playground o Modelos) para extraer el grafo")
        data = _parse(out)
        if not data:
            failed += 1
            continue
        for e in data.get("entidades") or []:
            if not isinstance(e, dict) or not str(e.get("nombre", "")).strip():
                continue
            key = norm(e["nombre"])
            # Descarta ruido típico de modelos pequeños: nombres vacíos, sólo cifras o demasiado largos
            if len(key) < 3 or key.isdigit() or len(key) > 60:
                continue
            ent = ents.setdefault(key, {"name": e["nombre"].strip(), "types": Counter(), "descs": [], "chunks": [],
                                        "sources": set()})
            ent["types"][str(e.get("tipo") or "concepto").lower()] += 1
            if e.get("descripcion") and len(ent["descs"]) < 4 and e["descripcion"] not in ent["descs"]:
                ent["descs"].append(str(e["descripcion"]).strip())
            if cid not in ent["chunks"]:
                ent["chunks"].append(cid)
            ent["sources"].add(meta.get("source", ""))
        for r in data.get("relaciones") or []:
            if not isinstance(r, dict):
                continue
            a, b = norm(str(r.get("origen", ""))), norm(str(r.get("destino", "")))
            if not a or not b or a == b or a not in ents or b not in ents:
                continue
            rel = str(r.get("relacion") or "relacionado con").strip()[:80]
            e = edges.setdefault((a, b, rel.lower()), {"source": a, "target": b, "relation": rel, "chunks": [], "weight": 0})
            e["weight"] += 1
            if cid not in e["chunks"]:
                e["chunks"].append(cid)

    # Comunidades (Louvain) sobre el grafo no dirigido
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(ents)
    for e in edges.values():
        G.add_edge(e["source"], e["target"], weight=G.get_edge_data(e["source"], e["target"], {}).get("weight", 0) + e["weight"])
    comms = []
    if G.number_of_edges():
        from networkx.algorithms.community import louvain_communities
        parts = sorted(louvain_communities(G, weight="weight", seed=42), key=len, reverse=True)
        big = [p for p in parts if len(p) >= 3][:12]
        for ci, members in enumerate(big):
            if task.cancel_requested:
                break
            task.update(0.85 + 0.14 * ci / max(1, len(big)), f"Resumiendo comunidad {ci + 1}/{len(big)}")
            top = sorted(members, key=lambda k: -G.degree(k))[:15]
            rels = [f"{ents[e['source']]['name']} —{e['relation']}→ {ents[e['target']]['name']}"
                    for e in edges.values() if e["source"] in members and e["target"] in members][:25]
            title, summary = f"Tema {ci + 1}", ""
            if summarize_communities:
                items = "\n".join([f"- {ents[k]['name']}: {(ents[k]['descs'] or [''])[0]}" for k in top] + rels)
                out = rag._llm(COMMUNITY_PROMPT.format(items=items[:4000]), 260, helper) or ""
                lines = [l.strip("#*: ") for l in out.strip().splitlines() if l.strip()]
                if lines:
                    # Los modelos a veces escriben «Título: X» o lo marcan en negrita
                    first = re.sub(r"^\W*(t[ií]tulo|title)\W*[:\-]?\s*", "", lines[0], flags=re.I).strip("*_# ")
                    title = (first or f"Tema {ci + 1}")[:80]
                    summary = " ".join(l for l in lines[1:] if not re.match(r"^\W*(resumen|summary)\W*:?$", l, re.I))
            comms.append({"id": ci, "title": title, "summary": summary, "entities": sorted(members, key=lambda k: -G.degree(k))})
    for ci, c in enumerate(comms):
        for k in c["entities"]:
            ents[k]["community"] = ci

    data = {
        "collection_id": coll.id, "embed_model": coll.embed_model,
        "entities": {k: {"name": v["name"], "type": v["types"].most_common(1)[0][0], "descs": v["descs"],
                         "chunks": v["chunks"], "sources": sorted(v["sources"]), "degree": G.degree(k),
                         "community": v.get("community")} for k, v in ents.items()},
        "edges": list(edges.values()), "communities": comms,
        "stats": {"chunks": len(chunks), "failed": failed, "entities": len(ents), "edges": len(edges),
                  "communities": len(comms)},
    }
    _path(coll.id).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with _lock:
        _cache.pop(coll.id, None)
    return data["stats"]


def exists(cid: int) -> bool:
    """¿Hay grafo construido? (sin cargarlo ni calcular embeddings)"""
    return _path(cid).exists()


def load(cid: int) -> dict | None:
    with _lock:
        if cid in _cache:
            return _cache[cid]
    p = _path(cid)
    if not p.exists():
        return None
    g = json.loads(p.read_text(encoding="utf-8"))
    keys = list(g["entities"])
    texts = [f"{g['entities'][k]['name']}: {(g['entities'][k]['descs'] or [''])[0]}" for k in keys]
    g["_keys"] = keys
    g["_vecs"] = np.array(rag.embed(g["embed_model"], texts, "passage")) if keys else np.zeros((0, 1))
    ctexts = [f"{c['title']}. {c['summary']}" for c in g["communities"]]
    g["_cvecs"] = np.array(rag.embed(g["embed_model"], ctexts, "passage")) if ctexts else np.zeros((0, 1))
    with _lock:
        _cache[cid] = g
    return g


def drop(cid: int) -> None:
    _path(cid).unlink(missing_ok=True)
    with _lock:
        _cache.pop(cid, None)


def query(coll, question: str, k_entities: int = 6, max_relations: int = 25) -> dict | None:
    """Subgrafo relevante para la pregunta: entidades semilla, vecinos, relaciones, fragmentos y comunidades."""
    g = load(coll.id)
    if not g or not g["_keys"]:
        return None
    qv = np.array(rag.embed(g["embed_model"], [question], "query")[0])
    sims = g["_vecs"] @ qv
    qn = f" {norm(question)} "
    for i, k in enumerate(g["_keys"]):  # coincidencia literal del nombre en la pregunta
        if len(k) > 2 and f" {k} " in qn:
            sims[i] += 0.3
    seeds = [g["_keys"][i] for i in np.argsort(-sims)[:k_entities] if sims[i] > 0.75]
    if not seeds:
        return None
    nbr = defaultdict(float)
    rels = []
    for e in g["edges"]:
        if e["source"] in seeds or e["target"] in seeds:
            rels.append(e)
            for x in (e["source"], e["target"]):
                if x not in seeds:
                    nbr[x] += e["weight"]
    rels.sort(key=lambda e: -(e["weight"] + (e["source"] in seeds) + (e["target"] in seeds)))
    rels = rels[:max_relations]
    neighbors = [k for k, _ in sorted(nbr.items(), key=lambda x: -x[1])[:10]]
    E = g["entities"]
    chunk_votes = Counter()
    for k in seeds:
        chunk_votes.update({c: 2 for c in E[k]["chunks"]})
    for k in neighbors:
        chunk_votes.update({c: 1 for c in E[k]["chunks"]})
    for e in rels:
        chunk_votes.update({c: 1 for c in e["chunks"]})
    comms = []
    if len(g["_cvecs"]):
        cs = g["_cvecs"] @ qv
        comms = [g["communities"][i] for i in np.argsort(-cs)[:2] if cs[i] > 0.78]
    lines = ["Entidades relevantes:"]
    lines += [f"- {E[k]['name']} ({E[k]['type']}): {'; '.join(E[k]['descs'][:2])}" for k in seeds + neighbors[:5]]
    if rels:
        lines += ["Relaciones:"] + [f"- {E[e['source']]['name']} —{e['relation']}→ {E[e['target']]['name']}" for e in rels]
    for c in comms:
        lines += [f"Tema «{c['title']}»: {c['summary']}"]
    return {"context": "\n".join(lines), "seeds": [E[k]["name"] for k in seeds], "neighbors": [E[k]["name"] for k in neighbors],
            "relations": len(rels), "chunk_ids": [c for c, _ in chunk_votes.most_common(8)],
            "communities": [c["title"] for c in comms]}


def view(cid: int, limit: int = 250) -> dict | None:
    """Nodos y aristas para visualizar (los más conectados)."""
    g = load(cid)
    if not g:
        return None
    E = g["entities"]
    keys = sorted(E, key=lambda k: -E[k]["degree"])[:limit]
    ks = set(keys)
    return {"nodes": [{"id": k, "name": E[k]["name"], "type": E[k]["type"], "degree": E[k]["degree"],
                       "community": E[k]["community"], "desc": (E[k]["descs"] or [""])[0], "sources": E[k]["sources"]} for k in keys],
            "links": [{"source": e["source"], "target": e["target"], "relation": e["relation"], "weight": e["weight"]}
                      for e in g["edges"] if e["source"] in ks and e["target"] in ks],
            "communities": [{k: v for k, v in c.items() if k != "entities"} | {"size": len(c["entities"])} for c in g["communities"]],
            "stats": g["stats"]}
