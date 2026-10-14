#!/usr/bin/env python3
"""
exp_correctivo.py — RAG correctivo (estilo CRAG): el LLM revisa lo recuperado antes de generar.

Flujo por pregunta:
  1. Recuperar con la configuración final (k=3 + enlace).
  2. Un LLM juzga si el contexto es suficiente para responder la pregunta.
  3. Si no lo es: el LLM reescribe la consulta, se recupera de nuevo con k=5, se combinan ambos
     contextos sin repetir fragmentos y se vuelve a juzgar.
  4. Si sigue siendo insuficiente, el sistema se abstendría de responder y derivaría a un humano.

Qué se mide, sobre 34 preguntas: las 26 con respuesta (evaluación y validación) y 8 sin respuesta
en el corpus (conjunto_sin_respuesta.json).
  - Veredicto verdadero de suficiencia: para las 26 con respuesta, "suficiente" si todos los textos
    de `requiere` aparecen en el contexto (el mismo criterio de la Fase 2); para las 8 sin
    respuesta, "insuficiente" por construcción.
  - Calidad del juez en el primer intento: aciertos, abstenciones de más (juzga insuficiente un
    contexto que sí alcanzaba) y respuestas peligrosas (juzga suficiente un contexto que no
    alcanza).
  - Efecto del lazo correctivo: recall final, preguntas sin respuesta que terminan en abstención
    y costo en llamadas al LLM y segundos.

Limitación: el criterio de `requiere` exige frases literales de la política, de modo que "suficiente"
es una definición estricta. Un contexto que alcanza para responder sin contener esa frase se
cuenta como insuficiente.

Uso:
    python exp_correctivo.py                       # requiere python indexar.py y Ollama
    python exp_correctivo.py --verificar-conjunto  # comprueba que las 8 preguntas realmente no están en el corpus
    python exp_correctivo.py --simulado --ruta-db /tmp/db_prueba   # prueba sin Ollama
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import base_conocimiento as bc
import llm_local
import medir_embeddings as me
import rag_ecomarket as rag

RAIZ = Path(__file__).parent
K_REINTENTO = 5

ESQ_GRADO = {"type": "object", "required": ["suficiente", "motivo"],
             "properties": {"suficiente": {"type": "boolean"}, "motivo": {"type": "string"}}}
ESQ_CONSULTA = {"type": "object", "required": ["consulta"], "properties": {"consulta": {"type": "string"}}}

PROMPT_GRADO = """Eres el revisor de un sistema de atención al cliente de EcoMarket. Decide si el \
CONTEXTO contiene la información necesaria para responder la PREGUNTA basándose solo en él.

Responde suficiente=false si falta un dato clave que la respuesta necesita (una cifra, una \
condición o una excepción), o si el contexto trata otro tema. Responde suficiente=true solo si \
con ese contexto se puede dar una respuesta completa y correcta.

CONTEXTO:
{contexto}

PREGUNTA: {pregunta}"""

PROMPT_REESCRIBIR = """Una búsqueda con la pregunta de un cliente no encontró información suficiente en \
la base de conocimiento de EcoMarket (políticas de devoluciones y envíos, preguntas frecuentes, \
catálogo y guías de cuidado). Reescríbela como una consulta de búsqueda más precisa: usa los \
términos que emplearía un documento de políticas y no agregues datos que el cliente no mencionó.

PREGUNTA ORIGINAL: {pregunta}"""


def sin_acentos(t):
    return me.sin_acentos(t)


def cubre(docs, requiere):
    union = sin_acentos(" ".join(d.page_content for d in docs))
    return all(me.contiene(union, r) for r in requiere)


def juzgar(llm, pregunta, docs):
    r = llm(PROMPT_GRADO.format(contexto=rag.formatear_contexto(docs), pregunta=pregunta), ESQ_GRADO)
    return r.get("suficiente") if isinstance(r, dict) and isinstance(r.get("suficiente"), bool) else None


def correctivo(llm, rec, pregunta):
    docs1 = rec.recuperar(pregunta)
    g1 = juzgar(llm, pregunta, docs1)
    if g1:
        return {"docs": docs1, "g1": g1, "g2": None, "final": True, "consulta2": None}
    r = llm(PROMPT_REESCRIBIR.format(pregunta=pregunta), ESQ_CONSULTA)
    consulta2 = r.get("consulta") if isinstance(r, dict) and r.get("consulta") else pregunta
    k0, rec.k = rec.k, K_REINTENTO
    try:
        docs2 = rec.recuperar(consulta2)
    finally:
        rec.k = k0
    vistos = {d.id for d in docs1}
    docs = docs1 + [d for d in docs2 if d.id not in vistos]
    g2 = juzgar(llm, pregunta, docs)
    return {"docs": docs, "g1": g1, "g2": g2, "final": bool(g2), "consulta2": consulta2}


def verificar_conjunto():
    sr = json.loads((RAIZ / "conjunto_sin_respuesta.json").read_text(encoding="utf-8"))["preguntas"]
    corpus = sin_acentos(" ".join(f["texto"] for f in bc.construir_fragmentos()))
    ok = True
    for p in sr:
        hallados = [t for t in p["ausentes"] if sin_acentos(t) in corpus]
        if hallados:
            ok = False
            print(f"  {p['id']}: el corpus SÍ contiene {hallados}; retirar o reformular la pregunta")
    print("Las 8 preguntas sin respuesta no aparecen en el corpus." if ok else "Hay preguntas que revisar.")
    return ok


def grado_falso(prompt, esquema):
    """Para --simulado."""
    if "consulta" in esquema["properties"]:
        return {"consulta": "política de devoluciones excepción producto dañado"}
    return {"suficiente": "[faq::" not in prompt, "motivo": "simulado"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default=llm_local.MODELO)
    ap.add_argument("--razonar", action="store_true", help="activa el razonamiento de qwen3 (más lento)")
    ap.add_argument("--ruta-db", default=str(rag.RUTA_DB))
    ap.add_argument("--verificar-conjunto", action="store_true")
    ap.add_argument("--simulado", action="store_true")
    a = ap.parse_args()
    if a.verificar_conjunto:
        raise SystemExit(0 if verificar_conjunto() else 1)
    assert verificar_conjunto(), "conjunto_sin_respuesta.json contiene preguntas con respuesta en el corpus"

    emb = None
    if a.simulado:
        from indexar import EmbeddingsSimulados
        emb = EmbeddingsSimulados()
    else:
        llm_local.comprobar(a.modelo)
    rec = rag.crear_recuperador(Path(a.ruta_db), rag.K, True, emb)
    llm = llm_local.LLMFalso(grado_falso) if a.simulado else llm_local.crear(a.modelo, a.razonar)

    def cargar(n):
        return json.loads((RAIZ / n).read_text(encoding="utf-8"))["preguntas"]
    con = cargar("conjunto_oro.json") + cargar("conjunto_validacion.json")
    sin = cargar("conjunto_sin_respuesta.json")

    filas = []
    for p in con + sin:
        tiene = "requiere" in p
        llamadas0, seg0 = llm.llamadas, llm.segundos
        t = correctivo(llm, rec, p["pregunta"])
        base_docs = rec.recuperar(p["pregunta"])
        real_suf = cubre(base_docs, p["requiere"]) if tiene else False
        filas.append({"id": p["id"], "con_respuesta": tiene, "critico": p.get("critico", False),
                      "suficiente_real": real_suf, "g1": t["g1"], "g2": t["g2"], "final": t["final"],
                      "recall_final": cubre(t["docs"], p["requiere"]) if tiene else None,
                      "consulta2": t["consulta2"], "llamadas": llm.llamadas - llamadas0,
                      "segundos": round(llm.segundos - seg0, 1),
                      "chars_final": sum(len(d.page_content) for d in t["docs"])})
        print(f"  {p['id']:<7} real={'suf' if real_suf else 'insuf':<5} juez1={t['g1']!s:<6} "
              f"final={t['final']!s:<6}", flush=True)

    c = [f for f in filas if f["con_respuesta"]]
    s = [f for f in filas if not f["con_respuesta"]]
    ok1 = sum(f["g1"] == f["suficiente_real"] for f in filas)
    print("\n== Calidad del juez en el primer intento (34 preguntas) ==")
    print(f"  aciertos                          {ok1}/{len(filas)}")
    print(f"  abstenciones de más (con respuesta, contexto suficiente, juez dice insuficiente): "
          f"{sum(f['suficiente_real'] and f['g1'] is False for f in c)}/{sum(f['suficiente_real'] for f in c)}")
    print(f"  respuestas peligrosas (sin respuesta o contexto insuficiente, juez dice suficiente): "
          f"{sum((not f['suficiente_real']) and f['g1'] is True for f in filas)}/{sum(not f['suficiente_real'] for f in filas)}")
    base_rec = sum(f["suficiente_real"] for f in c) / len(c)
    fin_rec = sum(bool(f["recall_final"]) for f in c) / len(c)
    print("\n== Efecto del lazo correctivo ==")
    print(f"  recall (26 con respuesta): línea base {base_rec:.2f} -> con lazo {fin_rec:.2f}")
    print(f"  con respuesta que terminan en abstención: {sum(not f['final'] for f in c)}/{len(c)}")
    print(f"  sin respuesta que terminan en abstención: {sum(not f['final'] for f in s)}/{len(s)}")
    print(f"  llamadas al LLM por pregunta (media): {sum(f['llamadas'] for f in filas) / len(filas):.2f}   "
          f"segundos por pregunta: {sum(f['segundos'] for f in filas) / len(filas):.1f}")
    print(f"  contexto final medio: {sum(f['chars_final'] for f in filas) / len(filas):.0f} caracteres")
    if a.simulado:
        print("\n(modo simulado: no se escribe ningún archivo de resultados)")
        return
    (RAIZ / "resultados_llm_correctivo.json").write_text(
        json.dumps({"modelo": a.modelo, "razonar": a.razonar, "filas": filas}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("\nDetalle guardado en resultados_llm_correctivo.json")


if __name__ == "__main__":
    main()
