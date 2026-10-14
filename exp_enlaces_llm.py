#!/usr/bin/env python3
"""
exp_enlaces_llm.py — Versión mínima de "GraphRAG": un LLM propone las relaciones entre documentos.

GraphRAG usa un LLM para extraer entidades y relaciones del corpus y armar un grafo. Aquí se
prueba solo la parte que el sistema ya usa: la relación FAQ -> sección de políticas (el enlace E).
Para cada una de las 18 preguntas frecuentes, el LLM ve la FAQ y las secciones de políticas y
propone cuáles la gobiernan. Se compara con la tabla escrita a mano (corpus/enlaces_faq_politica.json).

Se miden dos cosas:
  1. Concordancia de los enlaces: precisión, cobertura y F1 sobre los pares (FAQ, sección), y qué
     FAQ difieren de la tabla manual. Un enlace faltante es el error grave (la FAQ puede omitir una
     excepción, como la de producto dañado); uno de más solo añade ruido al contexto.
  2. Efecto en la recuperación: recall@3 con tres tablas de enlaces (ninguna, manual, propuesta
     por el LLM) sobre los conjuntos de evaluación y validación, con bge-m3.

Con la fusión F activa (la configuración final), las secciones 1.1, 1.2 y 1.3 son un solo fragmento,
así que un enlace a 1.2 trae el mismo texto que uno a 1.3: la fusión puede ocultar una omisión del
LLM. `--sin-fusion` mide los mismos enlaces sobre las secciones separadas, donde cada enlace
cuenta por sí solo.

Uso:
    python exp_enlaces_llm.py                 # requiere Ollama con qwen3:14b y bge-m3
    python exp_enlaces_llm.py --razonar       # con razonamiento de qwen3 (más lento)
    python exp_enlaces_llm.py --sin-fusion    # secciones 1.1-1.3 separadas: cada enlace cuenta solo
    python exp_enlaces_llm.py --prompt excepciones --sin-fusion   # prompt en dos pasos que pide excepciones omitidas
    python exp_enlaces_llm.py --propagar --sin-fusion             # + regla determinista: la excepción que cita a R sigue a R
    python exp_enlaces_llm.py --diagnostico --sin-fusion          # sí/no por cada par (FAQ, sección), 144 llamadas
    python exp_enlaces_llm.py --sin-fusion --modelo anthropic:claude-sonnet-5-5   # mismo experimento con un modelo de API
    python exp_enlaces_llm.py --simulado      # prueba la tubería sin Ollama
"""
import argparse
import json
import re
from pathlib import Path

import llm_local
import medir_embeddings as me
import medir_segmentacion as ms

RAIZ = Path(__file__).parent

# Dos formas de pedir los enlaces. "directo" pide las secciones que gobiernan la FAQ y advierte
# contra temas parecidos: favorece la coincidencia directa. "excepciones" separa la tarea en dos
# pasos y pide de forma explícita las excepciones que la FAQ omite. Comparar ambos dice si la
# omisión de la excepción 1.3 (observada con "directo") es un límite del modelo o del prompt.
PROMPTS = {}
PROMPTS["directo"] = """Eres analista de políticas de EcoMarket. Te muestro una pregunta frecuente (FAQ) \
redactada por el área de mercadeo y las secciones de las políticas oficiales.

Indica qué secciones de las políticas GOBIERNAN la respuesta de la FAQ: aquellas cuyas reglas \
condicionan, completan o corrigen lo que la FAQ afirma (por ejemplo, una excepción que la FAQ no \
menciona). No incluyas secciones que solo traten un tema parecido. Si ninguna aplica, devuelve \
una lista vacía.

Secciones válidas: {validas}

POLÍTICAS:
{politicas}

FAQ:
Pregunta: {pregunta}
Respuesta: {respuesta}"""

PROMPTS["excepciones"] = """Eres analista de políticas de EcoMarket. Te muestro una pregunta frecuente (FAQ) \
redactada por el área de mercadeo y las secciones de las políticas oficiales. La FAQ resume las \
políticas en lenguaje comercial y puede omitir condiciones o excepciones.

Trabaja en dos pasos:
Paso 1. Identifica la sección cuya regla coincide con lo que la FAQ afirma.
Paso 2. Revisa las demás secciones y busca cualquier excepción, condición o caso especial que \
cambiaría la respuesta de la FAQ para algún cliente y que la FAQ no menciona. Inclúyela aunque \
trate de un tema distinto al de la FAQ.

Devuelve las secciones de ambos pasos. Si ninguna sección se relaciona con la FAQ, devuelve una \
lista vacía.

Secciones válidas: {validas}

POLÍTICAS:
{politicas}

FAQ:
Pregunta: {pregunta}
Respuesta: {respuesta}"""


PROMPT_BINARIO = """Eres analista de políticas de EcoMarket. Te muestro UNA pregunta frecuente (FAQ) \
redactada por el área de mercadeo, que puede omitir condiciones o excepciones, y UNA sección de \
las políticas oficiales.

¿Esta sección gobierna la respuesta de la FAQ? Es decir: ¿sus reglas condicionan, completan o \
corrigen lo que la FAQ afirma, para algún cliente? Una sección que solo trata un tema parecido \
sin cambiar la respuesta no la gobierna.

SECCIÓN DE POLÍTICAS:
{seccion}

FAQ:
Pregunta: {pregunta}
Respuesta: {respuesta}"""

ESQ_BINARIO = {"type": "object", "required": ["gobierna"], "properties": {"gobierna": {"type": "boolean"}}}


def diagnosticar(llm, faqs, politicas):
    """Una pregunta binaria por cada par (FAQ, sección): 18 x 8 = 144 llamadas cortas. Convierte la
    tarea de 'elegir entre ocho' en 'juzgar una'; si aquí aparecen enlaces que la lista omitió,
    el límite estaba en el formato de selección y no en la comprensión."""
    propuesta, sin_veredicto = {}, 0
    for f in faqs:
        propuesta[f["id"]] = []
        for p in politicas:
            r = llm(PROMPT_BINARIO.format(seccion=p["texto"], pregunta=f["pregunta"], respuesta=f["respuesta"]),
                    ESQ_BINARIO)
            v = r.get("gobierna") if isinstance(r, dict) else None
            if v is None:
                sin_veredicto += 1
            elif v:
                propuesta[f["id"]].extend(c for c in p["claves"].split(",") if c)
        propuesta[f["id"]] = sorted(set(propuesta[f["id"]]))
    return propuesta, sin_veredicto


def esquema(validas):
    return {"type": "object", "required": ["secciones"], "properties": {
        "secciones": {"type": "array", "items": {"type": "string", "enum": validas}}}}


def proponer(llm, faqs, politicas, plantilla="directo"):
    validas = sorted({c for f in politicas for c in f["claves"].split(",") if c})
    texto = "\n\n".join(f"[Sección {f['claves'].replace(',', ', ')}]\n{f['texto']}" for f in politicas)
    propuesta = {}
    for f in faqs:
        r = llm(PROMPTS[plantilla].format(validas=", ".join(validas), politicas=texto,
                              pregunta=f["pregunta"], respuesta=f["respuesta"]), esquema(validas))
        secciones = r.get("secciones", []) if isinstance(r, dict) else []
        propuesta[f["id"]] = sorted({s for s in secciones if s in validas})
    return propuesta


def excepciones_estructurales(politicas):
    """Regla -> [excepciones], leída del propio texto de las políticas: una sección que dice que
    la restricción de la sección R 'no aplica' es una excepción de R. Es determinista y no usa
    retroalimentación; depende de que la política cite la regla que modifica."""
    mapa = {}
    for f in politicas:
        claves = [c for c in f["claves"].split(",") if c]
        for m in re.finditer(r"secci[oó]n\s+(\d+(?:\.\d+)?)[^.]{0,60}?no aplica", f["texto"], re.I | re.S):
            regla = m.group(1)
            for e in claves:
                if e != regla:
                    mapa.setdefault(regla, []).append(e)
    return {r: sorted(set(v)) for r, v in mapa.items()}


def propagar(propuesta, excepciones):
    """Toda FAQ enlazada a una regla queda enlazada también a sus excepciones. Devuelve la tabla
    ampliada y los enlaces añadidos."""
    salida, añadidos = {}, []
    for faq, secciones in propuesta.items():
        nuevo = set(secciones)
        for r in secciones:
            for e in excepciones.get(r, []):
                if e not in nuevo:
                    nuevo.add(e)
                    añadidos.append((faq, e))
        salida[faq] = sorted(nuevo)
    return salida, sorted(añadidos)


def comparar(manual, propuesta):
    pares_m = {(f, s) for f, ss in manual.items() for s in ss}
    pares_p = {(f, s) for f, ss in propuesta.items() for s in ss}
    tp = len(pares_m & pares_p)
    prec = tp / len(pares_p) if pares_p else 0.0
    cob = tp / len(pares_m) if pares_m else 0.0
    f1 = 2 * prec * cob / (prec + cob) if prec + cob else 0.0
    return {"precision": round(prec, 3), "cobertura": round(cob, 3), "f1": round(f1, 3),
            "faltan": sorted(pares_m - pares_p), "sobran": sorted(pares_p - pares_m),
            "faq_identicas": sum(set(manual.get(f, [])) == set(propuesta.get(f, [])) for f in
                                 {*manual, *propuesta})}


def fraccion(modelo, embed, cache, frags, mapa, enlace_activo, refs, lista):
    """recall@3 y críticas con una tabla de enlaces dada (ms.evaluar lee ms.REFS_FAQ)."""
    original = ms.REFS_FAQ
    ms.REFS_FAQ = refs
    try:
        r = ms.evaluar(frags, mapa, enlace_activo, lista, modelo, embed, cache)
    finally:
        ms.REFS_FAQ = original
    return r["recall"]["@3"], r["recall_criticas@3"], r["fallan@3"]


def llm_falso(prompt, esquema_):
    """Para --simulado: enlaza 1.2 a toda FAQ de devoluciones; en modo binario, 'sí' si la sección
    y la FAQ comparten la palabra 'devol'."""
    if "gobierna" in esquema_["properties"]:
        sec, faq = prompt.split("SECCIÓN DE POLÍTICAS:")[1].split("FAQ:")
        return {"gobierna": "devol" in sec.lower() and "devol" in faq.lower()}
    return {"secciones": ["1.2"] if "devol" in prompt.lower().split("faq:")[1] else []}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default=llm_local.MODELO)
    ap.add_argument("--razonar", action="store_true")
    ap.add_argument("--sin-fusion", action="store_true")
    ap.add_argument("--prompt", choices=list(PROMPTS), default="directo")
    ap.add_argument("--diagnostico", action="store_true",
                    help="pregunta sí/no por cada par (FAQ, sección) en lugar de pedir la lista; 144 llamadas")
    ap.add_argument("--propagar", action="store_true",
                    help="añade a la propuesta del LLM las excepciones que las políticas citan de forma explícita")
    ap.add_argument("--simulado", action="store_true")
    a = ap.parse_args()

    # Secciones de políticas que ven el modelo y la regla de propagación. Con --sin-fusion son las
    # ocho secciones separadas; sin él, el fragmento fusionado 1.1-1.3 aparece como un solo bloque
    # etiquetado "Sección 1.1, 1.2, 1.3". Las corridas anteriores a esta versión mostraron siempre
    # el bloque fusionado, incluso con --sin-fusion (ver fase3_rag.md, 6.1).
    frags_pol, _ = ms.construir(not a.sin_fusion, False)
    politicas = [{"claves": ",".join(c for c in f["claves"] if c), "texto": f["texto"]}
                 for f in frags_pol if f["fuente"] == "politicas"]
    faqs_json = json.loads((RAIZ / "corpus" / "faq.json").read_text(encoding="utf-8"))["preguntas"]
    manual = {f["id"]: ms.REFS_FAQ.get(f["id"], []) for f in faqs_json}

    if a.simulado:
        llm, modelo_emb, embed = llm_local.LLMFalso(llm_falso), "simulado-hash", me.embed_simulado
    else:
        llm_local.comprobar(a.modelo)
        llm, modelo_emb, embed = llm_local.crear(a.modelo, a.razonar), "bge-m3", me.embed_ollama
    if a.diagnostico:
        print(f"Diagnóstico binario: {len(faqs_json)} FAQ x {len(politicas)} secciones...", flush=True)
        propuesta, sin_veredicto = diagnosticar(llm, faqs_json, politicas)
        print(f"  respuestas sin veredicto interpretable: {sin_veredicto}")
        print("  enlaces por FAQ: " + ", ".join(f"{k} {v}" for k, v in propuesta.items()))
    else:
        print(f"Proponiendo enlaces para {len(faqs_json)} FAQ (prompt '{a.prompt}')...", flush=True)
        propuesta = proponer(llm, faqs_json, politicas, a.prompt)
    if a.propagar:
        exc = excepciones_estructurales(politicas)
        propuesta, añadidos = propagar(propuesta, exc)
        print(f"\n== Propagación de excepciones (regla determinista) ==\n  excepciones leídas de las políticas: {exc}")
        print(f"  enlaces añadidos a la propuesta del LLM: {añadidos or '—'}")
    cmp = comparar(manual, propuesta)

    print("\n== Concordancia con la tabla manual ==")
    print(f"  precisión {cmp['precision']:.2f} | cobertura {cmp['cobertura']:.2f} | F1 {cmp['f1']:.2f} | "
          f"FAQ idénticas {cmp['faq_identicas']}/{len(faqs_json)}")
    print(f"  enlaces que faltan (grave): {cmp['faltan'] or '—'}")
    print(f"  enlaces de más (ruido):     {cmp['sobran'] or '—'}")

    frags, mapa = ms.construir(not a.sin_fusion, False)
    cache, resultados = {}, {}
    print(f"\n== Efecto en la recuperación (recall@3 / críticas), "
          f"{'secciones 1.1-1.3 separadas' if a.sin_fusion else 'con fusión 1.1-1.3'} ==")
    for titulo, archivo in (("evaluación", "conjunto_oro.json"), ("validación", "conjunto_validacion.json")):
        lista = json.loads((RAIZ / archivo).read_text(encoding="utf-8"))["preguntas"]
        fila = {}
        for nombre, activo, refs in (("sin enlace", False, {}), ("manual", True, manual),
                                     ("propuesto por LLM", True, propuesta)):
            fila[nombre] = fraccion(modelo_emb, embed, cache, frags, mapa, activo, refs, lista)
        resultados[titulo] = fila
        print(f"  {titulo}: " + " | ".join(f"{n} {r:.2f}/{c:.2f}" for n, (r, c, _f) in fila.items()))
        for n, (_r, _c, fallan) in fila.items():
            if fallan and n == "propuesto por LLM":
                print(f"      falla con enlaces del LLM: {', '.join(fallan)}")
    print("\n" + llm_local.resumen_costo(llm))
    if a.simulado:
        print("(modo simulado: no se escribe ningún archivo de resultados)")
        return
    etiqueta_modelo = "_" + a.modelo.replace(":", "_").replace("/", "_") if a.modelo != llm_local.MODELO else ""
    destino = ("resultados_llm_enlaces" + etiqueta_modelo + ("_diagnostico" if a.diagnostico else "")
               + ("_excepciones" if a.prompt == "excepciones" else "")
               + ("_razonar" if a.razonar else "") + ("_propagar" if a.propagar else "")
               + ("_sin_fusion" if a.sin_fusion else "") + ".json")
    (RAIZ / destino).write_text(json.dumps(
        {"modelo": a.modelo, "razonar": a.razonar, "prompt": a.prompt, "diagnostico": a.diagnostico,
         "propagar": a.propagar, "fusion": not a.sin_fusion, "propuesta": propuesta, "manual": manual,
         "comparacion": cmp, "recuperacion": {t: {n: {"recall@3": r, "criticas@3": c, "fallan": f}
                                               for n, (r, c, f) in fila.items()}
                                              for t, fila in resultados.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Detalle guardado en {destino}")


if __name__ == "__main__":
    main()
