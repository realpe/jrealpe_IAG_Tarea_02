#!/usr/bin/env python3
"""
exp_juez.py — ¿Sirve qwen3:14b como juez de fidelidad? (evaluación de respuestas con LLM)

Pregunta que responde: dado un contexto y una respuesta, ¿el modelo detecta cuándo la respuesta
afirma algo que el contexto no respalda? Para saberlo se usan 18 casos con veredicto conocido
(casos_juez.json): 8 respuestas fieles y 10 infieles, entre ellas errores sutiles de un solo dato
(un plazo alterado, un cupón inventado, un saldo en lugar de reembolso al medio de pago).

Se comparan cuatro configuraciones: dos formas de pedir el veredicto y razonamiento apagado o
encendido.
  directo       el modelo responde {fiel, motivo}
  afirmaciones  el modelo lista cada afirmación de la respuesta y marca si está respaldada;
                el veredicto lo calcula el código (fiel = todas respaldadas), no el modelo

Métricas (positivo = respuesta infiel):
  detección        infieles detectadas / infieles reales
  falsas alarmas   fieles marcadas como infieles / fieles reales
  exactitud        aciertos / casos
  sin veredicto    respuestas del juez que no se pudieron interpretar

Uso:
    python exp_juez.py                    # las cuatro configuraciones (unos minutos)
    python exp_juez.py --configs directo  # solo las del prompt directo
    python exp_juez.py --simulado         # prueba la tubería sin Ollama (números sin significado)
"""
import argparse
import json
import re
from pathlib import Path

import base_conocimiento as bc
import llm_local

RAIZ = Path(__file__).parent

ESQ_DIRECTO = {"type": "object", "required": ["fiel", "motivo"],
               "properties": {"fiel": {"type": "boolean"}, "motivo": {"type": "string"}}}
ESQ_AFIRMACIONES = {"type": "object", "required": ["afirmaciones"], "properties": {
    "afirmaciones": {"type": "array", "items": {"type": "object", "required": ["texto", "respaldada"],
                     "properties": {"texto": {"type": "string"}, "respaldada": {"type": "boolean"}}}}}}

PROMPT_DIRECTO = """Eres un auditor de un sistema de atención al cliente. Revisas si una RESPUESTA \
está respaldada por el CONTEXTO.

La respuesta es fiel solo si TODA afirmación suya (cifras, plazos, montos, condiciones, \
excepciones, ofrecimientos) está en el contexto o se deduce directamente de él. Si agrega un dato \
que el contexto no contiene, o lo contradice, no es fiel. Responder que no se tiene la información \
es fiel cuando el contexto efectivamente no la contiene.

CONTEXTO:
{contexto}

PREGUNTA DEL CLIENTE: {pregunta}

RESPUESTA A REVISAR: {respuesta}"""

PROMPT_AFIRMACIONES = """Eres un auditor de un sistema de atención al cliente. Verificas una \
RESPUESTA contra el CONTEXTO, afirmación por afirmación.

Paso 1: separa la respuesta en afirmaciones individuales (una cifra, un plazo, un monto, una \
condición o un ofrecimiento por afirmación).
Paso 2: para cada una indica si el contexto la respalda o se deduce directamente de él. Una \
afirmación que el contexto no menciona, o que lo contradice, no está respaldada.

CONTEXTO:
{contexto}

PREGUNTA DEL CLIENTE: {pregunta}

RESPUESTA A REVISAR: {respuesta}"""

CONFIGS = {  # nombre -> (prompt, esquema, razonar)
    "directo": (PROMPT_DIRECTO, ESQ_DIRECTO, False),
    "directo+razonar": (PROMPT_DIRECTO, ESQ_DIRECTO, True),
    "afirmaciones": (PROMPT_AFIRMACIONES, ESQ_AFIRMACIONES, False),
    "afirmaciones+razonar": (PROMPT_AFIRMACIONES, ESQ_AFIRMACIONES, True),
}


def veredicto(resultado):
    """True/False según el juez; None si la respuesta no se pudo interpretar."""
    if not isinstance(resultado, dict):
        return None
    if "afirmaciones" in resultado:
        lista = resultado["afirmaciones"]
        if not isinstance(lista, list) or not lista:
            return None
        return all(bool(a.get("respaldada")) for a in lista if isinstance(a, dict))
    if isinstance(resultado.get("fiel"), bool):
        return resultado["fiel"]
    return None


def evaluar(nombre, caso_lista, textos, llm):
    prompt, esquema, _ = CONFIGS[nombre]
    filas = []
    for c in caso_lista:
        contexto = "\n\n".join(f"[{i}]\n{textos[i]}" for i in c["contexto"])
        r = llm(prompt.format(contexto=contexto, pregunta=c["pregunta"], respuesta=c["respuesta"]), esquema)
        filas.append({"id": c["id"], "real": c["fiel"], "juez": veredicto(r)})
    infieles = [f for f in filas if not f["real"]]
    fieles = [f for f in filas if f["real"]]
    return {
        "deteccion": round(sum(f["juez"] is False for f in infieles) / len(infieles), 3),
        "falsas_alarmas": round(sum(f["juez"] is False for f in fieles) / len(fieles), 3),
        "exactitud": round(sum(f["juez"] == f["real"] for f in filas) / len(filas), 3),
        "sin_veredicto": sum(f["juez"] is None for f in filas),
        "llamadas": llm.llamadas, "segundos": round(llm.segundos, 1),
        "errores": [f["id"] for f in filas if f["juez"] != f["real"]],
        "detalle": filas,
    }


def juez_falso(prompt, esquema):
    """Para --simulado: marca infiel si la respuesta contiene un número ausente del contexto."""
    contexto = prompt.split("CONTEXTO:")[1].split("PREGUNTA DEL CLIENTE")[0]
    respuesta = prompt.split("RESPUESTA A REVISAR:")[1]
    nuevos = set(re.findall(r"\d+", respuesta)) - set(re.findall(r"\d+", contexto))
    if "afirmaciones" in esquema["properties"]:
        return {"afirmaciones": [{"texto": "x", "respaldada": not nuevos}]}
    return {"fiel": not nuevos, "motivo": "simulado"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    ap.add_argument("--modelo", default=llm_local.MODELO)
    ap.add_argument("--simulado", action="store_true")
    a = ap.parse_args()

    casos = json.loads((RAIZ / "casos_juez.json").read_text(encoding="utf-8"))["casos"]
    textos = {f["id"]: f["texto"] for f in bc.construir_fragmentos()}
    faltan = {i for c in casos for i in c["contexto"]} - set(textos)
    assert not faltan, f"ids de contexto que no existen en el índice: {faltan}"
    if not a.simulado:
        llm_local.comprobar(a.modelo)

    salida = {}
    for nombre in a.configs:
        razonar = CONFIGS[nombre][2]
        llm = llm_local.LLMFalso(juez_falso) if a.simulado else llm_local.crear(a.modelo, razonar)
        print(f"Ejecutando {nombre} ({len(casos)} casos)...", flush=True)
        salida[nombre] = evaluar(nombre, casos, textos, llm)

    print(f"\n{'configuración':<24}{'detección':>10}{'falsas alarmas':>16}{'exactitud':>11}"
          f"{'sin veredicto':>15}{'segundos':>10}")
    for n, r in salida.items():
        print(f"{n:<24}{r['deteccion']:>10.2f}{r['falsas_alarmas']:>16.2f}{r['exactitud']:>11.2f}"
              f"{r['sin_veredicto']:>15}{r['segundos']:>10}")
    print("\nCasos en que el juez se equivocó (o no dio veredicto):")
    for n, r in salida.items():
        print(f"  {n:<24}{', '.join(r['errores']) or '—'}")
    if a.simulado:
        print("\n(modo simulado: no se escribe ningún archivo de resultados)")
        return
    (RAIZ / "resultados_llm_juez.json").write_text(
        json.dumps({"modelo": a.modelo, "resultados": salida}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nDetalle guardado en resultados_llm_juez.json")


if __name__ == "__main__":
    main()
