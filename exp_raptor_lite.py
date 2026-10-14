#!/usr/bin/env python3
"""
exp_raptor_lite.py — Versión mínima de RAPTOR: agrupar fragmentos y resumir cada grupo con un LLM.

RAPTOR construye un árbol: agrupa fragmentos por similitud, resume cada grupo con un LLM, y repite
sobre los resúmenes; en la consulta se puede recuperar a cualquier nivel. Aquí se construye un solo
nivel (54 fragmentos -> k grupos -> k resúmenes) y se miden los dos riesgos que importan en un
dominio de políticas:

  1. ¿Se pierden las cifras? Fracción de los números de los fragmentos de cada grupo (30 días,
     72 horas, $500.000...) que sobreviven en su resumen, y números que el resumen inventa.
  2. ¿El resumen enruta bien? Para cada pregunta, se toma el resumen más cercano y se mira si el
     grupo al que pertenece contiene, entre todos sus fragmentos, el texto que la respuesta
     necesita (`requiere`). Se compara con la recuperación por fragmento (top-3 sin enlace).
     Entregar un grupo completo cuesta más contexto: se reporta el tamaño medio.

Además se imprime la composición de cada grupo (de qué documentos vienen sus fragmentos) y cada
resumen, para leerlos: la calidad de un resumen no se agota en estas métricas.

El agrupamiento es k-means esférico (coseno) con semilla fija; con otra semilla o k los grupos cambian.

Uso:
    python exp_raptor_lite.py              # requiere Ollama con qwen3:14b y bge-m3
    python exp_raptor_lite.py --k 6
    python exp_raptor_lite.py --simulado   # prueba la tubería sin Ollama
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

import base_conocimiento as bc
import llm_local
import medir_embeddings as me

RAIZ = Path(__file__).parent
PROMPT = """Resume en español, en máximo 90 palabras, los siguientes fragmentos de la base de \
conocimiento de EcoMarket. Conserva TODAS las cifras, plazos, montos y condiciones tal como \
aparecen; no agregues información que no esté en los fragmentos.

FRAGMENTOS:
{fragmentos}"""


def kmeans(X, k, semilla=0, iteraciones=100):
    """K-means esférico: X ya viene normalizada, así que similitud = producto punto."""
    rng = np.random.default_rng(semilla)
    centros = [X[rng.integers(len(X))]]
    for _ in range(k - 1):                       # inicialización k-means++
        d = (1 - np.max(X @ np.array(centros).T, axis=1)).clip(min=0) ** 2
        p = d / d.sum() if d.sum() > 0 else None
        centros.append(X[rng.choice(len(X), p=p)])
    C = np.array(centros)
    for _ in range(iteraciones):
        asign = np.argmax(X @ C.T, axis=1)
        nuevo = np.array([X[asign == j].mean(axis=0) if (asign == j).any() else C[j] for j in range(k)])
        nuevo /= np.linalg.norm(nuevo, axis=1, keepdims=True)
        if np.allclose(nuevo, C):
            break
        C = nuevo
    return asign


# Cifras escritas en letras en el corpus (la guía dice "doscientos lavados", "un litro"). Se
# convierten a dígitos antes de comparar, para no contar como inventada una cifra reformateada.
PALABRAS_NUM = {"un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
                "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13,
                "catorce": 14, "quince": 15, "dieciseis": 16, "dieciséis": 16, "diecisiete": 17,
                "dieciocho": 18, "diecinueve": 19, "veinte": 20,
                "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
                "ochenta": 80, "noventa": 90, "cien": 100, "doscientos": 200, "trescientos": 300,
                "quinientos": 500, "mil": 1000}


def numeros(texto):
    """Conjunto de cifras del texto, normalizadas: sin separadores de miles ('89.000' y '89000'
    son la misma cifra) y con las escritas en letras convertidas a dígitos."""
    cifras = set()
    for m in re.findall(r"\d+(?:[.,]\d{3})*(?:[.,]\d{1,2})?", texto):
        # '5,5' o '2.5' con 1-2 decimales se conservan; '89.000' y '1.200.000' pierden los puntos
        cifras.add(re.sub(r"[.,](?=\d{3}(?:\D|$))", "", m))
    for palabra in re.findall(r"[a-záéíóú]+", texto.lower()):
        if palabra in PALABRAS_NUM:
            cifras.add(str(PALABRAS_NUM[palabra]))
    return cifras


def resumir_falso(prompt, _esquema):
    """Para --simulado: 'resume' quedándose con los primeros 200 caracteres."""
    return prompt.split("FRAGMENTOS:\n")[1][:200]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=8, help="número de grupos (defecto 8)")
    ap.add_argument("--modelo", default=llm_local.MODELO)
    ap.add_argument("--simulado", action="store_true")
    a = ap.parse_args()

    frags = bc.construir_fragmentos()
    textos = [f["texto"] for f in frags]
    norm = [me.sin_acentos(t) for t in textos]
    if a.simulado:
        llm, modelo_emb, embed = llm_local.LLMFalso(resumir_falso), "simulado-hash", me.embed_simulado
    else:
        llm_local.comprobar(a.modelo)
        llm, modelo_emb, embed = llm_local.crear(a.modelo), "bge-m3", me.embed_ollama

    X = embed(modelo_emb, textos, "document")
    grupo = kmeans(X, a.k)
    print(f"{len(frags)} fragmentos en {a.k} grupos. Resumiendo...", flush=True)

    resumenes, retencion = [], []
    for g in range(a.k):
        miembros = [i for i in range(len(frags)) if grupo[i] == g]
        if not miembros:
            resumenes.append("")
            continue
        r = llm(PROMPT.format(fragmentos="\n\n".join(textos[i] for i in miembros)))
        resumenes.append(r.strip())
        n_orig = numeros(" ".join(textos[i] for i in miembros))
        n_res = numeros(r)
        retencion.append({"grupo": g, "miembros": len(miembros), "ids": [frags[i]["id"] for i in miembros],
                          "numeros": len(n_orig), "conservados": len(n_orig & n_res),
                          "perdidos": sorted(n_orig - n_res), "inventados": sorted(n_res - n_orig)})
        origen = {}
        for i in miembros:
            origen[frags[i]["fuente"]] = origen.get(frags[i]["fuente"], 0) + 1
        print(f"\n[grupo {g}] {len(miembros)} fragmentos {origen}")
        print("  " + resumenes[-1].replace("\n", "\n  "))

    tot_o = sum(r["numeros"] for r in retencion)
    tot_c = sum(r["conservados"] for r in retencion)
    print(f"\n== Cifras ==\n  conservadas {tot_c}/{tot_o} ({tot_c / tot_o:.0%} de los números de los fragmentos)")
    inv = [(r["grupo"], r["inventados"]) for r in retencion if r["inventados"]]
    print(f"  números que el resumen añade y no estaban en los fragmentos: {inv or '—'}")
    print(f"  (la métrica solo ve cifras; un resumen puede conservarlas todas y cambiar su sentido)")
    perdidos = [r for r in retencion if r["conservados"] < r["numeros"]]
    print(f"  grupos que pierden alguna cifra: {len(perdidos)}/{len(retencion)}")

    R = embed(modelo_emb, [r or " " for r in resumenes], "document")
    texto_grupo = {g: " ".join(norm[i] for i in range(len(frags)) if grupo[i] == g) for g in range(a.k)}
    chars_grupo = {g: sum(len(textos[i]) for i in range(len(frags)) if grupo[i] == g) for g in range(a.k)}

    print("\n== Enrutamiento: ¿el grupo del mejor resumen contiene lo que la respuesta necesita? ==")
    salida = {}
    for titulo, archivo in (("evaluación", "conjunto_oro.json"), ("validación", "conjunto_validacion.json")):
        preguntas = json.loads((RAIZ / archivo).read_text(encoding="utf-8"))["preguntas"]
        Q = embed(modelo_emb, [p["pregunta"] for p in preguntas], "query")
        ok_grupo = ok_frag = 0
        tam, tam_frag = [], []
        for p, q in zip(preguntas, Q):
            g = int(np.argmax(R @ q))
            ok_grupo += all(me.contiene(texto_grupo[g], r) for r in p["requiere"])
            tam.append(chars_grupo[g])
            mejores = np.argsort(-(X @ q))[:3]
            top3 = " ".join(norm[i] for i in mejores)
            tam_frag.append(sum(len(textos[i]) for i in mejores))
            ok_frag += all(me.contiene(top3, r) for r in p["requiere"])
        n = len(preguntas)
        salida[titulo] = {"grupo_top1": round(ok_grupo / n, 3), "fragmentos_top3": round(ok_frag / n, 3),
                          "chars_grupo_medio": round(float(np.mean(tam))),
                          "chars_top3_medio": round(float(np.mean(tam_frag)))}
        print(f"  {titulo}: grupo del mejor resumen {ok_grupo / n:.2f} (entrega ~{np.mean(tam):.0f} caracteres) | "
              f"top-3 por fragmento {ok_frag / n:.2f} (entrega ~{np.mean(tam_frag):.0f} caracteres, sin enlace)")
    print("\n" + llm_local.resumen_costo(llm))
    if a.simulado:
        print("(modo simulado: no se escribe ningún archivo de resultados)")
        return
    (RAIZ / "resultados_llm_raptor.json").write_text(json.dumps(
        {"modelo": a.modelo, "k": a.k, "resumenes": resumenes, "cifras": retencion, "enrutamiento": salida},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("Detalle guardado en resultados_llm_raptor.json")


if __name__ == "__main__":
    main()
