#!/usr/bin/env python3
"""
medir_chunking.py — Evidencia de la Fase 2 (Taller 2): comparar estrategias de segmentación
(chunking) sobre el corpus de EcoMarket con el modelo de embedding ya elegido (bge-m3).

Las tres estrategias clásicas que pide el enunciado se aplican tal como se usarían "por
defecto": cada documento se lee como texto plano y se parte sin conocer su estructura.
    fijo 400/80      ventana de 400 caracteres con solape de 80
    fijo 800/160     ventana de 800 caracteres con solape de 160
    párrafos         corte en cada línea en blanco
    recursivo 500/50 RecursiveCharacterTextSplitter de LangChain (separadores \\n\\n, \\n, ' ')
    recursivo 1000/100
Se comparan con la estrategia propia, que conoce la forma de cada documento:
    por tipo         políticas por sección, catálogo por fila, FAQ por par, guía por párrafo
    por tipo + F+E   igual, con la fusión de 1.1-1.3 y el enlace FAQ -> política (Fase 2)

Métrica: la misma de medir_embeddings.py (los textos de `requiere` deben aparecer en la unión
de los k primeros fragmentos), sobre conjunto_oro.json y conjunto_validacion.json. Como cada
estrategia produce fragmentos de tamaño distinto, se reporta además cuántos caracteres llegan
al LLM con k=3 (volumen de contexto).

Uso:
    pip install langchain-text-splitters        # solo para las estrategias recursivas
    python medir_chunking.py                    # bge-m3
    python medir_chunking.py --modelo qwen3-embedding:4b
    python medir_chunking.py --simulado         # sin Ollama: valida la tubería
"""
import argparse
import json
import re

import requests

import medir_embeddings as me
import medir_segmentacion as ms

RAIZ = me.RAIZ
ARCHIVOS = {"politicas": "politicas.md", "catalogo": "catalogo_productos.csv",
            "faq": "faq.json", "guia": "guia_cuidado.md"}


# ------------------------------------------------------------------ cortadores de texto plano
def cortar_fijo(texto, tam, solape):
    """Ventana deslizante de `tam` caracteres; corta sin respetar palabras ni secciones."""
    paso = tam - solape
    trozos = [texto[i:i + tam] for i in range(0, max(len(texto) - solape, 1), paso)]
    return [t.strip() for t in trozos if t.strip()]


def cortar_parrafos(texto):
    return [p.strip() for p in re.split(r"\n\s*\n", texto) if p.strip()]


def cortador_recursivo(tam, solape):
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        return None
    return RecursiveCharacterTextSplitter(chunk_size=tam, chunk_overlap=solape).split_text


def fragmentos_planos(cortar):
    frags = []
    for fuente, archivo in ARCHIVOS.items():
        texto = (RAIZ / "corpus" / archivo).read_text(encoding="utf-8")
        for t in cortar(texto):
            frags.append({"fuente": fuente, "texto": t, "indexado": t})
    return frags


def estrategias():
    """Lista de (nombre, fragmentos, mapa, enlace)."""
    lista = []
    for nombre, cortar in [("fijo 400/80", lambda t: cortar_fijo(t, 400, 80)),
                           ("fijo 800/160", lambda t: cortar_fijo(t, 800, 160)),
                           ("párrafos", cortar_parrafos)]:
        lista.append((nombre, fragmentos_planos(cortar), {}, False))
    for tam, solape in ((500, 50), (1000, 100)):
        cortar = cortador_recursivo(tam, solape)
        if cortar is None:
            print("[omitido] estrategias recursivas: falta `pip install langchain-text-splitters`")
            break
        lista.append((f"recursivo {tam}/{solape}", fragmentos_planos(cortar), {}, False))
    frags, mapa = ms.construir(False, False)
    lista.append(("por tipo", frags, mapa, False))
    frags, mapa = ms.construir(True, False)
    lista.append(("por tipo + F+E", frags, mapa, True))
    return lista


# ------------------------------------------------------------------ salida
def imprimir(titulo, filas):
    print(f"\n{titulo}")
    print(f"{'estrategia':<18}{'frag.':>6}{'R@3':>6}{'R@5':>6}{'crít@3':>8}{'chars@3':>9}  fallan@3")
    print("-" * 82)
    for nombre, r in filas:
        print(f"{nombre:<18}{r['fragmentos_indice']:>6}{r['recall']['@3']:>6.2f}"
              f"{r['recall']['@5']:>6.2f}{r['recall_criticas@3']:>8.2f}{r['chars_ctx@3']:>9}  "
              f"{' '.join(i[-2:] for i in r['fallan@3'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--modelo", default="bge-m3")
    ap.add_argument("--simulado", action="store_true")
    args = ap.parse_args()

    conjuntos = {
        "ORO (18 preguntas)": "conjunto_oro.json",
        "VALIDACIÓN (8 preguntas)": "conjunto_validacion.json",
    }
    preguntas = {t: json.loads((RAIZ / a).read_text(encoding="utf-8"))["preguntas"]
                 for t, a in conjuntos.items()}

    if args.simulado:
        modelo, embed = "simulado-hash", me.embed_simulado
    else:
        try:
            instalados = me.modelos_instalados()
        except requests.RequestException as e:
            raise SystemExit(f"No se pudo consultar Ollama en {me.OLLAMA}: {e}")
        modelo = next((n for n in instalados if n == args.modelo or n.startswith(args.modelo + ":")), None)
        if modelo is None:
            raise SystemExit(f"{args.modelo} no está instalado (ollama pull {args.modelo})")
        embed = me.embed_ollama
    print(f"Modelo: {modelo}")

    lista = estrategias()
    resumen = ", ".join(f"{n}: {len(f)}" for n, f, _m, _e in lista)
    print(f"Fragmentos por estrategia -> {resumen}")

    # Precálculo de embeddings en una pasada por tipo (mismo esquema que medir_segmentacion.py)
    cache = {}
    q_pre, d_pre = ms.prefijos(modelo)
    docs = list(dict.fromkeys(d_pre + f["indexado"] for _n, fr, _m, _e in lista for f in fr))
    qs = list(dict.fromkeys(q_pre + p["pregunta"] for ps in preguntas.values() for p in ps))
    for tipo, textos in (("document", docs), ("query", qs)):
        for t, v in zip(textos, embed(modelo, textos, tipo)):
            cache[(modelo, tipo, t)] = v

    salida = {}
    for nombre, frags, mapa, enlace in lista:
        for titulo, ps in preguntas.items():
            salida.setdefault(titulo, []).append(
                (nombre, ms.evaluar(frags, mapa, enlace, ps, modelo, embed, cache)))
    for titulo, filas in salida.items():
        imprimir(titulo, filas)

    if args.simulado:
        print("\n(modo simulado: no se escribe ningún archivo de resultados)")
        return
    destino = f"resultados_chunking_{modelo.replace(':', '_')}.json"
    (RAIZ / destino).write_text(json.dumps(
        {"modelo": modelo, "resultados": {t: dict(f) for t, f in salida.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetalle guardado en {destino}")


if __name__ == "__main__":
    main()
