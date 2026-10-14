#!/usr/bin/env python3
"""
evaluar_rag.py — Mide el recall de la recuperación del sistema RAG real (índice Chroma +
bge-m3 + enlace) con el mismo criterio de la Fase 2, y lo compara con lo medido allí.

Criterio (idéntico a medir_segmentacion.py): una pregunta se considera recuperada si TODOS los
textos de `requiere` aparecen en la unión de los fragmentos entregados al modelo. No invoca al
modelo generador, por lo que termina en segundos.

Valores de referencia de la Fase 2 (bge-m3, por tipo + F + E, k=3):
    conjunto de evaluación : recall@3 1.00, críticas 1.00
    conjunto de validación : recall@3 0.88, críticas 0.80

Uso:
    python evaluar_rag.py                       # configuración final
    python evaluar_rag.py --sin-enlace          # para ver cuánto aporta E
    python evaluar_rag.py --k 5
"""
import argparse
import json
from pathlib import Path

import medir_embeddings as me
import rag_ecomarket as rag

RAIZ = Path(__file__).parent
REFERENCIA = {"conjunto_oro.json": (1.00, 1.00), "conjunto_validacion.json": (0.88, 0.80)}


def evaluar(recuperador, preguntas):
    detalle = []
    for p in preguntas:
        docs = recuperador.recuperar(p["pregunta"])
        union = me.sin_acentos(" ".join(d.page_content for d in docs))
        ok = all(me.contiene(union, r) for r in p["requiere"])
        detalle.append({"id": p["id"], "critico": p["critico"], "ok": ok,
                        "ids": [d.id for d in docs]})
    total = sum(d["ok"] for d in detalle) / len(detalle)
    crit = [d for d in detalle if d["critico"]]
    return total, sum(d["ok"] for d in crit) / len(crit), detalle


def cargar(nombre):
    return json.loads((RAIZ / nombre).read_text(encoding="utf-8"))["preguntas"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=rag.K)
    ap.add_argument("--sin-enlace", action="store_true")
    ap.add_argument("--ruta-db", default=str(rag.RUTA_DB))
    ap.add_argument("--simulado", action="store_true",
                    help="embeddings falsos (solo prueba la tubería; el recall no significa nada)")
    a = ap.parse_args()

    emb = None
    if a.simulado:
        from indexar import EmbeddingsSimulados
        emb = EmbeddingsSimulados()
    recuperador = rag.crear_recuperador(Path(a.ruta_db), a.k, not a.sin_enlace, emb)

    print(f"k={a.k} | enlace {'apagado' if a.sin_enlace else 'activo'}")
    for nombre, (ref_total, ref_crit) in REFERENCIA.items():
        total, crit, detalle = evaluar(recuperador, cargar(nombre))
        print(f"\n{nombre}: recall@{a.k} {total:.2f} (críticas {crit:.2f})"
              f"   [Fase 2 con k=3: {ref_total:.2f} / {ref_crit:.2f}]")
        for d in detalle:
            if not d["ok"]:
                print(f"   falla {d['id']}{' (crítica)' if d['critico'] else ''}: {', '.join(d['ids'])}")
