# EcoMarket RAG — Taller Práctico #2 (IA Generativa)

Sistema de Generación Aumentada por Recuperación (RAG) para el asistente de atención al cliente de EcoMarket. Extiende el asistente del Taller 1 con una base de conocimiento de cuatro tipos de documento, de modo que sus respuestas se apoyen en las políticas, el catálogo, las preguntas frecuentes y la guía de cuidado de la empresa.

## Decisiones de arquitectura

| Componente | Elección | Documento |
|---|---|---|
| Modelo de embeddings | `bge-m3` (código abierto, local con Ollama) | [Fase 1](fase1_componentes.md) |
| Base de datos vectorial | ChromaDB, modo persistente local | [Fase 1](fase1_componentes.md) |
| Segmentación | Por tipo de documento + fusión de las secciones 1.1–1.3 + enlace FAQ → política | [Fase 2](fase2_base_conocimiento.md) |
| Modelo generador | `qwen3:14b` (Ollama), el mismo del Taller 1 | [Fase 3](fase3_rag.md) |
| Orquestación | LangChain | [Fase 3](fase3_rag.md) |

Los modelos de embeddings se compararon sobre un conjunto de 18 preguntas de clientes y se validaron con 8 preguntas adicionales. La segmentación final alcanza recall@3 de 1.00 sobre el conjunto de evaluación y 0.88 sobre el de validación, frente a 0.83 y 0.75 de la mejor segmentación clásica evaluada, con 28 % a 38 % menos texto enviado al modelo. El detalle y las limitaciones están en los documentos de cada fase.

## Estado

| Fase | Contenido | Estado |
|---|---|---|
| 1 | Selección de componentes (embeddings y base vectorial) | Entregada |
| 2 | Base de conocimiento: documentos, segmentación e indexación | Entregada |
| 3 | Integración con LangChain y ejecución del código | En desarrollo: sistema y experimentos listos, [documento](fase3_rag.md); faltan la evaluación de respuestas y la comparación con el Taller 1 |

## Estructura del repositorio

```
taller2-rag/
├── README.md
├── fase1_componentes.md           Fase 1: selección y justificación de componentes
├── fase2_base_conocimiento.md     Fase 2: documentos, segmentación, indexación
├── fase3_rag.md                   Fase 3: sistema RAG, verificación y experimentos
├── requirements.txt               Dependencias con versiones fijadas
├── verificar_entorno.py           Comprueba paquetes, Ollama y modelos
├── corpus/
│   ├── politicas.md               Políticas de servicio (v3.1)
│   ├── catalogo_productos.csv     14 productos
│   ├── faq.json                   18 preguntas frecuentes (v2.4)
│   ├── guia_cuidado.md            Guía de cuidado de productos (v1.2)
│   └── enlaces_faq_politica.json  Tabla de enlaces FAQ → sección de políticas
├── conjunto_oro.json              18 preguntas de evaluación
├── conjunto_validacion.json       8 preguntas escritas antes de diseñar la fusión y el enlace
├── base_conocimiento.py           Construye los 54 fragmentos finales con ids y metadatos
├── indexar.py                     Crea o actualiza el índice Chroma (incremental, por hash)
├── rag_ecomarket.py               Sistema RAG: recuperación, enlace, prompt y generación
├── rag_ejemplo.py                 Script de ejemplo del curso, sin modificar (LangChain 0.x; no corre en este entorno)
├── rag_ejemplo_adaptado.py        El mismo script migrado a LangChain 1.x (línea base)
├── evaluar_rag.py                 Recall@3 de la recuperación sobre el índice real
├── llm_local.py                   Cliente mínimo de Ollama para los experimentos con LLM
├── exp_juez.py                    Experimento: qwen3:14b como juez de fidelidad
├── exp_correctivo.py              Experimento: RAG correctivo (el LLM revisa lo recuperado)
├── exp_enlaces_llm.py             Experimento: el LLM propone los enlaces FAQ -> política
├── exp_raptor_lite.py             Experimento: agrupar y resumir con LLM (RAPTOR de un nivel)
├── casos_juez.json                18 casos con veredicto conocido para el juez
├── conjunto_sin_respuesta.json    8 preguntas cuya respuesta no está en el corpus
├── medir_embeddings.py            Fase 1: comparación de modelos de embeddings
├── medir_chunking.py              Fase 2: comparación de estrategias de segmentación
├── medir_segmentacion.py          Fase 2: efecto de la fusión, el enlace y otras intervenciones
└── resultados_*.json              Resultados medidos, con el detalle por pregunta
```

## Requisitos

- Python 3.12
- [Ollama](https://ollama.com) en ejecución, con los modelos `bge-m3` y `qwen3:14b`

## Instalación

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull bge-m3
ollama pull qwen3:14b

python verificar_entorno.py
```

`verificar_entorno.py` revisa la versión de Python, los paquetes, que Ollama responda, que los modelos estén descargados y que el modelo de embeddings devuelva vectores de 1 024 dimensiones. Cada problema que encuentra indica el comando que lo corrige.

## Reproducir la evidencia de las Fases 1 y 2

Los resultados guardados en `resultados_*.json` se regeneran con los scripts de medición. Cada uno acepta `--simulado` para validar la tubería sin Ollama (los números de ese modo no tienen significado).

```bash
# Fase 1: comparación de modelos de embeddings (descargar antes los modelos que se quiera medir)
python medir_embeddings.py --modelos nomic-embed-text bge-m3

# Fase 2: comparación de estrategias de segmentación y de intervenciones sobre el índice
python medir_chunking.py
python medir_segmentacion.py
```

Los modelos de Voyage AI usados como referencia propietaria en la Fase 1 requieren una clave de API exportada como variable de entorno (`VOYAGE_API_KEY`) y se miden con `--modelos voyage:voyage-4`. La clave nunca se guarda en archivos del repositorio.

## Ejecutar el sistema RAG (Fase 3)

```bash
python indexar.py
python rag_ecomarket.py "¿Puedo devolver un termo que me llegó abollado?" --mostrar-contexto
python evaluar_rag.py
```

El script de ejemplo del curso está escrito para LangChain 0.x y no corre en este entorno; `rag_ejemplo_adaptado.py` es el mismo script, paso a paso, migrado a 1.x (`python rag_ejemplo_adaptado.py`, o `--evaluar` para medir su recuperación).

Opciones de `rag_ecomarket.py`: `--k`, `--sin-enlace`, `--prompt base|citas_exactas|estricto`, `--llm`, `--interactivo`. El detalle está en [fase3_rag.md](fase3_rag.md).

## Experimentos con el LLM local

Cuatro experimentos acotados que prueban técnicas que dependen del modelo generador. Cada uno acepta `--simulado` para validar la tubería sin Ollama.

```bash
python exp_juez.py                     # ¿detecta respuestas que el contexto no respalda?
python exp_correctivo.py               # el LLM juzga la recuperación y reintenta o se abstiene
python exp_enlaces_llm.py              # el LLM propone los enlaces y se comparan con la tabla manual
python exp_raptor_lite.py              # agrupar fragmentos y resumirlos: ¿se pierden las cifras?
```

Los resultados se guardan en `resultados_llm_*.json`. Cualquier experimento puede correr con un modelo de la API de Anthropic en lugar del local (`--modelo anthropic:claude-sonnet-5-5`); la clave se lee de la variable de entorno `ANTHROPIC_API_KEY` y, como la de Voyage, nunca se escribe en archivos del repositorio.

## Datos

El corpus es sintético y se redactó para este taller: sus documentos describen a una empresa ficticia. Las limitaciones de las evaluaciones que dependen de él se detallan en la sección de limitaciones de la Fase 2.
