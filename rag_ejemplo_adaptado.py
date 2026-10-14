# -*- coding: utf-8 -*-
"""
rag_ejemplo_adaptado.py — El `rag_ejemplo.py` del curso, migrado a LangChain 1.x y al corpus de EcoMarket.

Conserva los siete pasos y la estructura del script original. Cada cambio respecto al original
está marcado con "# CAMBIO:" y su motivo. Los cambios son de tres tipos:
  - API: el original usa la serie 0.x de LangChain (`langchain_community`, `langchain.chains`,
    `RetrievalQA`, `.run()`), que no existe en la 1.4.3 fijada en requirements.txt.
  - Modelos: el original usa `llama2` para embeddings y generación; aquí se usan los modelos
    elegidos en las Fases 1 y 3 (`bge-m3` y `qwen3:14b`).
  - Corpus: el original embebe un párrafo de ejemplo en el código; aquí se leen los cuatro
    documentos de corpus/ como texto plano.
Lo que NO cambia a propósito: el chunking recursivo de 1000/200 y la base vectorial en memoria,
para que este script sea la línea base "tal como propone el ejemplo" frente a rag_ecomarket.py.

Uso:
    python rag_ejemplo_adaptado.py             # consola interactiva, como el original
    python rag_ejemplo_adaptado.py --evaluar   # recall@k de su recuperación sobre las 26 preguntas
    python rag_ejemplo_adaptado.py --simulado  # prueba la tubería sin Ollama
"""
import argparse
import json
import sys
from pathlib import Path

# CAMBIO (API): en LangChain 1.x cada integración vive en su paquete.
#   langchain_community.llms.Ollama            -> langchain_ollama.ChatOllama
#   langchain_community.embeddings.OllamaEmbeddings -> langchain_ollama.OllamaEmbeddings
#   langchain_community.vectorstores.Chroma    -> langchain_chroma.Chroma
#   langchain.text_splitter                    -> langchain_text_splitters
#   langchain.chains.RetrievalQA               -> una cadena LCEL (ver paso 6)
#   langchain_community.document_loaders.TextLoader -> lectura directa con pathlib (paso 1)
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAIZ = Path(__file__).parent
OLLAMA = "http://localhost:11434"
MODELO_EMBEDDINGS = "bge-m3"     # CAMBIO (modelos): el original usa "llama2" para embeddings
MODELO_LLM = "qwen3:14b"         # CAMBIO (modelos): el original usa "llama2" para generar


def construir(simulado=False):
    # Paso 1: Definir la base de conocimiento de la empresa (documentos).
    # CAMBIO (corpus): el original define DOCUMENTO_ECO_MARKET como una cadena en el código.
    # Aquí se leen los cuatro documentos reales de corpus/ como texto plano, que es lo que
    # haría TextLoader. El CSV y el JSON entran tal cual (sin interpretar sus campos): es la
    # forma ingenua a propósito, para medir qué aporta la segmentación por tipo de la Fase 2.
    documentos = [(RAIZ / "corpus" / n).read_text(encoding="utf-8")
                  for n in ("politicas.md", "catalogo_productos.csv", "faq.json", "guia_cuidado.md")]

    # Paso 2: Crear el modelo de embeddings.
    if simulado:
        from indexar import EmbeddingsSimulados
        embeddings = EmbeddingsSimulados()
    else:
        from langchain_ollama import OllamaEmbeddings
        try:
            embeddings = OllamaEmbeddings(model=MODELO_EMBEDDINGS, base_url=OLLAMA)
        except Exception as e:  # noqa: BLE001 - mismo manejo que el original
            print(f"Error al inicializar el modelo de embeddings de Ollama: {e}")
            print(f"Asegúrate de tener Ollama corriendo y el modelo '{MODELO_EMBEDDINGS}' descargado.")
            sys.exit(1)

    # Paso 3: Dividir el documento en fragmentos (chunks) para una búsqueda eficiente.
    # Sin cambios: mismos parámetros del original (1000 caracteres, solapamiento 200).
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents(documentos)

    # Paso 4: Crear la base de datos vectorial en memoria y cargar los fragmentos.
    # Sin cambios de diseño: Chroma en memoria (sin persist_directory), como el original.
    print(f"Creando y cargando la base de datos vectorial ({len(docs)} fragmentos)...")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

    # Paso 5: Configurar el modelo de lenguaje (LLM).
    if simulado:
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        llm = FakeListChatModel(responses=["(respuesta simulada)"])
    else:
        from langchain_ollama import ChatOllama
        try:
            # CAMBIO (API): ChatOllama en lugar de Ollama (LLM de texto). reasoning=False apaga el
            # bloque <think> de qwen3; temperature=0 para respuestas repetibles.
            llm = ChatOllama(model=MODELO_LLM, base_url=OLLAMA, temperature=0, reasoning=False)
        except Exception as e:  # noqa: BLE001
            print(f"Error al inicializar el modelo LLM de Ollama: {e}")
            print(f"Asegúrate de tener Ollama corriendo y el modelo '{MODELO_LLM}' descargado.")
            sys.exit(1)

    # Paso 6: Crear el "chain" de RAG.
    # CAMBIO (API): RetrievalQA.from_chain_type(chain_type="stuff") ya no existe en 1.x. "stuff"
    # significa concatenar los documentos recuperados dentro del prompt; eso es exactamente lo
    # que hace esta cadena LCEL. El prompt es el que RetrievalQA usaba por defecto, traducido.
    retriever = vectorstore.as_retriever()          # k=4 por defecto, igual que el original
    prompt = ChatPromptTemplate.from_template(
        "Usa los siguientes fragmentos de contexto para responder la pregunta del final. "
        "Si no sabes la respuesta, di que no lo sabes; no intentes inventar una respuesta.\n\n"
        "{context}\n\nPregunta: {question}\nRespuesta útil:")

    def unir(docs_):
        return "\n\n".join(d.page_content for d in docs_)

    qa_chain = ({"context": retriever | unir, "question": RunnablePassthrough()}
                | prompt | llm | StrOutputParser())
    return qa_chain, retriever, docs


def evaluar(retriever, docs):
    """Recall@k de la recuperación de este script, con el criterio de la Fase 2."""
    import medir_embeddings as me
    print(f"Fragmentos en el índice: {len(docs)} | tamaño medio {sum(len(d.page_content) for d in docs) // len(docs)} caracteres")
    for nombre in ("conjunto_oro.json", "conjunto_validacion.json"):
        preguntas = json.loads((RAIZ / nombre).read_text(encoding="utf-8"))["preguntas"]
        for k in (3, 4):
            retriever.search_kwargs["k"] = k
            ok = crit_ok = 0
            crit = sum(p["critico"] for p in preguntas)
            chars = 0
            fallan = []
            for p in preguntas:
                rec = retriever.invoke(p["pregunta"])
                union = me.sin_acentos(" ".join(d.page_content for d in rec))
                chars += sum(len(d.page_content) for d in rec)
                if all(me.contiene(union, r) for r in p["requiere"]):
                    ok += 1
                    crit_ok += p["critico"]
                else:
                    fallan.append(p["id"])
            print(f"  {nombre} k={k}: recall {ok / len(preguntas):.2f} | críticas {crit_ok / crit:.2f} | "
                  f"contexto medio {chars // len(preguntas)} caracteres | fallan: {', '.join(fallan) or '—'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evaluar", action="store_true")
    ap.add_argument("--simulado", action="store_true")
    a = ap.parse_args()
    qa_chain, retriever, docs = construir(a.simulado)

    if a.evaluar:
        evaluar(retriever, docs)
        sys.exit(0)

    # Paso 7: Interactuar con el sistema RAG.
    print("\n¡El sistema RAG para EcoMarket está listo!")
    print("Escribe una pregunta y presiona Enter. Escribe 'salir' para terminar.")
    while True:
        try:
            pregunta = input("\nTu pregunta: ")
        except (EOFError, KeyboardInterrupt):
            break
        if pregunta.lower() == "salir":
            break
        print("Buscando y generando respuesta...")
        respuesta = qa_chain.invoke(pregunta)     # CAMBIO (API): .run() se reemplazó por .invoke()
        print("\nRespuesta del Asistente:\n")
        print(respuesta)
