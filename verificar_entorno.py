#!/usr/bin/env python3
"""
verificar_entorno.py — Comprueba que el entorno puede ejecutar el sistema RAG de EcoMarket.

Revisa, en este orden: versión de Python, paquetes instalados (con sus versiones), que Ollama
responda, que estén descargados los modelos necesarios, y que el modelo de embeddings devuelva
vectores del tamaño esperado. Cada fallo indica el comando que lo corrige. No genera texto con
el modelo grande, de modo que termina en pocos segundos.

Uso:  python verificar_entorno.py
"""
import sys
from importlib import metadata

import requests

OLLAMA = "http://localhost:11434"
MODELO_EMBEDDINGS = "bge-m3"
MODELO_LLM = "qwen3:14b"
DIM_ESPERADA = 1024
PAQUETES = ["numpy", "requests", "langchain", "langchain-core", "langchain-text-splitters",
            "langchain-ollama", "langchain-chroma", "chromadb"]

fallos = []


def ok(msg):
    print(f"  [ok]    {msg}")


def falla(msg, arreglo):
    print(f"  [FALLA] {msg}\n          -> {arreglo}")
    fallos.append(msg)


print("1. Python")
if sys.version_info >= (3, 10):
    ok(f"Python {sys.version.split()[0]}")
else:
    falla(f"Python {sys.version.split()[0]} es antiguo",
          "crear el entorno con python3.12 -m venv .venv")

print("2. Paquetes")
for p in PAQUETES:
    try:
        ok(f"{p} {metadata.version(p)}")
    except metadata.PackageNotFoundError:
        falla(f"falta {p}", "pip install -r requirements.txt")

print("3. Ollama")
instalados = []
try:
    r = requests.get(f"{OLLAMA}/api/tags", timeout=5)
    r.raise_for_status()
    instalados = [m["name"] for m in r.json()["models"]]
    ok(f"responde en {OLLAMA} ({len(instalados)} modelos instalados)")
except requests.RequestException as e:
    falla(f"Ollama no responde en {OLLAMA} ({type(e).__name__})",
          "abrir la aplicación de Ollama o ejecutar `ollama serve`")

print("4. Modelos")
for modelo in (MODELO_EMBEDDINGS, MODELO_LLM):
    if any(n == modelo or n.startswith(modelo + ":") for n in instalados):
        ok(f"{modelo} instalado")
    elif instalados:
        falla(f"{modelo} no está instalado", f"ollama pull {modelo}")

print("5. Embeddings")
if any(n.startswith(MODELO_EMBEDDINGS) for n in instalados):
    try:
        from langchain_ollama import OllamaEmbeddings
        vector = OllamaEmbeddings(model=MODELO_EMBEDDINGS, base_url=OLLAMA).embed_query("prueba")
        if len(vector) == DIM_ESPERADA:
            ok(f"{MODELO_EMBEDDINGS} devuelve vectores de {len(vector)} dimensiones")
        else:
            falla(f"dimensión {len(vector)} (se esperaba {DIM_ESPERADA})",
                  "revisar que el modelo descargado sea bge-m3")
    except Exception as e:  # noqa: BLE001 - el verificador debe informar cualquier error
        falla(f"no se pudo generar un embedding ({type(e).__name__}: {e})",
              "revisar la instalación de langchain-ollama y que Ollama esté activo")
else:
    print("  [omitido] requiere bge-m3 instalado")

print()
if fallos:
    print(f"Entorno incompleto: {len(fallos)} problema(s). Corrige los puntos marcados y vuelve a ejecutar.")
    sys.exit(1)
print("Entorno listo.")
