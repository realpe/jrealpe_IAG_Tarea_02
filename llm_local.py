#!/usr/bin/env python3
"""
llm_local.py — Cliente mínimo de Ollama para los experimentos con LLM (exp_*.py).

Llama directamente a /api/chat con `requests`, sin LangChain, para controlar tres cosas que
importan al comparar experimentos: temperatura 0 con semilla fija, el modo de razonamiento de
qwen3 (`think`) y la salida estructurada en JSON.

Salida JSON: con razonamiento apagado se envía el esquema en `format` (Ollama restringe la
generación a ese esquema). Con razonamiento encendido no se envía `format`, porque restringir
la gramática desde el primer token puede interferir con el bloque de razonamiento; en su lugar
el esquema se pide en el prompt y la respuesta se extrae del texto.

Cada instancia cuenta llamadas y segundos, para reportar el costo de cada técnica.
"""
import json
import os
import time

import requests

OLLAMA = "http://localhost:11434"
MODELO = "qwen3:14b"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
# USD por millón de tokens (platform.claude.com/docs/en/models/overview, consultado 2026-10-02).
PRECIO_ANTHROPIC = {"claude-haiku-4-5-20251001": (1, 5), "claude-sonnet-5-5": (2, 10),
                    "claude-opus-5-5": (4, 20), "claude-fable-5-1": (10, 50)}


def extraer_json(texto):
    """Primer objeto JSON del texto, o {'_error': texto} si no hay uno válido."""
    try:
        return json.loads(texto)
    except (json.JSONDecodeError, TypeError):
        pass
    i, j = texto.find("{"), texto.rfind("}")
    if i != -1 and j > i:
        try:
            return json.loads(texto[i:j + 1])
        except json.JSONDecodeError:
            pass
    return {"_error": texto}


class LLM:
    def __init__(self, modelo=MODELO, razonar=False, num_ctx=8192):
        self.modelo, self.razonar, self.num_ctx = modelo, razonar, num_ctx
        self.llamadas, self.segundos = 0, 0.0

    def __call__(self, prompt, esquema=None):
        """Devuelve texto, o un dict si se pasa `esquema` (JSON Schema)."""
        if esquema is not None:
            prompt += ("\n\nResponde ÚNICAMENTE con un objeto JSON válido que cumpla este esquema:\n"
                       + json.dumps(esquema, ensure_ascii=False))
        cuerpo = {"model": self.modelo, "stream": False, "think": self.razonar,
                  "messages": [{"role": "user", "content": prompt}],
                  "options": {"temperature": 0, "seed": 0, "num_ctx": self.num_ctx}}
        if esquema is not None and not self.razonar:
            cuerpo["format"] = esquema
        t0 = time.perf_counter()
        r = requests.post(f"{OLLAMA}/api/chat", json=cuerpo, timeout=900)
        r.raise_for_status()
        self.segundos += time.perf_counter() - t0
        self.llamadas += 1
        contenido = r.json()["message"]["content"]
        return extraer_json(contenido) if esquema is not None else contenido


class LLMAnthropic:
    """Mismo contrato que LLM, contra la API de Anthropic (modelo 'anthropic:<id>'). La clave se
    lee de la variable de entorno ANTHROPIC_API_KEY y nunca se escribe en archivos. La API no
    restringe la salida a un esquema, así que el JSON se pide en el prompt y se extrae del texto.
    `razonar` se ignora: el modo de razonamiento de estos modelos no se activa aquí. No admite
    `temperature`, así que las corridas no son necesariamente repetibles. Acumula tokens para
    estimar el costo."""

    def __init__(self, modelo, razonar=False, max_tokens=1024):
        self.modelo, self.max_tokens = modelo, max_tokens
        self.llamadas, self.segundos = 0, 0.0
        self.tokens_entrada, self.tokens_salida = 0, 0
        self.clave = os.environ.get("ANTHROPIC_API_KEY")
        if not self.clave:
            raise SystemExit("Falta la variable de entorno ANTHROPIC_API_KEY")

    def __call__(self, prompt, esquema=None):
        if esquema is not None:
            prompt += ("\n\nResponde ÚNICAMENTE con un objeto JSON válido que cumpla este esquema, "
                       "sin texto adicional:\n" + json.dumps(esquema, ensure_ascii=False))
        # Sin `temperature`: los modelos actuales de la API lo rechazan ("deprecated for this
        # model"). A diferencia de Ollama (temperatura 0 y semilla fija), aquí la repetibilidad
        # entre corridas no está garantizada.
        cuerpo = {"model": self.modelo, "max_tokens": self.max_tokens,
                  "messages": [{"role": "user", "content": prompt}]}
        cabeceras = {"x-api-key": self.clave, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"}
        t0 = time.perf_counter()
        for intento in range(5):
            r = requests.post(ANTHROPIC_URL, json=cuerpo, headers=cabeceras, timeout=120)
            if r.status_code in (429, 529):          # límite de tasa o sobrecarga: esperar y repetir
                time.sleep(float(r.headers.get("retry-after", 5 * (intento + 1))))
                continue
            break
        if r.status_code >= 400:
            # El cuerpo trae la causa (modelo inexistente, parámetro inválido, crédito agotado).
            raise SystemExit(f"API de Anthropic respondió {r.status_code}: {r.text[:600]}")
        self.segundos += time.perf_counter() - t0
        self.llamadas += 1
        datos = r.json()
        uso = datos.get("usage", {})
        self.tokens_entrada += uso.get("input_tokens", 0)
        self.tokens_salida += uso.get("output_tokens", 0)
        contenido = "".join(b.get("text", "") for b in datos["content"] if b.get("type") == "text")
        return extraer_json(contenido) if esquema is not None else contenido

    def costo_usd(self):
        pe, ps = PRECIO_ANTHROPIC.get(self.modelo, (0, 0))
        return (self.tokens_entrada * pe + self.tokens_salida * ps) / 1e6


def crear(modelo=MODELO, razonar=False):
    """LLM local (Ollama) o de API según el prefijo: 'anthropic:claude-sonnet-5-5'."""
    if modelo.startswith("anthropic:"):
        return LLMAnthropic(modelo.split(":", 1)[1], razonar)
    return LLM(modelo, razonar)


def comprobar(modelo=MODELO):
    """Verifica que el modelo esté disponible (Ollama instalado y modelo descargado, o clave de API)."""
    if modelo.startswith("anthropic:"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("Falta la variable de entorno ANTHROPIC_API_KEY (export ANTHROPIC_API_KEY=...)")
        return
    comprobar_ollama(modelo)


def resumen_costo(llm):
    """Línea para imprimir al final de un experimento."""
    base = f"Llamadas al LLM: {llm.llamadas} | segundos: {llm.segundos:.1f}"
    if isinstance(llm, LLMAnthropic):
        base += (f" | tokens {llm.tokens_entrada} entrada / {llm.tokens_salida} salida"
                 f" | costo aprox. USD {llm.costo_usd():.3f}")
    return base


class LLMFalso:
    """Sustituto para probar la tubería sin Ollama: `fn(prompt, esquema)` decide la respuesta."""

    def __init__(self, fn):
        self.fn, self.llamadas, self.segundos = fn, 0, 0.0

    def __call__(self, prompt, esquema=None):
        self.llamadas += 1
        return self.fn(prompt, esquema)


def comprobar_ollama(modelo=MODELO):
    try:
        r = requests.get(f"{OLLAMA}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(f"Ollama no responde en {OLLAMA} ({type(e).__name__})")
    nombres = [m["name"] for m in r.json()["models"]]
    if not any(n == modelo or n.startswith(modelo + ":") for n in nombres):
        raise SystemExit(f"{modelo} no está instalado (ollama pull {modelo})")
