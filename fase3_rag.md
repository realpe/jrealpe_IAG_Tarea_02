# Fase 3 — Sistema RAG con LangChain

Este documento describe el sistema RAG de EcoMarket, la verificación de que reproduce las mediciones de la Fase 2, los experimentos realizados sobre el código y el prompt, y las limitaciones y supuestos.

## 1. Alcance y estado

El sistema recupera fragmentos de la base de conocimiento (Fase 2), los entrega como contexto a `qwen3:14b`, el mismo modelo generador del Taller 1, y devuelve la respuesta junto con los fragmentos usados.

| Componente | Archivo | Estado |
|---|---|---|
| Construcción de fragmentos | `base_conocimiento.py` | Terminado |
| Indexación incremental en Chroma | `indexar.py` | Terminado |
| Recuperación, enlace, prompt y generación | `rag_ecomarket.py` | Terminado |
| Evaluación de la recuperación sobre el índice real | `evaluar_rag.py` | Terminado |
| Experimentos con el LLM local (juez, RAG correctivo, enlaces por LLM, resúmenes por grupo) | `exp_*.py`, `llm_local.py` | Terminado |
| Evaluación de respuestas completas (citas, abstención) | — | Pendiente |
| Integración con el clasificador del Taller 1 | — | Pendiente |
| Script del curso migrado a LangChain 1.x, como línea base | `rag_ejemplo_adaptado.py` | Terminado |
| Comparación con la implementación artesanal del Taller 1 | — | Pendiente |

El script de ejemplo del curso (`rag_ejemplo.py`) se recibió el 13 de octubre, cuando el sistema ya estaba construido. Está escrito para la serie 0.x de LangChain y no corre con la 1.4.3 fijada en `requirements.txt`: sus seis importaciones fallan (`langchain_community`, `langchain.text_splitter` y `langchain.chains` no existen en 1.x), usa `llama2` como modelo de embeddings y de generación, y `qa_chain.run()` fue retirado. La sección 2 bis muestra la correspondencia entre sus siete pasos y este sistema, y `rag_ejemplo_adaptado.py` es el script original migrado paso a paso, que se conserva como línea base.

## 2. Flujo de una consulta

```
pregunta
  [1] RECUPERAR   embedding de la pregunta (bge-m3) → k fragmentos más cercanos en Chroma
  [2] ENLAZAR     cada FAQ recuperada añade la sección de políticas que la gobierna
  [3] ARMAR       los fragmentos, con su id, se insertan en el prompt como CONTEXTO
  [4] GENERAR     qwen3:14b responde con ese contexto
  → respuesta + fragmentos usados
```

Los pasos 3 y 4 forman una cadena LCEL (`RunnableLambda | prompt | llm | StrOutputParser`). Los pasos 1 y 2 se ejecutan en paralelo con la conservación de la pregunta mediante `RunnableParallel`, y `.assign` añade la respuesta sin perder los fragmentos.

## 2 bis. Correspondencia con `rag_ejemplo.py`

El script del curso tiene siete pasos. La tabla indica dónde está cada uno en este sistema y qué se modificó. `rag_ejemplo_adaptado.py` conserva los siete pasos con las mismas decisiones de diseño del original (chunking recursivo de 1 000/200, Chroma en memoria, k=4, prompt de `RetrievalQA`) y cambia solo lo necesario para que corra: las importaciones, los modelos y la carga del corpus. Cada cambio está marcado en el código con `# CAMBIO:` y su motivo.

| Paso del original | Qué hace allí | En este sistema | Qué se modificó y por qué |
|---|---|---|---|
| 1. Base de conocimiento | Un párrafo de políticas embebido en el código | `corpus/` (4 documentos) y `base_conocimiento.py` | Cuatro tipos de documento reales; el CSV se serializa campo por campo y la FAQ por pares, porque el texto plano pierde la estructura (Fase 2) |
| 2. Embeddings | `OllamaEmbeddings(model="llama2")` | `OllamaEmbeddings(model="bge-m3")` | `llama2` es un modelo generativo usado como embedder; `bge-m3` se eligió entre 11 modelos medidos (Fase 1). Importación desde `langchain_ollama` |
| 3. Chunking | `RecursiveCharacterTextSplitter(1000, 200)` | Segmentación por tipo + fusión (`base_conocimiento.py`) | El recursivo 1000/100 dio recall@3 0.83 / 0.75 y la segmentación por tipo con F+E 1.00 / 0.88, con 38 % menos contexto (Fase 2) |
| 4. Base vectorial | `Chroma.from_documents` en memoria | `indexar.py`: `PersistentClient`, distancia coseno, indexación incremental por hash | En memoria obliga a reembeber en cada arranque; el índice persistente se actualiza solo con lo que cambió |
| 5. LLM | `Ollama(model="llama2")` | `ChatOllama(model="qwen3:14b", temperature=0, reasoning=False, num_ctx=8192)` | El mismo generador del Taller 1; temperatura 0 para comparar experimentos; `reasoning=False` evita el bloque `<think>` |
| 6. Cadena | `RetrievalQA.from_chain_type(chain_type="stuff")` + `as_retriever()` (k=4) | `Recuperador` (k=3 + enlace FAQ → política) y cadena LCEL en `crear_rag` | `RetrievalQA` no existe en 1.x; "stuff" equivale a concatenar los fragmentos en el prompt, que es lo que hace la cadena LCEL. Se añade el paso de enlace y el prompt propio con reglas de citación y abstención |
| 7. Interacción | Bucle `input()` con `qa_chain.run()` | `--interactivo` con `.invoke()`, más `--mostrar-contexto` y flags de experimento | `.run()` fue retirado; los flags permiten observar el efecto de cada cambio |

**Comparación medida.** `rag_ejemplo_adaptado.py --evaluar` mide la recuperación del script original, con los modelos de este proyecto y sobre el mismo corpus, con el criterio de la Fase 2. El corpus en texto plano con chunking recursivo de 1 000/200 produce 27 fragmentos de 793 caracteres en promedio, frente a los 54 de 390 de la segmentación por tipo.

| Sistema | Evaluación (recall / críticas) | Validación (recall / críticas) | Contexto medio por consulta (evaluación) |
|---|---|---|---|
| Script del curso, k=3 | 0.83 / 0.67 | 0.75 / 0.60 | 2 652 caracteres |
| Script del curso, k=4 (su valor por defecto) | 0.94 / 0.83 | 0.75 / 0.60 | 3 488 caracteres |
| `rag_ecomarket.py`, k=3 + enlace | 1.00 / 1.00 | 0.88 / 0.80 | 1 659 caracteres |

Con su k=4 por defecto, el script del curso se acerca en el conjunto de evaluación (falla solo `oro-07`) a costa de entregar el doble de contexto; en el conjunto de validación, el que se escribió sin conocer el diseño, se queda en 0.75 con `val-01` y `val-02` fallidas y 3 de 5 críticas. `val-01` (miel con el frasco quebrado) es el caso de la excepción por producto dañado: en el texto plano la FAQ de alimentos y la sección 1.3 quedan en fragmentos distintos y la similitud trae la FAQ. `oro-07` (reacción alérgica, que requiere la sección 1.4 de escalamiento) falla con ambos k; la explicación más probable, consistente con lo medido en la Fase 2 para el chunking recursivo, es que en fragmentos de 1 000 caracteres que mezclan varias secciones el texto de la 1.4 queda diluido entre el de devoluciones y envíos, y pierde en similitud; no se verificó fragmento a fragmento. Son los dos problemas que motivaron la segmentación por tipo, la fusión y el enlace de la Fase 2, y la diferencia entre las dos filas finales es lo que esas tres decisiones aportan: mejor recuperación en las preguntas críticas con la mitad del contexto.

## 3. Secciones del código

| Sección (`rag_ecomarket.py`) | Qué hace | Qué se puede modificar |
|---|---|---|
| `PROMPTS` | Tres plantillas con las variables `{contexto}` y `{pregunta}`: `base`, `citas_exactas` y `estricto` | Reglas del prompt: uso exclusivo del contexto, prevalencia de la política sobre la FAQ, formato de citas, abstención |
| `Recuperador.__init__` | Lee de Chroma los fragmentos de políticas y arma un diccionario sección → fragmento (`'1.1'`, `'1.2'` y `'1.3'` apuntan al fragmento fusionado) | Fuente del enlace (metadato `claves`) |
| `Recuperador.recuperar`, paso 1 | `similarity_search_with_score(pregunta, k)`; el valor devuelto es la distancia coseno, menor es más cercano | `--k` |
| `Recuperador.recuperar`, paso 2 | Si hay una FAQ entre los resultados, añade al final las secciones indicadas en su metadato `politica_ref`, sin repetir fragmentos | `--sin-enlace` |
| `formatear_contexto` | Antepone el id de cada fragmento para que el modelo lo cite | Formato del contexto |
| `crear_recuperador` | Abre el índice, verifica que no esté vacío y devuelve el recuperador. Se separa de `crear_rag` para evaluar la recuperación sin cargar el generador | Modelo de embeddings (debe coincidir con el usado en `indexar.py`) |
| `crear_rag` | Configura `ChatOllama` (`temperature=0`, `reasoning=False`, `num_ctx=8192`) y construye la cadena | `--llm`, `--prompt` |
| `citas_invalidas` | Compara los ids citados entre corchetes con los fragmentos entregados e imprime un aviso si alguno no corresponde | Expresión regular de ids |

Parámetros del generador. `temperature=0` hace las respuestas repetibles entre corridas, condición necesaria para comparar experimentos. `reasoning=False` desactiva el modo de razonamiento de qwen3, que antepone un bloque `<think>` a la respuesta. `num_ctx=8192` amplía el contexto por defecto de Ollama.

## 4. Verificación sobre el índice real

`evaluar_rag.py` aplica el criterio de la Fase 2 (todos los textos de `requiere` presentes en la unión de los fragmentos entregados) sobre el sistema real, sin invocar al generador.

| Configuración | Evaluación (recall@3 / críticas) | Validación (recall@3 / críticas) |
|---|---|---|
| Por tipo + fusión + enlace (final) | 1.00 / 1.00 | 0.88 / 0.80 |
| Sin enlace | 0.89 / 0.67 | 0.62 / 0.40 |

Los valores coinciden con los de `medir_segmentacion.py` y `medir_chunking.py`. La única pregunta que falla con la configuración final es `val-02` (sarpullido): recupera la guía de cuidado y no la política de higiene personal.

## 5. Experimentos

Pregunta A: «¿Puedo devolver un termo que me llegó abollado?». Pregunta B: «Compre un mix de frutos secos y llego con el empaque roto. Puedo devolverlo?» (`oro-01`). Pregunta C: «¿Hacen envíos a Ecuador?» (el corpus no contiene información sobre envíos internacionales). Todas las corridas usan `temperature=0`; cada fila es una sola corrida.

| # | Pregunta | Cambio | Observación |
|---|---|---|---|
| 1 | A | Configuración base (k=3, enlace, prompt `base`) | Acepta la devolución con 72 horas, fotografías y transporte a cargo de EcoMarket. Cita `[politicas::1.3]`, id que no existe |
| 2 | A | `--sin-enlace` | Contexto: `faq-02`, una guía de cuidado y `faq-04`. Acepta la devolución y cita `[faq::faq-04]`. No menciona que la restricción de empaque deja de aplicar |
| 3 | A | `--k 1` | Acepta la devolución y afirma que la restricción de empaque no aplica. Cita `[politicas::1.3]` |
| 4 | A | `--k 5` | Acepta la devolución con los mismos plazos. Cita `[politicas::1.3]` |
| 5 | A | `--prompt estricto` | Tres frases. Cita `[faq::faq-04]` y `[politicas::1.1-1.3]`, ambos válidos |
| 6 | A | `--prompt citas_exactas` | Cita `[politicas::1.1-1.3]`, válido; sin aviso |
| 7 | B | `--sin-enlace` | Contexto: `faq-03` (los frutos secos no admiten devolución), `faq-02` y `faq-04`. Acepta la devolución y cita `[faq::faq-04]`. No menciona `faq-03` |
| 8 | B | Con enlace | Se añade `politicas::1.1-1.3`. Acepta la devolución citando la excepción de la sección 1.3. Cita `[politicas::1.3]`; el verificador emite el aviso |
| 9 | C | Prompt `base` | Afirma «EcoMarket no ofrece envíos a Ecuador» y cita `[politicas::2]`, que habla de tiempos «en el resto del país» y no menciona envíos internacionales. Ofrece derivar a un agente humano |
| 10 | C | `--prompt estricto` | Responde «No tengo esa información; la derivo con un agente humano.» |

Las filas 3 y 4 no imprimieron el contexto. Que en la fila 3 el contexto incluya la sección 1.1 a 1.3 se deduce de la distancia: el fragmento más cercano de la fila 1 es `faq-02`, cuyo enlace apunta a las secciones 1.2 y 1.3.

### Hallazgos

1. **El enlace cambia el fundamento de la respuesta, no su veredicto, en las dos preguntas probadas.** Sin enlace, la FAQ de pedidos rotos (`faq-04`) bastó para que el modelo aceptara la devolución en las filas 2 y 7. Con enlace, la respuesta se apoya en la política y menciona la excepción por producto dañado (filas 1 y 8). En la fila 7 el contexto contenía además `faq-03`, que indica que los frutos secos no admiten devolución, y la respuesta no mencionó esa contradicción. La métrica de recall (sección 4) es más exigente que la respuesta del modelo, porque exige que aparezcan frases concretas de la política. Con una sola corrida por configuración, estas observaciones no permiten afirmar que el enlace nunca cambie un veredicto.

2. **Con el prompt `base`, el modelo cita ids inexistentes.** El fragmento fusionado se llama `politicas::1.1-1.3` y el modelo escribió `politicas::1.3`, tomando el número de sección del texto. Ocurrió en las cuatro corridas con prompt `base` que citaron esa sección (filas 1, 3, 4 y 8). Con `citas_exactas` y con `estricto` los ids citados fueron válidos (filas 5 y 6). La fusión de secciones, que mejoró la recuperación, dejó un id compuesto que el modelo tiende a abreviar.

3. **La abstención depende del prompt.** Ante una pregunta sin respuesta en el corpus, el prompt `base` afirmó un hecho no respaldado y el prompt `estricto` se abstuvo. El prompt `estricto` respondió correctamente la pregunta A (fila 5), por lo que en estas dos preguntas no hubo abstención excesiva.

4. **El enlace puede añadir contexto irrelevante.** En la pregunta C, la FAQ de pedidos rotos aparece entre los tres primeros resultados con distancia 0.51, y el enlace añade la política de devoluciones completa. El modelo no se dejó desviar, pero el enlace se activa sin importar la distancia de la FAQ.

## 6. Experimentos con el LLM local

En la revisión de técnicas quedaron fuera cuatro que dependen del modelo generador: usar un LLM como juez de las respuestas, un RAG correctivo en el que el LLM revisa lo recuperado, un grafo de relaciones entre documentos construido por el LLM (GraphRAG) y resúmenes jerárquicos de grupos de fragmentos (RAPTOR). Las dos últimas se probaron en una versión mínima que conserva la parte relevante para este corpus. Todas corren con `qwen3:14b` a temperatura 0 y semilla fija, y cada script acepta `--simulado` para validar la tubería sin Ollama. Los resultados completos están en `resultados_llm_*.json`.

### 6.1 El LLM propone los enlaces FAQ → política (`exp_enlaces_llm.py`)

Para cada una de las 18 preguntas frecuentes, el modelo ve la FAQ y las secciones de políticas y propone cuáles la gobiernan. La propuesta se compara con la tabla manual de la Fase 2 y se mide su efecto en la recuperación. 18 llamadas, 15 segundos.

| Medida | Resultado |
|---|---|
| FAQ con enlaces idénticos a la tabla manual | 14 de 18 |
| Precisión / cobertura / F1 sobre pares (FAQ, sección) | 0.64 / 0.64 / 0.64 |
| Enlaces que faltan | `faq-02 → 1.3`, `faq-03 → 1.3`, `faq-08 → 2.1`, `faq-09 → 2` |
| Enlaces de más | `faq-08 → 2`, `faq-08 → 3`, `faq-09 → 1.1`, `faq-09 → 1.3` |
| Recall@3 / críticas con la tabla del LLM (configuración final) | Evaluación 1.00 / 1.00; validación 0.88 / 0.80, iguales a la tabla manual |

Las dos omisiones graves son `faq-02 → 1.3` y `faq-03 → 1.3`: la sección 1.3 es la excepción por producto dañado, justo lo que la FAQ de higiene y la de alimentos omiten. El modelo enlazó esas FAQ con la sección 1.2 (la regla) y no con la 1.3 (la excepción). Que el recall sea idéntico se explica por la fusión F: las secciones 1.1 a 1.3 son un solo fragmento en el índice, de modo que un enlace a 1.2 trae el mismo texto que uno a 1.3. La fusión cubrió la omisión. Para saber cuánto habría costado sin ella, `--sin-fusion` mide los mismos enlaces sobre las secciones separadas:

| Secciones 1.1–1.3 separadas | Sin enlace | Tabla manual | Tabla del LLM |
|---|---|---|---|
| Evaluación (recall@3 / críticas) | 0.89 / 0.67 | 1.00 / 1.00 | 0.94 / 0.83 (falla `oro-04`) |
| Validación | 0.62 / 0.40 | 0.88 / 0.80 | 0.62 / 0.40 (fallan `val-01`, `val-02`, `val-03`) |

Sin la fusión, la tabla del LLM pierde en validación toda la ganancia del enlace: queda igual que no tener enlaces. Las tres preguntas que fallan son las de producto dañado en categorías restringidas (miel con el frasco quebrado, cepillo con las cerdas sueltas), que necesitan la sección 1.3. Las dos intervenciones de la Fase 2 se cubren mutuamente: con la fusión, el enlace del LLM alcanza; sin ella, no.

**¿Límite del modelo o del prompt?** El prompt anterior (`directo`) pide las secciones que gobiernan la FAQ y advierte contra temas parecidos, lo que favorece la coincidencia directa. Se probó un segundo prompt (`excepciones`) que avisa que la FAQ es de mercadeo y puede omitir condiciones, y pide dos pasos: la sección que coincide con lo que la FAQ afirma, y después cualquier excepción en las demás secciones que cambiaría la respuesta para algún cliente, aunque trate de otro tema. Mismo modelo, sin razonamiento, secciones separadas.

| Prompt | Precisión / cobertura / F1 | Enlaces graves que faltan | Enlaces de más | Validación (recall@3 / críticas) |
|---|---|---|---|---|
| `directo` | 0.64 / 0.64 / 0.64 | `faq-02 → 1.3`, `faq-03 → 1.3` | 4 | 0.62 / 0.40 |
| `excepciones` | 0.38 / 0.73 / 0.50 | `faq-02 → 1.3` | 13 | 0.75 / 0.60 |
| `excepciones` + razonamiento (317 s) | 0.61 / 0.73 / 0.67 | `faq-02 → 1.3` | 5 | 0.75 / 0.60 |
| Tabla manual | — | — | — | 0.88 / 0.80 |

El segundo prompt encontró la excepción para la FAQ de alimentos (`faq-03 → 1.3`) y no para la de higiene: a `faq-02` le enlazó la sección 1.4 (escalamiento por daño a la salud) en lugar de la 1.3, aunque la 1.3 dice de forma explícita "incluidos perecederos y productos de higiene". Una lectura posible es que la frase de la FAQ "por normativa sanitaria, un producto con el sello roto no puede reingresar al inventario" suena absoluta y el modelo no la contrastó con la excepción; es una hipótesis. El costo del segundo prompt fue la precisión: 13 enlaces de más, entre ellos `faq-05 → 1.1, 1.3, 1.4` y `faq-09 → 1.1, 1.3, 2.1, 4`, que no afectan el recall pero arrastran secciones completas a cada consulta en que aparezcan esas FAQ.

Con razonamiento, el modelo produjo la propuesta más limpia de las tres (13 de 18 idénticas, 5 enlaces de más) y volvió a omitir `faq-02 → 1.3`; para esa FAQ propuso únicamente la sección 1.2. Tres configuraciones del mismo modelo, con un costo de 1 a 17 segundos por FAQ, dejaron el mismo hueco.

Un factor de confusión que se detectó después: en esas corridas el modelo recibió las políticas con las secciones 1.1 a 1.3 fusionadas en un solo bloque etiquetado "Sección 1.1, 1.2, 1.3", y se le pidió elegir entre las tres por separado. Se corrigió el script para que `--sin-fusion` gobierne también lo que ve el modelo, y se repitió la corrida con el prompt `directo` sobre las ocho secciones separadas: mismas 14 de 18 FAQ idénticas, mismas omisiones `faq-02 → 1.3` y `faq-03 → 1.3`, mismo recall. La presentación no explica la omisión. Las variantes `excepciones` y con razonamiento no se repitieron con las secciones separadas.

La respuesta a la pregunta es "ambas cosas". Parte de la omisión era del prompt, porque la instrucción explícita recuperó un enlace. Parte es del modelo, porque la otra omisión persistió con razonamiento. Ninguna propuesta iguala a la tabla manual, y conviene decir por qué: la tabla se escribió con la retroalimentación de la evaluación (los fallos de `oro-01` y `oro-04` en la Fase 2) y con un objetivo claro, proteger el caso del producto dañado. El modelo solo tuvo el texto. Eso expone la limitación más seria del enlace manual: en un corpus nuevo, o para una FAQ que ninguna pregunta de evaluación toca, el hueco existiría y nadie lo vería hasta que un cliente cayera en él. El conjunto de validación, escrito antes de diseñar los enlaces, valida los enlaces que ya existen; no descubre los que faltan.

**Una regla que no depende de retroalimentación.** La sección 1.3 está escrita así: "La restricción de la sección 1.2 no aplica cuando el producto llegó dañado". La excepción cita la regla que modifica. De ahí sale una regla determinista (`--propagar`): si una sección E dice que la restricción de la sección R no aplica, toda FAQ enlazada a R se enlaza también a E. Leída del texto de las políticas, la regla produce `1.2 → 1.3`. Aplicada a la propuesta del modelo sin razonamiento, añade exactamente tres enlaces, `faq-02 → 1.3`, `faq-03 → 1.3` y `faq-05 → 1.3` (este último, ruido inofensivo), y corrige las dos omisiones graves.

| Secciones 1.1–1.3 separadas | F1 con la tabla manual | FAQ idénticas | Evaluación | Validación |
|---|---|---|---|---|
| Propuesta del LLM (`directo`) | 0.64 | 14 / 18 | 0.94 / 0.83 | 0.62 / 0.40 |
| Propuesta del LLM + propagación | 0.72 | 15 / 18 | 1.00 / 1.00 | 0.88 / 0.80 |
| Tabla manual | — | — | 1.00 / 1.00 | 0.88 / 0.80 |

Con la propagación, la tabla generada sin retroalimentación iguala en recuperación a la tabla manual; la única pregunta que falla es `val-02`, que tampoco resuelve la tabla manual. Las dos diferencias que quedan con la tabla manual (`faq-08`, `faq-09`) son las discutibles. El límite de la regla es la redacción: solo detecta excepciones que citan de forma explícita la sección que modifican; "esta restricción no aplica cuando..." sin número de sección no la activa.

**¿Capacidad del modelo o formato de la tarea?** Quedaba por separar dos explicaciones de la omisión: que `qwen3:14b` no entienda que la sección 1.3 modifica la FAQ de higiene, o que no la encuentre cuando debe elegir entre ocho secciones en un solo paso. Se hicieron dos pruebas más. La primera cambia el formato con el mismo modelo (`--diagnostico`): una pregunta sí/no por cada par (FAQ, sección), 144 llamadas cortas. La segunda cambia el modelo con el formato original: los mismos 18 prompts contra dos modelos de API (Claude Sonnet 5.5 y Claude Opus 5.5, `--modelo anthropic:<id>`). Todas con las secciones separadas.

| Modelo y formato | F1 | FAQ idénticas | `faq-02 → 1.3` | `faq-03 → 1.3` | Enlaces de más | Validación | Costo |
|---|---|---|---|---|---|---|---|
| qwen3:14b, lista | 0.64 | 14 / 18 | no | no | 4 | 0.62 / 0.40 | 13 s |
| qwen3:14b, lista + razonamiento | 0.67 | 13 / 18 | no | sí | 5 | 0.75 / 0.60 | 317 s |
| qwen3:14b, binario por par | 0.67 | 13 / 18 | **sí** | **sí** | 5 | **0.88 / 0.80** | 73 s |
| Claude Sonnet 5.5, lista | 0.60 | 11 / 18 | **sí** | **sí** | 10 | **0.88 / 0.80** | 46 s, USD 0.11 |
| Claude Opus 5.5, lista | 0.64 | 13 / 18 | **sí** | **sí** | 8 | **0.88 / 0.80** | 72 s, USD 0.23 |
| qwen3:14b, lista + propagación | 0.72 | 15 / 18 | sí (regla) | sí (regla) | 5 | **0.88 / 0.80** | 13 s |
| Tabla manual | — | — | sí | sí | — | 0.88 / 0.80 | — |

La respuesta es "ambas cosas", y ahora con evidencia. En formato binario, `qwen3:14b` responde "sí" a los dos pares que omitía en la lista: la comprensión está, y lo que fallaba era la selección entre ocho candidatos en un solo paso. Los dos modelos de API encuentran los dos enlaces en el formato de lista, así que un modelo más grande resuelve el formato que el de 14B no resuelve. Las tres vías (formato binario, modelo grande, regla de propagación) llegan al mismo recall que la tabla manual; en todas falla únicamente `val-02`, que la tabla manual tampoco resuelve. Las dos diferencias que persisten en todos los modelos (`faq-08 → 2.1`, `faq-09 → 2`) son las que se señalaron como discutibles: cinco configuraciones distintas discrepan de la tabla manual en el mismo sitio, lo que sugiere revisar esas dos filas de la tabla y no a los modelos.

El formato binario tiene su costo: ocho veces más llamadas que la lista, y una omisión nueva (`faq-06 → 2`, envío gratis) que el formato de lista sí encontraba. Los modelos de API dejan más ruido que el local (8 a 10 enlaces de más frente a 4); el enlace `faq-13 → 1.3` (garantía con producto dañado), que ambos añadieron, es defendible. Con la API no se puede fijar la temperatura (el parámetro está retirado para estos modelos), de modo que esas dos corridas no son necesariamente repetibles; las de Ollama sí (temperatura 0 y semilla fija). El costo total de las dos corridas de API fue USD 0.34.

Para un corpus mayor, la combinación razonable es la que se midió aquí: un modelo propone (el local en formato binario, o uno de API en formato de lista), la regla estructural propaga las excepciones que las políticas declaran, y una persona revisa el resto, empezando por las filas en que varios modelos coinciden en discrepar de la tabla.

Las otras diferencias son discutibles. `faq-09` (cancelar un pedido) dice que si el pedido ya salió de bodega la cancelación se tramita como devolución, y el modelo la enlazó con las secciones de devoluciones en lugar de la de envíos. `faq-08` (rastreo) se enlazó con la política de envíos completa y las reglas de comunicación en lugar de la subsección de retrasos. La tabla manual es la lectura de una persona y no una verdad de referencia; en estos dos casos la propuesta del modelo es defendible.

Conclusión: el modelo produce una tabla de enlaces razonable en segundos, pero falló en los dos enlaces que protegen el caso de seguridad. Confirma la posición de la Fase 2: el LLM puede proponer los enlaces y una persona debe revisarlos, con atención a las excepciones.

### 6.2 `qwen3:14b` como juez de fidelidad (`exp_juez.py`)

Se construyeron 18 casos con veredicto conocido (`casos_juez.json`): 8 respuestas fieles al contexto y 10 que afirman algo que el contexto no respalda. Entre las infieles hay errores de un solo dato (15 días en lugar de 30, 7 días en lugar de 72 horas, un cupón del 10 % inventado, "saldo en su cuenta" en lugar de reembolso al medio de pago) y una respuesta real del sistema: la que afirmó que EcoMarket no envía a Ecuador (`j09`). Se compararon dos formas de pedir el veredicto (directo: fiel sí o no; por afirmaciones: el modelo lista cada afirmación y marca si está respaldada, y el código calcula el veredicto) con el razonamiento de qwen3 apagado y encendido. Positivo significa respuesta infiel.

| Configuración | Detección | Falsas alarmas | Exactitud | Segundos (18 casos) | Errores |
|---|---|---|---|---|---|
| Directo | 0.90 | 0.00 | 0.94 | 42 | `j09` |
| Directo + razonamiento | 1.00 | 0.00 | 1.00 | 180 | — |
| Por afirmaciones | 1.00 | 0.25 | 0.89 | 55 | `j03`, `j04` |
| Por afirmaciones + razonamiento | 1.00 | 0.12 | 0.94 | 263 | `j04` |

Tres observaciones:

- Sin razonamiento, el único caso infiel que el juez directo dejó pasar fue `j09`, la respuesta sobre Ecuador que el mismo modelo había generado. Es una sola observación y no permite concluir que el modelo sea indulgente con sus propias respuestas, pero es el caso más parecido a los errores que el sistema produce en la práctica, y el más plausible de leer.
- El razonamiento llevó la exactitud a 18 de 18, a un costo de 10 segundos por caso frente a 2.3.
- El veredicto por afirmaciones produjo falsas alarmas en `j03` y `j04`, los dos casos en que la respuesta correcta es derivar al cliente a un agente humano. Al descomponer la respuesta, el modelo trató "lo derivo con un agente humano" como una afirmación que el contexto debía respaldar. Un juez por afirmaciones necesita una regla explícita para las frases de abstención y escalamiento.

Con 18 casos, un caso equivale a 0.10 de detección o 0.12 de falsas alarmas. Los casos infieles los redactó el autor, por lo que la prueba mide la detección de los tipos de error que se le ocurrieron. Dicho esto, el resultado es suficiente para usar el juez directo con razonamiento como herramienta de auditoría de muestras, y el directo sin razonamiento como filtro barato.

### 6.3 RAG correctivo (`exp_correctivo.py`)

El modelo juzga si el contexto recuperado (k=3 + enlace) alcanza para responder. Si no, reescribe la consulta, se recupera de nuevo con k=5, se combinan ambos contextos y se vuelve a juzgar; si sigue sin alcanzar, el sistema se abstiene y deriva a un humano. Se midió sobre 34 preguntas: las 26 de evaluación y validación, y 8 sin respuesta en el corpus (`conjunto_sin_respuesta.json`; el script verifica por código que sus temas no aparecen en ningún fragmento). "Suficiente" se define con el criterio de la Fase 2: el contexto contiene las frases de `requiere`.

| Medida | Resultado |
|---|---|
| Aciertos del juez en el primer intento | 31 de 34 |
| Abstenciones de más (contexto suficiente juzgado insuficiente) | 2 de 25: `oro-09`, `val-06` |
| Respuestas peligrosas (contexto insuficiente juzgado suficiente) | 1 de 9: `sr-07` |
| Recall sobre las 26 con respuesta | 0.96 en el primer intento → 1.00 tras el lazo |
| Preguntas sin respuesta que terminan en abstención | 7 de 8 |
| Preguntas con respuesta que terminan en abstención | 2 de 26 |
| Costo | 1.59 llamadas y 4.1 segundos por pregunta, antes de generar la respuesta |

`val-02` (sarpullido tras usar el jabón), la única pregunta que la recuperación no resolvía, terminó en abstención: el juez la declaró insuficiente, la reescritura recuperó el texto de la política (por eso el recall sube a 1.00) y el segundo juicio la volvió a declarar insuficiente. La política 1.4 indica que las menciones de daño a la salud se escalan a un humano, así que la abstención es el resultado correcto, aunque se llegó a él por el camino largo.

Las dos abstenciones de más, `oro-09` ("¿me pueden dar un descuento por la demora?") y `val-06` ("¿me aseguran que llega el viernes?"), son preguntas cuya respuesta correcta es una negativa fundada en la política (compensaciones prohibidas, fechas estimadas). El contexto las contenía y el juez las consideró insuficientes. Junto con las falsas alarmas del juez por afirmaciones en `j03` y `j04`, apunta a una misma dificultad: el modelo interpreta "suficiente para responder" como "suficiente para dar al cliente lo que pide". Es una hipótesis a partir de cuatro casos, no un resultado.

La respuesta peligrosa fue `sr-07` ("¿cuánto cuesta el envío si mi pedido es de $60.000?"). El contexto dice que el envío es gratis por encima de $120.000 y no da la tarifa por debajo; el juez lo consideró suficiente. Un generador podría responder con una tarifa inventada.

Frente al sistema base, donde el prompt `base` inventó la respuesta sobre Ecuador y el prompt `estricto` se abstuvo, el lazo correctivo se abstuvo en 7 de las 8 preguntas sin respuesta sin depender del prompt de generación. El costo es un 8 % de preguntas legítimas que terminan en abstención y 4 segundos más por consulta.

### 6.4 Resúmenes por grupo, RAPTOR de un nivel (`exp_raptor_lite.py`)

Los 54 fragmentos se agrupan en 8 grupos por similitud de embeddings (k-means esférico, semilla fija) y el modelo resume cada grupo en 90 palabras con la instrucción de conservar todas las cifras. Se mide cuántas cifras sobreviven y si el grupo del resumen más cercano a la pregunta contiene lo que la respuesta necesita, comparado con el top-3 por fragmento. 8 llamadas, 64 segundos.

El agrupamiento reprodujo en gran parte la estructura por tipo de documento: un grupo con 13 de los 14 productos del catálogo, dos grupos de guía de cuidado, uno de FAQ de cuenta y pedidos, uno de políticas de devoluciones y uno de envíos. Un grupo quedó con un solo fragmento (la vela de soya).

**Cifras.** La primera corrida reportó 27 de 68 cifras conservadas y 18 "inventadas". Al revisar, las inventadas eran cifras reformateadas: el catálogo escribe `89000` y el resumen `$89.000`; la guía escribe "doscientos lavados", "dieciocho meses" y "un litro" y el resumen las puso en dígitos. La métrica se corrigió para normalizar separadores de miles y convertir cifras en letras. En la segunda corrida (mismos grupos y mismos resúmenes, porque el agrupamiento y la generación son deterministas) quedaron 54 de 82 cifras conservadas (66 %), 5 de 8 grupos con alguna cifra perdida y una sola "inventada", el "18" de "dieciocho meses", que faltaba en la lista de cifras en letras; con la lista completa son 55 de 83 y ninguna inventada. El resumen del catálogo, por ejemplo, conserva los precios y pierde las vidas útiles, las garantías y si el producto es apto para lavavajillas.

**Sentido.** La métrica de cifras no detecta el problema más serio, que apareció al leer los resúmenes. El del grupo de devoluciones dice: "No se aceptan devoluciones fuera de plazo, por más de $500.000 COP, o con daño a la salud". La sección 1.4 dice que esos casos se escalan a un agente humano; el resumen los convirtió en rechazos. Conserva las cifras y cambia la política. El resumen del grupo de higiene y cuidado dice que "no se aceptan devoluciones de productos de higiene personal con el sello roto", sin la excepción por producto dañado, que quedó en otro grupo. El agrupamiento por similitud separa la regla de su excepción y el resumen reproduce exactamente la omisión de la FAQ de mercadeo que motivó el enlace E.

**Enrutamiento.**

| Conjunto | Grupo del mejor resumen | Top-3 por fragmento, sin enlace |
|---|---|---|
| Evaluación (recall / contexto) | 0.83 / ~2 850 caracteres | 0.89 / ~1 210 caracteres |
| Validación | 0.38 / ~3 350 caracteres | 0.62 / ~1 220 caracteres |

Recuperar por resumen y entregar el grupo completo rinde menos que recuperar por fragmento y cuesta entre 2.3 y 2.8 veces más contexto. Para este corpus la técnica no aporta: sus beneficios aparecen en documentos largos cuyas respuestas exigen sintetizar muchos fragmentos, y aquí cada respuesta cabe en una sección. Es un solo nivel del árbol con k=8 y una semilla; otros valores darían otros grupos.

### 6.5 Lectura conjunta

| Técnica | Aporta aquí | Costo | Riesgo observado |
|---|---|---|---|
| Juez LLM | Sí, como auditoría de muestras | 2 a 10 s por respuesta | No detectó una respuesta inventada por el propio modelo sin razonamiento; falsas alarmas en abstenciones |
| RAG correctivo | Sí, para abstenerse ante preguntas sin respuesta | 1.6 llamadas y 4 s por consulta | Abstiene en respuestas que son una negativa fundada; aceptó un contexto incompleto |
| Enlaces por LLM | Como borrador, más una regla determinista que propaga excepciones | 13 a 317 s por corpus; USD 0.11 a 0.23 con API | El modelo de 14B omite la excepción en formato de lista y la encuentra en formato binario; Sonnet y Opus la encuentran en lista; la regla estructural la recupera sin modelo. Las tres vías igualan la tabla manual |
| Resúmenes por grupo | No | 64 s por corpus | Reescribió una política y separó una regla de su excepción |

Un patrón atraviesa los tres primeros: el modelo maneja bien los casos en que la respuesta está en el contexto y peor los casos en que la respuesta correcta es "no", "se escala" o "no hay información". Son justamente los casos que en atención al cliente tienen consecuencias. Toda regla determinista del sistema (el enlace, el clasificador del Taller 1, la verificación de citas) existe para cubrir esos casos sin depender del modelo.

## 7. Limitaciones y supuestos

- Los experimentos son una corrida por configuración sobre tres preguntas. Sirven para observar el efecto de cada cambio y no para estimar tasas.
- Las métricas de recuperación (sección 4) y las observaciones de respuesta (sección 5) miden cosas distintas. Falta una evaluación de respuestas completas sobre los 26 casos que cuente citas inválidas, abstenciones y respuestas con la excepción correcta.
- El enlace FAQ → política no tiene umbral de distancia.
- `val-02` (sarpullido) no se recupera. Se supone que en el sistema completo lo atiende el clasificador de intención del Taller 1 mediante la categoría `humano_exclusivo`; esa hipótesis no se ha medido.
- Un solo modelo generador (`qwen3:14b`) y un solo modelo de embeddings (`bge-m3`) en la configuración final.
- Cada consulta es independiente: el sistema no mantiene historial de conversación.
- Los experimentos con el LLM local son una corrida por configuración, con 18 casos de juez redactados por el autor y 8 preguntas sin respuesta. Sirven para observar comportamientos, no para estimar tasas.
- Las corridas con modelos de API no fijan temperatura (el parámetro está retirado) y pueden variar entre ejecuciones; la clave se lee de una variable de entorno y no se guarda en el repositorio.
- El mismo modelo (`qwen3:14b`) genera las respuestas y las juzga. Una familia de errores propia del modelo puede pasarle inadvertida como juez; la prueba lo sugiere en un caso y no lo demuestra.
- El corpus es sintético (ver Fase 2).

## 8. Reproducción

```bash
python indexar.py                     # crea o actualiza el índice (54 fragmentos)
python rag_ejemplo_adaptado.py --evaluar   # línea base: el script del curso migrado a LangChain 1.x
python evaluar_rag.py                 # recall@3 sobre evaluación y validación
python evaluar_rag.py --sin-enlace

Q="¿Puedo devolver un termo que me llegó abollado?"
python rag_ecomarket.py "$Q" --mostrar-contexto
python rag_ecomarket.py "$Q" --sin-enlace --mostrar-contexto
python rag_ecomarket.py "$Q" --k 1
python rag_ecomarket.py "$Q" --prompt citas_exactas
python rag_ecomarket.py "¿Hacen envíos a Ecuador?" --prompt estricto --mostrar-contexto

# Experimentos con el LLM local (sección 6); cada uno acepta --simulado
python exp_enlaces_llm.py
python exp_enlaces_llm.py --sin-fusion
python exp_enlaces_llm.py --prompt excepciones --sin-fusion
python exp_enlaces_llm.py --prompt excepciones --sin-fusion --razonar
python exp_enlaces_llm.py --propagar --sin-fusion
python exp_enlaces_llm.py --diagnostico --sin-fusion
export ANTHROPIC_API_KEY=...   # solo en la terminal; nunca en archivos del repositorio
python exp_enlaces_llm.py --sin-fusion --modelo anthropic:claude-sonnet-5-5
python exp_enlaces_llm.py --sin-fusion --modelo anthropic:claude-opus-5-5
python exp_juez.py
python exp_correctivo.py
python exp_raptor_lite.py
```
