# Fase 2 — Creación de la base de conocimiento de documentos

**Taller Práctico #2 · IA Generativa · Caso EcoMarket**

**Resumen:** la base de conocimiento reúne cuatro tipos de documento (políticas, catálogo, preguntas frecuentes y guía de cuidado), cada uno segmentado según su forma, con las secciones 1.1 a 1.3 de las políticas fusionadas en un único fragmento y un enlace que añade al contexto la sección de políticas que gobierna cada pregunta frecuente recuperada. Sobre 26 preguntas de evaluación, esta estrategia supera a las tres segmentaciones clásicas (tamaño fijo, párrafos y recursiva) y entrega entre 28 % y 38 % menos texto al modelo que la mejor de ellas.

---

## 1. Identificación de documentos

El corpus contiene cuatro tipos de documento. Sus formatos y su naturaleza difieren a propósito, porque la estrategia de segmentación depende de esa forma.

| Documento | Formato | Contenido | Papel en el sistema | Riesgo principal |
|---|---|---|---|---|
| Políticas de servicio (`politicas.md`, v3.1) | Markdown con secciones, tablas y listas | Devoluciones, categorías elegibles, excepciones, envíos, reglas de comunicación, criterios de escalamiento | Fuente normativa. Define lo que el asistente puede y no puede hacer | Reglas de seguridad y excepciones repartidas en secciones distintas |
| Catálogo de productos (`catalogo_productos.csv`) | Tabla, 14 productos × 12 campos | Material, garantía, categoría, aptitud para lavavajillas, advertencias | Datos de producto consultables por SKU | Advertencias enterradas en campos de texto libre |
| Preguntas frecuentes (`faq.json`, v2.4) | JSON, 18 pares pregunta/respuesta | Respuestas en lenguaje comercial | Respuestas rápidas a consultas habituales | La redacta mercadeo y puede omitir excepciones que sí están en las políticas |
| Guía de cuidado (`guia_cuidado.md`, v1.2) | Texto narrativo con encabezados | Uso y mantenimiento por tipo de producto, explicaciones físicas de las reglas | Contexto explicativo | Sin estructura de campos: una advertencia puede quedar a mitad de un párrafo |

**Qué queda fuera de la base de conocimiento.** Los pedidos de los clientes (`pedidos.json`) son datos transaccionales que cambian con cada evento y contienen datos personales. Se consultan por llamada a función, como en el Taller 1, y no se indexan. La base de conocimiento contiene únicamente información semiestática, versionable y citable.

### Riesgos del corpus

- **Divergencia entre la FAQ y las políticas.** `faq-02` afirma que un producto de higiene abierto no se recibe, y omite la excepción por producto dañado que establece la sección 1.3 de las políticas. Si el asistente recibe solo esa respuesta, niega un derecho que el cliente tiene. Es el riesgo que motiva la estrategia de enlace de la sección 2.3.
- **Datos duplicados entre documentos.** El plazo de 30 días, el umbral de envío gratis, el tiempo de reembolso y la garantía aparecen en más de un documento. Una edición en uno sin actualizar los demás produce contradicciones. Las políticas se toman como fuente de verdad.
- **Versiones.** Cada fragmento conserva la versión de su documento de origen (sección 3) para poder auditar con qué versión se respondió.
- **Corpus sintético.** Los documentos se redactaron para este taller. Su vocabulario puede ser más regular que el de documentos reales de una empresa.

---

## 2. Segmentación (chunking)

### 2.1 Estrategias consideradas

El enunciado menciona tres estrategias: tamaño fijo, por párrafos y recursiva. Se evaluaron las tres, aplicadas como suelen usarse por defecto, es decir, leyendo cada archivo como texto plano sin conocer su estructura. Se añadió una cuarta, la segmentación por tipo de documento, que usa la unidad natural de cada formato.

| Estrategia | Descripción | Configuraciones medidas |
|---|---|---|
| Tamaño fijo | Ventana deslizante de N caracteres con solape | 400/80 y 800/160 |
| Por párrafos | Corte en cada línea en blanco | — |
| Recursiva | `RecursiveCharacterTextSplitter` de LangChain (separadores `\n\n`, `\n`, espacio) | 500/50 y 1000/100 |
| Por tipo de documento | Unidad natural de cada formato (sección 2.2) | sin solape |

### 2.2 Estrategia adoptada, por documento

| Documento | Unidad de fragmento | Fragmentos | Tamaño (mín / mediana / máx, caracteres) | Tratamiento |
|---|---|---|---|---|
| Políticas | Sección o subsección | 8 (6 tras la fusión) | 247 / 528 / 633 | Se antepone la ruta de títulos (`Políticas EcoMarket > 1. Política de devoluciones > 1.3 …`) para que cada fragmento conserve su contexto |
| Catálogo | Una fila = un producto | 14 | 376 / 408 / 463 | Se serializa como `campo: valor` para que el embedding vea los nombres de columna (por ejemplo `apto_lavavajillas: No`) |
| Preguntas frecuentes | Un par pregunta/respuesta | 18 | 133 / 239 / 356 | La pregunta se incluye en el fragmento porque es el mejor ancla semántica para la consulta de un cliente |
| Guía de cuidado | Un párrafo, precedido del encabezado de su sección | 16 | 242 / 484 / 768 | Un texto narrativo sin campos no ofrece otra unidad natural, y el párrafo mantiene cada advertencia junto al producto que nombra |

Ninguna estrategia usa solape, porque cada unidad es autocontenida.

**Fusión de las secciones 1.1 a 1.3 (F).** La regla por categoría (1.2) y su excepción por producto dañado (1.3) deben llegar juntas al modelo. Se indexan como un solo fragmento de 1 348 caracteres, el mayor del corpus, que conserva los subtítulos de cada sección.

### 2.3 Enlace entre la FAQ y las políticas (E)

Cada pregunta de la FAQ se asocia a la sección de políticas que la gobierna mediante una tabla (`corpus/enlaces_faq_politica.json`). Cuando la recuperación trae una pregunta de la FAQ, el sistema añade al contexto las secciones enlazadas. Por ejemplo, `faq-02` (higiene abierto no se devuelve) arrastra las secciones 1.2 y 1.3, y con ello la excepción que la FAQ omite.

**Ejemplo.** Consulta del cliente: «Abrí el shampoo sólido y venía con el envase partido. ¿Me lo reciben?». Con `bge-m3`, los tres primeros resultados son un párrafo de la guía de cuidado, `faq-02` y la ficha del shampoo en el catálogo. Sin el enlace, la única regla sobre devoluciones que llegaría al generador es la de `faq-02` (un producto de higiene abierto no se recibe), y la respuesta negaría un derecho que el cliente tiene. Con el enlace, `faq-02` incorpora el fragmento de devoluciones (secciones 1.1 a 1.3 fusionadas), que establece que un producto llegado dañado se repone o se reembolsa sin importar la categoría ni el estado del empaque, y el generador cuenta con la información para responder correctamente.

La tabla se mantiene aparte de `faq.json`. La FAQ conserva su forma original, y el área dueña de las políticas puede revisar los enlaces. Es un mecanismo determinista: no depende del ranking de similitud, de modo que sigue protegiendo el caso peligroso si se cambia el modelo de embeddings o el texto de la consulta.

### 2.4 Evidencia: comparación de estrategias

Todas las estrategias se midieron con `bge-m3` sobre los mismos conjuntos: 18 preguntas del conjunto oro y 8 del conjunto de validación, escrito antes de diseñar la fusión y el enlace. Una pregunta es un acierto en *k* si los textos que exige aparecen en la unión de los *k* primeros fragmentos recuperados. La columna de caracteres indica cuánto texto llega al modelo generador con *k*=3.

| Estrategia | Fragmentos | Oro R@3 | Oro críticas@3 | Validación R@3 | Validación críticas@3 | Caracteres al LLM (oro / validación) |
|---|---|---|---|---|---|---|
| Fijo 400/80 | 66 | 0.78 | 0.50 | 0.62 | 0.40 | 1 183 / 1 140 |
| Fijo 800/160 | 34 | 0.83 | 0.67 | 0.50 | 0.40 | 2 364 / 2 319 |
| Por párrafos | 50 | 0.78 | 0.50 | 0.38 | 0.20 | 4 834 / 3 296 |
| Recursiva 500/50 | 56 | 0.78 | 0.33 | 0.50 | 0.40 | 1 203 / 1 235 |
| Recursiva 1000/100 | 26 | 0.83 | 0.67 | 0.75 | 0.60 | 2 666 / 2 634 |
| Por tipo | 56 | 0.89 | 0.67 | 0.62 | 0.40 | 1 091 / 1 216 |
| **Por tipo + fusión + enlace** | 54 | **1.00** | **1.00** | **0.88** | **0.80** | 1 659 / 1 890 |

*Cómo leer la fila final:* «por tipo» es la segmentación de la sección 2.2; «fusión» reúne las secciones 1.1 a 1.3 en un solo fragmento (54 fragmentos en lugar de 56); «enlace» añade al contexto las secciones de políticas asociadas a cada fragmento de FAQ recuperado (sección 2.3). Las columnas R@3 y críticas@3 indican la proporción de preguntas (todas, o solo las críticas) cuyos tres primeros fragmentos contienen los textos requeridos.

**Lectura de los resultados**

1. **La segmentación por tipo con fusión y enlace obtiene el mejor recall y el mejor recall en preguntas críticas** en los dos conjuntos. Frente a la mejor estrategia clásica (recursiva 1000/100) entrega 38 % menos texto al modelo en el conjunto oro (1 659 frente a 2 666 caracteres) y 28 % menos en el de validación (1 890 frente a 2 634), con mejor recall en ambos.
2. **Por párrafos es la peor y la más costosa.** Esa estrategia supone prosa. El archivo JSON de la FAQ no tiene líneas en blanco y se convierte en un único fragmento de 5 715 caracteres, y los separadores `---` de las políticas generan cuatro fragmentos de tres caracteres. Con *k*=3 llegan más de cuatro veces más caracteres al modelo que con la segmentación por tipo (4 834 frente a 1 091), con el peor recall de validación.
3. **Los fragmentos grandes ayudan de forma parcial y se pagan en contexto.** Las ventanas de 800 y 1 000 caracteres mejoran a las de 400 y 500 en el conjunto oro. En el de validación solo mejora la recursiva de 1000/100 (0.75 frente a 0.50), y el tamaño fijo de 800/160 queda por debajo del de 400/80 (0.50 frente a 0.62). En todos los casos envían alrededor del doble de texto al modelo por consulta.

**Por qué fallan las estrategias clásicas.** Se contó, para cada estrategia, en cuántas preguntas con más de un texto requerido no existe ningún fragmento que los contenga juntos. Es el caso de 6 preguntas con párrafos (oro-01, oro-05, oro-07, val-01, val-02 y val-08) y de 5 con recursiva 500/50 (oro-01, oro-07, val-01, val-02 y val-08), y de ninguna con recursiva 1000/100, la estrategia clásica con mejor resultado de validación. La segmentación por tipo deja dos preguntas en ese estado (oro-01 y val-01) y son precisamente las dos que necesitan la regla y su excepción; la fusión las resuelve, y con ella el conteo llega a cero.

La falta de coherencia en los cortes también afecta al ranking cuando las frases sí caen en un mismo fragmento. En tamaño fijo 400/80, el fragmento que contiene la regla de escalamiento por daño a la salud (oro-07) empieza con el final de la sección 1.3 cortado a mitad de palabra (`rte de retorno lo asume EcoMarket.`), incluye el título de la sección 1.4 y termina con la lista interrumpida (`Superintendencia de Industria y Co`). Contiene los dos textos que exige la pregunta y no entra entre los tres primeros resultados. Con la segmentación por tipo, la sección 1.4 es un fragmento propio con su título y la pregunta se recupera.

### 2.5 Intervenciones evaluadas por separado

Cada pieza de la estrategia final se midió de forma aislada con `bge-m3` (R@3 / críticas@3), junto con una cuarta intervención que se descartó.

| Variante | Oro | Validación | Fragmentos al LLM (oro / validación) |
|---|---|---|---|
| Base (por tipo) | 0.89 / 0.67 | 0.62 / 0.40 | 3.00 / 3.00 |
| F: fusión 1.1–1.3 | 0.89 / 0.67 | 0.62 / 0.40 | 3.00 / 3.00 |
| E: enlace FAQ → política | 1.00 / 1.00 | 0.88 / 0.80 | 3.78 / 3.75 |
| **F + E** | **1.00 / 1.00** | **0.88 / 0.80** | 3.44 / 3.50 |
| P: frases típicas de cliente añadidas al índice de las políticas | 0.89 / 0.67 | 0.62 / 0.40 | 3.00 / 3.00 |
| F + E + P | 1.00 / 1.00 | 0.88 / 0.80 | 3.28 / 3.38 |

- **El enlace (E) es la intervención que mueve el recall.** La fusión (F) sola no cambia el recall en *k*=3 y combinada con E reduce el contexto, porque el fragmento fusionado ocupa un solo cupo (3.44 frente a 3.78 fragmentos por consulta en el conjunto oro).
- **Las frases típicas (P) se descartan.** Sumadas a F + E producen exactamente los mismos resultados, y cada política nueva exigiría redactar y mantener sus frases. Sin beneficio medido y con costo de mantenimiento no se incorporan. Es un resultado negativo que se conserva como parte de la evidencia.
- **El efecto de E se repite con otro modelo.** Con `qwen3-embedding:4b` y la instrucción de consulta genérica (Fase 1), la base da 0.89 / 0.67 en el conjunto oro y 0.62 / 0.40 en el de validación, igual que `bge-m3`. Con E ambos modelos suben a 1.00 / 1.00 y 0.88 / 0.80. Con la instrucción de dominio, `qwen3-embedding:4b` ya obtiene ese resultado sin E, y E no lo altera.

---

## 3. Indexación

El proceso que convierte los documentos en una base vectorial consultable sigue estos pasos:

1. **Carga y segmentación.** Se leen los cuatro documentos y cada uno se divide con la estrategia de la sección 2.2, con la fusión de las secciones 1.1 a 1.3. Cada fragmento recibe un identificador estable y un hash SHA-256 calculado sobre su texto, la versión del documento y su enlace a políticas (`politica_ref`): si cambia cualquiera de los tres, cambia el hash.
2. **Comparación con el índice.** Se leen los identificadores y hashes ya almacenados en la colección y se clasifica cada fragmento: nuevo (id ausente), cambiado (id presente con otro hash), intacto (mismo hash) o huérfano (id en el índice que ya no existe en el corpus).
3. **Vectorización.** Solo los fragmentos nuevos o cambiados se convierten en vectores de 1 024 dimensiones con `bge-m3`, servido por Ollama en lotes de 32. Los vectores se normalizan, de modo que la similitud coseno equivale al producto punto.
4. **Almacenamiento.** Los vectores nuevos o cambiados se cargan con `upsert` en una colección de ChromaDB persistente (métrica coseno) junto con el texto original y sus metadatos; los huérfanos se borran; los intactos no se tocan.
5. **Cuándo se ejecuta.** El reindexado se lanza a mano (`python indexar.py`) después de editar un documento; no hay un proceso que vigile los archivos. En una operación real se programaría (por ejemplo, en un cron o en el pipeline que publica las políticas), y el costo de cada ejecución es proporcional a lo que cambió: editar una respuesta de la FAQ reembebe un fragmento de 54.

**Metadatos por fragmento**

| Campo | Ejemplo | Uso |
|---|---|---|
| `id` | `politicas::1.1-1.3` · `faq::faq-02` · `catalogo::ECO-TER-1L` · `guia::acero-inoxidable-botellas-y-termos::1` | Identificador estable que permite actualizar un fragmento sin duplicarlo |
| `fuente` | `politicas` | Filtrar por documento |
| `version` | `3.1` | Auditar con qué versión del documento se respondió |
| `politica_ref` | `1.2,1.3` (solo en la FAQ) | Insumo del enlace de la sección 2.3, guardado como cadena separada por comas |
| `claves` | `1.1,1.2,1.3` (solo en políticas) | Secciones que cubre el fragmento; es lo que el enlace usa para encontrar el fragmento de una sección |
| `hash` | `a3f9…` | Detectar qué fragmentos cambiaron desde la última indexación |

En la consulta, el sistema vectoriza la pregunta con el mismo modelo, recupera los *k* = 3 fragmentos más cercanos, añade las secciones de políticas enlazadas a cada fragmento de FAQ recuperado y entrega el resultado al generador junto con el identificador y la versión de cada fragmento para citarlos. El script de indexado y el de consulta forman parte de la Fase 3.

---

## 4. Limitaciones

- **Alcance de la evidencia.** La evaluación usa 18 preguntas del conjunto oro y 8 de validación sobre un corpus sintético de 56 fragmentos. Una pregunta pesa 5.6 puntos porcentuales en el primer conjunto y 12.5 en el segundo. Con esa muestra, las diferencias pequeñas entre estrategias no se distinguen del azar, y las cifras describen esta prueba y no el rendimiento general de cada estrategia.
- **Diseño posterior a los fallos.** La fusión y el enlace se diseñaron después de analizar los fallos de las 18 preguntas del conjunto oro. Por eso se evaluaron también en el conjunto de validación, escrito antes. La mejora se repite allí (de 0.62 a 0.88 en recall@3 y de 0.40 a 0.80 en críticas), aunque con solo 8 preguntas.
- **Una pregunta sin resolver.** val-02 (un cliente describe un síntoma de piel sin mencionar la palabra "alergia") no se recupera con ningún modelo ni estrategia probados: la norma habla de "daño a la salud" y el cliente de un "sarpullido". La hipótesis de mitigación es que ese caso lo atienda antes el clasificador del Taller 1 (corte de seguridad de `humano_exclusivo`), sin pasar por la recuperación. Esa hipótesis se evalúa al integrar los sistemas en la Fase 3 y no se ha medido.
- **Sin modelo de lenguaje en la construcción del índice.** Las reglas de segmentación, la fusión de las secciones 1.1 a 1.3 y la tabla de enlaces se diseñaron manualmente, a partir del análisis de los documentos y de los fallos observados en la evaluación. La construcción del índice es determinista y no invoca un modelo de lenguaje. Existen alternativas que sí lo hacen y que no se evaluaron: segmentación semántica o guiada por un LLM, prefijos de contexto generados por un LLM para cada fragmento y propuestas automáticas de enlaces entre documentos. La única variante cercana que se midió, las frases típicas de cliente (P), no aportó mejoras. Para un corpus mayor, una extensión razonable es que un LLM proponga los enlaces y una persona los apruebe, porque los enlaces protegen casos de seguridad.
- **Mantenimiento del enlace.** La tabla de enlaces se construye a mano. Cada pregunta nueva de la FAQ exige decidir a qué sección de políticas se vincula, y un enlace omitido deja a esa pregunta sin la protección del mecanismo.
- **Qué mide la métrica.** El recall mide si el texto que exige cada pregunta llega al contexto. No evalúa la calidad de la respuesta generada, que se examina en la Fase 3.
- **Parámetros calibrados con este corpus.** Los tamaños de las estrategias clásicas se fijaron con valores habituales (400 a 1 000 caracteres) y no se optimizaron. Otros tamaños podrían cambiar el ranking de las estrategias clásicas.

---

## 5. Reproducibilidad

Los resultados de esta fase se regeneran con Ollama en ejecución y el modelo `bge-m3` descargado:

```bash
pip install numpy requests langchain-text-splitters
python medir_chunking.py        # comparación de estrategias de segmentación (sección 2.4)
python medir_segmentacion.py    # intervenciones F, E y P por separado (sección 2.5)
```

Los detalles por pregunta quedan en `resultados_chunking_bge-m3_latest.json` y `resultados_segmentacion.json`. Las mediciones de robustez con `qwen3-embedding:4b` (sección 2.5) usan `python medir_segmentacion.py --modelo qwen3-embedding:4b --instruccion-generica`, que guarda su resultado en `resultados_segmentacion_qwen3-embedding_4b_generica.json`.
