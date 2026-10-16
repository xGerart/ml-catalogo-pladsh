# Decisiones del proyecto — guía para la defensa

Cada decisión con su **porqué** y la **pregunta probable** del docente. Las cifras son
las del notebook final (`notebooks/proyecto_final.ipynb`, semilla 42).

## 1. Problema y datos

| Decisión | Por qué | Pregunta probable → respuesta corta |
|---|---|---|
| Clasificar productos por su **nombre** | 10 697 de 11 296 productos (95 %) están en «General»; el nombre es el único dato descriptivo (la descripción está vacía, la marca la tienen solo 602) | *¿Por qué no usaste el precio?* → Se solapa entre categorías y obliga a combinar texto con un número en el pipeline; no aportaba lo suficiente para justificar la complejidad |
| Descartar **ventas** | No describen qué *es* el producto; un producto nuevo no tiene ventas → el modelo aprendería «viejo vs. nuevo» | |
| Descartar **código de barras** | Es un identificador arbitrario; la llave correcta para volver al ERP es el `id` | |
| **Deduplicar** los 506 nombres repetidos antes de muestrear | Si el mismo texto cae en train y test, la métrica sale inflada (*data leakage*) | *¿Cómo evitaste fuga de datos?* → Deduplicado previo + pipelines que ajustan TF-IDF/χ² dentro de cada fold |
| Muestra **aleatoria** de 1 000 | Para que el test represente al catálogo; las ~600 categorías existentes están sesgadas (todas capilares) | *¿Por qué no usaste las categorías que ya existían?* → Solo cubren cabello; se usaron como **control externo**: 96,7 % de acuerdo |

## 2. Taxonomía y etiquetado

| Decisión | Por qué | Pregunta probable |
|---|---|---|
| Taxonomía definida **a mano** | k-means agrupaba por marca, no por tipo de producto (silhouette 0,02–0,04) | *¿Por qué no usaste los clústeres como categorías?* → Agrupan «todo ALFAPARF» junto: tintes, shampoo y aceites. No sirven como categoría de negocio |
| **Supervisado** y no no supervisado (sección 3b del notebook) | k-means (k = 8) sobre los 10 777 productos: un clúster de «de» se lleva el 71 %, los demás son marcas o una palabra; contra las etiquetas humanas ARI ≈ 0,008 (azar), pureza 0,43 | *¿Por qué no clustering?* → Las categorías las define el negocio (para qué sirve el producto) y esa información no está en la similitud de los nombres; el algoritmo encuentra marcas. Hay que enseñárselo con ejemplos etiquetados |
| Separar **Coloración** de **Cuidado capilar** | Juntas serían la clase más grande y son rubros distintos en la tienda | Frontera: matizantes (shampoo matizante = cuidado; tinte = coloración) |
| Regla «la herramienta va a su dominio» | Un pincel de uñas es de uñas, aunque sea «herramienta» | *¿Por qué la cosmetiquera es Maquillaje?* → Misma regla: accesorio del dominio maquillaje |
| **Etiquetado asistido** (LLM propone, humano revisa) | 1 000 etiquetas en ~40 min de revisión en vez de ~2 h | *¿Son confiables las etiquetas?* → Revisión humana de las 1 000 (0,4 % corregido) + control externo con categorías del ERP (96,7 %) |
| Mínimo **30 ejemplos** por clase, fijado *antes* de etiquetar | Con menos, las métricas por clase son inestables | *¿Por qué fusionaste Extensiones?* → Llegó a 28; la regla estaba fijada antes. Completarla a mano rompería la aleatoriedad |

## 3. Representación (feature extraction)

| Decisión | Por qué | Pregunta probable |
|---|---|---|
| **n-gramas de caracteres** (`char_wb`), no de palabras | Abreviaturas y errores: «ESM», «ESMALTE», «ESAMALTE» comparten «esm», «sma» | *¿Qué es `char_wb`?* → n-gramas de caracteres solo dentro de cada palabra (con bordes), no cruzando espacios |
| `sublinear_tf=True` | Usa log(1+tf): una palabra repetida no pesa linealmente más | |
| `min_df=2` | Descarta n-gramas que aparecen en un solo nombre (ruido, no generalizan) | |
| **Word2Vec del Spanish Billion Words Corpus** | Sugerencia del docente; es Word2Vec literal, español, licencia libre, citable | *¿Por qué preentrenado y no entrenarlo tú?* → 1 000 nombres de ~5 palabras (~5 000 tokens) es muy poco para entrenar embeddings |
| Vector del producto = **promedio** de sus palabras | Forma estándar y simple de pasar de palabras a documento | *¿Qué pierde el promedio?* → El orden y el peso relativo de las palabras |
| Recortar Word2Vec al vocabulario del catálogo | El modelo pesa 1,1 GB; Colab solo necesita ~4 000 palabras (4,5 MB) | |

## 4. Feature selection

| Decisión | Por qué | Pregunta probable |
|---|---|---|
| **χ²** sobre TF-IDF | Es un filtro (no entrena modelo), barato, exigido por la rúbrica | *¿Qué mide χ²?* → Dependencia entre la presencia de un n-grama y la clase |
| `k` dentro de la grilla | La selección se optimiza con la misma CV, sin fuga | |
| No aplicar χ² a Word2Vec | χ² exige features ≥ 0 y los promedios Word2Vec son negativos; además ya es denso (300 dims) | |
| `all` (sin selección) incluido en la grilla | Sin él no se puede afirmar si χ² ayuda o no | |
| **Resultado:** el mejor es `all` (0,739); k = 3500 da 0,729 | La diferencia (0,009) es menor que la desviación entre folds (≈ 0,04); la LogReg regularizada ya anula los n-gramas inútiles | *¿Entonces para qué sirvió?* → Se comparó contra no seleccionar y los datos dicen que no conviene descartar; es un hallazgo |

## 5. Modelos e hiperparámetros

| Decisión | Por qué | Pregunta probable |
|---|---|---|
| **Regresión logística** con TF-IDF | Lineal, interpretable, funciona muy bien en texto disperso de alta dimensión | *¿Qué es C?* → C = 1/λ: C grande = poca regularización |
| **SVM RBF** con Word2Vec | El espacio denso de 300 dims puede necesitar frontera no lineal | *¿Qué es γ?* → Ancho del kernel: γ grande = frontera más irregular (más varianza) |
| `StandardScaler` antes del SVM | El kernel RBF depende de distancias; sin escalar, dominan las dimensiones de mayor rango | |
| `probability=True` en el SVC | Para obtener probabilidades y calcular AUC/ROC | |
| Grillas en **escala logarítmica** | El efecto de C es multiplicativo | |
| Tuning con **5 × 3** repeated stratified k-fold | Cada combinación se entrena 15 veces; 10 × 10 sería 100 veces por combinación | *¿Por qué no nested CV?* → La rúbrica lo desaconseja por tiempo; el test apartado cumple el rol de estimación final |
| **F1-macro** como métrica | Clases desbalanceadas (6:1); accuracy premiaría acertar solo «Uñas» | *¿Diferencia entre macro y weighted?* → Macro promedia clases por igual; weighted pondera por soporte |

## 5b. Optuna (extra de la rúbrica)

| Decisión | Por qué | Pregunta probable → respuesta corta |
|---|---|---|
| Optuna con **TPE** (optimización bayesiana) | Busca en rangos continuos y concentra los intentos donde el modelo anduvo bien | *¿Qué es TPE?* → Separa los intentos en «buenos» y «malos», modela la densidad de los hiperparámetros en cada grupo y propone donde la razón buenos/malos es mayor |
| Misma CV 5 × 3, misma métrica y semilla fija | Para comparar justo contra la grilla y que sea reproducible | *¿Usaste el test?* → No; igual que la grilla, solo el 80 % de entrenamiento |
| 40 intentos para TF-IDF (grilla: 120) y 20 para SVM (grilla: 24) | Mostrar si llega al óptimo con menos evaluaciones | *¿Y llegó?* → TF-IDF: 0,7386 vs 0,7387 con un tercio; SVM: 0,6472 vs 0,6404 |
| **No** se reemplazaron los hiperparámetros de la grilla en el resto del notebook | Las mejoras (0,0001 y 0,007) son menores que la variación entre folds (≈ 0,04) | *¿Por qué no usaste lo de Optuna?* → No cambia ninguna conclusión; Optuna sirve como confirmación y muestra eficiencia |
| Importancia con **PED-ANOVA**, fijado explícitamente | Es el default de Optuna 5; fijarlo evita que cambie con la versión | *¿Qué hiperparámetro importa más?* → λ (80 %), luego n-gramas (11 %) y k (8 %) |

## 6. Ajustes hechos al ver los resultados

| Qué se vio | Ajuste | Por qué |
|---|---|---|
| Grilla de χ² llegaba a k = 5000, pero los folds tienen 3 760–4 674 n-gramas | k bajado a 250–3500 | k > features = «todas»: la selección no se evaluaba |
| Con C hasta 10³, el óptimo de LogReg cayó en el **borde** (C = 1000) | Grilla extendida a 10⁴ | Óptimo en el borde = no sabes si hay uno mejor afuera; extendida, quedó en C ≈ 464 (interior) |
| Con k hasta 3500, el mejor k también cayó en el **borde** | Se agregó `all` (sin selección) | Mismo criterio que con C; `all` resultó el mejor |
| La revisión independiente señaló que el p de Wilcoxon es optimista | Se agregó el t-test corregido de Nadeau–Bengio | El p corregido (≈ 4,3 × 10⁻⁶) es el defendible |
| `RocCurveDisplay` cambió de API en sklearn 1.7 | ROC dibujada con `roc_curve` + matplotlib | Mismo resultado en local (1.9) y en Colab |
| 10 × 10 tardó < 1 min | Se mantuvo 10 × 10 | Es lo que recomienda la rúbrica |

## 7. Comparación estadística

| Decisión | Por qué | Pregunta probable |
|---|---|---|
| **Mismos 100 folds** para ambos modelos | Hace el test **pareado** | *¿Por qué pareado?* → Cada fold es la misma partición para ambos; se comparan diferencias fold a fold |
| **Wilcoxon** y no t-test | No asume normalidad de las diferencias | *¿H₀?* → Las diferencias son simétricas alrededor de 0 (ningún modelo es sistemáticamente mejor) |
| **Estratificado** | Con clases desbalanceadas un fold podría quedar sin una clase | |
| **t-test corregido de Nadeau–Bengio** además de Wilcoxon | Los folds comparten datos → la varianza de las diferencias se subestima; la corrección la infla por (1/J + n_test/n_train) | *¿Tu p-value es real?* → El de Wilcoxon (10⁻¹⁷) es optimista; el corregido (4,3 × 10⁻⁶) es el defendible, y también rechaza H₀ |
| Sesgo de selección declarado | La comparación usa los mismos datos del tuning y la grilla de TF-IDF (120) es mayor que la de Word2Vec (24) | *¿Eso no favorece a TF-IDF?* → Levemente; por eso el test apartado: F1 0,75 en test vs 0,74–0,76 en CV, no hubo inflación |

**Resultado:** 0,759 vs 0,648 F1-macro; Wilcoxon p ≈ 1,2 × 10⁻¹⁷; t corregido p ≈ 4,3 × 10⁻⁶ → TF-IDF + LogReg es mejor.

## 8. Lectura de resultados

- **Curvas de aprendizaje → alta varianza:** train ≈ 1, validación ≈ 0,74 y aún subiendo → más datos ayudarían.
- **Matriz de confusión:** fallan Herramientas (F1 0,50) y Piel y cuerpo (0,50) por vocabulario compartido («mascarilla», «crema», «pinza», «lima»).
- **AUC alto (0,97) con F1-macro menor (0,75):** el modelo ordena bien, pero el umbral argmax falla en clases chicas.
- **t-SNE:** TF-IDF forma islas por línea de producto; Word2Vec mezcla más las clases.
- **Por qué pierde Word2Vec:** las palabras sin vector son las marcas, que son lo más informativo.

## 8b. Aplicación al catálogo completo

| Decisión | Por qué | Pregunta probable |
|---|---|---|
| Reentrenar el modelo final con los 1 000 etiquetados | Ya se eligieron los hiperparámetros; usar todos los datos da el mejor modelo para producción | *¿No es trampa entrenar con el test?* → No: el test ya cumplió su rol (estimar el desempeño); para *usar* el modelo se aprovecha todo |
| Umbral de confianza **0,8** elegido con predicciones **out-of-fold** | Elegirlo mirando el test sería usar el test para decidir | *¿Por qué 0,8?* → Out-of-fold: cubre el 72 % con 93,6 % de acierto; el test lo confirma (77 %, 96,8 %) |
| Revisión manual bajo el umbral | Las clases difíciles (Otros, Piel y cuerpo) concentran las dudas | *¿Cuánto trabajo ahorra?* → Se revisan 2 079 de 9 777 productos: ~79 % menos |
| Validación externa con la categoría del ERP | 534 productos ya categorizados por la tienda, que el modelo nunca vio | *¿Cómo sabes que acierta fuera de tu muestra?* → 93,3 % de coincidencia; el error típico es Tinte → Cuidado capilar, la frontera prevista |

## 9. Preguntas trampa probables

- *«¿Por qué el eje x es λ y no C?»* → En sklearn el hiperparámetro es C, pero C = 1/λ. Se grafica contra λ y con el error (1 − F1) para leerlo como la curva *Loss vs. λ* de la teoría: sobreajuste a la izquierda, subajuste a la derecha, óptimo en el fondo de la U.
- *«¿Esas curvas de optimización son del train?»* → Hay dos: entrenamiento (azul) y validación (naranja), ambas promediadas sobre los 15 folds de la CV 5 × 3. El hiperparámetro se elige donde el error de validación es mínimo; la de entrenamiento muestra el sobreajuste (su error llega a 0 mientras el de validación se queda en ~0,26).
- *«¿Usaste el test para elegir hiperparámetros?»* → No. El test se usó una sola vez, al final.
- *«¿Por qué no balanceaste las clases (SMOTE, class_weight)?»* → Se priorizó F1-macro y estratificación; balancear es una mejora posible, no se probó para no agrandar el alcance.
- *«¿Las etiquetas las hizo la IA?»* → Las propuso; las revisó una persona fila por fila y se midió contra las categorías puestas por la tienda.
- *«¿Entonces el modelo etiquetó el catálogo?»* → Sí: 7 698 productos con confianza ≥ 0,8; los otros 2 079 van a revisión manual. Las 1 000 etiquetas de entrenamiento las propuso un LLM y las revisé yo; el modelo de ML aprendió de ellas.
- *«¿Por qué χ² si no mejoró?»* → La rúbrica lo pide y el resultado es informativo: con regularización, descartar features no ayuda.
- *«¿De dónde sale la muestra de 1 000?»* → Del catálogo deduplicado (los nombres repetidos se eliminan antes, para evitar fuga entre entrenamiento y test), con muestreo aleatorio simple y semilla 42. La sección 2 del notebook lo reconstruye desde los datos publicados y sale exactamente la misma muestra.
