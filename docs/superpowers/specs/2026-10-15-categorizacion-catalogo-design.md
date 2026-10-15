# Categorización automática del catálogo — diseño

Proyecto individual de Aprendizaje de Máquina (maestría en Ciencia de Datos, USFQ).
Entrega y defensa: **viernes 2026-10-16**. Rúbrica completa en `RUBRICA.md`.

## 1. Objetivo y criterios de éxito

- **Problema:** 10 697 de 11 296 productos de una tienda de belleza (ERP Pladsh,
  tenant de Silvia Chicaiza) están en la categoría comodín «General». Las ~600
  categorizadas son todas capilares. El único dato útil de cada producto es su
  nombre (texto corto con abreviaturas). Se clasifica cada producto en una
  macro-categoría a partir del nombre.
- **Éxito:** puntaje completo en los ítems de la rúbrica (7/10 están en optimización
  de hiperparámetros y comparación estadística), poder defender cada decisión, y
  como objetivo secundario un modelo útil para la tienda.
- **Fuera de alcance:** aprendizaje semi-supervisado, Jev, redes neuronales,
  precio/ventas/código de barras como features, escribir las categorías de vuelta
  en el ERP.

## 2. Estructura del repo (público en GitHub)

```
ml-catalogo-pladsh/
├── README.md
├── RUBRICA.md
├── pyproject.toml                 # entorno con uv
├── data/
│   ├── raw/                       # export completo de prod — .gitignore
│   └── processed/
│       ├── productos_etiquetados.csv   # id, nombre, etiqueta — se publica
│       └── w2v_vocab.npz          # vectores Word2Vec solo del vocabulario del catálogo (numpy; Colab no necesita gensim)
├── notebooks/
│   ├── 01_eda_taxonomia.ipynb     # exploración + clustering para la taxonomía
│   └── proyecto_final.ipynb       # entregable (Colab)
├── scripts/
│   └── extraer_vocab_w2v.py       # recorta el Word2Vec completo al vocabulario
├── src/catalogo/
│   ├── data.py                    # carga, limpieza, deduplicado, muestreo
│   ├── features.py                # MeanEmbeddingVectorizer (sklearn)
│   └── evaluation.py              # CV repetida, Wilcoxon, métricas, gráficos
├── tests/
└── reports/figures/
```

Regla: en `src/` va solo la plomería reutilizable; las decisiones evaluadas
(pipelines, grillas, gráficos, test, interpretación) se ven en el notebook final.

## 3. Datos y etiquetado

- **Fuente:** `data/raw/productos.csv` (export de prod del 2026-10-06). Se usan
  solo `id` y `nombre`; se excluye el producto inactivo.
- **Deduplicado** por nombre normalizado (mayúsculas, espacios colapsados y
  recortados); se conserva el `id` menor. Ocurre antes de muestrear, para que un
  mismo texto no quede en train y test.
- **Taxonomía:** en `01_eda_taxonomia`, TF-IDF de n-gramas de caracteres + k-means
  sobre todo el catálogo; se proponen 6–9 macro-categorías + «Otros» con
  definición y ejemplos. El estudiante aprueba la lista.
- **Muestra:** 1 000 productos al azar del catálogo deduplicado (semilla fija). Las
  ~600 categorías capilares existentes no se agregan; los productos de la muestra
  que ya las tienen sirven de control (deben quedar «Capilar»).
- **Etiquetado asistido:** Claude propone una etiqueta por producto con un prompt
  fijo guardado en el repo; el estudiante revisa un Excel `id | nombre | propuesta |
  final` (desplegable cerrado, `final` prellenado, revisión ordenada por propuesta),
  corrige o marca «No sé» (excluido). Se reporta el % de correcciones.
- **Clases chicas:** una categoría con < 30 ejemplos se fusiona en «Otros».
- **Salida publicada:** solo `id, nombre, etiqueta`.

## 4. Features y modelos

- **Técnica 1 (rúbrica opción 1):** `TfidfVectorizer(analyzer="char_wb", 2–5)` →
  `SelectKBest(chi2, k)` (feature selection tipo filtro) →
  `LogisticRegression` multiclase L2. Hiperparámetros: `C = 1/λ` (gráfica principal),
  `k`, rango de n-gramas.
- **Técnica 2 (rúbrica opción 2, transfer learning):** vectores Word2Vec del
  Spanish Billion Words Corpus (Cardellino, 2016), recortados al vocabulario del
  catálogo por `scripts/extraer_vocab_w2v.py`; vector del producto = promedio de
  sus palabras (cero si ninguna tiene vector) → `StandardScaler` →
  `SVC(kernel="rbf")`. Hiperparámetros: `C`, `γ`. Sin chi² (features negativas);
  la omisión se justifica en el notebook.
- El EDA reporta el % de tokens sin vector Word2Vec y cuántos productos quedan sin
  ninguno.

## 5. Validación y comparación

1. Split estratificado 80/20; el 20 % de test no se toca hasta el final.
2. Optimización en el 80 %: `GridSearchCV` + `RepeatedStratifiedKFold(5 folds × 3
   rep.)`, métrica F1-macro. Gráfica del hiperparámetro y `learning_curve` por
   técnica.
3. Comparación: ambos modelos optimizados con `RepeatedStratifiedKFold(10 × 10)`,
   mismos folds y `random_state`; Wilcoxon pareado sobre F1-macro, α = 0,05,
   reportar p-value. Si en Colab tarda más de ~15 min, bajar a 5 × 2 (permitido).
4. Test: Accuracy, Precision, Recall, F1 macro y weighted, AUC macro y micro, curva
   ROC multiclase.
5. Limitación a declarar: los scores de CV repetida no son independientes.

## 6. Notebook final (`proyecto_final.ipynb`)

Setup (clona repo, agrega `src/` al path —sin `pip install`, que en Colab puede exigir reiniciar el runtime—, semillas) → Introducción → Datos y etiquetado →
EDA (balance, largo de nombres y skewness, word cloud, palabras frecuentes por
clase, cobertura Word2Vec) → Feature extraction → Feature selection (chi²) →
Pipelines → Optimización → Comparación estadística + métricas de test → t-SNE de
ambos espacios coloreado por clase → Conclusiones → Referencias IEEE.
Los prompts de IA usados para el código van como comentarios en las celdas que
corresponden; el prompt de etiquetado, en «Datos y etiquetado».

## 7. Verificación

- Ejecución completa del notebook final en Colab con un runtime limpio.
- pytest de `src/`: el deduplicado no deja nombres repetidos; el transformer devuelve
  la forma correcta y ceros para tokens desconocidos; el split no comparte ids entre
  train y test.
- `data/raw/` nunca se commitea (`.gitignore`); el CSV publicado tiene solo 3
  columnas.

## 8. Cronograma

| Cuándo | Qué |
|---|---|
| Jue mañana | Fase 0 (repo, entorno) + fase 1 (limpieza, EDA) + fase 2 (taxonomía → aprobación) |
| Jue mediodía | Fase 3: propuesta de etiquetas → revisión del estudiante (~40 min) |
| Jue tarde | Fases 4–5: features, pipelines, optimización, Wilcoxon, t-SNE |
| Jue noche | Fase 6: notebook final, README, prueba en Colab |
| Vie | Fase 7: preparar la defensa |
