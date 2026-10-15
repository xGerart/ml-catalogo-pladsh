# Categorización automática de un catálogo de retail

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xGerart/ml-catalogo-pladsh/blob/main/notebooks/proyecto_final.ipynb)

Proyecto individual de Aprendizaje de Máquina (Maestría en Ciencia de Datos, USFQ).

**Problema.** En una tienda de productos de belleza, 10 697 de 11 296 productos del ERP
no tienen categoría. El único dato disponible es el nombre: texto corto, con
abreviaturas y marcas. Se clasifica cada producto en una de 8 macro-categorías y se
comparan dos técnicas: **TF-IDF de n-gramas de caracteres + χ² + regresión logística**
contra **Word2Vec preentrenado + SVM**.

## Resultados

| | TF-IDF + LogReg | Word2Vec + SVM |
|---|---|---|
| F1-macro, CV 10 × 10 (media) | **0,759** | 0,648 |
| F1-macro, test | **0,750** | 0,582 |
| Accuracy, test | **0,825** | 0,695 |
| AUC macro, test | **0,973** | 0,910 |

Sobre los 100 folds: Wilcoxon **p ≈ 1,2 × 10⁻¹⁷** y t-test corregido de Nadeau–Bengio
**p ≈ 4,3 × 10⁻⁶** → la diferencia es significativa. Word2Vec pierde porque no tiene vector para las marcas, que son las
palabras más informativas del catálogo.

## Estructura

- `notebooks/proyecto_final.ipynb` — entregable (abre en Colab con el botón de arriba)
- `notebooks/01_eda_taxonomia.ipynb` — clustering exploratorio usado para definir la taxonomía
- `src/catalogo/` — carga de datos, etiquetado, features y evaluación (probado con pytest)
- `data/processed/` — dataset etiquetado (`id, nombre, etiqueta`) y vectores Word2Vec recortados
- `etiquetado/` — taxonomía, prompt del etiquetado asistido y resumen de la revisión
- `scripts/` — muestreo, construcción del dataset y recorte del Word2Vec
- `DECISIONES.md` — por qué se tomó cada decisión
- `docs/superpowers/` — diseño (spec) y plan de implementación

## Uso local

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/jupyter lab notebooks/proyecto_final.ipynb
```

## Datos

Nombres de producto de un ERP en producción. Solo se publican `id`, `nombre` y
`etiqueta` de una muestra de 1 000 productos; no hay datos de clientes ni de ventas.
Vectores: Spanish Billion Words Corpus Embeddings (C. Cardellino, 2016), recortados al
vocabulario del catálogo.
