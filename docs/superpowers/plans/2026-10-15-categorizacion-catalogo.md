# Categorización automática del catálogo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clasificar productos de un catálogo de retail en macro-categorías a partir de su nombre, comparando estadísticamente TF-IDF + regresión logística contra Word2Vec preentrenado + SVM, entregado como notebook de Colab sobre un repo público.

**Architecture:** Paquete `src/catalogo/` con tres módulos de plomería (datos, features, evaluación) probados con pytest; dos notebooks (exploración/taxonomía y entregable final) donde se ven todas las decisiones evaluadas. El etiquetado es asistido: Claude propone, el estudiante revisa un Excel. Los vectores Word2Vec se recortan localmente al vocabulario del catálogo y se publican como `.npz`.

**Tech Stack:** Python 3.12 (uv), pandas, numpy, scikit-learn, scipy, matplotlib, wordcloud, openpyxl; gensim y jupytext solo en dev; pytest.

**Spec:** `docs/superpowers/specs/2026-10-15-categorizacion-catalogo-design.md` (rúbrica en `RUBRICA.md`).

## Orden de ejecución

Tasks 1 → 2 → 3 → 4 (hasta el Step 8, Excel generado). Mientras el estudiante revisa
el Excel (Task 4, Step 9), se ejecutan **Task 5 y Task 6**, que no dependen de las
etiquetas. Al volver la revisión: Task 4 Steps 10–11 → Task 7 → Task 8.

## Global Constraints

- Raíz del proyecto: `/Users/gerart/Documents/ml-catalogo-pladsh` (fuera del repo del ERP). Todos los paths del plan son relativos a ella.
- Python 3.12 en el venv (`uv venv --python 3.12`); el código no puede depender de gensim en runtime (Colab).
- `data/raw/` e `data/interim/` NUNCA se commitean; el único dato publicado es `data/processed/productos_etiquetados.csv` con exactamente las columnas `id,nombre,etiqueta`, más `data/processed/w2v_vocab.npz`.
- Semilla global `SEED = 42` en todo muestreo, split y CV.
- Muestra de etiquetado: 1 000 productos al azar del catálogo deduplicado.
- Clase con < 30 ejemplos → se fusiona en «Otros». Etiqueta de exclusión: «No sé».
- Validación: test 20 % estratificado; tuning con `RepeatedStratifiedKFold(n_splits=5, n_repeats=3)`; comparación con `RepeatedStratifiedKFold(n_splits=10, n_repeats=10)` (bajar a 5×2 si en Colab tarda > ~15 min); métrica de optimización F1-macro; Wilcoxon α = 0,05.
- Vectores: `https://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.bin.gz` (1,1 GB, Word2Vec binario, 300 dims).
- Copy en español neutral con tildes; referencias IEEE; prompts de IA usados para código como comentarios en la celda donde se usó.
- Repo GitHub `xGerart/ml-catalogo-pladsh`, público. **Publicar (crear repo/push) requiere confirmación explícita del estudiante.**

## Review Focus

- Producto cuyo nombre no tiene ninguna palabra con vector Word2Vec (solo abreviaturas/códigos) → vector cero, nunca NaN; el SVM no debe romperse. Test en Task 5.
- Nombre que tras tokenizar queda vacío (solo dígitos/símbolos, ej. «# 14 12ML») → tokenizer devuelve `[]` y TF-IDF no falla. Test en Task 5.
- Revisión del Excel incompleta (celda `final` vacía) o con un valor fuera de la lista → `build_labeled_dataset` falla ruidoso con el id del problema, nunca rellena en silencio. Test en Task 4.
- Clase que tras fusionar sigue con < 13 ejemplos («Otros» chico) → tras el 80/20 quedaría con < 10 en train y la CV estratificada de 10 folds no tendría un ejemplo por fold; `build_labeled_dataset` falla ruidoso. Test en Task 4.
- Dos llamadas de CV con el mismo `random_state` deben generar exactamente los mismos folds, o el Wilcoxon pareado deja de ser pareado. Test en Task 6.

---

### Task 1: Scaffold del repo y entorno

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/catalogo/__init__.py`, `tests/test_smoke.py`
- Move: `data/productos.csv` → `data/raw/productos.csv`; `data/README.md` → `data/raw/README.md`

**Interfaces:**
- Produces: paquete importable `catalogo` (vía `pythonpath = ["src"]` en pytest), venv en `.venv/`.

- [ ] **Step 1: Mover los datos crudos y crear carpetas**

```bash
cd /Users/gerart/Documents/ml-catalogo-pladsh
mkdir -p data/raw data/interim data/processed notebooks scripts src/catalogo tests reports/figures etiquetado
mv data/productos.csv data/raw/productos.csv
mv data/README.md data/raw/README.md
rm -f .DS_Store
```

- [ ] **Step 2: Escribir `pyproject.toml`**

```toml
[project]
name = "catalogo"
version = "0.1.0"
description = "Categorización automática de un catálogo de retail a partir del nombre del producto"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.0",
    "scikit-learn>=1.4",
    "scipy>=1.11",
    "matplotlib>=3.8",
    "wordcloud>=1.9",
    "openpyxl>=3.1",
]

[project.optional-dependencies]
dev = ["pytest>=8", "jupyter", "ipykernel", "jupytext>=1.16", "gensim>=4.3.3"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 3: Escribir `.gitignore`**

```gitignore
.venv/
__pycache__/
.ipynb_checkpoints/
.DS_Store
data/raw/
data/interim/
*.bin.gz
```

- [ ] **Step 4: Crear el paquete y el smoke test**

`src/catalogo/__init__.py`:
```python
"""Categorización automática del catálogo a partir del nombre del producto."""

SEED = 42
```

`tests/test_smoke.py`:
```python
import catalogo


def test_paquete_importa_y_fija_semilla():
    assert catalogo.SEED == 42
```

- [ ] **Step 5: Crear el venv, instalar y correr el test**

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
.venv/bin/pytest -q
```
Expected: `1 passed`.

- [ ] **Step 6: git init y primer commit (spec, plan, rúbrica, scaffold)**

```bash
git init -b main
git add .gitignore pyproject.toml RUBRICA.md docs src tests
git status --short   # verificar que data/raw NO aparece
git commit -m "chore: scaffold del proyecto, spec y plan"
```

---

### Task 2: Carga, limpieza, deduplicado y muestreo (`data.py`)

**Files:**
- Create: `src/catalogo/data.py`, `tests/test_data.py`

**Interfaces:**
- Produces:
  - `normalize_name(name: str) -> str`
  - `deduplicate(df: pd.DataFrame) -> pd.DataFrame` (cols `id, nombre`)
  - `load_catalog(path: Path) -> pd.DataFrame` (cols `id, nombre`, activos y deduplicados)
  - `sample_for_labeling(df: pd.DataFrame, n: int = 1000, seed: int = SEED) -> pd.DataFrame`
  - `find_root(start: Path | None = None) -> Path` (sube hasta encontrar `pyproject.toml`)

- [ ] **Step 1: Escribir los tests**

`tests/test_data.py`:
```python
from pathlib import Path

import pandas as pd
import pytest

from catalogo.data import (
    deduplicate,
    find_root,
    load_catalog,
    normalize_name,
    sample_for_labeling,
)


def test_normalize_name_colapsa_espacios_y_mayusculas():
    assert normalize_name("  esmalte   master\tnails ") == "ESMALTE MASTER NAILS"


def test_deduplicate_conserva_id_menor_y_no_deja_repetidos():
    df = pd.DataFrame({"id": [7, 3, 5], "nombre": ["Shampoo X", "SHAMPOO  X", "Gel Y"]})
    out = deduplicate(df)
    assert out["nombre"].is_unique
    assert set(out["id"]) == {3, 5}


def test_deduplicate_descarta_nombres_vacios():
    df = pd.DataFrame({"id": [1, 2], "nombre": ["   ", "Gel"]})
    assert deduplicate(df)["id"].tolist() == [2]


def test_load_catalog_excluye_inactivos(tmp_path: Path):
    csv = tmp_path / "p.csv"
    pd.DataFrame(
        {"id": [1, 2], "nombre": ["A", "B"], "is_active": ["t", "f"], "precio_venta": [1, 2]}
    ).to_csv(csv, index=False)
    out = load_catalog(csv)
    assert out.columns.tolist() == ["id", "nombre"]
    assert out["id"].tolist() == [1]


def test_sample_for_labeling_es_reproducible():
    df = pd.DataFrame({"id": range(50), "nombre": [f"P{i}" for i in range(50)]})
    a = sample_for_labeling(df, n=10, seed=42)
    b = sample_for_labeling(df, n=10, seed=42)
    assert a["id"].tolist() == b["id"].tolist()


def test_sample_for_labeling_falla_si_n_supera_el_catalogo():
    df = pd.DataFrame({"id": [1], "nombre": ["A"]})
    with pytest.raises(ValueError, match="1000"):
        sample_for_labeling(df, n=1000)


def test_find_root_encuentra_pyproject():
    assert (find_root() / "pyproject.toml").exists()
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `.venv/bin/pytest tests/test_data.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'catalogo.data'`.

- [ ] **Step 3: Implementar `src/catalogo/data.py`**

```python
"""Carga, limpieza, deduplicado y muestreo del catálogo."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from catalogo import SEED


def find_root(start: Path | None = None) -> Path:
    """Sube desde `start` (o el cwd) hasta el directorio que contiene pyproject.toml."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(f"No se encontró pyproject.toml desde {here}")


def normalize_name(name: str) -> str:
    """Mayúsculas y espacios colapsados: dos nombres que solo difieren en eso son el mismo."""
    return re.sub(r"\s+", " ", str(name)).strip().upper()


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por nombre normalizado, conservando el id menor.

    Se hace ANTES de muestrear: así un mismo texto nunca queda en train y test a la vez.
    """
    out = df[["id", "nombre"]].assign(nombre=df["nombre"].map(normalize_name))
    out = out[out["nombre"] != ""]
    out = out.sort_values("id").drop_duplicates("nombre", keep="first")
    return out.reset_index(drop=True)


def load_catalog(path: Path) -> pd.DataFrame:
    """Lee el export crudo del ERP y devuelve los productos activos deduplicados (id, nombre)."""
    df = pd.read_csv(path, usecols=["id", "nombre", "is_active"], dtype={"nombre": str})
    df = df[df["is_active"] == "t"]
    return deduplicate(df)


def sample_for_labeling(df: pd.DataFrame, n: int = 1000, seed: int = SEED) -> pd.DataFrame:
    """Muestra aleatoria simple y reproducible para etiquetar."""
    if n > len(df):
        raise ValueError(f"Se pidieron {n} productos pero el catálogo tiene {len(df)}")
    return df.sample(n=n, random_state=seed).reset_index(drop=True)
```

- [ ] **Step 4: Correr los tests**

Run: `.venv/bin/pytest tests/test_data.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Verificar contra los datos reales**

```bash
.venv/bin/python -c "
from catalogo.data import load_catalog, find_root
df = load_catalog(find_root() / 'data/raw/productos.csv')
print(len(df), df['nombre'].is_unique)"
```
Expected: ~10 790 filas (11 295 activos − ~506 duplicados) y `True`.

- [ ] **Step 6: Commit**

```bash
git add src/catalogo/data.py tests/test_data.py
git commit -m "feat(data): carga, deduplicado y muestreo del catálogo"
```

---

### Task 3: EDA exploratorio y taxonomía (puerta humana)

**Files:**
- Create: `notebooks/01_eda_taxonomia.py` (fuente jupytext, formato percent) → `notebooks/01_eda_taxonomia.ipynb`
- Create: `src/catalogo/taxonomy.py`, `etiquetado/taxonomia.md`

**Interfaces:**
- Consumes: `load_catalog`, `find_root` (Task 2).
- Produces: `catalogo.taxonomy.CATEGORIES: list[str]` (aprobadas por el estudiante, incluye `"Otros"`), `catalogo.taxonomy.NO_SE = "No sé"`, `catalogo.taxonomy.OTROS = "Otros"`.

- [ ] **Step 1: Escribir `notebooks/01_eda_taxonomia.py`**

```python
# %% [markdown]
# # 01 · EDA exploratorio y propuesta de taxonomía
# Clustering de los nombres del catálogo completo para descubrir las familias de
# productos escondidas en «General». No es parte de la nota: sirve para definir las
# categorías que se van a etiquetar.

# %%
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from catalogo import SEED
from catalogo.data import find_root, load_catalog

ROOT = find_root()
df = load_catalog(ROOT / "data/raw/productos.csv")
len(df)

# %%
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)
X = vec.fit_transform(df["nombre"])
X.shape

# %%
ks = range(6, 21, 2)
sil = {}
for k in ks:
    km = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit(X)
    sil[k] = silhouette_score(X, km.labels_, sample_size=3000, random_state=SEED)
pd.Series(sil, name="silhouette")

# %%
K = max(sil, key=sil.get)
km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(X)
df["cluster"] = km.labels_
terms = np.array(vec.get_feature_names_out())
for c in range(K):
    top = terms[km.cluster_centers_[c].argsort()[::-1][:12]]
    sample = df.loc[df["cluster"] == c, "nombre"].sample(
        min(8, (df["cluster"] == c).sum()), random_state=SEED
    )
    print(f"\n=== Clúster {c} · {(df['cluster'] == c).sum()} productos")
    print("n-gramas:", ", ".join(top))
    print("\n".join("  - " + s for s in sample))
```

- [ ] **Step 2: Convertir y ejecutar**

```bash
.venv/bin/jupytext --to ipynb notebooks/01_eda_taxonomia.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_taxonomia.ipynb
```
Expected: ejecuta sin errores; la última celda imprime un bloque por clúster.

- [ ] **Step 3: Redactar la propuesta de taxonomía**

Leyendo la salida de los clústeres (y una muestra adicional de nombres si un clúster es ambiguo), escribir `etiquetado/taxonomia.md` con 6–9 macro-categorías + «Otros». Para cada una: **nombre**, **definición de una línea**, **3–5 ejemplos reales** del catálogo y **casos frontera** (ej. «Tinte» ¿Capilar o aparte?). Esta lista es contenido que decide el estudiante, no código: no se inventa antes de ver los clústeres.

- [ ] **Step 4: PUERTA HUMANA — aprobación de la taxonomía**

Presentar `etiquetado/taxonomia.md` al estudiante y esperar su aprobación o cambios. No continuar hasta tener la lista final.

- [ ] **Step 5: Fijar las categorías aprobadas en `src/catalogo/taxonomy.py`**

```python
"""Taxonomía aprobada por el estudiante (ver etiquetado/taxonomia.md)."""

NO_SE = "No sé"
OTROS = "Otros"

# Orden = orden en el desplegable del Excel y en los gráficos.
CATEGORIES: list[str] = [
    # ← las categorías aprobadas en el Step 4, copiadas literalmente de taxonomia.md
    OTROS,
]
```
Las entradas de la lista son exactamente los nombres aprobados en `etiquetado/taxonomia.md` (sin añadir ni renombrar).

- [ ] **Step 6: Test de invariantes de la taxonomía**

`tests/test_taxonomy.py`:
```python
from catalogo.taxonomy import CATEGORIES, NO_SE, OTROS


def test_taxonomia_tiene_otros_y_no_incluye_no_se():
    assert OTROS in CATEGORIES
    assert NO_SE not in CATEGORIES


def test_taxonomia_sin_duplicados_y_tamano_acordado():
    assert len(CATEGORIES) == len(set(CATEGORIES))
    assert 7 <= len(CATEGORIES) <= 10  # 6–9 macro-categorías + «Otros»
```
Run: `.venv/bin/pytest tests/test_taxonomy.py -q` → Expected: `2 passed`.

- [ ] **Step 7: Commit**

```bash
git add notebooks/01_eda_taxonomia.py notebooks/01_eda_taxonomia.ipynb src/catalogo/taxonomy.py etiquetado/taxonomia.md tests/test_taxonomy.py
git commit -m "feat(taxonomia): EDA exploratorio y categorías aprobadas"
```

---

### Task 4: Etiquetado asistido y dataset publicado

**Files:**
- Create: `src/catalogo/labeling.py`, `tests/test_labeling.py`, `etiquetado/prompt_etiquetado.md`, `scripts/preparar_etiquetado.py`, `scripts/construir_dataset.py`
- Generated (no commit): `data/interim/muestra.csv`, `data/interim/propuestas.csv`, `data/interim/revision.xlsx`
- Generated (commit): `data/processed/productos_etiquetados.csv`, `etiquetado/resumen.json`

**Interfaces:**
- Consumes: `load_catalog`, `sample_for_labeling`, `find_root` (Task 2); `CATEGORIES`, `NO_SE`, `OTROS` (Task 3).
- Produces:
  - `write_review_workbook(df: pd.DataFrame, categories: list[str], path: Path) -> None` (df con `id, nombre, propuesta`)
  - `read_review_workbook(path: Path) -> pd.DataFrame` (cols `id, nombre, propuesta, final`)
  - `build_labeled_dataset(review: pd.DataFrame, categories: list[str], min_count: int = 30) -> tuple[pd.DataFrame, dict]` → (`id, nombre, etiqueta`, resumen)
  - `load_labeled(path: Path) -> pd.DataFrame`

- [ ] **Step 1: Escribir los tests**

`tests/test_labeling.py`:
```python
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import load_workbook

from catalogo.labeling import (
    build_labeled_dataset,
    read_review_workbook,
    write_review_workbook,
)
from catalogo.taxonomy import NO_SE, OTROS

CATS = ["Uñas", "Capilar", OTROS]


def _review(finals, propuestas=None):
    n = len(finals)
    return pd.DataFrame(
        {
            "id": range(1, n + 1),
            "nombre": [f"P{i}" for i in range(n)],
            "propuesta": propuestas or finals,
            "final": finals,
        }
    )


def test_workbook_ida_y_vuelta_con_desplegable(tmp_path: Path):
    df = pd.DataFrame({"id": [2, 1], "nombre": ["ESMALTE", "SHAMPOO"], "propuesta": ["Uñas", "Capilar"]})
    path = tmp_path / "r.xlsx"
    write_review_workbook(df, CATS, path)
    ws = load_workbook(path).active
    assert ws.data_validations.dataValidation, "falta el desplegable"
    back = read_review_workbook(path)
    assert back.columns.tolist() == ["id", "nombre", "propuesta", "final"]
    assert (back["final"] == back["propuesta"]).all()
    assert back["propuesta"].tolist() == sorted(back["propuesta"])  # ordenado por propuesta


def test_build_excluye_no_se_y_cuenta_correcciones():
    finals = ["Uñas"] * 30 + ["Capilar"] * 30 + [OTROS] * 30 + [NO_SE]
    props = ["Uñas"] * 29 + ["Capilar"] + ["Capilar"] * 30 + [OTROS] * 30 + ["Uñas"]
    out, resumen = build_labeled_dataset(_review(finals, props), CATS)
    assert out.columns.tolist() == ["id", "nombre", "etiqueta"]
    assert len(out) == 90
    assert resumen["excluidos_no_se"] == 1
    assert resumen["corregidos"] == 1  # el «No sé» no cuenta como corrección


def test_build_fusiona_clases_chicas_en_otros():
    finals = ["Uñas"] * 5 + ["Capilar"] * 40 + [OTROS] * 30
    out, resumen = build_labeled_dataset(_review(finals), CATS)
    assert "Uñas" not in set(out["etiqueta"])
    assert (out["etiqueta"] == OTROS).sum() == 35
    assert resumen["fusionadas_en_otros"] == ["Uñas"]


def test_build_falla_si_hay_celdas_vacias():
    finals = ["Capilar"] * 40 + [None]
    with pytest.raises(ValueError, match="41"):
        build_labeled_dataset(_review(finals, ["Capilar"] * 41), CATS)


def test_build_falla_con_etiqueta_fuera_de_la_lista():
    finals = ["Capilar"] * 40 + ["Maquillaje"]
    with pytest.raises(ValueError, match="Maquillaje"):
        build_labeled_dataset(_review(finals, ["Capilar"] * 41), CATS)


def test_build_falla_si_otros_queda_con_menos_de_13():
    finals = ["Capilar"] * 40 + [OTROS] * 3
    with pytest.raises(ValueError, match="Otros"):
        build_labeled_dataset(_review(finals), CATS)
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `.venv/bin/pytest tests/test_labeling.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'catalogo.labeling'`.

- [ ] **Step 3: Implementar `src/catalogo/labeling.py`**

```python
"""Excel de revisión del etiquetado asistido y construcción del dataset publicado."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.worksheet.datavalidation import DataValidation

from catalogo.taxonomy import NO_SE, OTROS

COLUMNS = ["id", "nombre", "propuesta", "final"]
MIN_STRATIFIED = 13  # tras el 80/20 quedan ≥ 10 por clase en train: lo que pide la CV de 10 folds


def write_review_workbook(df: pd.DataFrame, categories: list[str], path: Path) -> None:
    """Excel ordenado por propuesta, con `final` prellenado y desplegable cerrado."""
    rows = df.sort_values(["propuesta", "nombre"])
    wb = Workbook()
    ws = wb.active
    ws.title = "revision"
    ws.append(COLUMNS)
    for r in rows.itertuples(index=False):
        ws.append([r.id, r.nombre, r.propuesta, r.propuesta])
    opciones = ",".join([*categories, NO_SE])
    dv = DataValidation(type="list", formula1=f'"{opciones}"', allow_blank=False)
    dv.error = "Elige una categoría de la lista"
    ws.add_data_validation(dv)
    dv.add(f"D2:D{len(rows) + 1}")
    ws.column_dimensions["B"].width = 70
    ws.column_dimensions["C"].width = 18
    ws.column_dimensions["D"].width = 18
    ws.freeze_panes = "A2"
    wb.save(path)


def read_review_workbook(path: Path) -> pd.DataFrame:
    ws = load_workbook(path, read_only=True).active
    values = list(ws.values)
    return pd.DataFrame(values[1:], columns=list(values[0]))[COLUMNS]


def build_labeled_dataset(
    review: pd.DataFrame, categories: list[str], min_count: int = 30
) -> tuple[pd.DataFrame, dict]:
    """Valida la revisión, excluye «No sé», fusiona clases chicas en «Otros»."""
    vacias = review.loc[review["final"].isna() | (review["final"].astype(str).str.strip() == ""), "id"]
    if len(vacias):
        raise ValueError(f"Revisión incompleta: celdas `final` vacías en ids {vacias.tolist()}")
    validas = set(categories) | {NO_SE}
    fuera = sorted(set(review["final"]) - validas)
    if fuera:
        raise ValueError(f"Etiquetas fuera de la taxonomía: {fuera}")

    no_se = review["final"] == NO_SE
    df = review[~no_se].copy()
    corregidos = int((df["final"] != df["propuesta"]).sum())

    counts = df["final"].value_counts()
    chicas = sorted(c for c, n in counts.items() if n < min_count and c != OTROS)
    df["etiqueta"] = df["final"].where(~df["final"].isin(chicas), OTROS)

    final_counts = df["etiqueta"].value_counts()
    insuficientes = final_counts[final_counts < MIN_STRATIFIED]
    if len(insuficientes):
        raise ValueError(
            f"Clases con menos de {MIN_STRATIFIED} ejemplos tras fusionar: {insuficientes.to_dict()}"
        )

    resumen = {
        "revisados": int(len(review)),
        "excluidos_no_se": int(no_se.sum()),
        "etiquetados": int(len(df)),
        "corregidos": corregidos,
        "pct_corregidos": round(100 * corregidos / len(df), 2),
        "fusionadas_en_otros": chicas,
        "distribucion": {k: int(v) for k, v in final_counts.items()},
    }
    out = df[["id", "nombre", "etiqueta"]].sort_values("id").reset_index(drop=True)
    return out, resumen


def load_labeled(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"nombre": str, "etiqueta": str})
```

- [ ] **Step 4: Correr los tests**

Run: `.venv/bin/pytest tests/test_labeling.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Escribir el prompt de etiquetado `etiquetado/prompt_etiquetado.md`**

```markdown
# Prompt usado para proponer etiquetas (Claude, Anthropic)

Eres un experto en productos de belleza y cuidado personal de una tienda en Ecuador.
Clasifica cada producto en UNA de estas categorías, usando solo su nombre:

<lista de CATEGORIES con la definición de cada una, copiada de etiquetado/taxonomia.md>

Reglas:
- Responde solo con la categoría exacta de la lista.
- Si el nombre no permite decidir con razonable seguridad, responde «Otros».
- Los casos frontera se resuelven como indica taxonomia.md.

Formato: una línea por producto, `id,categoría`.
```
(La lista se rellena con el contenido literal aprobado en Task 3; el archivo final no contiene el marcador `<...>`.)

- [ ] **Step 6: Generar la muestra**

`scripts/preparar_etiquetado.py`:
```python
"""Paso 1 del etiquetado: muestra 1 000 productos al azar del catálogo deduplicado."""

from catalogo.data import find_root, load_catalog, sample_for_labeling

ROOT = find_root()
muestra = sample_for_labeling(load_catalog(ROOT / "data/raw/productos.csv"), n=1000)
muestra.to_csv(ROOT / "data/interim/muestra.csv", index=False)
print(f"{len(muestra)} productos → data/interim/muestra.csv")
```
Run: `.venv/bin/python scripts/preparar_etiquetado.py` → Expected: `1000 productos → data/interim/muestra.csv`.

- [ ] **Step 7: Claude propone las etiquetas**

Claude lee `data/interim/muestra.csv` en lotes de 200 y, aplicando literalmente `etiquetado/prompt_etiquetado.md`, escribe `data/interim/propuestas.csv` con columnas `id,propuesta`. Verificar:
```bash
.venv/bin/python -c "
import pandas as pd
from catalogo.taxonomy import CATEGORIES
m = pd.read_csv('data/interim/muestra.csv'); p = pd.read_csv('data/interim/propuestas.csv')
assert set(m.id) == set(p.id) and len(p) == 1000, 'faltan o sobran ids'
assert set(p.propuesta) <= set(CATEGORIES), set(p.propuesta) - set(CATEGORIES)
print(p.propuesta.value_counts())"
```

- [ ] **Step 8: Generar el Excel de revisión**

```bash
.venv/bin/python -c "
import pandas as pd
from catalogo.data import find_root
from catalogo.labeling import write_review_workbook
from catalogo.taxonomy import CATEGORIES
R = find_root()
df = pd.read_csv(R/'data/interim/muestra.csv').merge(pd.read_csv(R/'data/interim/propuestas.csv'), on='id')
write_review_workbook(df, CATEGORIES, R/'data/interim/revision.xlsx')
print('OK')"
open data/interim/revision.xlsx
```

- [ ] **Step 9: PUERTA HUMANA — el estudiante revisa el Excel**

Instrucciones al estudiante: recorrer por bloques de `propuesta`; cambiar `final` solo donde no esté de acuerdo; «No sé» si el nombre no alcanza; guardar el archivo. Esperar confirmación.

- [ ] **Step 10: Construir y verificar el dataset publicado**

`scripts/construir_dataset.py`:
```python
"""Paso final del etiquetado: valida la revisión y escribe el CSV publicado."""

import json

import pandas as pd

from catalogo.data import find_root
from catalogo.labeling import build_labeled_dataset, read_review_workbook
from catalogo.taxonomy import CATEGORIES

ROOT = find_root()
review = read_review_workbook(ROOT / "data/interim/revision.xlsx")
dataset, resumen = build_labeled_dataset(review, CATEGORIES)
dataset.to_csv(ROOT / "data/processed/productos_etiquetados.csv", index=False)

# Control: los productos de la muestra que en el ERP ya tenían categoría (todas capilares)
# deberían quedar «Capilar». Mide la propuesta y la revisión contra una referencia externa.
CAPILAR = "Capilar"
if CAPILAR not in CATEGORIES:
    raise ValueError("La taxonomía no tiene «Capilar»: ajustar el nombre del control")
raw = pd.read_csv(ROOT / "data/raw/productos.csv", usecols=["id", "categoria"])
control = review.merge(raw, on="id")
control = control[control["categoria"] != "General"]
resumen["control_capilar"] = {
    "productos": int(len(control)),
    "propuesta_capilar_pct": round(100 * (control["propuesta"] == CAPILAR).mean(), 2),
    "final_capilar_pct": round(100 * (control["final"] == CAPILAR).mean(), 2),
}
(ROOT / "etiquetado/resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2))
print(json.dumps(resumen, ensure_ascii=False, indent=2))
```
Run: `.venv/bin/python scripts/construir_dataset.py`
Expected: imprime el resumen; `head -1 data/processed/productos_etiquetados.csv` muestra exactamente `id,nombre,etiqueta`.

- [ ] **Step 11: Commit**

```bash
git add src/catalogo/labeling.py tests/test_labeling.py etiquetado scripts/preparar_etiquetado.py scripts/construir_dataset.py data/processed/productos_etiquetados.csv
git status --short   # data/interim NO debe aparecer
git commit -m "feat(etiquetado): etiquetado asistido y dataset publicado"
```

---

### Task 5: Features — tokenizer, vectores Word2Vec recortados y `MeanEmbeddingVectorizer`

**Files:**
- Create: `src/catalogo/features.py`, `tests/test_features.py`, `scripts/extraer_vocab_w2v.py`
- Generated (commit): `data/processed/w2v_vocab.npz`

**Interfaces:**
- Consumes: `load_catalog`, `find_root` (Task 2).
- Produces:
  - `tokenize(text: str) -> list[str]`
  - `class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin)` con `__init__(self, vectors_path: str)`, `fit(X, y=None)`, `transform(X) -> np.ndarray (n, dim) float32`
  - `vocabulary_coverage(texts, words: set[str]) -> dict` con claves `tokens_total`, `tokens_sin_vector`, `pct_tokens_sin_vector`, `productos_sin_vector`, `pct_productos_sin_vector`, `top_sin_vector` (lista de (token, frecuencia), 20)

- [ ] **Step 1: Escribir los tests**

`tests/test_features.py`:
```python
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

from catalogo.features import MeanEmbeddingVectorizer, tokenize, vocabulary_coverage


@pytest.fixture
def npz(tmp_path: Path) -> str:
    path = tmp_path / "v.npz"
    np.savez(
        path,
        words=np.array(["esmalte", "uñas", "shampoo"]),
        vectors=np.array([[1, 0], [0, 1], [3, 3]], dtype=np.float32),
    )
    return str(path)


def test_tokenize_minusculas_tildes_y_descarta_numeros():
    assert tokenize("ESMALTE Uñas #14 12ML") == ["esmalte", "uñas", "ml"]


def test_tokenize_nombre_sin_letras_devuelve_vacio():
    assert tokenize("# 14 - 250") == []


def test_transform_promedia_vectores_conocidos(npz):
    X = MeanEmbeddingVectorizer(npz).fit(["x"]).transform(["ESMALTE UÑAS", "SHAMPOO XYZ"])
    np.testing.assert_allclose(X, [[0.5, 0.5], [3, 3]])
    assert X.dtype == np.float32


def test_transform_sin_palabras_conocidas_da_cero_y_no_nan(npz):
    X = MeanEmbeddingVectorizer(npz).fit(["x"]).transform(["ESM UV LED", "# 14"])
    assert not np.isnan(X).any()
    np.testing.assert_array_equal(X, np.zeros((2, 2)))


def test_es_clonable_por_sklearn(npz):
    est = clone(MeanEmbeddingVectorizer(npz))
    assert est.vectors_path == npz


def test_vocabulary_coverage():
    cov = vocabulary_coverage(["ESMALTE ROJO", "ESM UV"], {"esmalte", "rojo"})
    assert cov["tokens_total"] == 4
    assert cov["tokens_sin_vector"] == 2
    assert cov["productos_sin_vector"] == 1
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `.venv/bin/pytest tests/test_features.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'catalogo.features'`.

- [ ] **Step 3: Implementar `src/catalogo/features.py`**

```python
"""Tokenizer y representación Word2Vec (promedio de vectores) compatible con sklearn."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

TOKEN_RE = re.compile(r"[a-záéíóúüñ]+")


def tokenize(text: str) -> list[str]:
    """Palabras alfabéticas en minúscula de al menos 2 letras (descarta números y códigos)."""
    return [t for t in TOKEN_RE.findall(str(text).lower()) if len(t) >= 2]


class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """Vector de un producto = promedio de los vectores Word2Vec de sus palabras.

    Un nombre sin ninguna palabra con vector queda en cero: es información
    («el modelo preentrenado no conoce este producto»), no un error.
    """

    def __init__(self, vectors_path: str):
        self.vectors_path = vectors_path

    def fit(self, X, y=None):
        data = np.load(self.vectors_path)
        self.index_ = {w: i for i, w in enumerate(data["words"].tolist())}
        self.matrix_ = data["vectors"].astype(np.float32)
        return self

    def transform(self, X):
        out = np.zeros((len(X), self.matrix_.shape[1]), dtype=np.float32)
        for i, text in enumerate(X):
            idx = [self.index_[t] for t in tokenize(text) if t in self.index_]
            if idx:
                out[i] = self.matrix_[idx].mean(axis=0)
        return out


def vocabulary_coverage(texts, words: set[str]) -> dict:
    """Qué tanto del catálogo cubre el modelo preentrenado."""
    faltantes: Counter[str] = Counter()
    total = sin_vector = productos_sin = 0
    for text in texts:
        toks = tokenize(text)
        missing = [t for t in toks if t not in words]
        total += len(toks)
        sin_vector += len(missing)
        faltantes.update(missing)
        if len(missing) == len(toks):
            productos_sin += 1
    n = len(texts)
    return {
        "tokens_total": total,
        "tokens_sin_vector": sin_vector,
        "pct_tokens_sin_vector": round(100 * sin_vector / max(total, 1), 2),
        "productos_sin_vector": productos_sin,
        "pct_productos_sin_vector": round(100 * productos_sin / max(n, 1), 2),
        "top_sin_vector": faltantes.most_common(20),
    }
```

- [ ] **Step 4: Correr los tests**

Run: `.venv/bin/pytest tests/test_features.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Escribir `scripts/extraer_vocab_w2v.py`**

```python
"""Recorta el Word2Vec del Spanish Billion Words Corpus al vocabulario del catálogo.

El modelo completo pesa 1,1 GB; Colab solo necesita los vectores de las palabras que
aparecen en los nombres de producto. Corre una vez, localmente (requiere gensim).
Fuente: C. Cardellino, "Spanish Billion Words Corpus and Embeddings", 2016.
"""

import urllib.request

import numpy as np
from gensim.models import KeyedVectors

from catalogo.data import find_root, load_catalog
from catalogo.features import tokenize

URL = "https://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.bin.gz"
ROOT = find_root()
RAW = ROOT / "data/raw/SBW-vectors-300-min5.bin.gz"
OUT = ROOT / "data/processed/w2v_vocab.npz"

if not RAW.exists():
    print("Descargando 1,1 GB…")
    urllib.request.urlretrieve(URL, RAW)

kv = KeyedVectors.load_word2vec_format(str(RAW), binary=True)
vocab = sorted({t for name in load_catalog(ROOT / "data/raw/productos.csv")["nombre"] for t in tokenize(name)})

words, vectors = [], []
for w in vocab:
    for variant in (w, w.capitalize(), w.upper()):
        if variant in kv.key_to_index:
            words.append(w)
            vectors.append(kv[variant])
            break

np.savez_compressed(OUT, words=np.array(words), vectors=np.array(vectors, dtype=np.float32))
print(f"{len(words)}/{len(vocab)} palabras del catálogo con vector → {OUT.relative_to(ROOT)}")
```

- [ ] **Step 6: Ejecutarlo**

Run: `.venv/bin/python scripts/extraer_vocab_w2v.py`
Expected: imprime `N/M palabras del catálogo con vector`; `ls -lh data/processed/w2v_vocab.npz` < 25 MB.

- [ ] **Step 7: Commit**

```bash
git add src/catalogo/features.py tests/test_features.py scripts/extraer_vocab_w2v.py data/processed/w2v_vocab.npz
git commit -m "feat(features): Word2Vec recortado y MeanEmbeddingVectorizer"
```

---

### Task 6: Evaluación — CV repetida, Wilcoxon, métricas y gráficos (`evaluation.py`)

**Files:**
- Create: `src/catalogo/evaluation.py`, `tests/test_evaluation.py`

**Interfaces:**
- Produces:
  - `repeated_cv_scores(estimators: dict[str, BaseEstimator], X, y, n_splits=10, n_repeats=10, seed=SEED, scoring="f1_macro") -> pd.DataFrame` (una columna por modelo, una fila por fold)
  - `wilcoxon_test(scores: pd.DataFrame, a: str, b: str, alpha=0.05) -> dict` con `estadistico, p_value, alpha, significativo, media_a, media_b, mejor`
  - `classification_metrics(y_true, y_pred, y_proba, classes) -> dict` con `accuracy, precision_macro, recall_macro, f1_macro, f1_weighted, auc_macro, auc_micro`
  - `plot_param_curve(search: GridSearchCV, param: str, ax, log=True, label=None)`
  - `plot_learning_curve(estimator, X, y, cv, ax, scoring="f1_macro", title="")`
  - `plot_roc_multiclass(y_true, y_proba, classes, ax, title="")`
  - `plot_tsne(X, labels, ax, title="", seed=SEED)`

- [ ] **Step 1: Escribir los tests**

`tests/test_evaluation.py`:
```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from catalogo.evaluation import (
    classification_metrics,
    plot_learning_curve,
    plot_param_curve,
    plot_roc_multiclass,
    plot_tsne,
    repeated_cv_scores,
    wilcoxon_test,
)

X, y = make_classification(n_samples=300, n_features=8, n_informative=5, n_classes=3, random_state=0)
MODELS = {"logreg": LogisticRegression(max_iter=500), "dummy": DummyClassifier()}


def test_repeated_cv_scores_forma_y_folds_identicos():
    a = repeated_cv_scores(MODELS, X, y, n_splits=5, n_repeats=2)
    b = repeated_cv_scores(MODELS, X, y, n_splits=5, n_repeats=2)
    assert a.shape == (10, 2)
    pd.testing.assert_frame_equal(a, b)  # mismos folds → resultados idénticos


def test_wilcoxon_detecta_diferencia_real():
    scores = repeated_cv_scores(MODELS, X, y, n_splits=5, n_repeats=2)
    res = wilcoxon_test(scores, "logreg", "dummy")
    assert res["significativo"] and res["mejor"] == "logreg"
    assert 0 <= res["p_value"] < 0.05


def test_classification_metrics_claves_y_rangos():
    clf = LogisticRegression(max_iter=500).fit(X, y)
    m = classification_metrics(y, clf.predict(X), clf.predict_proba(X), clf.classes_)
    assert set(m) == {"accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "auc_macro", "auc_micro"}
    assert all(0 <= v <= 1 for v in m.values())


def test_graficos_no_fallan():
    search = GridSearchCV(LogisticRegression(max_iter=500), {"C": [0.1, 1, 10]}, cv=3, scoring="f1_macro").fit(X, y)
    fig, axes = plt.subplots(1, 4)
    plot_param_curve(search, "C", axes[0])
    plot_learning_curve(LogisticRegression(max_iter=500), X, y, StratifiedKFold(3), axes[1])
    clf = search.best_estimator_
    plot_roc_multiclass(y, clf.predict_proba(X), clf.classes_, axes[2])
    plot_tsne(X, y, axes[3])
    plt.close(fig)
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `.venv/bin/pytest tests/test_evaluation.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'catalogo.evaluation'`.

- [ ] **Step 3: Implementar `src/catalogo/evaluation.py`**

```python
"""Validación cruzada repetida, test de Wilcoxon, métricas y gráficos."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.manifold import TSNE
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import label_binarize

from catalogo import SEED


def repeated_cv_scores(estimators, X, y, n_splits=10, n_repeats=10, seed=SEED, scoring="f1_macro"):
    """Mismos folds para todos los modelos: requisito para un test pareado."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return pd.DataFrame(
        {
            name: cross_val_score(clone(est), X, y, cv=cv, scoring=scoring, n_jobs=-1)
            for name, est in estimators.items()
        }
    )


def wilcoxon_test(scores: pd.DataFrame, a: str, b: str, alpha: float = 0.05) -> dict:
    stat, p = wilcoxon(scores[a], scores[b])
    media_a, media_b = float(scores[a].mean()), float(scores[b].mean())
    return {
        "estadistico": float(stat),
        "p_value": float(p),
        "alpha": alpha,
        "significativo": bool(p < alpha),
        "media_a": media_a,
        "media_b": media_b,
        "mejor": a if media_a >= media_b else b,
    }


def classification_metrics(y_true, y_pred, y_proba, classes) -> dict:
    y_bin = label_binarize(y_true, classes=classes)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "auc_macro": roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=classes),
        "auc_micro": roc_auc_score(y_bin, y_proba, average="micro"),
    }


def plot_param_curve(search, param: str, ax, log: bool = True, label: str | None = None):
    """Score medio ± std vs. un hiperparámetro, fijando los demás en su mejor valor."""
    res = pd.DataFrame(search.cv_results_)
    key = f"param_{param}"
    for other, best in search.best_params_.items():
        if other != param:
            res = res[res[f"param_{other}"].astype(str) == str(best)]
    res = res.sort_values(key)
    x = res[key].astype(float)
    ax.errorbar(x, res["mean_test_score"], yerr=res["std_test_score"], marker="o", capsize=3, label=label)
    if log:
        ax.set_xscale("log")
    ax.axvline(float(search.best_params_[param]), ls="--", c="grey")
    ax.set_xlabel(param)
    ax.set_ylabel(f"{search.scoring} (validación)")


def plot_learning_curve(estimator, X, y, cv, ax, scoring="f1_macro", title=""):
    sizes, train, val = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 6)
    )
    for scores, name in ((train, "entrenamiento"), (val, "validación")):
        m, s = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(sizes, m, marker="o", label=name)
        ax.fill_between(sizes, m - s, m + s, alpha=0.2)
    ax.set_xlabel("ejemplos de entrenamiento")
    ax.set_ylabel(scoring)
    ax.set_title(title)
    ax.legend()


def plot_roc_multiclass(y_true, y_proba, classes, ax, title=""):
    y_bin = label_binarize(y_true, classes=classes)
    for i, c in enumerate(classes):
        RocCurveDisplay.from_predictions(y_bin[:, i], y_proba[:, i], name=str(c), ax=ax)
    RocCurveDisplay.from_predictions(y_bin.ravel(), y_proba.ravel(), name="micro-promedio", ax=ax, linestyle="--")
    ax.plot([0, 1], [0, 1], c="grey", lw=0.8)
    ax.set_title(title)


def plot_tsne(X, labels, ax, title="", seed=SEED):
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    emb = TSNE(n_components=2, random_state=seed, init="pca", perplexity=30).fit_transform(X)
    labels = np.asarray(labels)
    for c in np.unique(labels):
        mask = labels == c
        ax.scatter(emb[mask, 0], emb[mask, 1], s=8, label=str(c), alpha=0.7)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=7)
```

- [ ] **Step 4: Correr los tests**

Run: `.venv/bin/pytest tests/test_evaluation.py -q`
Expected: `4 passed`. Luego toda la suite: `.venv/bin/pytest -q` → todo verde.

- [ ] **Step 5: Commit**

```bash
git add src/catalogo/evaluation.py tests/test_evaluation.py
git commit -m "feat(evaluation): CV repetida, Wilcoxon, métricas y gráficos"
```

---

### Task 7: Notebook final (`proyecto_final`)

**Files:**
- Create: `notebooks/proyecto_final.py` (fuente jupytext) → `notebooks/proyecto_final.ipynb`
- Generated: figuras en `reports/figures/`

**Interfaces:**
- Consumes: `find_root`, `load_labeled` (Tasks 2, 4); `tokenize`, `MeanEmbeddingVectorizer`, `vocabulary_coverage` (Task 5); todo `evaluation` (Task 6); `etiquetado/resumen.json`, `etiquetado/prompt_etiquetado.md`.

- [ ] **Step 1: Escribir las celdas de setup, datos y EDA**

```python
# %% [markdown]
# # Categorización automática de un catálogo de retail a partir del nombre del producto
# Proyecto individual · Aprendizaje de Máquina · Maestría en Ciencia de Datos (USFQ)

# %%
# Setup: en Colab clona el repo y usa src/ directamente (sin pip install).
import sys
from pathlib import Path

if "google.colab" in sys.modules:
    import os
    import subprocess
    subprocess.run(["git", "clone", "-q", "https://github.com/xGerart/ml-catalogo-pladsh.git"], check=False)
    os.chdir("ml-catalogo-pladsh")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wordcloud"], check=True)
sys.path.insert(0, str(Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent) + "/src")

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from wordcloud import WordCloud

from catalogo import SEED
from catalogo.data import find_root
from catalogo.evaluation import (
    classification_metrics, plot_learning_curve, plot_param_curve,
    plot_roc_multiclass, plot_tsne, repeated_cv_scores, wilcoxon_test,
)
from catalogo.features import MeanEmbeddingVectorizer, tokenize, vocabulary_coverage
from catalogo.labeling import load_labeled

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = find_root()
FIG = ROOT / "reports/figures"
W2V = str(ROOT / "data/processed/w2v_vocab.npz")
np.random.seed(SEED)

# %% [markdown]
# ## 1. Introducción
# (Redactar: contexto de la tienda; 10 697 de 11 296 productos en «General»; por qué
# importa —reportes por categoría, reposición—; qué se hace: dos técnicas comparadas
# estadísticamente. Tres párrafos, cifras reales del EDA.)

# %%
df = load_labeled(ROOT / "data/processed/productos_etiquetados.csv")
resumen = json.loads((ROOT / "etiquetado/resumen.json").read_text())
resumen

# %% [markdown]
# ## 2. Datos y etiquetado
# (Redactar: origen —ERP en producción, export del 2026-10-06—; deduplicado; muestra
# aleatoria de 1 000; etiquetado asistido; % corregido = `resumen["pct_corregidos"]`;
# fusiones en «Otros».) Prompt usado para proponer etiquetas: ver
# `etiquetado/prompt_etiquetado.md` (citado también abajo).

# %% [markdown]
# ## 3. Análisis exploratorio (EDA)

# %%
fig, ax = plt.subplots(figsize=(8, 4))
df["etiqueta"].value_counts().plot.barh(ax=ax)
ax.set_title("Balance de clases")
ax.set_xlabel("productos")
fig.savefig(FIG / "balance.png", bbox_inches="tight")

# %%
largo = df["nombre"].str.split().str.len()
print(f"Palabras por nombre: media {largo.mean():.2f}, mediana {largo.median()}, skewness {largo.skew():.2f}")
fig, ax = plt.subplots(figsize=(6, 3))
largo.plot.hist(bins=20, ax=ax)
ax.set_xlabel("palabras por nombre")
fig.savefig(FIG / "largo_nombres.png", bbox_inches="tight")

# %%
fig, ax = plt.subplots(figsize=(10, 5))
wc = WordCloud(width=1000, height=500, background_color="white", random_state=SEED)
ax.imshow(wc.generate(" ".join(t for n in df["nombre"] for t in tokenize(n))))
ax.axis("off")
fig.savefig(FIG / "wordcloud.png", bbox_inches="tight")

# %%
from collections import Counter

top = {
    cat: Counter(t for n in grupo["nombre"] for t in tokenize(n)).most_common(8)
    for cat, grupo in df.groupby("etiqueta")
}
pd.DataFrame({c: [f"{w} ({n})" for w, n in ws] for c, ws in top.items()})

# %%
words = set(np.load(W2V)["words"].tolist())
cov = vocabulary_coverage(df["nombre"].tolist(), words)
cov
```

- [ ] **Step 2: Escribir las celdas de feature extraction, selection, pipelines y optimización**

```python
# %% [markdown]
# ## 4. Feature extraction
# Técnica 1 — TF-IDF de n-gramas de caracteres (opción 1 de la rúbrica).
# Técnica 2 — Word2Vec preentrenado del Spanish Billion Words Corpus, promedio por
# producto (opción 2: transfer learning). (Redactar por qué cada una, citando la
# cobertura `cov` medida arriba.)
#
# ## 5. Feature selection
# χ² (filtro) sobre TF-IDF, con k optimizado. No aplica a Word2Vec: χ² exige features
# no negativas y la representación preentrenada ya es densa y de baja dimensión.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    df["nombre"], df["etiqueta"], test_size=0.2, stratify=df["etiqueta"], random_state=SEED
)
len(X_train), len(X_test)

# %% [markdown]
# ## 6. Pipelines

# %%
pipe_tfidf = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", sublinear_tf=True, min_df=2)),
    ("chi2", SelectKBest(chi2)),
    ("clf", LogisticRegression(max_iter=3000)),
])
pipe_w2v = Pipeline([
    ("w2v", MeanEmbeddingVectorizer(W2V)),
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", probability=True, random_state=SEED)),
])

# %% [markdown]
# ## 7. Optimización de hiperparámetros
# Repeated stratified k-fold (5 × 3) sobre el 80 % de entrenamiento, métrica F1-macro.

# %%
cv_tune = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
grid_tfidf = {
    "tfidf__ngram_range": [(2, 4), (2, 5)],
    "chi2__k": [500, 2000, 5000],
    "clf__C": np.logspace(-2, 3, 8),  # C = 1/λ
}
grid_w2v = {"clf__C": np.logspace(-2, 3, 8), "clf__gamma": ["scale", 1e-3, 1e-2]}
search_tfidf = GridSearchCV(pipe_tfidf, grid_tfidf, cv=cv_tune, scoring="f1_macro", n_jobs=-1).fit(X_train, y_train)
search_w2v = GridSearchCV(pipe_w2v, grid_w2v, cv=cv_tune, scoring="f1_macro", n_jobs=-1).fit(X_train, y_train)
search_tfidf.best_params_, search_tfidf.best_score_, search_w2v.best_params_, search_w2v.best_score_

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
plot_param_curve(search_tfidf, "clf__C", axes[0])
axes[0].set_title("TF-IDF + LogReg: C = 1/λ")
plot_param_curve(search_w2v, "clf__C", axes[1])
axes[1].set_title("Word2Vec + SVM: C")
plot_param_curve(search_tfidf, "chi2__k", axes[2], log=False)
axes[2].set_title("TF-IDF: k de χ²")
fig.savefig(FIG / "optimizacion.png", bbox_inches="tight")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_learning_curve(search_tfidf.best_estimator_, X_train, y_train, cv_tune, axes[0], title="TF-IDF + LogReg")
plot_learning_curve(search_w2v.best_estimator_, X_train, y_train, cv_tune, axes[1], title="Word2Vec + SVM")
fig.savefig(FIG / "curvas_aprendizaje.png", bbox_inches="tight")
# (Redactar el diagnóstico sesgo vs. varianza que muestren las curvas.)
```

- [ ] **Step 3: Escribir las celdas de comparación, test, t-SNE, conclusiones y referencias**

```python
# %% [markdown]
# ## 8. Comparación estadística: Wilcoxon sobre repeated stratified 10 × 10 CV
# Mismos folds para ambos modelos (pareado). H0: no hay diferencia en F1-macro.
# Limitación: los 100 scores no son independientes (los folds comparten datos).

# %%
N_SPLITS, N_REPEATS = 10, 10  # bajar a 5, 2 si tarda > ~15 min en Colab (permitido por la rúbrica)
scores = repeated_cv_scores(
    {"TF-IDF + LogReg": search_tfidf.best_estimator_, "Word2Vec + SVM": search_w2v.best_estimator_},
    X_train, y_train, n_splits=N_SPLITS, n_repeats=N_REPEATS,
)
res = wilcoxon_test(scores, "TF-IDF + LogReg", "Word2Vec + SVM")
fig, ax = plt.subplots(figsize=(6, 4))
scores.boxplot(ax=ax)
ax.set_ylabel("F1-macro por fold")
fig.savefig(FIG / "comparacion.png", bbox_inches="tight")
res

# %%
filas = {}
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (name, search) in zip(axes, [("TF-IDF + LogReg", search_tfidf), ("Word2Vec + SVM", search_w2v)]):
    est = search.best_estimator_
    proba = est.predict_proba(X_test)
    filas[name] = classification_metrics(y_test, est.predict(X_test), proba, est.classes_)
    plot_roc_multiclass(y_test, proba, est.classes_, ax, title=name)
fig.savefig(FIG / "roc.png", bbox_inches="tight")
pd.DataFrame(filas).round(3)

# %% [markdown]
# ## 9. Visualización con t-SNE (color = categoría)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
X_tfidf = search_tfidf.best_estimator_[:-1].transform(df["nombre"])
X_w2v = search_w2v.best_estimator_[:-1].transform(df["nombre"])
plot_tsne(X_tfidf, df["etiqueta"], axes[0], title="Espacio TF-IDF (+χ²)")
plot_tsne(X_w2v, df["etiqueta"], axes[1], title="Espacio Word2Vec")
fig.savefig(FIG / "tsne.png", bbox_inches="tight")

# %% [markdown]
# ## 10. Conclusiones
# (Redactar con los resultados obtenidos: qué técnica ganó y si la diferencia es
# significativa —p-value—; qué clases se confunden; cómo la cobertura de Word2Vec
# explica el resultado; qué diagnosticaron las curvas de aprendizaje; limitaciones.
# Sin teoría.)
#
# ## 11. Referencias
# [1] C. Cardellino, "Spanish Billion Words Corpus and Embeddings," 2016. [Online]. Available: https://crscardellino.ar/SBWCE/
# [2] T. Mikolov, K. Chen, G. Corrado, and J. Dean, "Efficient estimation of word representations in vector space," arXiv:1301.3781, 2013.
# [3] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011.
# [4] F. Wilcoxon, "Individual comparisons by ranking methods," Biometrics Bulletin, vol. 1, no. 6, pp. 80–83, 1945.
# [5] L. van der Maaten and G. Hinton, "Visualizing data using t-SNE," J. Mach. Learn. Res., vol. 9, pp. 2579–2605, 2008.
# [6] Anthropic, "Claude," 2026. Asistente usado para proponer etiquetas y escribir código (prompts en las celdas correspondientes).
```

- [ ] **Step 4: Convertir, ejecutar de punta a punta y medir el tiempo**

```bash
.venv/bin/jupytext --to ipynb notebooks/proyecto_final.py
time .venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/proyecto_final.ipynb
ls reports/figures
```
Expected: ejecuta sin errores; existen `balance.png, largo_nombres.png, wordcloud.png, optimizacion.png, curvas_aprendizaje.png, comparacion.png, roc.png, tsne.png`. Si el tiempo total pasa de ~20 min, bajar `N_SPLITS, N_REPEATS = 5, 2` y reducir la grilla TF-IDF (`ngram_range` a uno solo) y re-ejecutar.

- [ ] **Step 5: Redactar los textos con los resultados reales**

Reemplazar cada bloque «(Redactar: …)» por texto final escrito con las cifras que produjo la ejecución (introducción, datos y etiquetado con el prompt citado, explicación de features, diagnóstico de curvas, interpretación del Wilcoxon, conclusiones). Agregar como comentario `# Prompt (IA): …` en las celdas de código escritas con asistencia. Re-ejecutar Step 4 para que el notebook guardado tenga las salidas.

- [ ] **Step 6: Commit**

```bash
git add notebooks/proyecto_final.py notebooks/proyecto_final.ipynb reports/figures
git commit -m "feat(notebook): notebook final con optimización, comparación y t-SNE"
```

---

### Task 8: README, publicación y prueba en Colab

**Files:**
- Create: `README.md`

- [ ] **Step 1: Escribir `README.md`**

```markdown
# Categorización automática de un catálogo de retail

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xGerart/ml-catalogo-pladsh/blob/main/notebooks/proyecto_final.ipynb)

Proyecto individual de Aprendizaje de Máquina (Maestría en Ciencia de Datos, USFQ).

**Problema.** En una tienda de belleza, 10 697 de 11 296 productos del ERP no tienen
categoría. El único dato disponible es el nombre (texto corto con abreviaturas). Se
clasifica cada producto en una macro-categoría comparando dos técnicas:
TF-IDF + χ² + regresión logística contra Word2Vec preentrenado + SVM.

**Resultados.** <tabla de F1-macro/AUC en test y p-value del Wilcoxon, copiada del notebook>

## Estructura
- `notebooks/proyecto_final.ipynb` — entregable (abre en Colab con el botón de arriba)
- `notebooks/01_eda_taxonomia.ipynb` — clustering exploratorio para definir las categorías
- `src/catalogo/` — datos, features y evaluación (probados con pytest)
- `data/processed/` — dataset etiquetado (`id, nombre, etiqueta`) y vectores Word2Vec recortados
- `etiquetado/` — taxonomía, prompt de etiquetado asistido y resumen de la revisión
- `scripts/` — preparación del etiquetado y recorte del Word2Vec

## Uso local
    uv venv --python 3.12 && uv pip install -e ".[dev]"
    .venv/bin/pytest -q
    .venv/bin/jupyter lab notebooks/proyecto_final.ipynb

## Datos
Nombres de producto de un ERP en producción, con autorización de la tienda. Solo se
publican id, nombre y etiqueta. Vectores: Spanish Billion Words Corpus (Cardellino, 2016).
```
(La línea de resultados se completa con las cifras reales del notebook; el archivo final no contiene `<…>`.)

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README del proyecto"
```

- [ ] **Step 3: PUERTA HUMANA — confirmar publicación**

Preguntar al estudiante si se crea el repo público `xGerart/ml-catalogo-pladsh` y se hace push (es un paso hacia afuera, irreversible en la práctica). Con su sí:
```bash
gh repo create xGerart/ml-catalogo-pladsh --public --source . --push
```

- [ ] **Step 4: Prueba en Colab con runtime limpio**

El estudiante abre el badge del README, ejecuta *Entorno de ejecución → Ejecutar todo* y confirma que termina sin errores y en cuánto tiempo. Si falla, se corrige, se commitea y se repite.
