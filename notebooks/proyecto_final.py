# %% [markdown]
# # Categorización automática de un catálogo de retail a partir del nombre del producto
# Proyecto individual · Aprendizaje de Máquina · Maestría en Ciencia de Datos (USFQ)

# %%
# Prompt (IA): «Arma la celda de setup para que el notebook corra igual en Colab y en
# local: en Colab clona el repo para tener los datos publicados; fija la semilla. Todo el
# código del análisis se define dentro del notebook.»
import os
import sys
from pathlib import Path

if "google.colab" in sys.modules:
    import subprocess
    subprocess.run(["git", "clone", "-q", "https://github.com/xGerart/ml-catalogo-pladsh.git"], check=False)
    os.chdir("ml-catalogo-pladsh")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wordcloud", "optuna"], check=True)

import json
import re
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import wilcoxon
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, auc, classification_report, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_val_score,
    learning_curve, train_test_split,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from wordcloud import WordCloud

SEED = 42  # semilla única de todo el proyecto
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="k=.*is greater than n_features")
# Raíz del proyecto: la carpeta clonada en Colab, o la carpeta padre si se ejecuta desde notebooks/.
ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
FIG = ROOT / "reports/figures"
# Ruta relativa: no deja la ruta local de la máquina en las salidas del notebook.
W2V = os.path.relpath(ROOT / "data/processed/w2v_vocab.npz")
np.random.seed(SEED)
print(f"Python {sys.version.split()[0]} · scikit-learn {sklearn.__version__} · numpy {np.__version__} · "
      f"pandas {pd.__version__} · semilla {SEED}")

# %% [markdown]
# ## 1. Introducción
#
# **Contexto.** Una tienda de productos de belleza en Ecuador opera con un ERP propio. De
# sus **11 296 productos, 10 697 (95 %) están en la categoría comodín «General»**: se
# cargaron desde el sistema anterior sin categoría. Las ~600 que sí la tienen son todas
# capilares (Shampoo, Mascarilla, Matizantes…), pero el catálogo incluye uñas,
# maquillaje, pestañas, skincare, herramientas y accesorios.
#
# **Por qué importa.** Sin categoría no hay reportes de ventas por rubro, ni análisis de
# margen por línea, ni reglas de reposición por familia de producto. Categorizar 10 697
# productos a mano es inviable, y el único dato disponible de cada uno es su **nombre**:
# texto corto (≈5 palabras), en mayúsculas, con abreviaturas («ESM UV LED #12 12ML»),
# marcas y códigos.
#
# **Qué se hace.** Se plantea como **clasificación supervisada multiclase de texto
# corto**: (1) se define una taxonomía de 8 macro-categorías; (2) se etiqueta una muestra
# aleatoria de 1 000 productos; (3) se comparan dos técnicas —**TF-IDF de n-gramas de
# caracteres + χ² + regresión logística** contra **Word2Vec preentrenado + SVM**—, con
# optimización de hiperparámetros por validación cruzada repetida y un **test de
# Wilcoxon** para decidir si la diferencia es estadísticamente significativa; (4) se
# aplica el mejor modelo a los 9 777 productos restantes, con un umbral de confianza que
# separa lo que se categoriza solo de lo que necesita revisión manual.

# %% [markdown]
# ### Flujo del proyecto
#
# De la entrada (nombres de producto exportados del ERP) a la salida (catálogo
# categorizado): limpieza, etiquetado asistido de una muestra, dos técnicas optimizadas y
# comparadas estadísticamente, evaluación en un test apartado y aplicación de la técnica
# ganadora al resto del catálogo. El test (magenta) solo se usa al final. El diagrama lo
# genera `scripts/diagrama_flujo.py` (SVG), exportado a PNG para que se vea en Colab.

# %%
# Prompt (IA): «Muestra el diagrama de flujo del proyecto en el notebook de forma que se vea en Colab.»
# Se muestra en PNG: Colab no siempre renderiza salidas SVG. El SVG fuente está en el
# mismo directorio (el README lo usa) y el PNG se exporta de él con un navegador.
from IPython.display import Image, display

display(Image(filename=str(ROOT / "reports/figures/diagrama_flujo.png"), width=900))

# %% [markdown]
# ## 2. Datos y etiquetado
#
# **Origen.** Export de solo lectura del ERP en producción (2026-10-06). Solo se usan
# `id` y `nombre`; precio, ventas y código de barras se descartaron a propósito: las
# ventas no dicen qué *es* un producto y un producto nuevo no las tiene; el código de
# barras es un identificador arbitrario.
#
# **Deduplicado.** 506 nombres estaban repetidos. Se normalizaron (mayúsculas, espacios)
# y se dejó uno por nombre **antes** de muestrear: si el mismo texto cae en
# entrenamiento y en test, la métrica sale inflada (*data leakage*). Quedan 10 777
# productos únicos.
#
# **Taxonomía.** Se exploró con k-means sobre n-gramas de caracteres (sección 3b).
# Hallazgo: los clústeres agrupan por **marca** (ALFAPARF, Salon In) o por una palabra
# frecuente, y no por tipo de producto (silhouette 0,02–0,04), así que la taxonomía se
# definió con criterio humano por *para qué sirve* el producto (`etiquetado/taxonomia.md`).
# Ese mismo hallazgo justifica un modelo supervisado.
#
# **Muestra y etiquetado asistido.** 1 000 productos al azar (semilla 42) del catálogo
# deduplicado. Un LLM (Claude [7]) propuso una etiqueta por producto con un prompt fijo
# (`etiquetado/prompt_etiquetado.md`, reproducido abajo) y el autor revisó las 1 000 en
# una planilla: **corrigió 4 (0,4 %)**, todas cosmetiqueras y maletines que pasaron de
# «Otros» a «Maquillaje».
#
# **Control externo.** 61 productos de la muestra ya tenían categoría en el ERP (puesta
# por personal de la tienda). El 96,7 % quedó en una categoría capilar, como corresponde;
# los 2 restantes («BIOAQUA RICE RAW PULP», «DBELESA GOTAS…») son de skincare y el ERP los
# tenía como «Aceite» y «Serum».
#
# **Clase fusionada.** «Extensiones y accesorios de cabello» tuvo 28 ejemplos, menos que
# el mínimo de 30 fijado antes de etiquetar, y se fusionó en «Otros». Quedan **8 clases**.

# %%
# Prompt (IA): «Carga el dataset etiquetado y el resumen de la revisión del etiquetado.»
df = pd.read_csv(ROOT / "data/processed/productos_etiquetados.csv", dtype={"nombre": str, "etiqueta": str})
resumen = json.loads((ROOT / "etiquetado/resumen.json").read_text())
resumen

# %% [markdown]
# **Procedencia de la muestra.** La celda siguiente contiene el código que define qué
# productos se etiquetaron: normalización y deduplicado de los nombres **antes** de
# muestrear (para que un mismo texto nunca quede en entrenamiento y en test), y muestreo
# aleatorio simple de 1 000 productos con la semilla 42. Como verificación, reconstruye la
# muestra a partir de los datos publicados (los 1 000 etiquetados más los 9 777 restantes)
# y comprueba que salen exactamente los mismos productos.

# %%
# Prompt (IA): «Muestra cómo se obtuvo la muestra de 1 000: normalización y deduplicado de
# nombres antes de muestrear y muestreo aleatorio simple con semilla; verifica que se
# reconstruye exactamente desde los datos publicados.»
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


def sample_for_labeling(df: pd.DataFrame, n: int = 1000, seed: int = SEED) -> pd.DataFrame:
    """Muestra aleatoria simple y reproducible para etiquetar."""
    if n > len(df):
        raise ValueError(f"Se pidieron {n} productos pero el catálogo tiene {len(df)}")
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


catalogo_dedup = deduplicate(pd.concat(
    [df[["id", "nombre"]], pd.read_csv(ROOT / "data/processed/catalogo_sin_etiquetar.csv")[["id", "nombre"]]]
))
muestra = sample_for_labeling(catalogo_dedup, n=1000)
print(f"Catálogo deduplicado: {len(catalogo_dedup)} productos únicos")
print(f"Muestra reconstruida con semilla {SEED} idéntica a los 1 000 etiquetados: {set(muestra['id']) == set(df['id'])}")

# %%
# Prompt (IA): «Muestra el prompt con el que se propusieron las etiquetas.»
print((ROOT / "etiquetado/prompt_etiquetado.md").read_text())

# %% [markdown]
# ## 3. Análisis exploratorio (EDA)
#
# Los datos se manejan con pandas [10] y NumPy [9]; los gráficos, con matplotlib [12] y
# wordcloud [14].

# %%
# Prompt (IA): «Grafica el balance de clases del dataset etiquetado.»
fig, ax = plt.subplots(figsize=(8, 4))
df["etiqueta"].value_counts().plot.barh(ax=ax)
ax.set_title("Balance de clases")
ax.set_xlabel("productos")
fig.savefig(FIG / "balance.png", bbox_inches="tight")

# %% [markdown]
# **¿Está balanceada?** No. «Uñas» tiene 318 ejemplos y «Herramientas y equipos de
# cabello» 52: una razón de 6 a 1. Consecuencias que se aplican en todo el notebook:
# (1) la métrica de optimización es **F1-macro**, que pesa igual a todas las clases (la
# accuracy premiaría acertar solo «Uñas»); (2) todos los splits y la validación cruzada
# son **estratificados**, para que cada fold conserve la proporción de clases.

# %%
# Prompt (IA): «Calcula y grafica la distribución del largo de los nombres con su skewness.»
largo = df["nombre"].str.split().str.len()
print(f"Palabras por nombre: media {largo.mean():.2f}, mediana {largo.median()}, skewness {largo.skew():.2f}")
fig, ax = plt.subplots(figsize=(6, 3))
largo.plot.hist(bins=20, ax=ax)
ax.set_xlabel("palabras por nombre")
fig.savefig(FIG / "largo_nombres.png", bbox_inches="tight")

# %% [markdown]
# **¿Está normalizada? ¿Hace falta una transformación no lineal?** El largo de los
# nombres tiene asimetría positiva (skewness ≈ 1,66): la mayoría tiene 4–6 palabras y
# unos pocos son descripciones largas. **No se transforma** porque el largo no entra como
# feature. Lo que sí se normaliza es la representación: TF-IDF con `sublinear_tf`
# (log de la frecuencia) y normalización L2 por documento; los vectores Word2Vec se
# estandarizan con `StandardScaler` porque el kernel RBF del SVM depende de distancias.
# **¿Estacionalidad?** No aplica: no es una serie de tiempo.

# %% [markdown]
# **Tokenizador.** Para el word cloud, las palabras frecuentes y Word2Vec se necesita
# partir cada nombre en palabras: minúsculas, solo letras (con tildes y ñ) y de al menos
# 2 letras, así se descartan números, códigos y unidades sueltas («ESM UV LED #12 12ML» →
# esm, uv, led, ml). TF-IDF no lo usa: trabaja con n-gramas de caracteres.

# %%
# Prompt (IA): «Escribe un tokenizador que devuelva las palabras alfabéticas en minúscula
# de al menos 2 letras, con tildes y ñ, descartando números y códigos.»
TOKEN_RE = re.compile(r"[a-záéíóúüñ]+")


def tokenize(text: str) -> list[str]:
    """Palabras alfabéticas en minúscula de al menos 2 letras (descarta números y códigos)."""
    return [t for t in TOKEN_RE.findall(str(text).lower()) if len(t) >= 2]


tokenize("ESM UV LED #12 12ML")

# %%
# Prompt (IA): «Genera un word cloud con las palabras de los nombres de producto.»
fig, ax = plt.subplots(figsize=(10, 5))
wc = WordCloud(width=1000, height=500, background_color="white", random_state=SEED)
ax.imshow(wc.generate(" ".join(t for n in df["nombre"] for t in tokenize(n))))
ax.axis("off")
fig.savefig(FIG / "wordcloud.png", bbox_inches="tight")

# %%
# Prompt (IA): «Muestra las palabras más frecuentes de cada categoría en una tabla.»
top = {
    cat: Counter(t for n in grupo["nombre"] for t in tokenize(n)).most_common(8)
    for cat, grupo in df.groupby("etiqueta")
}
pd.DataFrame({c: [f"{w} ({n})" for w, n in ws] for c, ws in top.items()})

# %% [markdown]
# **Palabras frecuentes.** Cada clase tiene palabras muy distintivas («esmalte», «nails»,
# «gel», «acrilico» en Uñas; «tinte», «color» en Coloración) y **marcas** que casi
# definen la clase (alfaparf, thyms, farmavita en Coloración). Las unidades («ml», «oz»)
# y conectores («de», «para») aparecen en todas y aportan poco; TF-IDF les baja el peso
# solo por aparecer en muchos documentos.

# %% [markdown]
# ### 3b. ¿Por qué supervisado y no no supervisado? Clustering exploratorio
#
# Antes de etiquetar a mano, la pregunta natural es si las categorías se pueden
# **descubrir** sin etiquetas. Se aplica k-means [21] sobre TF-IDF de n-gramas de
# caracteres (3–5) —cada nombre se representa por los fragmentos de 3 a 5 letras que
# contiene, con más peso para los fragmentos raros; es la misma representación de la
# técnica 1 y se detalla en la sección 4— a los **10 777 productos** del catálogo deduplicado (los 1 000
# etiquetados más los 9 777 restantes) y se mide: (1) el **silhouette** [22] para varios
# k, que indica si existen grupos naturales bien separados (va de −1 a 1; cerca de 0 =
# grupos solapados); (2) qué agrupa cada clúster con k = 8, el mismo número de clases; y
# (3) cuánto coinciden esos clústeres con las etiquetas humanas de los 1 000 productos,
# con el **ARI** [23] (*adjusted Rand index*: 0 = coincidencia al azar, 1 = idénticos), la
# **NMI** (información mutua normalizada) y la **pureza** (fracción de productos que
# pertenecen a la clase mayoritaria de su clúster).

# %%
# Prompt (IA): «Agrega al EDA el clustering exploratorio: k-means sobre TF-IDF de n-gramas
# de caracteres del catálogo completo, silhouette para varios k, y compara los clústeres de
# k = 8 con las etiquetas humanas (ARI, NMI, pureza) para justificar el enfoque supervisado.»
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

catalogo_completo = pd.concat(
    [df[["id", "nombre"]], pd.read_csv(ROOT / "data/processed/catalogo_sin_etiquetar.csv")[["id", "nombre"]]],
    ignore_index=True,
)
vec_km = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)
X_km = vec_km.fit_transform(catalogo_completo["nombre"])

# Silhouette sobre una muestra de 3 000 puntos: calcularlo con los 10 777 es cuadrático en memoria.
silhouette = {}
for k in range(6, 21, 2):
    grupos = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit_predict(X_km)
    silhouette[k] = silhouette_score(X_km, grupos, sample_size=3000, random_state=SEED)

# k = 8: el mismo número de clases que la taxonomía, para comparar en igualdad de condiciones.
km = KMeans(n_clusters=8, n_init=10, random_state=SEED).fit(X_km)
catalogo_completo["cluster"] = km.labels_
terminos = np.array(vec_km.get_feature_names_out())


def palabra_clave(c):
    """La palabra más frecuente del clúster que contiene su n-grama de mayor peso en el centroide."""
    ngrama = terminos[km.cluster_centers_[c].argmax()].strip()
    nombres = catalogo_completo.loc[catalogo_completo["cluster"] == c, "nombre"]
    palabras = Counter(t for n in nombres for t in tokenize(n) if ngrama in t)
    return palabras.most_common(1)[0][0] if palabras else ngrama


nombres_cluster = {c: f"«{palabra_clave(c)}» ({(km.labels_ == c).sum()} productos)" for c in range(8)}

# Comparación con las etiquetas humanas de los 1 000 productos etiquetados.
comp = df.merge(catalogo_completo[["id", "cluster"]], on="id")
cruce = pd.crosstab(comp["cluster"].map(nombres_cluster), comp["etiqueta"])
ari = adjusted_rand_score(comp["etiqueta"], comp["cluster"])
nmi = normalized_mutual_info_score(comp["etiqueta"], comp["cluster"])
pureza = cruce.max(axis=1).sum() / cruce.to_numpy().sum()
print(f"Tamaño de los clústeres (k = 8): {catalogo_completo['cluster'].value_counts().tolist()}")
print(f"Contra las etiquetas humanas: ARI = {ari:.3f} · NMI = {nmi:.2f} · pureza = {pureza:.2f} "
      f"(predecir siempre la clase mayor: {df['etiqueta'].value_counts(normalize=True).max():.2f})")

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), layout="constrained", gridspec_kw={"width_ratios": [1, 2.2]})
axes[0].plot(list(silhouette), list(silhouette.values()), marker="o")
axes[0].set_ylim(0, 0.2)
axes[0].set_xlabel("k (número de clústeres)")
axes[0].set_ylabel("silhouette")
axes[0].set_title("Silhouette de k-means: casi 0 para todo k")
im = axes[1].imshow(cruce.to_numpy(), cmap="Blues", aspect="auto")
axes[1].set_xticks(range(len(cruce.columns)), cruce.columns, rotation=30, ha="right")
axes[1].set_yticks(range(len(cruce.index)), cruce.index)
for (r, c), v in np.ndenumerate(cruce.to_numpy()):
    if v:
        axes[1].text(c, r, v, ha="center", va="center", fontsize=8,
                     color="white" if v > cruce.to_numpy().max() / 2 else "black")
axes[1].set_xlabel("etiqueta humana")
axes[1].set_ylabel("clúster: palabra que lo define (tamaño en el catálogo)")
axes[1].set_title(f"Clústeres k = 8 vs. etiquetas humanas (ARI = {ari:.3f})")
fig.savefig(FIG / "clustering.png", bbox_inches="tight")

# %% [markdown]
# **Lectura: el no supervisado no encuentra las categorías.**
# - **No hay grupos naturales.** El silhouette queda entre 0,02 y 0,04 para todo k de 6 a
#   20: los clústeres están casi totalmente solapados.
# - **Un clúster se lleva casi todo.** Con k = 8, un solo clúster junta 7 698 productos
#   (71 %), definido por conectores y unidades («de», «ml»): un cajón de sastre.
# - **Los demás agrupan por marca o por una palabra, no por tipo de producto.** Hay un
#   clúster ALFAPARF y otro de la marca Salon In (SALONIN) que mezclan tintes, shampoo y tratamientos (Coloración
#   y Cuidado capilar juntos); y «Uñas» queda partida en tres clústeres (esmalte, nails,
#   acrílico).
# - **Contra las etiquetas humanas, coincidencia de azar:** ARI ≈ 0,008 (0 = azar), NMI ≈
#   0,30 y pureza 0,43, apenas sobre el 0,32 que se obtiene diciendo siempre «Uñas». El
#   modelo supervisado de la sección 8 llega a accuracy 0,825 y F1-macro 0,75 en un test
#   que nunca vio.
#
# **Conclusión metodológica.** Las categorías que la tienda necesita están definidas por
# el **negocio** (para qué sirve el producto), y esa información no está en la estructura
# de similitud de los nombres: el algoritmo encuentra marcas y palabras frecuentes. Por
# eso el problema se plantea como **clasificación supervisada**: se le enseña al modelo
# la definición con 1 000 ejemplos etiquetados. El clustering se usó solo como exploración
# para entender el catálogo y definir la taxonomía.

# %% [markdown]
# ## 4. Feature extraction
#
# **Técnica 1 — TF-IDF de n-gramas de caracteres** [6] (opción 1 de la rúbrica). Se usan
# n-gramas de *caracteres* (`char_wb`) y no de palabras porque los nombres están llenos
# de abreviaturas y errores de tipeo: «ESM», «ESMALTE», «ESAMALTE» y «ESMALETE» comparten
# los fragmentos «esm» y «sma», mientras que como palabras serían cuatro features sin
# relación. `min_df=2` descarta n-gramas que aparecen en un solo nombre (ruido).
#
# **Técnica 2 — Word2Vec preentrenado** (opción 2 de la rúbrica: transfer learning),
# sugerencia del docente. Vectores de 300 dimensiones del *Spanish Billion Words Corpus*
# [1], entrenados con Word2Vec [2]. El vector de un producto es el **promedio** de los
# vectores de sus palabras. Ventaja esperada: semejanza semántica («mascarilla» cerca de
# «tratamiento») aunque no compartan letras. Como el modelo completo pesa 1,1 GB, se
# recortó al vocabulario del catálogo con gensim [13] (`scripts/extraer_vocab_w2v.py`, con verificación
# SHA-256 del archivo descargado).

# %% [markdown]
# **Transformador Word2Vec.** sklearn no trae un transformador «promedio de Word2Vec», así
# que se escribe uno compatible con `Pipeline`: hereda de `BaseEstimator` (para que
# `GridSearchCV` lo pueda clonar) y de `TransformerMixin` (que aporta `fit_transform`).
# `fit` solo carga los vectores ya entrenados —eso es el *transfer learning*: no aprende
# nada de los 1 000 nombres— y `transform` promedia los vectores de las palabras de cada
# nombre. Un nombre sin ninguna palabra conocida queda en el vector cero.

# %%
# Prompt (IA): «Escribe un transformador de sklearn que convierta cada nombre en el promedio
# de los vectores Word2Vec de sus palabras, cargando los vectores desde un .npz, y que deje
# en cero los nombres sin ninguna palabra conocida.»
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


MeanEmbeddingVectorizer(W2V).fit_transform(["ESMALTE ROJO", "ALFAPARF 8.1"])[:, :4]

# %%
# Prompt (IA): «Muestra los vecinos más cercanos de algunas palabras en el espacio Word2Vec.»
# Vecinos más cercanos en el espacio Word2Vec: muestran qué "sabe" el modelo preentrenado.
data = np.load(W2V)
vocab, V = data["words"].tolist(), data["vectors"]
Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
pd.DataFrame({
    q: [vocab[i] for i in (Vn @ Vn[vocab.index(q)]).argsort()[::-1][1:7]]
    for q in ["esmalte", "tinte", "pestañas", "labial", "shampoo"]
})

# %% [markdown]
# Los vecinos tienen sentido en general («pestañas» → rizador, cejas, postizas), pero
# también muestran ruido del corpus general («shampoo» → pedicure, manicure).

# %%
# Prompt (IA): «Mide qué porcentaje de las palabras del dataset no tiene vector en el Word2Vec preentrenado.»
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


words = set(np.load(W2V)["words"].tolist())
cov = vocabulary_coverage(df["nombre"].tolist(), words)
cov

# %% [markdown]
# **Cobertura de Word2Vec.** El 11,6 % de las palabras del dataset no tiene vector
# preentrenado, y solo 4 productos (0,4 %) quedan sin ninguna. Pero las palabras sin
# vector **son justamente las marcas** (alfaparf, rodher, amybeauty, masglo, thyms…): el
# modelo preentrenado se construyó con texto general en español y nunca las vio. Esto
# anticipa la principal debilidad de la técnica 2.

# %% [markdown]
# ## 5. Separación de los datos y feature selection
#
# **Split.** 80 % entrenamiento (800) / 20 % test (200), estratificado. Va aquí porque la
# selección de features es el primer paso que **usa las etiquetas**: desde este punto todo
# se calcula solo con el entrenamiento. El test **no se toca** hasta la sección 8: toda
# decisión (hiperparámetros, comparación) se toma con validación cruzada sobre el 80 %.

# %%
# Prompt (IA): «Separa 80/20 estratificado con semilla fija; el test no se toca hasta el final.»
X_train, X_test, y_train, y_test = train_test_split(
    df["nombre"], df["etiqueta"], test_size=0.2, stratify=df["etiqueta"], random_state=SEED
)
len(X_train), len(X_test)

# %% [markdown]
# **Feature selection con χ² (tipo filtro).** χ² puntúa cada n-grama según su dependencia
# con la clase, sin entrenar un modelo, y se quedan los *k* mejores. En el modelo, χ² va
# **dentro del pipeline** y *k* entra a la grilla de hiperparámetros (sección 7), así que la
# selección se ajusta en cada fold y se optimiza con la misma validación cruzada. La celda
# siguiente lo muestra sobre el entrenamiento completo, solo para ver qué puntúa χ² y cómo
# se reparten los puntajes. **No se aplica a Word2Vec**: χ² exige features no negativas y
# los promedios de Word2Vec toman valores negativos; además, la representación
# preentrenada ya es densa y de baja dimensión (300), como admite la rúbrica para
# transfer learning.

# %%
# Prompt (IA): «Muestra la selección χ² sobre el entrenamiento: los n-gramas con mayor puntaje y la
# clase con la que más se asocian, y un gráfico de los puntajes ordenados con los k de la grilla.»
tfidf_demo = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
X_demo = tfidf_demo.fit_transform(X_train)
chi2_scores, _ = chi2(X_demo, y_train)
ngramas = tfidf_demo.get_feature_names_out()
# Clase asociada a cada n-grama: la de mayor TF-IDF promedio entre sus productos.
clases_demo = np.sort(y_train.unique())
Y = label_binarize(y_train, classes=clases_demo)
medias = np.asarray((X_demo.T @ Y).T) / Y.sum(axis=0)[:, None]
orden = np.argsort(chi2_scores)[::-1]
acumulado = np.cumsum(chi2_scores[orden]) / chi2_scores.sum()
print(f"{X_demo.shape[1]} n-gramas en el entrenamiento")
for k in [250, 1000, 3500]:
    print(f"  los {k} mejores concentran el {100 * acumulado[k - 1]:.0f} % del puntaje χ² total")

fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
ax.plot(np.arange(1, len(orden) + 1), chi2_scores[orden])
ax.set_xscale("log")
ax.set_yscale("log")
for k in [250, 500, 1000, 2000, 3500, len(orden)]:
    ax.axvline(k, ls="--", c="grey", lw=0.8)
    ax.text(k, ax.get_ylim()[1], f" k={k}" if k < len(orden) else " all", fontsize=8, va="top")
ax.set_xlabel("posición del n-grama ordenado por puntaje χ² (escala log)")
ax.set_ylabel("puntaje χ² (escala log)")
ax.set_title("Puntaje χ² de cada n-grama, con los k de la grilla")
fig.savefig(FIG / "chi2.png", bbox_inches="tight")
plt.show()

pd.DataFrame({
    "n-grama": [repr(ngramas[i]) for i in orden[:15]],
    "χ²": chi2_scores[orden[:15]].round(1),
    "clase asociada": [clases_demo[medias[:, i].argmax()] for i in orden[:15]],
})

# %% [markdown]
# **Lectura de χ².** Los n-gramas con mayor puntaje son fragmentos de palabras que
# definen una clase («tint», « pes», «taña»): χ² encuentra lo esperado. Pasados los ~40
# más distintivos, el puntaje cae de forma **gradual**, sin un codo claro en el rango de la
# grilla: la información está repartida entre miles de n-gramas (los 250 mejores
# concentran el 18 % del puntaje total; los 1 000 mejores, el 42 %). Por eso no se fija *k* a ojo: se deja como hiperparámetro, con la opción `all`
# (sin selección). En la sección 7 la validación cruzada elige `all`: con este volumen de
# features, la regularización L2 de la regresión logística ya controla las features poco
# informativas y descartarlas no mejora el F1.

# %% [markdown]
# ## 6. Pipelines
#
# Cada técnica es un `Pipeline` de sklearn [3], [18]: la extracción de features se ajusta **dentro
# de cada fold** de la validación cruzada. Si TF-IDF o χ² se ajustaran con todos los
# datos antes de la CV, verían los folds de validación (data leakage).

# %%
# Prompt (IA): «Define los dos pipelines de sklearn: TF-IDF char_wb → SelectKBest(chi2)
# → LogisticRegression, y MeanEmbeddingVectorizer → StandardScaler → SVC RBF con
# probability=True para poder calcular AUC.»
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
display(pipe_tfidf, pipe_w2v)

# %% [markdown]
# ## 7. Optimización de hiperparámetros
#
# Búsqueda en grilla con **repeated stratified k-fold (5 folds × 3 repeticiones)** [15] sobre
# el 80 % de entrenamiento, métrica F1-macro. Se usan 5 folds (y no 10) para el *tuning*
# porque cada combinación de la grilla se entrena 15 veces; las 10 × 10 se reservan para
# la comparación final.

# %%
# Prompt (IA): «Optimiza ambos pipelines con GridSearchCV y RepeatedStratifiedKFold
# 5×3, scoring f1_macro; para LogReg grilla de C en escala log, k de chi2 y ngram_range.
# Muestra en tablas la mejor combinación de cada técnica y las 5 mejores de cada grilla.»
cv_tune = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
grid_tfidf = {
    "tfidf__ngram_range": [(2, 4), (2, 5)],
    "chi2__k": [250, 500, 1000, 2000, 3500, "all"],  # "all" = sin selección, para comparar
    "clf__C": np.logspace(-2, 4, 10),  # C = 1/λ; hasta 10⁴ para que el óptimo no caiga en el borde
}
grid_w2v = {"clf__C": np.logspace(-2, 3, 8), "clf__gamma": ["scale", 1e-3, 1e-2]}
# return_train_score=True: guarda también el F1 de entrenamiento de cada fold, para graficar
# entrenamiento contra validación y ver dónde empieza el sobreajuste.
search_tfidf = GridSearchCV(
    pipe_tfidf, grid_tfidf, cv=cv_tune, scoring="f1_macro", n_jobs=-1, return_train_score=True
).fit(X_train, y_train)
search_w2v = GridSearchCV(
    pipe_w2v, grid_w2v, cv=cv_tune, scoring="f1_macro", n_jobs=-1, return_train_score=True
).fit(X_train, y_train)


def resumen_busqueda(search, nombre):
    """Mejor combinación de una búsqueda: hiperparámetros y F1 de validación y de entrenamiento."""
    r, i = search.cv_results_, search.best_index_
    return {
        "técnica": nombre,
        "mejores hiperparámetros": ", ".join(f"{k.split('__')[1]}={v:.4g}" if isinstance(v, float) else f"{k.split('__')[1]}={v}"
                                             for k, v in search.best_params_.items()),
        "combinaciones": len(r["params"]),
        "F1-macro validación": f"{r['mean_test_score'][i]:.3f} ± {r['std_test_score'][i]:.3f}",
        "F1-macro entrenamiento": f"{r['mean_train_score'][i]:.3f}",
    }


def mejores_combinaciones(search, n=5):
    """Las n mejores combinaciones de la grilla, ordenadas por F1 de validación."""
    r = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(n)
    params = r.filter(like="param_").rename(columns=lambda c: c.split("__")[1])
    if "C" in params:
        params.insert(0, "λ = 1/C", (1 / params["C"].astype(float)).map("{:.4g}".format))
        params["C"] = params["C"].astype(float).map("{:.4g}".format)
    return params.assign(
        **{"F1 validación": r["mean_test_score"].round(4), "± std": r["std_test_score"].round(4),
           "F1 entrenamiento": r["mean_train_score"].round(4)}
    ).reset_index(drop=True)


display(pd.DataFrame([resumen_busqueda(search_tfidf, "TF-IDF + LogReg"), resumen_busqueda(search_w2v, "Word2Vec + SVM")]))
print("TF-IDF + LogReg — 5 mejores combinaciones")
display(mejores_combinaciones(search_tfidf))
print("Word2Vec + SVM — 5 mejores combinaciones")
display(mejores_combinaciones(search_w2v))

# %% [markdown]
# **Resultado de la grilla.** TF-IDF + LogReg: mejor F1-macro de validación **0,739 ± 0,045**
# con n-gramas 2–5, sin selección (`k = all`) y C ≈ 464 (λ ≈ 0,002), entre 120
# combinaciones. Word2Vec + SVM: **0,640 ± 0,037** con C ≈ 7,2 y γ = 10⁻³, entre 24. En
# ambas técnicas las 5 mejores combinaciones quedan a menos de 0,01 entre sí, bastante
# menos que la desviación entre folds (≈ 0,04): el óptimo es una **meseta**, no un punto
# frágil. El F1 de entrenamiento (1,000 y 0,977) muy por encima del de validación es la
# brecha de sobreajuste que muestran las gráficas siguientes.

# %% [markdown]
# **Funciones de gráfico de la optimización.** `plot_param_curve` toma los resultados de
# `GridSearchCV` (`cv_results_`), fija los demás hiperparámetros en su mejor valor y
# dibuja el F1 (o el error = 1 − F1) de entrenamiento y de validación, con la media ± la
# desviación estándar de los folds, contra el hiperparámetro elegido. `plot_learning_curve`
# usa `sklearn.model_selection.learning_curve` [16] para entrenar con tamaños crecientes del
# entrenamiento y medir ambos errores.

# %%
# Prompt (IA): «Escribe una función que grafique el score o el error de entrenamiento y de
# validación (media ± std) contra un hiperparámetro a partir de un GridSearchCV, fijando los
# demás en su mejor valor, con valores no numéricos como líneas horizontales; y otra que
# grafique la curva de aprendizaje con learning_curve.»
def plot_param_curve(search, param: str, ax, log: bool = True, label: str | None = None,
                     as_error: bool = False, x_transform=None, x_label: str | None = None):
    """Score (o error) de entrenamiento y de validación (media ± std) vs. un hiperparámetro.

    Los demás hiperparámetros se fijan en su mejor valor. La curva de entrenamiento solo
    aparece si el GridSearchCV se corrió con return_train_score=True. Valores no
    numéricos de la grilla (p. ej. k="all") se dibujan como líneas horizontales.

    as_error=True grafica 1 − score (con F1, el error); x_transform permite graficar, por
    ejemplo, λ = 1/C en vez de C, para leer la curva como en la teoría (Loss vs. λ).
    """
    res = pd.DataFrame(search.cv_results_)
    key = f"param_{param}"
    for other, best in search.best_params_.items():
        if other != param:
            res = res[res[f"param_{other}"].astype(str) == str(best)]
    tx = x_transform or (lambda v: v)
    ty = (lambda v: 1 - v) if as_error else (lambda v: v)
    numeric = pd.to_numeric(res[key], errors="coerce")
    num = res[numeric.notna()].assign(_x=tx(numeric[numeric.notna()].astype(float))).sort_values("_x")
    series = [("validación", "test", "tab:orange")]
    if "mean_train_score" in res:
        series.insert(0, ("entrenamiento", "train", "tab:blue"))
    for nombre, split, color in series:
        ax.errorbar(num["_x"], ty(num[f"mean_{split}_score"]), yerr=num[f"std_{split}_score"],
                    marker="o", capsize=3, color=color, label=nombre)
        for _, row in res[numeric.isna()].iterrows():
            ax.axhline(ty(row[f"mean_{split}_score"]), ls=":", color=color, label=f"{nombre} ({param} = {row[key]})")
    if log:
        ax.set_xscale("log")
    best = search.best_params_[param]
    if pd.notna(pd.to_numeric(best, errors="coerce")):
        ax.axvline(float(tx(float(best))), ls="--", c="grey", label="óptimo")
    ax.set_xlabel(x_label or param)
    metric = str(search.scoring)
    ax.set_ylabel(f"error = 1 − {metric}" if as_error else metric)
    ax.legend(fontsize=8)


def plot_learning_curve(estimator, X, y, cv, ax, scoring="f1_macro", title="", as_error: bool = False):
    """Curva de aprendizaje: score (o error = 1 − score) de entrenamiento y validación vs. m."""
    sizes, train, val = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 6)
    )
    if as_error:
        train, val = 1 - train, 1 - val
    for scores, name in ((train, "entrenamiento"), (val, "validación")):
        m, s = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(sizes, m, marker="o", label=name)
        ax.fill_between(sizes, m - s, m + s, alpha=0.2)
    ax.set_xlabel("m (ejemplos de entrenamiento)")
    ax.set_ylabel(f"error = 1 − {scoring}" if as_error else scoring)
    ax.set_title(title)
    ax.legend()
    return sizes, train.mean(axis=1), val.mean(axis=1)


# Prompt (IA): «Grafica la optimización como en la teoría del curso: error = 1 − F1-macro
# de entrenamiento y validación contra λ = 1/C, marcando sobreajuste, subajuste y óptimo.»
def marcar_regiones(ax):
    ax.text(0.02, 0.97, "sobreajuste\n(varianza alta)", transform=ax.transAxes, va="top", fontsize=9, color="tab:red")
    ax.text(0.98, 0.5, "subajuste\n(sesgo alto)", transform=ax.transAxes, va="center", ha="right", fontsize=9, color="tab:red")


fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")
plot_param_curve(search_tfidf, "clf__C", axes[0], as_error=True, x_transform=lambda c: 1 / c, x_label="λ = 1/C")
axes[0].set_title("TF-IDF + LogReg: regularización λ")
marcar_regiones(axes[0])
plot_param_curve(search_w2v, "clf__C", axes[1], as_error=True, x_transform=lambda c: 1 / c, x_label="λ = 1/C")
axes[1].set_title("Word2Vec + SVM: regularización λ")
marcar_regiones(axes[1])
plot_param_curve(search_tfidf, "chi2__k", axes[2], log=False, as_error=True, x_label="k (n-gramas conservados por χ²)")
axes[2].set_title("TF-IDF: selección de features k")
fig.savefig(FIG / "optimizacion.png", bbox_inches="tight")

# %% [markdown]
# **Lectura de la optimización.** Se grafica el **error = 1 − F1-macro** de
# **entrenamiento** (azul) y de **validación** (naranja) contra la fuerza de la
# regularización **λ = 1/C**, igual que la curva *Loss vs. λ* de la teoría. Cada punto es
# el promedio de los 15 folds de la CV 5 × 3 y las barras son ± una desviación estándar.
# El hiperparámetro se elige donde el **error de validación es mínimo**, nunca mirando el
# de entrenamiento.
# - **λ (LogReg).** A la derecha (λ ≥ 20, regularización fuerte) ambos errores son altos
#   y casi iguales (≈ 0,94: el modelo predice una sola clase): **subajuste**, sesgo alto.
#   Al bajar λ, ambos bajan. A la izquierda (λ ≤ 0,2) el error de entrenamiento llega a 0
#   y el de validación se queda en ≈ 0,26 y vuelve a subir levemente en λ = 10⁻⁴: la
#   brecha entre ambos es **sobreajuste**, varianza alta. El óptimo de validación es
#   **λ ≈ 0,002 (C ≈ 464)**, con error ≈ 0,261 (F1 ≈ 0,739), en el interior de la grilla.
# - **λ (SVM).** Mismo patrón. Óptimo **λ ≈ 0,14 (C ≈ 7,2) con γ = 10⁻³**, error ≈ 0,360
#   (F1 ≈ 0,640); con menos regularización el error de validación queda plano.
# - **k de χ².** Con pocos n-gramas (k = 250) el modelo subajusta: error de
#   entrenamiento 0,26 y de validación 0,46. Al crecer k ambos bajan; el de entrenamiento
#   llega a 0 en k ≈ 2000, pero el de validación **sigue bajando** hasta el mejor
#   resultado, **sin selección** (líneas punteadas, `all`: error 0,261). La diferencia con
#   k = 3500 (0,009) es menor que la desviación entre folds (≈ 0,04): **χ² no aporta**.
#   La regresión logística regularizada ya asigna peso casi cero a los n-gramas
#   irrelevantes. Es un resultado, no un error: se comparó la selección contra no
#   seleccionar y los datos dicen que conviene conservar todos los n-gramas.

# %% [markdown]
# **¿Dónde cae el óptimo dentro de la grilla?** El valor absoluto de C no se compara entre
# técnicas: LogReg y SVM minimizan pérdidas distintas (logística vs. *hinge*) sobre
# representaciones de escala distinta (TF-IDF de norma 1 repartida en miles de n-gramas vs.
# 300 dimensiones estandarizadas), así que «C = 464» y «C = 7,2» no están en la misma unidad.
# Lo que sí se compara es **la posición del óptimo en su propio recorrido**: si queda en el
# interior, dónde empieza a saturar el F1 de entrenamiento y qué brecha deja. Las tablas
# recorren C con los demás hiperparámetros en su mejor valor; los mapas de calor muestran la
# grilla completa (TF-IDF con n-gramas 2–5).

# %%
# Prompt (IA): «Muestra el recorrido completo de C de cada técnica (los demás hiperparámetros
# en su mejor valor) con F1 de validación y de entrenamiento, la brecha y la posición en la
# grilla; y mapas de calor del F1 de validación de C × k (LogReg) y C × γ (SVM) con el
# óptimo marcado.»
def recorrido(search, param):
    """F1 de validación y de entrenamiento para cada valor de param, con los demás en su mejor valor."""
    r = pd.DataFrame(search.cv_results_)
    for otro, mejor in search.best_params_.items():
        if otro != param:
            r = r[r[f"param_{otro}"].astype(str) == str(mejor)]
    r = r.assign(valor=r[f"param_{param}"].astype(float)).sort_values("valor")
    return pd.DataFrame({
        "posición": [f"{i} de {len(r)}" for i in range(1, len(r) + 1)],
        "C": r["valor"].map("{:.4g}".format).values,
        "λ = 1/C": (1 / r["valor"]).map("{:.4g}".format).values,
        "F1 validación": r["mean_test_score"].round(3).values,
        "± std": r["std_test_score"].round(3).values,
        "F1 entrenamiento": r["mean_train_score"].round(3).values,
        "brecha": (r["mean_train_score"] - r["mean_test_score"]).round(3).values,
        "": np.where(np.isclose(r["valor"], search.best_params_[param]), "← óptimo", ""),
    })


def mapa_grilla(search, filas, columnas, ax, fijos=None, orden=None, etiquetas=None, titulo=""):
    """Mapa de calor del F1 de validación sobre dos hiperparámetros, con el óptimo recuadrado.

    orden fija el orden de las columnas (p. ej. γ = scale entre 10⁻³ y 10⁻², donde cae su valor).
    """
    r = pd.DataFrame(search.cv_results_)
    for p, v in (fijos or {}).items():
        r = r[r[f"param_{p}"].astype(str) == str(v)]
    tabla = r.pivot_table(index=r[f"param_{filas}"].astype(float), columns=r[f"param_{columnas}"].astype(str),
                          values="mean_test_score", sort=False)
    tabla = tabla.sort_index()[orden or sorted(tabla.columns, key=lambda c: (c == "all", pd.to_numeric(c, errors="coerce")))]
    im = ax.imshow(tabla.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(tabla.shape[1]), [(etiquetas or {}).get(c, c) for c in tabla.columns])
    ax.set_yticks(range(tabla.shape[0]), [f"{c:.3g}" for c in tabla.index])
    for (i, j), v in np.ndenumerate(tabla.values):
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color="white" if v < tabla.values.mean() else "black")
    i = list(tabla.index).index(min(tabla.index, key=lambda c: abs(c - search.best_params_[filas])))
    j = list(tabla.columns).index(str(search.best_params_[columnas]))
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="red", lw=2.5))
    ax.set_xlabel(columnas.split("__")[1]); ax.set_ylabel("C"); ax.set_title(titulo)
    plt.colorbar(im, ax=ax, label="F1-macro validación")


print("TF-IDF + LogReg — recorrido de C (k = all, n-gramas 2–5)")
display(recorrido(search_tfidf, "clf__C"))
print("Word2Vec + SVM — recorrido de C (γ = 10⁻³)")
display(recorrido(search_w2v, "clf__C"))
Z = search_w2v.best_estimator_[:2].transform(X_train)
gamma_scale = 1 / (Z.shape[1] * Z.var())
print(f"γ = 'scale' equivale a 1 / (n_features · Var(X)) = 1 / ({Z.shape[1]} · {Z.var():.3f}) = {gamma_scale:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), layout="constrained")
mapa_grilla(search_tfidf, "clf__C", "chi2__k", axes[0], fijos={"tfidf__ngram_range": (2, 5)},
            titulo="TF-IDF + LogReg: F1 validación, C × k (n-gramas 2–5)")
mapa_grilla(search_w2v, "clf__C", "clf__gamma", axes[1], orden=["0.001", "scale", "0.01"],
            etiquetas={"scale": f"scale (≈ {gamma_scale:.4f})"}, titulo="Word2Vec + SVM: F1 validación, C × γ")
fig.savefig(FIG / "grilla_mapas.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Lectura: dónde cae cada óptimo.**
# - **Los dos extremos de C se comportan igual en ambas técnicas.** Con C ≤ 0,05
#   (regularización muy fuerte) el F1 de entrenamiento y el de validación valen 0,06: el
#   modelo predice una sola clase (**subajuste**). Con C grande el F1 de entrenamiento llega
#   a ≈ 1 y la validación deja de mejorar: ya no queda regularización que quitar.
# - **LogReg, C ≈ 464 (posición 8 de 10).** El entrenamiento ya está en 1,000 desde
#   C ≈ 21,5, y entre 21,5 y 464 la validación es una **meseta** (0,737 → 0,739: una
#   diferencia de 0,002 frente a una desviación de 0,045). Pasado 464 la validación **baja**
#   (0,726 y 0,716), así que el óptimo está en el interior. Es un C alto, pero no es un
#   sobreajuste mayor: cualquier C de la meseta da el mismo F1, y la brecha
#   (≈ 0,26) es la misma que con C = 21,5.
# - **SVM, C ≈ 7,2 (posición 5 de 8).** Es el último valor antes de que el entrenamiento
#   sature (0,977; desde C ≈ 37 pasa a 0,996). Con más C la validación queda plana
#   (0,633–0,634), sin mejora, y la brecha crece de 0,34 a 0,36.
# - **Por qué no comparar 464 con 7,2.** Las pérdidas y la escala de los datos son distintas,
#   así que C no está en la misma unidad. Leídos en su propio recorrido, los dos óptimos caen
#   en el mismo lugar: donde el F1 de entrenamiento ya se acerca a 1 y la validación deja
#   de subir. De hecho, la brecha del SVM (0,34) es **mayor** que la de LogReg (0,26).
# - **γ del SVM.** γ = 10⁻² cae a ≈ 0,45 con cualquier C: el kernel es demasiado estrecho y
#   la frontera se ajusta punto a punto. `scale` (≈ 0,0033, el valor por defecto) da 0,632 y
#   10⁻³ da 0,640: un γ menor (frontera más suave) conviene. 10⁻³ es el borde de la grilla.
#   Optuna buscó γ en continuo entre 10⁻⁴ y 10⁻¹ (§7b) y encontró su mejor valor en
#   γ ≈ 0,0016, junto a ese borde, así que el óptimo no estaba más abajo.
# - **k de χ² (mapa izquierdo).** Para cada C razonable (≥ 4,6) la columna `all` es la mejor
#   o empata con 3500: lo mismo que mostró la curva de k.

# %%
# Prompt (IA): «Grafica las curvas de aprendizaje como error contra m (tamaño del
# entrenamiento) para ambas técnicas, marcando la brecha entre entrenamiento y validación.»
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
for ax, (nombre, search) in zip(axes, [("TF-IDF + LogReg", search_tfidf), ("Word2Vec + SVM", search_w2v)]):
    m, err_train, err_val = plot_learning_curve(
        search.best_estimator_, X_train, y_train, cv_tune, ax, title=nombre, as_error=True
    )
    ax.annotate("", xy=(m[-1], err_train[-1]), xytext=(m[-1], err_val[-1]),
                arrowprops=dict(arrowstyle="<->", color="black"))
    ax.text(m[-1] * 0.97, (err_train[-1] + err_val[-1]) / 2, "brecha\n(varianza)", ha="right", va="center", fontsize=9)
fig.savefig(FIG / "curvas_aprendizaje.png", bbox_inches="tight")

# %% [markdown]
# **Diagnóstico sesgo vs. varianza** (curvas con `learning_curve` [16]). Igual que la curva *high variance* de la teoría: en
# ambas técnicas el error de entrenamiento es ≈ 0 y el de validación queda muy por
# encima (≈ 0,26 y ≈ 0,36), con una **brecha** grande: **alta varianza**. El error de
# validación **sigue bajando** al llegar a 640 ejemplos y no se ha estabilizado: **más
# datos etiquetados mejorarían el modelo**. Es la acción con mejor retorno (más que
# cambiar de algoritmo), y es barata gracias al etiquetado asistido. Word2Vec + SVM
# además tiene un piso de error más alto: su problema no es solo de varianza sino de
# representación (no conoce las marcas).

# %% [markdown]
# ### 7b. Optimización con Optuna (extra)
#
# La búsqueda en grilla prueba **todas** las combinaciones de una lista fija de valores.
# **Optuna** [19] hace optimización bayesiana con TPE (*Tree-structured Parzen
# Estimator*) [20]: con los intentos ya evaluados modela qué zonas del espacio dan
# buenos resultados y muestrea más ahí. Eso permite buscar en **rangos continuos** (λ
# puede tomar cualquier valor entre los extremos, no solo los de la grilla) con menos
# evaluaciones.
#
# Para que la comparación con la grilla sea justa se usa **exactamente lo mismo**: el
# mismo 80 % de entrenamiento, la misma CV repeated stratified 5 × 3, la misma métrica
# (F1-macro) y semilla fija en el muestreador. El test sigue sin tocarse. Se dan **40
# intentos** a TF-IDF (la grilla evaluó 120 combinaciones) y **20** al SVM (la grilla, 24).

# %%
# Prompt (IA): «Optimiza las dos técnicas con Optuna (TPE con semilla fija) usando la misma
# CV 5×3 y F1-macro que la grilla, buscando λ en escala log continua; compara con la grilla.»
import optuna
import optuna.visualization.matplotlib as ov

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def objetivo_tfidf(trial):
    lam = trial.suggest_float("lambda", 1e-5, 1e2, log=True)
    n_max = trial.suggest_categorical("ngram_max", [4, 5])
    k = trial.suggest_categorical("k", [250, 500, 1000, 2000, 3500, "all"])
    pipe = clone(pipe_tfidf).set_params(tfidf__ngram_range=(2, n_max), chi2__k=k, clf__C=1 / lam)
    return cross_val_score(pipe, X_train, y_train, cv=cv_tune, scoring="f1_macro", n_jobs=-1).mean()


def objetivo_w2v(trial):
    lam = trial.suggest_float("lambda", 1e-3, 1e2, log=True)
    gamma = trial.suggest_float("gamma", 1e-4, 1e-1, log=True)
    pipe = clone(pipe_w2v).set_params(clf__C=1 / lam, clf__gamma=gamma)
    return cross_val_score(pipe, X_train, y_train, cv=cv_tune, scoring="f1_macro", n_jobs=-1).mean()


estudio_tfidf = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
estudio_tfidf.optimize(objetivo_tfidf, n_trials=40)
estudio_w2v = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
estudio_w2v.optimize(objetivo_w2v, n_trials=20)

bt, bw = search_tfidf.best_params_, search_w2v.best_params_
comparacion_optuna = pd.DataFrame([
    {"técnica": "TF-IDF + LogReg", "método": "grilla", "evaluaciones": len(search_tfidf.cv_results_["params"]),
     "F1-macro CV": search_tfidf.best_score_, "λ": 1 / bt["clf__C"], "n-gramas": f"2–{bt['tfidf__ngram_range'][1]}",
     "k": bt["chi2__k"], "γ": None},
    {"técnica": "TF-IDF + LogReg", "método": "Optuna (TPE)", "evaluaciones": len(estudio_tfidf.trials),
     "F1-macro CV": estudio_tfidf.best_value, "λ": estudio_tfidf.best_params["lambda"],
     "n-gramas": f"2–{estudio_tfidf.best_params['ngram_max']}", "k": estudio_tfidf.best_params["k"], "γ": None},
    {"técnica": "Word2Vec + SVM", "método": "grilla", "evaluaciones": len(search_w2v.cv_results_["params"]),
     "F1-macro CV": search_w2v.best_score_, "λ": 1 / bw["clf__C"], "n-gramas": None, "k": None, "γ": bw["clf__gamma"]},
    {"técnica": "Word2Vec + SVM", "método": "Optuna (TPE)", "evaluaciones": len(estudio_w2v.trials),
     "F1-macro CV": estudio_w2v.best_value, "λ": estudio_w2v.best_params["lambda"], "n-gramas": None, "k": None,
     "γ": estudio_w2v.best_params["gamma"]},
])
comparacion_optuna.style.format({"F1-macro CV": "{:.4f}", "λ": "{:.4g}", "γ": "{:.4g}"}, na_rep="—")

# %%
# Prompt (IA): «Grafica el historial de optimización de ambos estudios, la importancia de
# los hiperparámetros de TF-IDF y el contorno λ vs. γ del SVM con la API matplotlib de Optuna.»
def guardar(ax, titulo, archivo, ylabel=None, xlabel=None):
    """Reemplaza título y etiquetas (Optuna los pone en inglés) y guarda la figura."""
    ax.set_title("", loc="left")  # Optuna pone su título en inglés a la izquierda
    ax.set_title("", loc="right")
    ax.set_title(titulo)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.figure.savefig(FIG / archivo, bbox_inches="tight")
    plt.show()


# Los gráficos de Optuna activan el estilo «ggplot» de matplotlib para TODO el notebook, y su
# paleta tiene solo 7 colores: con 8 clases, dos categorías saldrían del mismo color en los
# gráficos siguientes. style.context deja ese estilo solo dentro de este bloque.
with plt.style.context("ggplot"):
    for estudio, nombre, archivo in [(estudio_tfidf, "TF-IDF + LogReg", "optuna_historial_tfidf.png"),
                                     (estudio_w2v, "Word2Vec + SVM", "optuna_historial_w2v.png")]:
        ax = ov.plot_optimization_history(estudio)
        ax.legend(["F1-macro de cada intento", "mejor hasta ese intento"], fontsize=8)
        guardar(ax, f"Optuna · {nombre}: historial", archivo, ylabel="F1-macro (CV 5 × 3)", xlabel="intento")
    # Evaluador fijado explícitamente (es el default de Optuna 5) para que no cambie con la versión.
    ax = ov.plot_param_importances(estudio_tfidf, evaluator=optuna.importance.PedAnovaImportanceEvaluator())
    ax.get_legend().remove()
    guardar(ax, "Optuna · TF-IDF + LogReg: importancia de hiperparámetros", "optuna_importancia_tfidf.png",
            xlabel="importancia (PED-ANOVA)", ylabel="hiperparámetro")
    ax = ov.plot_contour(estudio_w2v, params=["gamma", "lambda"])
    guardar(ax, "Optuna · Word2Vec + SVM: F1-macro según λ y γ", "optuna_contorno_w2v.png", ylabel="λ = 1/C", xlabel="γ")

# %% [markdown]
# **Lectura de Optuna.**
# - **TF-IDF + LogReg: mismo resultado con un tercio de las evaluaciones.** Optuna llega a
#   F1-macro 0,7386 con 40 intentos; la grilla, a 0,7387 con 120 combinaciones. Ambos
#   eligen n-gramas 2–5 y sin selección (`k = all`); Optuna elige λ ≈ 0,0095 y la grilla
#   λ ≈ 0,0022, dos puntos de la misma **meseta** que ya se veía en la curva de λ (con
#   λ entre 10⁻³ y 10⁻¹ el error de validación es prácticamente igual). El historial muestra
#   que el mejor valor se alcanza casi desde el inicio; los intentos malos son la
#   exploración de λ grandes (subajuste).
# - **Word2Vec + SVM: Optuna mejora levemente a la grilla.** F1-macro 0,6472 con 20
#   intentos contra 0,6404 con 24, porque busca en un rango continuo y encuentra un punto
#   **entre** los valores de la grilla (λ ≈ 0,060 y γ ≈ 0,0016; la grilla solo tenía
#   λ ≈ 0,139 y γ = 0,001 cerca de ahí). El contorno muestra la zona buena (λ < ~0,3 y
#   γ ≤ ~2 × 10⁻³) y que TPE concentra sus intentos en ella.
# - **Importancia de hiperparámetros (TF-IDF, PED-ANOVA):** λ explica el 80 % de la
#   variación del F1, el largo de los n-gramas el 11 % y k apenas el 8 %. La
#   regularización es lo que más importa, y es coherente con que χ² no aporte.
# - **Consecuencia para el resto del notebook:** las mejoras de Optuna (0,0001 y 0,007 de
#   F1) son mucho menores que la variación entre folds (≈ 0,04) y que la distancia entre
#   las dos técnicas (≈ 0,10). Por eso la comparación estadística y el modelo final usan
#   los hiperparámetros de la grilla: cambiarlos no altera ninguna conclusión. Lo que
#   Optuna aporta es **eficiencia** (mismo óptimo con menos evaluaciones) y la
#   confirmación de que la grilla no dejó un óptimo mejor sin explorar.

# %% [markdown]
# ## 8. Comparación estadística: Wilcoxon sobre repeated stratified 10 × 10 CV
#
# Los dos modelos ya optimizados se evalúan con **los mismos 100 folds**
# (`RepeatedStratifiedKFold`, 10 folds × 10 repeticiones, `random_state=42`). Se usa
# validación **estratificada** porque las clases están desbalanceadas: un fold sin
# ejemplos de «Herramientas» daría un F1-macro sin sentido. Como cada fold produce un par
# de scores (uno por modelo), el test es **pareado**: Wilcoxon de rangos con signo [4],
# que no asume normalidad de las diferencias (implementación de SciPy [11], [17]).
#
# - H₀: las diferencias de F1-macro entre ambos modelos son simétricas alrededor de 0
#   (ninguno es sistemáticamente mejor).
# - H₁: no lo son. α = 0,05.
#
# **Limitaciones y cómo se tratan.** (1) Los 100 scores **no son independientes**: los
# folds comparten datos, así que la varianza de las diferencias se subestima y el
# p-value de Wilcoxon sale optimista. Por eso se reporta además el **t-test corregido de
# Nadeau y Bengio** [8], que infla esa varianza por (1/J + n_test/n_train) y es el
# estándar para comparar modelos con CV repetida. (2) La comparación usa los mismos 800
# ejemplos con los que se optimizaron los hiperparámetros, y la grilla de TF-IDF (120
# combinaciones) es más grande que la de Word2Vec (24): el sesgo de selección favorece
# levemente a TF-IDF. El test de la sección siguiente, con 200 productos nunca vistos,
# controla ese sesgo.
#
# **Las tres funciones de la comparación.** `repeated_cv_scores` evalúa ambos modelos con
# **un mismo** objeto `RepeatedStratifiedKFold` de semilla fija, así los dos ven
# exactamente los mismos 100 folds (requisito del test pareado). `wilcoxon_test` aplica
# `scipy.stats.wilcoxon` sobre los 100 pares. `corrected_resampled_ttest` implementa la
# fórmula de Nadeau y Bengio: t = d̄ / √((1/J + n_test/n_train) · s²), con J − 1 grados de
# libertad.

# %%
# Prompt (IA): «Escribe una función que evalúe varios modelos con RepeatedStratifiedKFold
# usando los mismos folds, otra que aplique scipy.stats.wilcoxon a dos columnas de scores y
# otra con el t-test corregido de Nadeau y Bengio para CV repetida.»
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


def corrected_resampled_ttest(scores: pd.DataFrame, a: str, b: str, n_train: int, n_test: int, alpha: float = 0.05) -> dict:
    """t-test con la corrección de Nadeau y Bengio (2003) para CV repetida.

    Los folds comparten datos, así que la varianza de las diferencias se subestima; la
    corrección la infla por (1/J + n_test/n_train) en lugar de 1/J.
    """
    from scipy.stats import t as t_dist

    d = (scores[a] - scores[b]).to_numpy()
    J = len(d)
    var = d.var(ddof=1)
    t_stat = d.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    p = 2 * t_dist.sf(abs(t_stat), df=J - 1)
    return {
        "t": float(t_stat),
        "p_value": float(p),
        "alpha": alpha,
        "significativo": bool(p < alpha),
        "mejor": a if d.mean() >= 0 else b,
    }


# Prompt (IA): «Compara los dos mejores modelos con repeated stratified 10x10 CV usando
# los mismos folds y aplica Wilcoxon pareado con alpha 0.05; grafica la distribución.»
N_SPLITS, N_REPEATS = 10, 10  # bajar a 5, 2 si tarda > ~15 min en Colab (permitido por la rúbrica)
scores = repeated_cv_scores(
    {"TF-IDF + LogReg": search_tfidf.best_estimator_, "Word2Vec + SVM": search_w2v.best_estimator_},
    X_train, y_train, n_splits=N_SPLITS, n_repeats=N_REPEATS,
)
res = wilcoxon_test(scores, "TF-IDF + LogReg", "Word2Vec + SVM")
n_test_fold = len(X_train) // N_SPLITS
res_nb = corrected_resampled_ttest(
    scores, "TF-IDF + LogReg", "Word2Vec + SVM", n_train=len(X_train) - n_test_fold, n_test=n_test_fold
)
fig, ax = plt.subplots(figsize=(6, 4))
scores.boxplot(ax=ax)
ax.set_ylabel("F1-macro por fold")
fig.savefig(FIG / "comparacion.png", bbox_inches="tight")
print(f"F1-macro medio: TF-IDF + LogReg {res['media_a']:.3f} · Word2Vec + SVM {res['media_b']:.3f}")
print(f"Wilcoxon: estadístico {res['estadistico']:.0f}, p = {res['p_value']:.2e}")
print(f"t corregido (Nadeau–Bengio): t = {res_nb['t']:.2f}, p = {res_nb['p_value']:.2e}")

# %% [markdown]
# **Resultado.** F1-macro medio en los 100 folds: **0,759 (TF-IDF + LogReg)** contra
# **0,648 (Word2Vec + SVM)**. Wilcoxon: **p ≈ 1,2 × 10⁻¹⁷**; t corregido de
# Nadeau–Bengio: **t = 4,87, p ≈ 4,3 × 10⁻⁶**. Ambos < 0,05: se rechaza H₀ y la ventaja
# de TF-IDF + LogReg es estadísticamente significativa. El p corregido es once órdenes de
# magnitud mayor que el de Wilcoxon —eso es lo que cuesta la dependencia entre folds—,
# pero la conclusión se mantiene.
#
# ### Métricas en el conjunto de test (200 productos nunca vistos)

# %%
# Prompt (IA): «Escribe una función con todas las métricas de clasificación de la rúbrica
# (accuracy, precision, recall, F1 macro y weighted, AUC macro y micro) y otra que dibuje la
# curva ROC one-vs-rest por clase más el micro-promedio con roc_curve.»
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


def plot_roc_multiclass(y_true, y_proba, classes, ax, title=""):
    """Curva ROC one-vs-rest por clase más el micro-promedio.

    Se dibuja con roc_curve + ax.plot (no con RocCurveDisplay) porque la API de estilo
    de RocCurveDisplay cambió en sklearn 1.7 y Colab puede traer otra versión.
    """
    y_bin = label_binarize(y_true, classes=classes)
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        # tab10: un color distinto por clase (el ciclo por defecto repite colores con 8 clases).
        ax.plot(fpr, tpr, lw=1, color=plt.cm.tab10(i % 10), label=f"{c} (AUC {auc(fpr, tpr):.2f})")
    fpr, tpr, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    ax.plot(fpr, tpr, ls="--", lw=2, c="black", label=f"micro-promedio (AUC {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], c="grey", lw=0.8)
    ax.set_xlabel("tasa de falsos positivos")
    ax.set_ylabel("tasa de verdaderos positivos")
    ax.set_title(title)
    ax.legend(fontsize=7)


# Prompt (IA): «Calcula en el test todas las métricas de la rúbrica para ambos modelos.»
modelos = {"TF-IDF + LogReg": search_tfidf.best_estimator_, "Word2Vec + SVM": search_w2v.best_estimator_}
proba_por_modelo = {name: est.predict_proba(X_test) for name, est in modelos.items()}
pd.DataFrame({
    name: classification_metrics(y_test, est.predict(X_test), proba_por_modelo[name], est.classes_)
    for name, est in modelos.items()
}).round(3)

# %% [markdown]
# **Curvas ROC (one-vs-rest).** Para cada clase se trata el problema como «esta clase
# contra todas las demás» y se barre el umbral de la probabilidad predicha: cada punto es
# (tasa de falsos positivos, tasa de verdaderos positivos) para un umbral. Una curva
# pegada a la esquina superior izquierda separa bien esa clase; la diagonal gris es un
# clasificador al azar (AUC = 0,5). La línea negra punteada es el **micro-promedio**: junta
# todas las decisiones clase-contra-resto en una sola curva.

# %%
# Prompt (IA): «Grafica la curva ROC multiclase de ambos modelos en el test.»
fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
for ax, (name, est) in zip(axes, modelos.items()):
    plot_roc_multiclass(y_test, proba_por_modelo[name], est.classes_, ax, title=f"Curvas ROC en el test — {name}")
fig.savefig(FIG / "roc.png", bbox_inches="tight")

# %% [markdown]
# **Lectura de la ROC.** TF-IDF + LogReg tiene curvas pegadas a la esquina superior
# izquierda: micro-promedio AUC 0,98 y por clase entre 0,93 y 1,00. Las más bajas son
# «Herramientas y equipos de cabello» (0,93) y «Piel y cuerpo» (0,94), las mismas clases
# con peor F1. Word2Vec + SVM queda por debajo en todas (micro-promedio 0,93, por clase de
# 0,81 a 0,99), con «Piel y cuerpo» (0,81) y «Pestañas y cejas» (0,86) como las peores.
# Que el AUC sea alto mientras el F1 es más bajo indica que el modelo **ordena** bien
# (la clase correcta suele tener una probabilidad alta) aunque al elegir la clase de
# mayor probabilidad todavía se equivoque en las clases chicas.

# %%
# Prompt (IA): «Muestra el reporte por clase y la matriz de confusión del mejor modelo en el test.»
best = search_tfidf.best_estimator_
print(classification_report(y_test, best.predict(X_test), digits=2))
fig, ax = plt.subplots(figsize=(9, 8))
ConfusionMatrixDisplay.from_estimator(best, X_test, y_test, ax=ax, xticks_rotation=45, colorbar=False)
ax.set_title("Matriz de confusión — TF-IDF + LogReg (test)")
fig.savefig(FIG / "matriz_confusion.png", bbox_inches="tight")

# %% [markdown]
# **Lectura del test.** TF-IDF + LogReg: accuracy 0,825, F1-macro 0,75, F1-weighted 0,81,
# AUC macro 0,97 y micro 0,98, muy cerca de lo estimado por validación cruzada (0,74–0,76):
# el sesgo de selección de hiperparámetros no infló el resultado. La distancia entre
# F1-macro (0,75) y F1-weighted (0,81) confirma que fallan las clases chicas:
# - **Muy bien:** Coloración capilar (F1 0,96), Maquillaje (0,96), Uñas (0,89): tienen
#   vocabulario y marcas propias.
# - **Mal:** Herramientas y equipos de cabello (F1 0,50) y Piel y cuerpo (0,50). Las
#   herramientas se confunden con Uñas (pinzas, limas, lámparas) y con Cuidado capilar;
#   Piel y cuerpo se confunde con Cuidado capilar porque comparten palabras («mascarilla»,
#   «crema», «aceite») que significan cosas distintas según sea para la cara o el cabello.

# %% [markdown]
# ## 9. Visualización con t-SNE (color = categoría)

# %%
# Prompt (IA): «Escribe una función que proyecte una matriz de features a 2D con t-SNE
# (semilla fija, inicialización PCA), la dibuje coloreada por etiqueta y devuelva las coordenadas.»
def plot_tsne(X, labels, ax, title="", seed=SEED):
    """t-SNE 2D coloreado por etiqueta; devuelve las coordenadas para poder anotar puntos."""
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
    return emb


# Prompt (IA): «Visualiza con t-SNE los dos espacios de features coloreados por la
# etiqueta y anota en el de TF-IDF el nombre de algunos productos: uno típico de varias
# islas y uno de la zona mezclada.»
fig, axes = plt.subplots(1, 2, figsize=(16, 7), layout="constrained")
X_tfidf = search_tfidf.best_estimator_[:-1].transform(df["nombre"])
X_w2v = search_w2v.best_estimator_[:-1].transform(df["nombre"])
emb_tfidf = plot_tsne(X_tfidf, df["etiqueta"], axes[0], title="Espacio TF-IDF (+χ²)")
plot_tsne(X_w2v, df["etiqueta"], axes[1], title="Espacio Word2Vec")

# Producto "típico" de una clase: el que tiene más vecinos de su misma clase en el mapa.
# Producto "ambiguo": el que tiene menos vecinos de su clase.
vecinos = NearestNeighbors(n_neighbors=11).fit(emb_tfidf).kneighbors(emb_tfidf, return_distance=False)[:, 1:]
etiquetas = df["etiqueta"].to_numpy()
pureza = (etiquetas[vecinos] == etiquetas[:, None]).mean(axis=1)
elegidos = [
    int(np.flatnonzero(etiquetas == c)[pureza[etiquetas == c].argmax()])
    for c in ["Coloración capilar", "Uñas", "Maquillaje", "Pestañas y cejas"]
]
elegidos.append(int(np.flatnonzero(etiquetas == "Piel y cuerpo")[pureza[etiquetas == "Piel y cuerpo"].argmin()]))
desplazamientos = [(15, 15), (15, 15), (15, 25), (15, 15), (20, -30)]
for idx, offset in zip(elegidos, desplazamientos):
    x, y_ = emb_tfidf[idx]
    nombre = df["nombre"].iloc[idx]
    axes[0].annotate(
        nombre[:30] + ("…" if len(nombre) > 30 else ""), xy=(x, y_), xytext=offset, textcoords="offset points",
        fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85),
        arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
    )
fig.savefig(FIG / "tsne.png", bbox_inches="tight")

# %% [markdown]
# **Lectura.** En el espacio TF-IDF aparecen **islas compactas y de un solo color**
# (líneas de producto como los tintes de una marca o los esmaltes de otra; los nombres
# anotados son productos típicos de cada isla). La anotación de «Piel y cuerpo» muestra
# un caso ambiguo: «B.CREAT.SK CREMA ANTIEDAD…» es una crema de una marca de maquillaje
# (Beauty Creations), y por la marca cae en la zona de Maquillaje. Mientras tanto,
# «Otros», «Herramientas» y «Piel y cuerpo» quedan dispersas en el centro, mezcladas: es
# la misma confusión que muestra la matriz. En el espacio Word2Vec las clases se
# solapan mucho más; solo «Uñas» y parte de «Cuidado capilar» forman regiones propias.
# t-SNE preserva vecindarios locales y no distancias globales, así que se lee la
# separación de grupos, no la posición absoluta [5].

# %% [markdown]
# ## 10. Aplicación al catálogo completo
#
# El objetivo de negocio era categorizar los productos que quedaron en «General». Se
# aplica el mejor modelo (TF-IDF + LogReg) a los **9 777 productos del catálogo
# deduplicado que no forman parte de los 1 000 etiquetados**.
#
# **Umbral de confianza.** La confianza de una predicción es la probabilidad que el
# modelo asigna a la clase elegida. Antes de usarla hay que comprobar que significa
# algo, y el umbral **no se elige mirando el test** (eso sería usar el test para tomar
# una decisión): se usan predicciones *out-of-fold* sobre el 80 % de entrenamiento
# (`cross_val_predict`, 5 folds estratificados) y se mide qué tan seguido acierta el
# modelo cuando su confianza supera cada umbral.

# %%
# Prompt (IA): «Elige el umbral de confianza con predicciones out-of-fold del entrenamiento, sin usar el test.»
best = search_tfidf.best_estimator_
proba_oof = cross_val_predict(
    best, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    method="predict_proba", n_jobs=-1,
)
clases = np.sort(y_train.unique())
conf_oof = proba_oof.max(axis=1)
acierto_oof = clases[proba_oof.argmax(axis=1)] == y_train.to_numpy()
pd.DataFrame([
    {
        "umbral": u,
        "% que lo supera": round(100 * (conf_oof >= u).mean(), 1),
        "accuracy sobre esos": round(acierto_oof[conf_oof >= u].mean(), 3),
    }
    for u in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)
])

# %%
# Prompt (IA): «Confirma en el test, que no participó en la elección, el umbral elegido.»
# Confirmación en el test (200 productos nunca vistos) con el umbral elegido.
UMBRAL = 0.8
proba_test = best.predict_proba(X_test)
conf_test = proba_test.max(axis=1)
acierto_test = best.classes_[proba_test.argmax(axis=1)] == y_test.to_numpy()
print(f"Test con confianza ≥ {UMBRAL}: {100 * (conf_test >= UMBRAL).mean():.1f} % de los productos, "
      f"accuracy {acierto_test[conf_test >= UMBRAL].mean():.3f}")

# %% [markdown]
# **Lectura del umbral.** Con predicciones out-of-fold, exigir confianza ≥ 0,8 deja el 72 %
# de los productos con un **93,6 % de acierto** (contra 79,8 % sin umbral). En el test, que
# no participó en la elección, ese umbral cubre el 77 % con un **96,8 %** de acierto: la
# confianza del modelo es informativa y el umbral generaliza.

# %%
# Prompt (IA): «Aplica el modelo final reentrenado al catálogo restante y grafica confianza y predicciones por categoría.»
catalogo = pd.read_csv(ROOT / "data/processed/catalogo_sin_etiquetar.csv")
# Modelo final: mismos hiperparámetros, reentrenado con los 1 000 productos etiquetados.
modelo_final = clone(best).fit(df["nombre"], df["etiqueta"])
proba = modelo_final.predict_proba(catalogo["nombre"])
catalogo["prediccion"] = modelo_final.classes_[proba.argmax(axis=1)]
catalogo["confianza"] = proba.max(axis=1)
catalogo["revision"] = np.where(catalogo["confianza"] >= UMBRAL, "automática", "manual")

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), layout="constrained")
catalogo["confianza"].plot.hist(bins=30, ax=axes[0])
axes[0].set_ylabel("productos")
axes[0].axvline(UMBRAL, ls="--", c="grey")
axes[0].set_xlabel("confianza de la predicción")
axes[0].set_title("Confianza en el catálogo completo")
pd.crosstab(catalogo["prediccion"], catalogo["revision"]).sort_values("automática").plot.barh(stacked=True, ax=axes[1])
axes[1].set_title(f"Predicciones por categoría (umbral {UMBRAL})")
axes[1].set_xlabel("productos")
axes[1].set_ylabel("")
fig.savefig(FIG / "catalogo_completo.png", bbox_inches="tight")
catalogo["revision"].value_counts()

# %% [markdown]
# **Lectura del catálogo.** De los 9 777 productos, **7 698 (78,7 %) se categorizan
# automáticamente** y **2 079 (21,3 %) van a revisión manual**. Revisar 2 079 productos en
# vez de 9 777 reduce el trabajo manual en ~79 %. «Otros» y «Piel y cuerpo» tienen la mayor
# proporción de revisión manual; «Coloración capilar» casi no la necesita. Coincide con el
# F1 por clase del test.

# %%
# Prompt (IA): «Dibuja un mapa t-SNE del catálogo completo coloreado por categoría predicha y por confianza.»
# Mapa del catálogo: TF-IDF → SVD (50 dims) → t-SNE. SVD primero porque t-SNE sobre miles
# de features dispersas es lento y pesado en memoria; es la práctica habitual.
X_cat = modelo_final[:-1].transform(catalogo["nombre"])
emb = TSNE(n_components=2, random_state=SEED, init="pca", perplexity=40).fit_transform(
    TruncatedSVD(n_components=50, random_state=SEED).fit_transform(X_cat)
)
fig, axes = plt.subplots(1, 2, figsize=(16, 7), layout="constrained")
auto = (catalogo["revision"] == "automática").to_numpy()
# Paleta sin grises: el gris queda reservado para «revisión manual».
paleta = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
axes[0].scatter(emb[~auto, 0], emb[~auto, 1], s=3, c="#d0d0d0", label="revisión manual")
for color, cat in zip(paleta, modelo_final.classes_):
    m = auto & (catalogo["prediccion"] == cat).to_numpy()
    axes[0].scatter(emb[m, 0], emb[m, 1], s=3, alpha=0.7, c=color, label=cat)
axes[0].set_title("Catálogo completo: categoría predicha (gris = confianza < 0,8)")
axes[0].legend(markerscale=4, fontsize=8, loc="lower left")
sc = axes[1].scatter(emb[:, 0], emb[:, 1], s=3, c=catalogo["confianza"], cmap="viridis")
fig.colorbar(sc, ax=axes[1], label="confianza")
axes[1].set_title("Catálogo completo: confianza de la predicción")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
fig.savefig(FIG / "catalogo_tsne.png", bbox_inches="tight")

# %% [markdown]
# **Lectura del mapa.** Cada punto es un producto; productos con nombres parecidos quedan
# cerca. Las islas compactas de un solo color son líneas de producto que el modelo
# reconoce con seguridad (esmaltes de una marca, tintes numerados); los puntos grises
# —revisión manual— se concentran en las zonas donde se mezclan colores, que son
# justamente los nombres ambiguos. El mapa de la derecha muestra lo mismo desde la
# confianza: las islas son amarillas (confianza ≈ 1) y las zonas mezcladas, oscuras.

# %%
# Prompt (IA): «Muestra los productos más representativos que el modelo asignó a cada categoría.»
# Ejemplos por categoría: los 3 productos con mayor confianza en cada una.
ejemplos = (
    catalogo[catalogo["revision"] == "automática"]
    .sort_values("confianza", ascending=False)
    .groupby("prediccion")
    .head(3)
    .sort_values(["prediccion", "confianza"], ascending=[True, False])
)
ejemplos[["prediccion", "nombre", "confianza"]].round(3).reset_index(drop=True)

# %%
# Prompt (IA): «Muestra las predicciones más seguras y las más dudosas del catálogo.»
# Ejemplos: las predicciones más seguras y las más dudosas.
cols = ["nombre", "prediccion", "confianza"]
pd.concat({
    "más seguras": catalogo.nlargest(8, "confianza")[cols].reset_index(drop=True),
    "más dudosas": catalogo.nsmallest(8, "confianza")[cols].reset_index(drop=True),
}, axis=1).round(2)

# %% [markdown]
# **Lectura de los extremos.** Las predicciones más seguras son líneas de producto con
# marca y vocabulario inconfundibles; las más dudosas son productos realmente ambiguos o
# fuera de la taxonomía («SAL DE LAVANDA», «ENVASE PARA GUARDAR PEGAMENTO»): justo los que
# conviene que revise una persona.

# %%
# Prompt (IA): «Compara las predicciones con las categorías que la tienda ya tenía en el ERP, con un mapa de calor.»
# Validación externa: productos que el personal de la tienda ya había categorizado en el
# ERP (todas categorías capilares). El modelo nunca vio esas etiquetas.
CAPILARES = {"Cuidado capilar", "Coloración capilar"}
ya = catalogo[catalogo["categoria_erp"] != "General"]
print(f"{len(ya)} productos con categoría previa en el ERP")
print(f"Predichos como capilares: {100 * ya['prediccion'].isin(CAPILARES).mean():.1f} %")
tabla = pd.crosstab(ya["categoria_erp"], ya["prediccion"])
tabla = tabla.loc[tabla.sum(axis=1).sort_values(ascending=False).index].head(12)
fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
im = ax.imshow(tabla.to_numpy(), cmap="Blues", aspect="auto")
ax.set_xticks(range(len(tabla.columns)), tabla.columns, rotation=35, ha="right")
ax.set_yticks(range(len(tabla.index)), tabla.index)
for (r, c), v in np.ndenumerate(tabla.to_numpy()):
    if v:
        ax.text(c, r, v, ha="center", va="center", fontsize=8,
                color="white" if v > tabla.to_numpy().max() / 2 else "black")
ax.set_xlabel("categoría predicha por el modelo")
ax.set_ylabel("categoría previa en el ERP (puesta por la tienda)")
ax.set_title("Validación externa: ERP vs. modelo")
fig.savefig(FIG / "validacion_erp.png", bbox_inches="tight")

# %% [markdown]
# **Lectura de la validación externa.** De los 534 productos que el personal de la tienda
# ya había categorizado (todos capilares), el modelo predijo como capilares el **93,3 %**,
# sin haber visto esas etiquetas. El error más frecuente está en la frontera prevista en la
# taxonomía: de los productos que el ERP tiene como «Tinte», 11 se predijeron como
# «Cuidado capilar».

# %% [markdown]
# ## 11. Conclusiones
#
# 1. **TF-IDF de n-gramas de caracteres + regresión logística supera a Word2Vec
#    preentrenado + SVM** en este problema, con diferencia significativa (F1-macro 0,759
#    vs 0,648 en 10 × 10 CV; t corregido p ≈ 4,3 × 10⁻⁶, Wilcoxon p ≈ 1,2 × 10⁻¹⁷) y en
#    test (F1-macro 0,75 vs 0,58).
# 2. **La causa es la representación, no el clasificador.** El conocimiento general del
#    idioma que trae Word2Vec no compensa que desconozca las marcas, que son las palabras
#    más informativas del catálogo (11,6 % de palabras sin vector, casi todas marcas), ni
#    las abreviaturas. Los n-gramas de caracteres capturan ambas cosas.
# 3. **El modelo tiene alta varianza y la curva de validación no se ha estabilizado**:
#    etiquetar más productos es la mejora más rentable, y el etiquetado asistido la hace
#    barata (0,4 % de corrección manual en esta muestra).
# 4. **Las clases difíciles son las de vocabulario compartido** (Herramientas, Piel y
#    cuerpo, Otros). Para usarlo en la tienda conviene aplicar el modelo con un umbral de
#    confianza y mandar a revisión manual los casos dudosos, en vez de categorizar todo
#    automáticamente.
# 5. **Aplicado al catálogo, el modelo categoriza automáticamente el 78,7 %** de los
#    9 777 productos (confianza ≥ 0,8, con ~94–97 % de acierto esperado) y deja 2 079 para
#    revisión manual. Contra las categorías que la tienda ya tenía, coincide en el
#    93,3 %. El trabajo manual se reduce en ~79 %.
# 6. **Optuna (TPE) confirma la optimización por grilla**: llega al mismo óptimo de
#    TF-IDF con un tercio de las evaluaciones (40 contra 120) y mejora al SVM en solo
#    0,007 de F1, dentro de la variación entre folds. La grilla no dejó un óptimo mejor sin
#    explorar, y λ es el hiperparámetro que más importa (80 %).
# 7. **χ² no mejoró el modelo**: el mejor resultado fue sin selección (0,739) y la
#    diferencia con k = 3500 es menor que la variación entre folds. Con regresión
#    logística regularizada conviene conservar todos los n-gramas.
# 8. **El enfoque supervisado está justificado**: k-means sobre el catálogo completo no
#    encuentra grupos naturales (silhouette 0,02–0,04), agrupa por marca o por palabras
#    frecuentes y coincide con las etiquetas humanas al nivel del azar (ARI ≈ 0,008).
#
# **Limitaciones:** una sola tienda; etiquetas de un solo revisor (sin acuerdo entre
# anotadores); scores de CV repetida no independientes; la taxonomía es de esta tienda.

# %% [markdown]
# ## 12. Referencias
#
# [1] C. Cardellino, "Spanish Billion Words Corpus and Embeddings," 2016. [Online]. Available: https://crscardellino.ar/SBWCE/
#
# [2] T. Mikolov, K. Chen, G. Corrado, and J. Dean, "Efficient estimation of word representations in vector space," arXiv:1301.3781, 2013.
#
# [3] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.
#
# [4] F. Wilcoxon, "Individual comparisons by ranking methods," *Biometrics Bulletin*, vol. 1, no. 6, pp. 80–83, 1945.
#
# [5] L. van der Maaten and G. Hinton, "Visualizing data using t-SNE," *J. Mach. Learn. Res.*, vol. 9, pp. 2579–2605, 2008.
#
# [6] G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," *Inf. Process. Manage.*, vol. 24, no. 5, pp. 513–523, 1988.
#
# [7] Anthropic, "Claude," 2026. Asistente usado para proponer etiquetas y escribir código (prompts en las celdas correspondientes).
#
# [8] C. Nadeau and Y. Bengio, "Inference for the generalization error," *Mach. Learn.*, vol. 52, no. 3, pp. 239–281, 2003.
#
# [9] C. R. Harris *et al.*, "Array programming with NumPy," *Nature*, vol. 585, pp. 357–362, 2020.
#
# [10] W. McKinney, "Data structures for statistical computing in Python," in *Proc. 9th Python in Science Conf.*, 2010, pp. 56–61.
#
# [11] P. Virtanen *et al.*, "SciPy 1.0: Fundamental algorithms for scientific computing in Python," *Nature Methods*, vol. 17, pp. 261–272, 2020.
#
# [12] J. D. Hunter, "Matplotlib: A 2D graphics environment," *Comput. Sci. Eng.*, vol. 9, no. 3, pp. 90–95, 2007.
#
# [13] R. Řehůřek and P. Sojka, "Software framework for topic modelling with large corpora," in *Proc. LREC 2010 Workshop on New Challenges for NLP Frameworks*, 2010, pp. 45–50.
#
# [14] A. Mueller, "WordCloud for Python," 2023. [Online]. Available: https://github.com/amueller/word_cloud
#
# [15] scikit-learn developers, "sklearn.model_selection.RepeatedStratifiedKFold." [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html
#
# [16] scikit-learn developers, "sklearn.model_selection.learning_curve." [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html
#
# [17] SciPy developers, "scipy.stats.wilcoxon." [Online]. Available: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
#
# [18] scikit-learn developers, "Pipelines and composite estimators." [Online]. Available: https://scikit-learn.org/stable/modules/compose.html
#
# [19] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A next-generation hyperparameter optimization framework," in *Proc. 25th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2019, pp. 2623–2631.
#
# [20] J. Bergstra, R. Bardenet, Y. Bengio, and B. Kégl, "Algorithms for hyper-parameter optimization," in *Adv. Neural Inf. Process. Syst. 24*, 2011, pp. 2546–2554.
#
# [21] S. Lloyd, "Least squares quantization in PCM," *IEEE Trans. Inf. Theory*, vol. 28, no. 2, pp. 129–137, 1982.
#
# [22] P. J. Rousseeuw, "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis," *J. Comput. Appl. Math.*, vol. 20, pp. 53–65, 1987.
#
# [23] L. Hubert and P. Arabie, "Comparing partitions," *J. Classification*, vol. 2, no. 1, pp. 193–218, 1985.
