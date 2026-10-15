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
