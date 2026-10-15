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
