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
