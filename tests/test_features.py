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
