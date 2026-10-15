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


def test_unlabeled_catalog_excluye_etiquetados_y_conserva_categoria_erp(tmp_path: Path):
    from catalogo.data import unlabeled_catalog

    csv = tmp_path / "p.csv"
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "nombre": ["Esmalte", "Tinte", "Gel", "esmalte"],
            "is_active": ["t", "t", "t", "t"],
            "categoria": ["General", "Tinte", "General", "General"],
        }
    ).to_csv(csv, index=False)
    labeled = pd.DataFrame({"id": [3], "nombre": ["GEL"], "etiqueta": ["Uñas"]})
    out = unlabeled_catalog(csv, labeled)
    assert out.columns.tolist() == ["id", "nombre", "categoria_erp"]
    assert out["id"].tolist() == [1, 2]  # 3 está etiquetado; 4 es duplicado de 1
    assert out.loc[out["id"] == 2, "categoria_erp"].item() == "Tinte"
