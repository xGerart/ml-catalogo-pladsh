from pathlib import Path

import pandas as pd
import pytest
from openpyxl import load_workbook

from catalogo.labeling import (
    build_labeled_dataset,
    check_review_matches_sample,
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


def test_workbook_escribe_nombres_con_prefijo_de_formula_como_texto(tmp_path: Path):
    df = pd.DataFrame({"id": [1], "nombre": ["=SUMA(A1)"], "propuesta": ["Uñas"]})
    path = tmp_path / "r.xlsx"
    write_review_workbook(df, CATS, path)
    cell = load_workbook(path).active["B2"]
    assert cell.data_type == "s"
    assert read_review_workbook(path)["nombre"].tolist() == ["=SUMA(A1)"]


def test_check_review_matches_sample_detecta_nombre_editado():
    muestra = pd.DataFrame({"id": [1, 2], "nombre": ["ESMALTE", "GEL"]})
    review = pd.DataFrame(
        {"id": [1, 2], "nombre": ["ESMALTE", "GEL EDITADO"], "propuesta": ["Uñas"] * 2, "final": ["Uñas"] * 2}
    )
    with pytest.raises(ValueError, match="2"):
        check_review_matches_sample(review, muestra)


def test_check_review_matches_sample_detecta_ids_faltantes():
    muestra = pd.DataFrame({"id": [1, 2], "nombre": ["ESMALTE", "GEL"]})
    review = pd.DataFrame({"id": [1], "nombre": ["ESMALTE"], "propuesta": ["Uñas"], "final": ["Uñas"]})
    with pytest.raises(ValueError, match="ids"):
        check_review_matches_sample(review, muestra)
