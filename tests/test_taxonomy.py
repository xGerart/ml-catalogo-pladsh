from catalogo.taxonomy import CATEGORIES, NO_SE, OTROS


def test_taxonomia_tiene_otros_y_no_incluye_no_se():
    assert OTROS in CATEGORIES
    assert NO_SE not in CATEGORIES


def test_taxonomia_sin_duplicados_y_tamano_acordado():
    assert len(CATEGORIES) == len(set(CATEGORIES))
    assert 7 <= len(CATEGORIES) <= 10  # 6–9 macro-categorías + «Otros»
