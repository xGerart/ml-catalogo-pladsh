import catalogo


def test_paquete_importa_y_fija_semilla():
    assert catalogo.SEED == 42
