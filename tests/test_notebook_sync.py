"""El notebook define sus propias funciones (la rúbrica pide todo el código dentro de él).

Este test verifica que cada una sea idéntica a su versión en src/, que es la que cubren
los demás tests: si alguien cambia una copia y no la otra, falla.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/proyecto_final.py"
MODULES = ["data", "features", "evaluation", "labeling"]


def _definitions(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: ast.dump(node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


NOTEBOOK_DEFS = _definitions(NOTEBOOK)
SRC_DEFS = {
    name: dump
    for module in MODULES
    for name, dump in _definitions(ROOT / f"src/catalogo/{module}.py").items()
}
# find_root y load_labeled no hacen falta en el notebook (rutas fijas y un read_csv).
SOLO_EN_SRC = {"find_root", "load_labeled"}


@pytest.mark.parametrize("name", sorted(set(SRC_DEFS) - SOLO_EN_SRC))
def test_funcion_del_notebook_identica_a_src(name):
    assert name in NOTEBOOK_DEFS, f"{name} no está definida en el notebook"
    assert NOTEBOOK_DEFS[name] == SRC_DEFS[name], f"{name} difiere entre el notebook y src/"
