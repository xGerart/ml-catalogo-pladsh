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
# Las de src/ que no están aquí son de preparación de datos (carga del export del ERP,
# planilla de etiquetado): no forman parte del análisis de ML del notebook.
EN_EL_NOTEBOOK = {
    "normalize_name", "deduplicate", "sample_for_labeling",
    "tokenize", "vocabulary_coverage", "MeanEmbeddingVectorizer",
    "plot_param_curve", "plot_learning_curve",
    "repeated_cv_scores", "wilcoxon_test", "corrected_resampled_ttest",
    "classification_metrics", "plot_roc_multiclass", "plot_tsne",
}


def test_el_notebook_no_define_otras_funciones_de_src():
    assert (set(NOTEBOOK_DEFS) & set(SRC_DEFS)) == EN_EL_NOTEBOOK


@pytest.mark.parametrize("name", sorted(EN_EL_NOTEBOOK))
def test_funcion_del_notebook_identica_a_src(name):
    assert name in NOTEBOOK_DEFS, f"{name} no está definida en el notebook"
    assert NOTEBOOK_DEFS[name] == SRC_DEFS[name], f"{name} difiere entre el notebook y src/"
