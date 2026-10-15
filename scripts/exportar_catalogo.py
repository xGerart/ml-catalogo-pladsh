"""Exporta el catálogo deduplicado sin los 1 000 productos etiquetados, para aplicarle el modelo."""

from catalogo.data import find_root, unlabeled_catalog
from catalogo.labeling import load_labeled

ROOT = find_root()
labeled = load_labeled(ROOT / "data/processed/productos_etiquetados.csv")
catalogo = unlabeled_catalog(ROOT / "data/raw/productos.csv", labeled)
catalogo.to_csv(ROOT / "data/processed/catalogo_sin_etiquetar.csv", index=False)
print(f"{len(catalogo)} productos → data/processed/catalogo_sin_etiquetar.csv")
print(catalogo["categoria_erp"].eq("General").value_counts().rename({True: "General", False: "ya categorizados"}))
