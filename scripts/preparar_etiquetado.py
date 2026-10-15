"""Paso 1 del etiquetado: muestra 1 000 productos al azar del catálogo deduplicado."""

from catalogo.data import find_root, load_catalog, sample_for_labeling

ROOT = find_root()
muestra = sample_for_labeling(load_catalog(ROOT / "data/raw/productos.csv"), n=1000)
muestra.to_csv(ROOT / "data/interim/muestra.csv", index=False)
print(f"{len(muestra)} productos → data/interim/muestra.csv")
