"""Paso final del etiquetado: valida la revisión y escribe el CSV publicado."""

import json

import pandas as pd

from catalogo.data import find_root
from catalogo.labeling import build_labeled_dataset, check_review_matches_sample, read_review_workbook
from catalogo.taxonomy import CATEGORIES

ROOT = find_root()
review = read_review_workbook(ROOT / "data/interim/revision.xlsx")
check_review_matches_sample(review, pd.read_csv(ROOT / "data/interim/muestra.csv"))
dataset, resumen = build_labeled_dataset(review, CATEGORIES)
dataset.to_csv(ROOT / "data/processed/productos_etiquetados.csv", index=False)

# Control: los productos de la muestra que en el ERP ya tenían categoría (todas capilares)
# deberían quedar en una de las dos categorías capilares de la taxonomía.
CAPILARES = {"Cuidado capilar", "Coloración capilar"}
if not CAPILARES <= set(CATEGORIES):
    raise ValueError("La taxonomía no tiene las categorías capilares del control")
raw = pd.read_csv(ROOT / "data/raw/productos.csv", usecols=["id", "categoria"])
control = review.merge(raw, on="id")
control = control[control["categoria"] != "General"]
resumen["control_capilar"] = {
    "productos": int(len(control)),
    "propuesta_capilar_pct": round(100 * control["propuesta"].isin(CAPILARES).mean(), 2),
    "final_capilar_pct": round(100 * control["final"].isin(CAPILARES).mean(), 2),
    "fuera_de_capilar": control.loc[~control["final"].isin(CAPILARES), ["id", "nombre", "categoria", "final"]].to_dict("records"),
}
(ROOT / "etiquetado/resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2))
print(json.dumps(resumen, ensure_ascii=False, indent=2))
