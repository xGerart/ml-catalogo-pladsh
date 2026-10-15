"""Carga, limpieza, deduplicado y muestreo del catálogo."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from catalogo import SEED


def find_root(start: Path | None = None) -> Path:
    """Sube desde `start` (o el cwd) hasta el directorio que contiene pyproject.toml."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(f"No se encontró pyproject.toml desde {here}")


def normalize_name(name: str) -> str:
    """Mayúsculas y espacios colapsados: dos nombres que solo difieren en eso son el mismo."""
    return re.sub(r"\s+", " ", str(name)).strip().upper()


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por nombre normalizado, conservando el id menor.

    Se hace ANTES de muestrear: así un mismo texto nunca queda en train y test a la vez.
    """
    out = df[["id", "nombre"]].assign(nombre=df["nombre"].map(normalize_name))
    out = out[out["nombre"] != ""]
    out = out.sort_values("id").drop_duplicates("nombre", keep="first")
    return out.reset_index(drop=True)


def load_catalog(path: Path) -> pd.DataFrame:
    """Lee el export crudo del ERP y devuelve los productos activos deduplicados (id, nombre)."""
    df = pd.read_csv(path, usecols=["id", "nombre", "is_active"], dtype={"nombre": str})
    df = df[df["is_active"] == "t"]
    return deduplicate(df)


def sample_for_labeling(df: pd.DataFrame, n: int = 1000, seed: int = SEED) -> pd.DataFrame:
    """Muestra aleatoria simple y reproducible para etiquetar."""
    if n > len(df):
        raise ValueError(f"Se pidieron {n} productos pero el catálogo tiene {len(df)}")
    return df.sample(n=n, random_state=seed).reset_index(drop=True)
