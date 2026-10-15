"""Validación cruzada repetida, test de Wilcoxon, métricas y gráficos."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import label_binarize

from catalogo import SEED


def repeated_cv_scores(estimators, X, y, n_splits=10, n_repeats=10, seed=SEED, scoring="f1_macro"):
    """Mismos folds para todos los modelos: requisito para un test pareado."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return pd.DataFrame(
        {
            name: cross_val_score(clone(est), X, y, cv=cv, scoring=scoring, n_jobs=-1)
            for name, est in estimators.items()
        }
    )


def wilcoxon_test(scores: pd.DataFrame, a: str, b: str, alpha: float = 0.05) -> dict:
    stat, p = wilcoxon(scores[a], scores[b])
    media_a, media_b = float(scores[a].mean()), float(scores[b].mean())
    return {
        "estadistico": float(stat),
        "p_value": float(p),
        "alpha": alpha,
        "significativo": bool(p < alpha),
        "media_a": media_a,
        "media_b": media_b,
        "mejor": a if media_a >= media_b else b,
    }


def classification_metrics(y_true, y_pred, y_proba, classes) -> dict:
    y_bin = label_binarize(y_true, classes=classes)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "auc_macro": roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=classes),
        "auc_micro": roc_auc_score(y_bin, y_proba, average="micro"),
    }


def plot_param_curve(search, param: str, ax, log: bool = True, label: str | None = None):
    """Score medio ± std vs. un hiperparámetro, fijando los demás en su mejor valor."""
    res = pd.DataFrame(search.cv_results_)
    key = f"param_{param}"
    for other, best in search.best_params_.items():
        if other != param:
            res = res[res[f"param_{other}"].astype(str) == str(best)]
    res = res.sort_values(key)
    x = res[key].astype(float)
    ax.errorbar(x, res["mean_test_score"], yerr=res["std_test_score"], marker="o", capsize=3, label=label)
    if log:
        ax.set_xscale("log")
    ax.axvline(float(search.best_params_[param]), ls="--", c="grey")
    ax.set_xlabel(param)
    ax.set_ylabel(f"{search.scoring} (validación)")


def plot_learning_curve(estimator, X, y, cv, ax, scoring="f1_macro", title=""):
    sizes, train, val = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 6)
    )
    for scores, name in ((train, "entrenamiento"), (val, "validación")):
        m, s = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(sizes, m, marker="o", label=name)
        ax.fill_between(sizes, m - s, m + s, alpha=0.2)
    ax.set_xlabel("ejemplos de entrenamiento")
    ax.set_ylabel(scoring)
    ax.set_title(title)
    ax.legend()


def plot_roc_multiclass(y_true, y_proba, classes, ax, title=""):
    """Curva ROC one-vs-rest por clase más el micro-promedio.

    Se dibuja con roc_curve + ax.plot (no con RocCurveDisplay) porque la API de estilo
    de RocCurveDisplay cambió en sklearn 1.7 y Colab puede traer otra versión.
    """
    y_bin = label_binarize(y_true, classes=classes)
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, lw=1, label=f"{c} (AUC {auc(fpr, tpr):.2f})")
    fpr, tpr, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    ax.plot(fpr, tpr, ls="--", lw=2, c="black", label=f"micro-promedio (AUC {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], c="grey", lw=0.8)
    ax.set_xlabel("tasa de falsos positivos")
    ax.set_ylabel("tasa de verdaderos positivos")
    ax.set_title(title)
    ax.legend(fontsize=7)


def plot_tsne(X, labels, ax, title="", seed=SEED):
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    emb = TSNE(n_components=2, random_state=seed, init="pca", perplexity=30).fit_transform(X)
    labels = np.asarray(labels)
    for c in np.unique(labels):
        mask = labels == c
        ax.scatter(emb[mask, 0], emb[mask, 1], s=8, label=str(c), alpha=0.7)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=7)
