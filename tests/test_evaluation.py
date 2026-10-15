import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from catalogo.evaluation import (
    classification_metrics,
    plot_learning_curve,
    plot_param_curve,
    plot_roc_multiclass,
    plot_tsne,
    repeated_cv_scores,
    wilcoxon_test,
)

X, y = make_classification(n_samples=300, n_features=8, n_informative=5, n_classes=3, random_state=0)
MODELS = {"logreg": LogisticRegression(max_iter=500), "dummy": DummyClassifier()}


def test_repeated_cv_scores_forma_y_folds_identicos():
    a = repeated_cv_scores(MODELS, X, y, n_splits=5, n_repeats=2)
    b = repeated_cv_scores(MODELS, X, y, n_splits=5, n_repeats=2)
    assert a.shape == (10, 2)
    pd.testing.assert_frame_equal(a, b)  # mismos folds → resultados idénticos


def test_wilcoxon_detecta_diferencia_real():
    scores = repeated_cv_scores(MODELS, X, y, n_splits=5, n_repeats=2)
    res = wilcoxon_test(scores, "logreg", "dummy")
    assert res["significativo"] and res["mejor"] == "logreg"
    assert 0 <= res["p_value"] < 0.05


def test_classification_metrics_claves_y_rangos():
    clf = LogisticRegression(max_iter=500).fit(X, y)
    m = classification_metrics(y, clf.predict(X), clf.predict_proba(X), clf.classes_)
    assert set(m) == {"accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "auc_macro", "auc_micro"}
    assert all(0 <= v <= 1 for v in m.values())


def test_graficos_no_fallan():
    search = GridSearchCV(LogisticRegression(max_iter=500), {"C": [0.1, 1, 10]}, cv=3, scoring="f1_macro").fit(X, y)
    fig, axes = plt.subplots(1, 4)
    plot_param_curve(search, "C", axes[0])
    plot_learning_curve(LogisticRegression(max_iter=500), X, y, StratifiedKFold(3), axes[1])
    clf = search.best_estimator_
    plot_roc_multiclass(y, clf.predict_proba(X), clf.classes_, axes[2])
    plot_tsne(X, y, axes[3])
    plt.close(fig)


def test_plot_param_curve_acepta_valores_no_numericos_como_referencia():
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("sel", SelectKBest(f_classif)), ("clf", LogisticRegression(max_iter=500))])
    search = GridSearchCV(pipe, {"sel__k": [2, 4, "all"]}, cv=3, scoring="f1_macro").fit(X, y)
    fig, ax = plt.subplots()
    plot_param_curve(search, "sel__k", ax, log=False)
    assert any(line.get_label() == "validación (sel__k = all)" for line in ax.get_lines())
    plt.close(fig)


def test_corrected_resampled_ttest_es_mas_conservador_que_sin_corregir():
    from scipy.stats import ttest_rel

    from catalogo.evaluation import corrected_resampled_ttest

    scores = repeated_cv_scores(MODELS, X, y, n_splits=5, n_repeats=2)
    res = corrected_resampled_ttest(scores, "logreg", "dummy", n_train=240, n_test=60)
    naive = ttest_rel(scores["logreg"], scores["dummy"]).pvalue
    assert res["p_value"] > naive
    assert res["significativo"] and res["mejor"] == "logreg"


def test_plot_param_curve_dibuja_entrenamiento_y_validacion():
    search = GridSearchCV(
        LogisticRegression(max_iter=500), {"C": [0.1, 1, 10]}, cv=3, scoring="f1_macro", return_train_score=True
    ).fit(X, y)
    fig, ax = plt.subplots()
    plot_param_curve(search, "C", ax)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "entrenamiento" in labels and "validación" in labels
    plt.close(fig)


def test_plot_param_curve_como_error_contra_lambda():
    search = GridSearchCV(
        LogisticRegression(max_iter=500), {"C": [0.1, 1, 10]}, cv=3, scoring="f1_macro", return_train_score=True
    ).fit(X, y)
    fig, ax = plt.subplots()
    plot_param_curve(search, "C", ax, as_error=True, x_transform=lambda c: 1 / c, x_label="λ = 1/C")
    val = next(c for c in ax.containers if c.get_label() == "validación")
    xs, ys = val.lines[0].get_data()
    assert sorted(xs.tolist()) == [0.1, 1.0, 10.0]  # 1/C
    best_c = search.best_params_["C"]
    i = list(xs).index(1 / best_c)
    assert abs(ys[i] - (1 - search.best_score_)) < 1e-12  # error = 1 − F1 de validación
    assert ax.get_xlabel() == "λ = 1/C"
    plt.close(fig)


def test_plot_learning_curve_como_error():
    fig, ax = plt.subplots()
    plot_learning_curve(LogisticRegression(max_iter=500), X, y, StratifiedKFold(3), ax, as_error=True)
    train = next(l for l in ax.get_lines() if l.get_label() == "entrenamiento")
    assert all(0 <= v <= 1 for v in train.get_ydata())
    assert "error" in ax.get_ylabel()
    plt.close(fig)


def test_plot_tsne_devuelve_las_coordenadas_2d():
    fig, ax = plt.subplots()
    emb = plot_tsne(X, y, ax)
    assert emb.shape == (len(X), 2)
    plt.close(fig)
