# Rúbrica del proyecto ML (transcrita de Notion, 2026-10-14)

Fuente: https://pointy-whistle-142.notion.site/Proyecto-ML-734cbbc3faae41c9a3f2425105baab07
(13 bloques de la página no son públicos y no se pudieron leer.)

## Reglas generales
- Objetivo: usar ML para resolver un problema con la metodología correcta del curso.
- Todo en **una notebook de Google Colab**. No hay informe aparte: todo va en la notebook.
- **Repositorio en GitHub con un README pertinente.**
- Defensa **obligatoria**: si un miembro del grupo no se presenta, ese estudiante
  saca 0. La nota es por grupo.

## Ítems (total 10 puntos)

| Ítem | Pts | Qué piden |
|---|---|---|
| Introducción | 0.1 | Contexto del problema; aterrizar qué se va a hacer; resumen |
| EDA | 0.6 | Mínimo: ¿data balanceada? ¿normalizada? ¿estacionalidad (series)? ¿transformaciones no lineales (skewness)? Si es texto: word cloud y palabras frecuentes |
| Feature Extraction | 0.25 | Para data no estructurada. Texto: (1) Bag of words TF-IDF → ML; (2) texto → modelo preentrenado → embedding → ML (transfer learning); (3) otros propuestos |
| Feature Selection | 0.25 | Técnica tipo *filter* en el pipeline. **Se puede omitir si el problema es de texto o usa transfer learning** |
| Optimización de hiperparámetros | **3.5** | En validación con **repeated k-fold CV** (nested no recomendado por tiempo). Gráfica de la optimización de algún hiperparámetro (ej.: λ) **y** curva de aprendizaje (`sklearn.model_selection.learning_curve`). Extra opcional: Optuna |
| Comparación estadística de 2 técnicas | **3.5** | Test de hipótesis (Wilcoxon o t-test) sobre repeated k-fold CV; por defecto 10 repeticiones × 10 folds (o 2×5 si es muy costoso). `RepeatedStratifiedKFold` con `random_state` fijo; justificar si se estratifica o no. `scipy.stats.wilcoxon` con α = 0.05, reportar p-value y concluir. Métricas de clasificación: Accuracy, AUC, curva ROC, Precision, Recall, AUC ROC macro/micro, F1 macro y F1 weighted |
| Pipelines | 0.5 | `sklearn.pipeline`. Si hay red neuronal: código modular (Lightning / PyTorch, o Keras) |
| Visualización con t-SNE | 1 | Espacio de características en 2D; en supervisado, color = label |
| Idioma | 0.1 | Ortografía y redacción |
| Referencias | 0.1 | Formato IEEE, de teoría y de código. **Si se usó IA para el código, poner los prompts como comentarios de Python donde corresponde** |
| Conclusiones | 0.1 | Pertinentes; la teoría no es una conclusión |

## Criterios (se aplican a cada ítem por separado)

| Implementación | Defensa | % del puntaje del ítem |
|---|---|---|
| Completa | Excelente | 100 |
| Completa | Regular | 75 |
| Parcial | Excelente | 50 |
| Parcial | Regular | 25 |
| Completa | Insuficiente | 25 |
| Parcial | Insuficiente | 0 |
| Inexistente | — | 0 |
