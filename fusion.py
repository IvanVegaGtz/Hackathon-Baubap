import numpy as np
from sklearn.metrics import brier_score_loss
from itertools import combinations




def select_best_combination(y_true, **models):
    """
    Selecciona la mejor combinación de probabilidades entre modelos
    utilizando la métrica Brier Score.
    
    Args:
    y_true (numpy.ndarray): Etiquetas verdaderas.
    **models: Diccionario de modelos con nombres como claves y probabilidades como valores (numpy.ndarray).

    Returns:
    best_combination (tuple): Tupla con las probabilidades de la mejor combinación.
    best_model_names (tuple): Tupla con los nombres de los arrays de la mejor combinación.
    best_brier_score (float): Brier Score de la mejor combinación.
    """

    model_names = list(models.keys())
    all_combinations = [] 

    for k in range(1, len(model_names) + 1):
        all_combinations.extend(combinations(model_names, k))

    best_brier_score = float('inf')
    best_combination = None
    best_model_names = None

    for combination in all_combinations:
        probs_combination = [models[model] for model in combination]
        avg_probs = np.mean(probs_combination, axis=0)
        current_brier_score = brier_score_loss(y_true, avg_probs)
        print(combination)
        print(current_brier_score)
        if current_brier_score < best_brier_score:
            best_brier_score = current_brier_score
            best_combination = probs_combination
            best_model_names = combination

    return best_combination, best_model_names, best_brier_score