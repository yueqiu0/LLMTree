import numpy as np
from .meta_rule import MetaRule


def calculate_gini_impurity(y_values):
    if len(y_values) == 0:
        return 0
    classes, counts = np.unique(y_values, return_counts=True)
    probabilities = counts / len(y_values)
    return 1 - np.sum(probabilities**2)


def calculate_meta_rule_gini(
    meta_rule: MetaRule, X: np.ndarray, y: np.ndarray, samples: np.ndarray
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    if len(samples) == 0:
        return 0.0, 0.0, 0.0, np.array([]), np.array([])
    node_X = X[samples]
    node_y = y[samples]
    feature_idx = meta_rule.feature_idx
    split_value = meta_rule.split_value
    current_gini = calculate_gini_impurity(node_y)
    feature_values = node_X[:, feature_idx]
    if meta_rule.is_categorical:
        left_mask = feature_values == split_value
    else:
        left_mask = feature_values < split_value
    right_mask = ~left_mask

    left_y = node_y[left_mask]
    right_y = node_y[right_mask]
    if len(left_y) == 0 or len(right_y) == 0:
        return 0.0, 0.0, 0.0, left_mask, right_mask
    left_gini = calculate_gini_impurity(left_y)
    right_gini = calculate_gini_impurity(right_y)

    n = len(node_y)
    weighted_gini = (len(left_y) / n) * left_gini + (len(right_y) / n) * right_gini
    gini_gain = current_gini - weighted_gini
    return gini_gain, left_gini, right_gini, left_mask, right_mask
