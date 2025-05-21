import numpy as np
from scipy.stats import chi2_contingency
from typing import List, Dict, Set
from ..dataset import DatasetMeta
from .. import logger
from .meta_rule import MetaRule

def calculate_weight_factor(depth: int) -> float:
    alpha = 0.8
    return alpha

def calculate_gini_impurity(y_values):
    if len(y_values) == 0:
        return 0
    
    classes, counts = np.unique(y_values, return_counts=True)
    probabilities = counts / len(y_values)
    return 1 - np.sum(probabilities ** 2)

def calculate_gini_scores(
    X: np.ndarray, 
    y: np.ndarray,
    meta: DatasetMeta
) -> Dict[int, float]:
    scores = {}
    
    for feat_idx in range(X.shape[1]):
        feature_name = meta.features[feat_idx].name if feat_idx < len(meta.features) else f"特征{feat_idx}"
        is_categorical = meta.features[feat_idx].is_categorical if feat_idx < len(meta.features) else False
        
        unique_values = np.unique(X[:, feat_idx])
        
        if len(unique_values) <= 1:
            scores[feat_idx] = 0.0
            continue
        
        best_gain = 0.0
        best_split = None
        
        if is_categorical:
            for val in unique_values:
                subset_indices = X[:, feat_idx] == val
                subset_size = np.sum(subset_indices)
                
                if subset_size == 0:
                    continue
                    
                subset_labels = y[subset_indices]
                subset_gini = 1.0
                
                for label in np.unique(y):
                    p = np.sum(subset_labels == label) / subset_size if subset_size > 0 else 0
                    subset_gini -= p * p
                
                weight = subset_size / len(y)
                weighted_gini = weight * subset_gini
                gain = calculate_gini_impurity(y) - weighted_gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = val
        else:
            sorted_values = np.sort(unique_values)
            decimal_places = 1