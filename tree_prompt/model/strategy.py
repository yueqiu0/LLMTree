import numpy as np
import jinja2
from itertools import product
from functools import reduce
import random
from tqdm import tqdm
from collections import Counter
import re

from ..dataset import DatasetMeta
from ..runner import Runner
from ..prompt import Serializer, TabularSerializer, ListSerializer, TextSerializer
from .tree import DecisionTree, RandomForest, TreeBase, RulePath, Node
from .. import logger
from .feature_selection import calculate_gini_impurity, calculate_meta_rule_gini
from .meta_rule import MetaRule

supervision_template = jinja2.Template(open("template/supervision.jinja").read())

_mu = 2
_threshold = 0.70  # default value

def set_threshold(value: float):
    """Set the threshold for modifying leaf node labels"""
    global _threshold
    _threshold = value
    logger.log(f"Leaf node label modification threshold set to: {value}")

def set_mu(mu: int):
    global _mu
    _mu = mu

def _get_feature_values(
    meta: DatasetMeta, x: np.ndarray, hist_nbins: int
) -> list[list]:
    feature_values = []

    for i in range(meta.feature_count()):
        feature = meta.features[i]
        if feature.is_categorical:
            feature_values.append(np.unique(x[:, i]))
        else:
            values = np.sort(np.unique(x[:, i]))
            if len(values) > hist_nbins:
                nums, values = np.histogram(
                    values,
                    hist_nbins,
                )
                values = values[np.where(nums > 0)[0] + 1]
                feature_values.append(values)
            else:
                if len(x) > 1:
                    feature_values.append(values[1:])
                else:
                    feature_values.append(values)

    return feature_values

class TrainStrategy:
    def __init__(self) -> None:
        self.train_x = None
        self.train_y = None
        self.tree = None
        self.hist_nbins = 10
        self._meta_instance = None

        # Initialize meta_rules as empty list
        self.meta_rules = []

        # Token counters
        self.supervision_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self.meta_rule_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self.evaluation_tokens = []
        self.evaluation_count = 0

    def set_train_data(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        self.train_x = train_x
        self.train_y = train_y
        
        if not hasattr(self, '_meta_instance'):
            try:
                self._meta_instance = self._meta
            except NotImplementedError:
                logger.log("Error: Subclass must implement _meta attribute")
                raise
                
        self._feature_shuffle_map = getattr(self._meta, 'feature_shuffle_map', None)
        if self._feature_shuffle_map:
            logger.log(f"Feature mapping saved in strategy: {self._feature_shuffle_map}")
        else:
            logger.log("Warning: No feature mapping found")
        
        self.split_values = _get_feature_values(self._meta, train_x, self.hist_nbins)
        self._create_tree(train_x)

    def _create_tree(self, train_x: np.ndarray) -> None:
        self.tree = DecisionTree(self.max_depth, self._meta.categories_map)
        self.tree.set_train_data(train_x)

    def step(self) -> tuple[bool, float]:
        next_split = self.tree.next_to_split()
        
        if next_split is None:
            logger.log("No splittable nodes found, training completed")
            return False, None
        
        depth = next_split.depth
        samples = next_split.get_samples()
        
        if depth >= self.max_depth:
            logger.log(f"Node depth({depth}) reached max depth({self.max_depth}), auto set as leaf")
            
            if samples is not None and len(samples) > 0:
                node_y = self.train_y[samples]
                most_common_label = np.argmax(np.bincount(node_y))
                logger.log(f"Prediction based on samples: {most_common_label}")
            else:
                most_common_label = -1
                logger.log("No samples found, initial set to unknown class(-1)")
            
            verified_prediction = self._llm_verify_leaf_node(next_split, most_common_label)
            logger.log(f"LLM verification result: {verified_prediction}")
            
            next_split.is_leaf = True
            next_split.leaf_class = verified_prediction
            next_split.prediction = verified_prediction
            next_split.freeze()
            
            logger.log(f"Max depth leaf node setup complete: {verified_prediction}")
            return True, None
        
        if samples is None or len(samples) == 0:
            logger.log("Node has no samples, initial set to unknown class(-1)")
            
            initial_prediction = -1
            verified_prediction = self._llm_verify_leaf_node(next_split, initial_prediction)
            
            logger.log(f"LLM verification result(empty node): {verified_prediction}")
            
            next_split.is_leaf = True
            next_split.leaf_class = verified_prediction
            next_split.prediction = verified_prediction
            next_split.freeze()
            
            logger.log(f"Empty node setup complete, final class: {verified_prediction}")
            return True, None
        
        node_y = self.train_y[samples]
        
        if len(set(node_y)) == 1:
            original_prediction = node_y[0]
            logger.log(f"All samples have same label, set as leaf node: {original_prediction}")
            
            verified_prediction = self._llm_verify_leaf_node(next_split, original_prediction)
            logger.log(f"LLM verification result: {verified_prediction}")
            
            next_split.is_leaf = True
            next_split.leaf_class = verified_prediction
            next_split.prediction = verified_prediction
            next_split.freeze()
            
            logger.log(f"Leaf node setup complete: leaf_class={next_split.leaf_class}, is_leaf={next_split.is_leaf}")
            return True, None
        
        unique_labels = np.unique(node_y)
        if len(unique_labels) == 1:
            logger.log(f"All samples have same label, set as leaf node: {unique_labels[0]}")
            next_split.is_leaf = True
            next_split.prediction = unique_labels[0]
            next_split.freeze()
            return True, None
        
        best_meta_rule, best_gain = self._select_meta_rule(next_split, self.meta_rules)
        
        shallow_depth = depth <= 1  
        is_small_sample = len(samples) <= 5

        if best_meta_rule is None:
            logger.log("No available meta rules, freezing node")
            
            most_common_label = np.argmax(np.bincount(node_y))
            logger.log(f"Node initial prediction: {most_common_label}")
            
            verified_prediction = self._llm_verify_leaf_node(next_split, most_common_label)
            logger.log(f"LLM verification result: {verified_prediction}")
            
            next_split.prediction = verified_prediction
            next_split.leaf_class = verified_prediction
            next_split.is_leaf = True
            next_split.freeze()
            
            logger.log(f"Node final setup: {verified_prediction}")
            return True, None
        elif best_gain <= 0:
            if shallow_depth and is_small_sample:
                logger.log(f"Shallow node(depth={depth}), small sample({len(samples)}), using rule: {best_meta_rule}")
            else:
                logger.log(f"Zero gain with non-shallow node(depth={depth}, samples={len(samples)}), freezing node")
                
                most_common_label = np.argmax(np.bincount(node_y))
                logger.log(f"Node initial prediction: {most_common_label}")
                
                verified_prediction = self._llm_verify_leaf_node(next_split, most_common_label)
                logger.log(f"LLM verification result: {verified_prediction}")
                
                next_split.prediction = verified_prediction
                next_split.leaf_class = verified_prediction
                next_split.is_leaf = True
                next_split.freeze()
                
                logger.log(f"Node final setup: {verified_prediction}")
                return True, None

        logger.log(f"Using rule: {best_meta_rule}, Gini gain: {best_gain:.4f}")
        best_feature = best_meta_rule.feature_idx
        split_value = best_meta_rule.split_value
        is_categorical = best_meta_rule.is_categorical
        
        if hasattr(self, '_meta') and self._meta:
            feature_name = self._meta.features[best_feature].name if best_feature < len(self._meta.features) else f"Unknown({best_feature})"
            logger.log(f"Selected feature: {best_feature} ({feature_name}), split: {split_value}, categorical: {is_categorical}")
        
        node_X = self.train_x[samples]
        if is_categorical:
            left_mask = node_X[:, best_feature] == split_value
        else:
            left_mask = node_X[:, best_feature] < split_value
        right_mask = ~left_mask
        
        left_y = node_y[left_mask]
        right_y = node_y[right_mask]
        
        left_class = np.argmax(np.bincount(left_y)) if len(left_y) > 0 else -1
        right_class = np.argmax(np.bincount(right_y)) if len(right_y) > 0 else -1
        
        logger.log(f"Left child label: {left_class}, Right child label: {right_class}")
        
        next_split.split(best_feature, split_value, is_categorical, left_class, right_class)
        
        logger.log(f"Node split complete, feature: {best_feature}, split: {split_value}")
        
        return True, 0.0
    def predict_tree(self, x: np.ndarray) -> list[int]:
        """Make predictions using decision tree"""
        if hasattr(self, 'random_forest'):
            if self.random_forest is None:
                logger.log("Warning: Random forest not initialized, returning default prediction")
                default_value = 0
                return [default_value] * len(x)
        elif hasattr(self, 'tree'):
            if self.tree is None:
                logger.log("Warning: Decision tree not initialized, returning default prediction")
                default_value = 0
                return [default_value] * len(x)
            
        results = self.predict_tree_raw(x)
        for i, res in enumerate(results):
            if res < 0:
                valid_labels = [label.value for label in self._meta.labels]
                selected_label = random.choice(valid_labels)
                logger.log(f"Encountered unknown prediction(-1), randomly selecting label: {selected_label}")
                results[i] = selected_label
        return results

    def predict_tree_raw(self, x: np.ndarray) -> list[int]:
        """Raw tree prediction containing -1 results"""
        if hasattr(self, 'random_forest'):
            if self.random_forest is None:
                logger.log("Warning: Random forest not initialized, returning default prediction")
                return [-1] * len(x)
        elif hasattr(self, 'tree'):
            if self.tree is None:
                logger.log("Warning: Decision tree not initialized, returning default prediction")
                return [-1] * len(x)
                
        ret = []
        if hasattr(self, 'random_forest') and self.random_forest is not None:
            for xx in x:
                predicted_value = self.random_forest.predict_one(xx)
                ret.append(predicted_value)
        elif hasattr(self, 'tree') and self.tree is not None:
            for xx in x:
                predicted_value = self.tree.predict_one(xx)
                ret.append(predicted_value)
        else:
            return [-1] * len(x)
            
        return ret

    def export(self) -> any:
        """Export model with feature mapping information"""
        if hasattr(self, '_feature_shuffle_map') and self._feature_shuffle_map:
            logger.log(f"Exporting feature mapping to JSON: {self._feature_shuffle_map}")
        else:
            logger.log("Warning: No feature mapping information during export")
        
        return {
            "type": self.strategy_type(),
            "model": self.tree.export_dict() if hasattr(self, "tree") else None,
            "args": {
                "max_depth": self.max_depth,
                "feature_shuffle_map": self._feature_shuffle_map if hasattr(self, '_feature_shuffle_map') else None
            },
            "prompt": self._gen_prompt(
                self.examples, (self.train_x, self.train_y)
            ),
        }

    def _assign_leaf_values(self, node, feature_idx, split_value):
        samples = node.get_samples()
        node_x = self.train_x[samples]
        node_y = self.train_y[samples]
        
        unique_labels, label_counts = np.unique(node_y, return_counts=True)
        label_dist = {int(label): count for label, count in zip(unique_labels, label_counts)}
        logger.log(f"Label distribution before assigning leaf values(node {id(node)}): {label_dist}, total samples: {len(samples)}")
        
        if self._meta.features[feature_idx].is_categorical:
            left_mask = node_x[:, feature_idx] == split_value
        else:
            left_mask = node_x[:, feature_idx] < split_value
        
        right_mask = ~left_mask
        
        left_y = node_y[left_mask] if np.any(left_mask) else []
        right_y = node_y[right_mask] if np.any(right_mask) else []
        
        if len(left_y) > 0:
            left_unique, left_counts = np.unique(left_y, return_counts=True)
            left_dist = {int(label): count for label, count in zip(left_unique, left_counts)}
            logger.log(f"Left child label distribution: {left_dist}, samples: {len(left_y)}")
        
        if len(right_y) > 0:
            right_unique, right_counts = np.unique(right_y, return_counts=True)
            right_dist = {int(label): count for label, count in zip(right_unique, right_counts)}
            logger.log(f"Right child label distribution: {right_dist}, samples: {len(right_y)}")
        
        valid_labels = []
        for label_info in self._meta.labels:
            valid_labels.append(label_info.value)
        logger.log(f"Valid label values: {valid_labels}")
        
        if len(left_y) > 0:
            left_label_counts = {}
            for label in left_y:
                label_int = int(label)
                left_label_counts[label_int] = left_label_counts.get(label_int, 0) + 1
            logger.log(f"Left child label counts: {left_label_counts}")
            
            if left_label_counts:
                left_class = max(left_label_counts.items(), key=lambda x: x[1])[0]
                logger.log(f"Left child majority class: {left_class}")
        else:
            left_class = valid_labels[0] if valid_labels else 0
            logger.log(f"Left child empty, using default: {left_class}")

        if len(right_y) > 0:
            right_label_counts = {}
            for label in right_y:
                label_int = int(label)
                right_label_counts[label_int] = right_label_counts.get(label_int, 0) + 1
            logger.log(f"Right child label counts: {right_label_counts}")
            
            if right_label_counts:
                right_class = max(right_label_counts.items(), key=lambda x: x[1])[0]
                logger.log(f"Right child majority class: {right_class}")
        else:
            right_class = valid_labels[0] if valid_labels else 0
            logger.log(f"Right child empty, using default: {right_class}")

        logger.log(f"Final assignment - Left: {left_class}, Right: {right_class}")
        return left_class, right_class

    def _process_categorical_feature(self, feature_idx, node):
        samples = node.get_samples()
        node_data = self.train_x[samples, feature_idx]
        unique_values = np.unique(node_data)
        
        logger.log(f"Feature {feature_idx} value distribution: {unique_values}")
        
        if len(unique_values) < 1:
            logger.log(f"Feature {feature_idx} has no valid values")
            return None
        
        node_labels = self.train_y[samples]
        unique_labels = np.unique(node_labels)
        
        label_counts = {int(label): np.sum(node_labels == label) for label in unique_labels}
        logger.log(f"Node label distribution: {label_counts}")
        
        if len(unique_labels) <= 1:
            logger.log(f"Node already pure with label: {unique_labels[0]}")
            return None
        
        parent_gini = 1.0
        total = len(node_labels)
        label_counts = {}
        for label in node_labels:
            label = int(label)
            label_counts[label] = label_counts.get(label, 0) + 1
        
        for _, count in label_counts.items():
            p = count / total
            parent_gini -= p * p
        
        logger.log(f"Parent node Gini: {parent_gini:.4f}")
        
        best_gain = -float('inf')
        best_value = None
        min_samples_leaf = max(1, int(0.05 * len(node_labels)))
        
        for value in unique_values:
            left_mask = node_data == value
            right_mask = ~left_mask
            
            left_labels = node_labels[left_mask]
            right_labels = node_labels[right_mask]
            
            if len(left_labels) < min_samples_leaf or len(right_labels) < min_samples_leaf:
                logger.log(f"  Value {value} skipped due to insufficient samples")
                continue
            
            left_counts = {int(label): np.sum(left_labels == label) for label in np.unique(left_labels)}
            right_counts = {int(label): np.sum(right_labels == label) for label in np.unique(right_labels)}
            logger.log(f"  Value {value}: left({len(left_labels)}) {left_counts}, right({len(right_labels)}) {right_counts}")
            
            try:
                left_gini = 1.0
                left_total = len(left_labels)
                left_label_counts = {}
                for label in left_labels:
                    label = int(label)
                    left_label_counts[label] = left_label_counts.get(label, 0) + 1
                
                for _, count in left_label_counts.items():
                    p = count / left_total
                    left_gini -= p * p
                    
                right_gini = 1.0
                right_total = len(right_labels)
                right_label_counts = {}
                for label in right_labels:
                    label = int(label)
                    right_label_counts[label] = right_label_counts.get(label, 0) + 1
                
                for _, count in right_label_counts.items():
                    p = count / right_total
                    right_gini -= p * p
                    
                logger.log(f"  Left Gini: {left_gini:.4f}, Right Gini: {right_gini:.4f}")
                
                n_left = len(left_labels)
                n_right = len(right_labels)
                n_total = len(node_labels)
                
                weighted_gini = (n_left/n_total)*left_gini + (n_right/n_total)*right_gini
                gain = parent_gini - weighted_gini
                logger.log(f"  Information gain: {gain:.4f}")
                
                if gain > 0.0001 and gain > best_gain:
                    best_gain = gain
                    best_value = value
                    logger.log(f"  ✓ Current best split: {value}, gain: {gain:.4f}")
            except Exception as e:
                logger.log(f"  Error processing value {value}: {str(e)}")
                continue
        
        if best_value is not None:
            logger.log(f"Final split value: {best_value}, gain: {best_gain:.4f}")
        else:
            if len(unique_values) > 1:
                values, counts = np.unique(node_data, return_counts=True)
                best_value = values[np.argmax(counts)]
                logger.log(f"No valid split found, using most frequent value: {best_value}")
            else:
                logger.log(f"No valid split found")
        
        return best_value
    def _split_node(self, feature_idx, split_value, node):
        samples = node.get_samples()
        node_data = self.train_x[samples, feature_idx]
        node_labels = self.train_y[samples]
        
        label_counts = {}
        for label in node_labels:
            label_int = int(label)
            label_counts[label_int] = label_counts.get(label_int, 0) + 1
        logger.log(f"Label distribution before splitting: {label_counts}")
        
        if self._meta.features[feature_idx].is_categorical:
            left_mask = node_data == split_value
        else:
            left_mask = node_data < split_value
        right_mask = ~left_mask
        
        left_samples = samples[left_mask]
        right_samples = samples[right_mask]
        left_labels = node_labels[left_mask]
        right_labels = node_labels[right_mask]
        
        left_counts = {}
        for label in left_labels:
            label_int = int(label)
            left_counts[label_int] = left_counts.get(label_int, 0) + 1
            
        right_counts = {}
        for label in right_labels:
            label_int = int(label)
            right_counts[label_int] = right_counts.get(label_int, 0) + 1
        
        logger.log(f"Left child label distribution: {left_counts}")
        logger.log(f"Right child label distribution: {right_counts}")

    def _serialize_rule(self, rule):
        if rule is None:
            return None
        
        conditions = []
        for feature_id, condition in rule.conditions.items():
            feature_name = self._meta.features[feature_id].name
            if condition.is_categorical:
                values = list(condition.categories)
                if len(values) == 0:
                    feature_values = []
                    if hasattr(self.tree, 'categories_map') and self.tree.categories_map and feature_id in self.tree.categories_map:
                        feature_values = list(self.tree.categories_map[feature_id])
                    else:
                        for label in self._meta.features[feature_id].labels:
                            feature_values.append(label.name)
                    
                    split_value = None
                    def find_split_value(node):
                        if node is None or node.is_leaf:
                            return None
                        if node.split_feature == feature_id:
                            return node.split_value
                        left_result = find_split_value(node.left_child)
                        if left_result is not None:
                            return left_result
                        return find_split_value(node.right_child)
                    
                    split_value = find_split_value(self.tree.root_node)
                    
                    if split_value is not None:
                        conditions.append(f"{feature_name} != {split_value}")
                    else:
                        conditions.append(f"{feature_name} not in any split values")
                        logger.log(f"Warning: Unable to determine split value for {feature_name}")
                elif len(values) == 1:
                    conditions.append(f"{feature_name} = {values[0]}")
                else:
                    values_str = ", ".join([str(v) for v in values])
                    conditions.append(f"{feature_name} in [{values_str}]")
            else:
                lower_bound = condition.lower
                upper_bound = condition.upper
                
                if lower_bound is not None and upper_bound is not None:
                    conditions.append(f"{lower_bound} ≤ {feature_name} < {upper_bound}")
                elif lower_bound is not None:
                    conditions.append(f"{feature_name} ≥ {lower_bound}")
                elif upper_bound is not None:
                    conditions.append(f"{feature_name} < {upper_bound}")
        
        label_name = None
        rule_value = rule.value
        
        if rule_value == -1:
            label_name = "unknown"
        else:
            for label in self._meta.labels:
                if label.value == rule_value:
                    label_name = label.name
                    break
        
        if label_name is None:
            label_name = f"unknown"
            logger.log(f"Warning: Label value {rule_value} not found")
        
        if conditions:
            return f"IF {' AND '.join(conditions)} THEN {label_name}"
        else:
            return f"{label_name}"

    def _get_path_to_node(self, node):
        path = []
        current = node
        
        while hasattr(current, 'parent') and current.parent is not None:
            parent = current.parent
            if not hasattr(parent, 'split_feature') or parent.split_feature is None:
                break
            
            feature_idx = parent.split_feature
            split_value = parent.split_value
            
            is_left = parent.left_child == current
            
            feature_name = f"Feature {feature_idx}"
            unit_info = ""
            
            if hasattr(self, '_meta') and self._meta and feature_idx < len(self._meta.features):
                feature = self._meta.features[feature_idx]
                feature_name = feature.name
                
                if hasattr(feature, 'desc') and feature.desc:
                    import re
                    unit_match = re.search(r'$(.*?)$$', feature.desc.strip())
                    if unit_match:
                        unit = unit_match.group(1)
                        unit_info = f" ({unit})"
                    else:
                        unit_match = re.search(r'$(.*?)$', feature.desc)
                        if unit_match:
                            unit = unit_match.group(1)
                            unit_info = f" ({unit})"
            
            if hasattr(parent, 'is_categorical') and parent.is_categorical:
                if is_left:
                    rule = f"{feature_name} = {split_value}"
                else:
                    rule = f"{feature_name} != {split_value}"
            else:
                if is_left:
                    rule = f"{feature_name} < {split_value}{unit_info}"
                else:
                    rule = f"{feature_name} >= {split_value}{unit_info}"
            
            path.append(rule)
            current = parent
        
        path.reverse()
        return path

    def _llm_verify_leaf_node(self, node_or_rules, prediction):
        path_rules = None
        if isinstance(node_or_rules, list):
            path_rules = node_or_rules
        else:
            path_rules = self._get_path_to_node(node_or_rules)
        
        if not path_rules:
            logger.log("Cannot retrieve node path rules, setting to unknown class(-1)")
            return -1
        
        feature_descriptions = []
        if hasattr(self, '_meta') and self._meta:
            for i, feature in enumerate(self._meta.features):
                feature_type = "Categorical" if feature.is_categorical else "Numerical"
                desc = feature.desc if hasattr(feature, 'desc') and feature.desc else ""
                feature_desc = f"Feature {i}: {feature.name} (Type: {feature_type}) - {desc}"
                
                if feature.is_categorical and hasattr(feature, 'categories') and feature.categories:
                    feature_desc += "\n    Possible values:"
                    for cat_value, cat_desc in feature.categories.items():
                        feature_desc += f"\n    - {cat_value}: {cat_desc}"
                    
                feature_descriptions.append(feature_desc)
        
        label_descriptions = []
        if hasattr(self, '_meta') and self._meta:
            for label in self._meta.labels:
                description = ""
                if hasattr(label, 'meaning') and label.meaning:
                    description = label.meaning
                elif hasattr(label, 'desc') and label.desc:
                    description = label.desc
                
                label_descriptions.append(f"Label {label.value}: {label.name} - {description}")
        
        prompt = supervision_template.render(
            domain_expertise=self._get_domain_expertise(),
            label_meaning=self._meta.label_meaning if hasattr(self._meta, 'label_meaning') else "Classification evaluation",
            feature_descriptions=feature_descriptions,
            label_descriptions=label_descriptions,
            path_rules=path_rules,
            labels=self._meta.labels
        )
        
        logger.log(f"Sending LLM verification request with path rules: {' AND '.join(path_rules)}")
        
        if not hasattr(self, 'llm_runner') or self.llm_runner is None:
            from ..runner import Runner
            if hasattr(self, '_runner') and isinstance(self._runner, Runner):
                self.llm_runner = self._runner
            else:
                logger.log("LLM runner not configured, skipping verification")
                return prediction
        
        try:
            response_gen = self.llm_runner.run([prompt])
            response_data, token_info = next(response_gen)
            response = response_data[0]
            
            logger.log(f"LLM response: {response}")
            
            confidence_scores = {}
            highest_confidence = 0
            best_label = prediction
            
            for line in response.split('\n'):
                if line.startswith('Label '):
                    parts = line.split(':')
                    if len(parts) >= 2:
                        label_str = parts[0].strip().replace('Label ', '')
                        try:
                            label = int(label_str)
                            confidence_match = re.search(r'(\d+\.\d+|\d+)', parts[1].strip())
                            if confidence_match:
                                confidence = float(confidence_match.group(1))
                                confidence_scores[label] = confidence
                                
                                if confidence > highest_confidence:
                                    highest_confidence = confidence
                                    best_label = label
                        except ValueError:
                            continue

            self.supervision_tokens["prompt"] += token_info["prompt_tokens"]
            self.supervision_tokens["completion"] += token_info["completion_tokens"]
            self.supervision_tokens["total"] += token_info["total_tokens"]
            tree_index = getattr(self, '_tree_index', -1)
            tree_info = f"Tree{tree_index}: " if tree_index >= 0 else ""
            logger.log(f"{tree_info}Supervision tokens - Input: {token_info['prompt_tokens']}, Output: {token_info['completion_tokens']}, Total: {token_info['total_tokens']}")
            
            if highest_confidence >= _threshold and best_label != prediction:
                logger.log(f"LLM suggests label change: {prediction} -> {best_label} (confidence: {highest_confidence})")
                return best_label
            else:
                return prediction
            
        except Exception as e:
            logger.log(f"Error during LLM verification: {e}")
            return prediction

    def _get_domain_expertise(self):
        if hasattr(self, '_meta') and hasattr(self._meta, 'target') and self._meta.target:
            return self._meta.target
        
        if hasattr(self, '_meta') and hasattr(self._meta, 'name') and self._meta.name:
            return f"{self._meta.name} classification"
        
        return "data analysis"

    def get_token_stats(self):
        build_tokens = {
            "prompt": self.supervision_tokens["prompt"] + self.meta_rule_tokens["prompt"],
            "completion": self.supervision_tokens["completion"] + self.meta_rule_tokens["completion"],
            "total": self.supervision_tokens["total"] + self.meta_rule_tokens["total"]
        }
        return {
            "build_tokens": build_tokens,
            "supervision_tokens": self.supervision_tokens,
            "meta_rule_tokens": self.meta_rule_tokens,
            "evaluation_tokens": self.evaluation_tokens
        }

    def reset_token_stats(self):
        self.supervision_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self.meta_rule_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self.evaluation_tokens = []
        self.evaluation_count = 0
        logger.log("Token statistics reset")
        
    def reset_for_new_training(self):
        self.reset_token_stats()
        
        if hasattr(self, "sub_strategies") and self.sub_strategies:
            for sub_strategy in self.sub_strategies:
                sub_strategy.reset_token_stats()
            logger.log(f"Reset token stats for {len(self.sub_strategies)} sub-strategies")
        
        logger.log(f"FeatureBaggingStrategy reset with preserved meta-rule cache")
        return self
class UnknownClassStrategy(TrainStrategy):
    def __init__(
        self,
        runner: Runner,
        master_template: jinja2.Template,
        serializer: Serializer,
        max_depth: int,
        train_batch: int,
        hist_nbins: int = 10,
    ) -> None:
        super().__init__()
        self.runner = runner
        self.master_template = master_template
        self.serializer = serializer
        self._meta_instance = serializer.meta
        self.max_depth = max_depth
        self.train_batch = train_batch
        self.hist_nbins = hist_nbins
        self.llm_runner = runner

    @property
    def _meta(self) -> DatasetMeta:
        return self.serializer.meta

    def _get_available_predictions(self) -> list[int]:
        return [-1, *range(self._meta.label_count())]

    def _gen_prompt(
        self, x: np.ndarray, examples: tuple[np.ndarray, np.ndarray] = None
    ) -> str:
        rules = self._get_tree_rules()
        if rules is None:
            return None

        x_test_str = [self.serializer.serialize(xx, None) for xx in x]
        examples_str = [self.serializer.serialize(x, y) for x, y in zip(*examples)] if examples else []

        prompt = self.master_template.render(
            meta=self._meta,
            examples=examples_str,
            rules=rules,
            format_desc=self.serializer.format_desc(),
            prediction_intro=self.serializer.answer_requirement(len(x_test_str)),
            tests=x_test_str,
            unknown_correction_example=self.serializer.unknown_correction_example(),
        )

        return prompt

    def _predict_llm_with_tree_batched(
        self, prompts: list[str], expected_lens: list[int]
    ) -> list[list[int]]:
        all_results = []
        for i, (resp_candidates, token_info) in enumerate(self.runner.run(prompts)):
            for resp in resp_candidates:
                results = self.serializer.answer_decoder.decode(resp)
                results = [self._meta.get_label_value(r) for r in results]
            
            self.evaluation_tokens.append(token_info)
            tree_index = getattr(self, '_tree_index', -1)
            tree_info = f"Tree{tree_index}: " if tree_index >= 0 else ""
            logger.log(f"{tree_info}Evaluation tokens - Input: {token_info['prompt_tokens']}, Output: {token_info['completion_tokens']}, Total: {token_info['total_tokens']}")

            if None in results or len(results) != expected_lens[i]:
                continue

            all_results.append(results)
            break
        else:
            logger.log("No valid response found")
            return None

        return all_results

    def _get_tree_rules(self) -> list[str]:
        rules: list[str] = []
        paths: list[RulePath] = self.tree.export_paths()
        if not paths:
            return None
            
        for r in paths:
            processed_rule = self._serialize_rule(r)
            if processed_rule:
                rules.append((processed_rule, len(r.conditions)))

        return [rule[0] for rule in sorted(rules, key=lambda x: x[1])]

    def export(self) -> any:
        return {
            "type": "unknown_class",
            "model": self.tree.export_nodes_dict(),
            "args": {"max_depth": self.max_depth, "categories": None},
            "prompt": self._gen_prompt([], (self.train_x, self.train_y)),
        }

    @staticmethod
    def load(
        model_dict: dict,
        runner: Runner = None,
        master_template: jinja2.Template = None,
        serializer: Serializer = None,
    ) -> "UnknownClassStrategy":
        categories_map = {int(k): set(v) for k, v in model_dict["args"]["categories"].items()}
        tree = DecisionTree.load_nodes_dict(
            model_dict["model"], model_dict["args"]["max_depth"], categories_map
        )
        strategy = UnknownClassStrategy(
            runner=runner,
            master_template=master_template,
            serializer=serializer,
            max_depth=tree.max_depth,
            train_batch=1024,
            hist_nbins=1024,
        )
        strategy.tree = tree
        return strategy

    def get_tree(self) -> DecisionTree:
        return self.tree

    def _create_meta_rules_prompt(self, max_depth: int) -> str:
        num_rules_required = max(10, 2**max_depth - 1)
        train_examples = []
        
        if self.train_x is not None and self.train_y is not None:
            sample_count = min(20, len(self.train_y))
            indices = list(range(len(self.train_y)))
            
            if len(indices) > sample_count:
                random.shuffle(indices)
                indices = indices[:sample_count]
            
            for idx in indices:
                x_row = self.train_x[idx]
                y_val = self.train_y[idx]
                feature_str = []
                
                for f_idx, f_val in enumerate(x_row):
                    feature = self._meta.features[f_idx]
                    if feature.is_categorical:
                        cat_value = str(f_val)
                        if f_val in feature.categories:
                            cat_value = f_val
                        elif isinstance(f_val, (int, float)) and feature.categories:
                            cat_keys = list(feature.categories.keys())
                            if f_val < len(cat_keys):
                                cat_value = cat_keys[int(f_val)]
                        feature_str.append(f"{feature.name}={cat_value}")
                    else:
                        feature_str.append(f"{feature.name}={f_val}")
                
                label_name = next((label.name for label in self._meta.labels if label.value == y_val), "unknown")
                train_examples.append(f"{', '.join(feature_str)} ⟶ {label_name}")

        template = jinja2.Environment(loader=jinja2.FileSystemLoader('./template')).get_template('meta_rule.jinja')
        return template.render(
            meta=self._meta, 
            num_rules_required=num_rules_required,
            train_examples=train_examples
        )

    def _get_meta_rules(self, max_depth: int, runner: Runner = None) -> list[MetaRule]:
        tree_index = getattr(self, '_tree_index', -1)
        cache_key = f"{self._meta.name}_tree{tree_index}_meta_rules_{max_depth}" if tree_index >=0 else f"{self._meta.name}_meta_rules_{max_depth}"
        
        if not hasattr(self, '_instance_cached_meta_rules'):
            self._instance_cached_meta_rules = {}
            
        if cache_key in self._instance_cached_meta_rules:
            logger.log(f"Using cached meta rules: {len(self._instance_cached_meta_rules[cache_key])} rules")
            return self._instance_cached_meta_rules[cache_key]
        
        runner = runner or self.runner
        prompt = self._create_meta_rules_prompt(max_depth)
        meta_rules = []

        for responses, token_info in runner.run([prompt]):
            for response in responses:
                for line in response.strip().split('\n'):
                    if parsed_rule := MetaRule.parse_rule(line.strip(), self._meta):
                        meta_rules.append(parsed_rule)
        
        meta_rules.sort(key=lambda x: x.confidence, reverse=True)
        self._instance_cached_meta_rules[cache_key] = meta_rules
        
        self.meta_rule_tokens["prompt"] += token_info["prompt_tokens"]
        self.meta_rule_tokens["completion"] += token_info["completion_tokens"]
        self.meta_rule_tokens["total"] += token_info["total_tokens"]
        logger.log(f"Meta rule prompt tokens - Input: {token_info['prompt_tokens']}, Output: {token_info['completion_tokens']}")

        return meta_rules

    def _select_meta_rule(self, node, meta_rules: list[MetaRule], mu: int = 2) -> tuple[MetaRule, float]:
        mu = _mu
        samples = node.get_samples()
        is_small_sample = len(samples) <= 5
        used_features = set()

        current = node
        while hasattr(current, 'parent') and current.parent:
            if hasattr(current.parent, 'split_feature'):
                used_features.add(current.parent.split_feature)
            current = current.parent

        if not meta_rules:
            return None, 0.0

        max_confidence = meta_rules[0].confidence
        best_gain, best_rule, best_confidence = -1, None, -1
        best_small_sample_rule, best_small_sample_confidence = None, -1

        for level in itertools.count(0):
            min_confidence = max_confidence - mu - level * (mu + 1)
            if min_confidence < 0:
                break

            usable_rules = [
                rule for rule in meta_rules
                if rule.feature_idx not in used_features
                and min_confidence <= rule.confidence <= (max_confidence - level * (mu + 1))
            ]

            if not usable_rules:
                continue

            has_non_zero_gain = False
            for rule in usable_rules:
                gain, left_gini, right_gini, left_mask, right_mask = calculate_meta_rule_gini(
                    rule, self.train_x, self.train_y, samples
                )

                logger.log(f"Rule '{rule}' Gini gain: {gain:.4f}")
                logger.log(f"  Left: samples={np.sum(left_mask)}, Gini={left_gini:.4f}")
                logger.log(f"  Right: samples={np.sum(right_mask)}, Gini={right_gini:.4f}")

                if is_small_sample and (np.sum(left_mask) == 0 or np.sum(right_mask) == 0):
                    if rule.confidence > best_small_sample_confidence:
                        best_small_sample_rule, best_small_sample_confidence = rule, rule.confidence

                if not is_small_sample and (np.sum(left_mask) == 0 or np.sum(right_mask) == 0):
                    continue

                if gain > 0:
                    has_non_zero_gain = True

                if gain > best_gain or (gain == best_gain and rule.confidence > best_confidence):
                    best_gain, best_rule, best_confidence = gain, rule, rule.confidence

            if has_non_zero_gain:
                break

        if is_small_sample and not best_rule and best_small_sample_rule:
            logger.log(f"Small sample case: Using rule {best_small_sample_rule}")
            return best_small_sample_rule, 0.001

        if best_rule:
            logger.log(f"Best meta rule: {best_rule} with gain {best_gain:.4f}")
        else:
            logger.log("No valid meta rules found")

        return best_rule, best_gain
    def set_train_data(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        super().set_train_data(train_x, train_y)
        
        self.meta_rules = self._get_meta_rules(self.max_depth, self.runner)
        logger.log(f"Obtained meta rules count: {len(self.meta_rules)}")
        for rule in self.meta_rules[:10]:
            logger.log(f"Rule: {rule}")

    def reset_for_new_training(self):
        self.tree = None
        self.reset_token_stats()
        logger.log(f"UnknownClassStrategy reset for new training")
        return self


class KnownClassStrategy(UnknownClassStrategy):
    def __init__(
        self,
        runner: Runner,
        master_template: jinja2.Template,
        serializer: Serializer,
        max_depth: int,
        train_batch: int,
        hist_nbins: int = 10,
    ) -> None:
        super().__init__(
            runner, 
            master_template, 
            serializer, 
            max_depth, 
            train_batch, 
            hist_nbins
        )

    def _get_available_predictions(self) -> list[int]:
        return [*range(self._meta.label_count())]


class FeatureBaggingStrategy(TrainStrategy):
    def __init__(
        self,
        all_meta: DatasetMeta,
        runner: Runner,
        template: jinja2.Template,
        serializer_type: str,
        num_trees: int,
        max_depth: int,
        train_batch: int,
        hist_nbins: int = 10,
    ) -> None:
        super().__init__()
        self.runner = runner
        self.template = template
        self.all_meta = all_meta
        self.max_depth = max_depth
        self.train_batch = train_batch
        self.hist_nbins = hist_nbins

        self.supervision_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self.meta_rule_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self.evaluation_tokens = []
        self.evaluation_count = 0

        self._feature_shuffle_map = getattr(all_meta, 'feature_shuffle_map', {})
        if self._feature_shuffle_map:
            logger.log(f"Initial feature mapping: {self._feature_shuffle_map}")
        
        self.trees = []
        self.trees_active = []
        self.random_forest = None

        self.num_trees = min(num_trees, self.all_meta.feature_count())

        def _split(a, n):
            k, m = divmod(len(a), n)
            return (
                a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
            )

        self.feature_groups = list(
            _split(range(self.all_meta.feature_count()), self.num_trees)
        )
        self.feature_groups = [list(x) for x in self.feature_groups]

        if serializer_type == "tabular":
            self.serializer = TabularSerializer(all_meta)
        elif serializer_type == "list":
            self.serializer = ListSerializer(all_meta)
        elif serializer_type == "text":
            self.serializer = TextSerializer(all_meta)

        self.sub_metas = []
        for i, feature_idxes in enumerate(self.feature_groups):
            sub_meta = DatasetMeta(
                name=self.all_meta.name,
                desc=self.all_meta.desc,
                target=self.all_meta.target,
                label_meaning=self.all_meta.label_meaning,
                features=[self.all_meta.features[idx] for idx in feature_idxes],
                labels=self.all_meta.labels
            )
            
            if hasattr(self.all_meta, 'feature_shuffle_map') and self.all_meta.feature_shuffle_map:
                sub_meta.feature_shuffle_map = {}
                for local_idx, global_idx in enumerate(feature_idxes):
                    if global_idx in self.all_meta.feature_shuffle_map:
                        sub_meta.feature_shuffle_map[local_idx] = self.all_meta.feature_shuffle_map[global_idx]
                    else:
                        sub_meta.feature_shuffle_map[local_idx] = global_idx
                        
                logger.log(f"Tree {i+1} feature map: {sub_meta.feature_shuffle_map}")
            
            self.sub_metas.append(sub_meta)

        self.sub_strategies: list[UnknownClassStrategy] = []
        for i in range(self.num_trees):
            if serializer_type == "tabular":
                serializer = TabularSerializer(self.sub_metas[i])
            elif serializer_type == "list":
                serializer = ListSerializer(self.sub_metas[i])
            elif serializer_type == "text":
                serializer = TextSerializer(self.sub_metas[i])

            self.sub_strategies.append(
                UnknownClassStrategy(
                    runner=runner,
                    master_template=template,
                    serializer=serializer,
                    max_depth=max_depth,
                    train_batch=train_batch,
                    hist_nbins=hist_nbins,
                )
            )
            self.sub_strategies[i]._tree_index = i
        
        self.trees_active = [True] * self.num_trees

    def set_train_data(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        self.train_x = train_x
        self.train_y = train_y
        
        logger.log(f"Main training data shape: x={train_x.shape}, y={train_y.shape}")
        
        self.trees = []
        self.trees_active = [True] * self.num_trees
        self.random_forest = None

        for i, feature_group in enumerate(self.feature_groups):
            if hasattr(self.sub_strategies[i], 'tree'):
                self.sub_strategies[i].tree = None
            
            sub_x = train_x[:, feature_group]
            logger.log(f"Sub-strategy {i+1} data shape: x={sub_x.shape}, features: {feature_group}")
            
            self.sub_strategies[i].set_train_data(sub_x, train_y)
            
            global_feature_names = [self.all_meta.features[idx].name for idx in feature_group]
            logger.log(f"Sub-strategy {i+1} features: {global_feature_names}")
            
            child_meta_rules = self.sub_strategies[i].meta_rules
            
            feature_name_to_local_idx = {}
            for local_idx, global_idx in enumerate(feature_group):
                feature_name = self.all_meta.features[global_idx].name
                feature_name_to_local_idx[feature_name] = local_idx
            
            self.sub_strategies[i]._feature_name_to_local_idx = feature_name_to_local_idx
            logger.log(f"Sub-strategy {i+1} feature map: {feature_name_to_local_idx}")
            
            filtered_rules = []
            for rule in child_meta_rules:
                if hasattr(rule, 'feature_name') and rule.feature_name:
                    feature_name = rule.feature_name
                else:
                    continue
                
                if feature_name in feature_name_to_local_idx:
                    local_idx = feature_name_to_local_idx[feature_name]
                    new_rule = MetaRule(
                        feature_idx=local_idx,
                        feature_name=feature_name,
                        split_value=rule.split_value,
                        is_categorical=rule.is_categorical,
                        confidence=rule.confidence
                    )
                    filtered_rules.append(new_rule)
            
            self.sub_strategies[i].meta_rules = filtered_rules
            logger.log(f"Sub-strategy {i+1}: Filtered {len(filtered_rules)}/{len(child_meta_rules)} rules")
            
            for r in filtered_rules[:3]:
                logger.log(f"  Rule: {r}")

        if self._feature_shuffle_map:
            logger.log(f"Feature shuffle map preserved: {self._feature_shuffle_map}")

    def step(self) -> tuple[bool, list[float]]:
        losses = []
        continue_step = False

        if len(self.trees) != len(self.sub_strategies):
            self.trees = [None] * len(self.sub_strategies)
            logger.log(f"Initialized {len(self.trees)} sub-trees")

        for i, sub_strategy in enumerate(tqdm(self.sub_strategies, desc="Trees")):
            if not self.trees_active[i]:
                continue
            logger.log(
                f"Training tree {i + 1}/{self.num_trees} (features: {self.feature_groups[i]})"
            )
            this_continue_step, loss = sub_strategy.step()
            self.trees_active[i] = this_continue_step
            continue_step = continue_step or this_continue_step
            losses.append(loss)
            
            self.trees[i] = sub_strategy.get_tree()
            logger.log(f"Updated random forest sub-tree {i+1}")
            
            if not this_continue_step:
                logger.log(f"Tree {i+1} training completed")

        if all(tree is not None for tree in self.trees):
            self.random_forest = RandomForest(
                self.trees, self.feature_groups, len(self.all_meta.labels)
            )
            logger.log(f"Rebuilt random forest with {len(self.trees)} trees")
        else:
            uninitialized = [i for i, tree in enumerate(self.trees) if tree is None]
            logger.log(f"Uninitialized sub-trees at indices {uninitialized}")

        return continue_step, losses
    def predict_llm_with_tree(
        self, x: np.ndarray, with_examples: bool = False
    ) -> list[int]:
        prompt = self._gen_prompt(
            x,
            self.sub_strategies,
            (self.train_x, self.train_y) if with_examples else None,
        )
        if not prompt:
            raise RuntimeError("Invalid tree rules!")

        for resp_candidates, token_info in self.runner.run([prompt]):
            for resp in resp_candidates:
                results = self.serializer.answer_decoder.decode(resp)
                results = [self.all_meta.get_label_value(r) for r in results]

                if None in results or len(results) != len(x):
                    continue

                self.evaluation_tokens.append(token_info)
                logger.log(f"Random forest evaluation tokens - Input: {token_info['prompt_tokens']}, Output: {token_info['completion_tokens']}, Total: {token_info['total_tokens']}")
                return results
            else:
                logger.log(f"No valid response found in: {resp_candidates}")
                return None

    def _get_tree_rules(self, sub_strategies: list[UnknownClassStrategy]) -> list[str]:
        all_rules = []
        
        for i, sub_strategy in enumerate(sub_strategies):
            if tree_rules := sub_strategy._get_tree_rules():
                all_rules.extend([f"Tree {i+1}: {rule}" for rule in tree_rules])
        
        return all_rules

    def _gen_prompt(
        self,
        x: np.ndarray,
        sub_strategies: list[TrainStrategy],
        examples: tuple[np.ndarray, np.ndarray] = None,
    ) -> str:
        rules = self._get_tree_rules(sub_strategies)
        if not rules:
            return None

        x_test_str = [self.serializer.serialize(xx, None) for xx in x]
        examples_str = [self.serializer.serialize(x, y) for x, y in zip(*examples)] if examples else []

        return self.template.render(
            meta=self.all_meta,
            examples=examples_str,
            rules=rules,
            format_desc=self.serializer.format_desc(),
            prediction_intro=self.serializer.answer_requirement(len(x_test_str)),
            tests=x_test_str,
            unknown_correction_example=self.serializer.unknown_correction_example(),
        )

    def get_tree(self) -> TreeBase:
        return self.random_forest

    @property
    def _meta(self) -> DatasetMeta:
        return self.all_meta

    def export(self) -> any:
        if hasattr(self, '_feature_shuffle_map') and self._feature_shuffle_map:
            logger.log(f"Exporting feature map: {self._feature_shuffle_map}")
        
        return {
            "type": "random_forest",
            "model": self.random_forest.export_dict(),
            "args": {
                "max_depth": self.max_depth, 
                "num_trees": self.num_trees,
                "feature_shuffle_map": getattr(self, '_feature_shuffle_map', None)
            },
            "prompt": self._gen_prompt([], self.sub_strategies, (self.train_x, self.train_y)),
        }

    def _merge_token_stats_from_sub_strategies(self):
        current_evaluation_tokens = self.evaluation_tokens.copy()
        self.supervision_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self.meta_rule_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self.evaluation_tokens = []
        
        logger.log("Merging sub-strategy token stats...")
        
        for i, sub_strategy in enumerate(self.sub_strategies):
            self.supervision_tokens["prompt"] += sub_strategy.supervision_tokens["prompt"]
            self.supervision_tokens["completion"] += sub_strategy.supervision_tokens["completion"]
            self.supervision_tokens["total"] += sub_strategy.supervision_tokens["total"]
            
            self.meta_rule_tokens["prompt"] += sub_strategy.meta_rule_tokens["prompt"]
            self.meta_rule_tokens["completion"] += sub_strategy.meta_rule_tokens["completion"]
            self.meta_rule_tokens["total"] += sub_strategy.meta_rule_tokens["total"]
            
            self.evaluation_tokens.extend(sub_strategy.evaluation_tokens)
            
            logger.log(f"Sub-strategy {i+1} tokens: "
                    f"meta={sub_strategy.meta_rule_tokens['total']}, "
                    f"supervision={sub_strategy.supervision_tokens['total']}")

        self.evaluation_tokens.extend(current_evaluation_tokens)

    def get_token_stats(self):
        self._merge_token_stats_from_sub_strategies()
        
        build_tokens = {
            "prompt": self.supervision_tokens["prompt"] + self.meta_rule_tokens["prompt"],
            "completion": self.supervision_tokens["completion"] + self.meta_rule_tokens["completion"],
            "total": self.supervision_tokens["total"] + self.meta_rule_tokens["total"]
        }
        
        logger.log(f"Forest token usage: meta={self.meta_rule_tokens['total']}, supervision={self.supervision_tokens['total']}")
        
        return {
            "build_tokens": build_tokens,
            "supervision_tokens": self.supervision_tokens,
            "meta_rule_tokens": self.meta_rule_tokens,
            "evaluation_tokens": self.evaluation_tokens
        }

    def _get_label_name(self, rule_value):
        if rule_value < 0:
            return "unknown"
        
        label_name = next((label.name for label in self._meta.labels if label.value == rule_value), None)
        if not label_name:
            label_name = "unknown"
            logger.log(f"Warning: Label value {rule_value} not found")
        
        return label_name

    def reset_for_new_training(self):
        self.reset_token_stats()
        
        if hasattr(self, "sub_strategies") and self.sub_strategies:
            for sub_strategy in self.sub_strategies:
                sub_strategy.reset_token_stats()
            logger.log(f"Reset token stats for {len(self.sub_strategies)} sub-strategies")
        
        logger.log("FeatureBaggingStrategy reset with meta-rule cache")
        return self


def clean_llm_response(response):
    if not response:
        return None
    cleaned = re.sub(r'<think>.*?', '', response, flags=re.DOTALL)
    return cleaned.lstrip('\n')