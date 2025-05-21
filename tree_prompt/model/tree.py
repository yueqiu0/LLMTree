import numpy as np
from typing import Set
from .. import logger


class Condition:
    def __init__(self) -> None:
        self.feature: int = None
        self.lower: float = None
        self.upper: float = None
        self.categories: set[str | int] = None

    @property
    def is_categorical(self) -> bool:
        return self.categories is not None

    @staticmethod
    def numerical(feature: int, lower: float, upper: float) -> "Condition":
        cond = Condition()
        cond.feature = feature
        cond.lower = lower
        cond.upper = upper
        return cond

    @staticmethod
    def categorical(feature: int, categories: set[str | int]) -> "Condition":
        cond = Condition()
        cond.feature = feature
        cond.categories = categories
        return cond

    def merged(self, other: "Condition") -> "Condition":
        assert self.feature == other.feature

        if self.is_categorical:
            intersection = self.categories.intersection(other.categories)
            if not intersection or len(intersection) == len(self.categories):
                return None
            return Condition.categorical(self.feature, intersection)

        if self.lower is None:
            lower = other.lower
        elif other.lower is None:
            lower = self.lower
        elif other.lower <= self.lower:
            return None
        else:
            lower = other.lower
            
        if self.upper is None:
            upper = other.upper
        elif other.upper is None:
            upper = self.upper
        elif other.upper >= self.upper:
            return None
        else:
            upper = other.upper

        if upper and lower and upper <= lower:
            return None

        return Condition.numerical(self.feature, lower, upper)

    def __repr__(self) -> str:
        return self.__dict__.__repr__()


class RulePath:
    def __init__(self) -> None:
        self.conditions: dict[int, Condition] = {}
        self.value = None

    @staticmethod
    def from_conditions(conditions: list[Condition], value: float) -> "RulePath":
        rule = RulePath()
        rule.value = value
        
        for cond in conditions:
            if cond.feature not in rule.conditions:
                rule.conditions[cond.feature] = cond
            elif not rule.conditions[cond.feature]:
                return None
            else:
                merged = rule.conditions[cond.feature].merged(cond)
                if not merged:
                    return None
                rule.conditions[cond.feature] = merged
        
        return rule

    def __repr__(self) -> str:
        return self.__dict__.__repr__()


class Node:
    def __init__(self, parent: "Node", depth: int, leaf_class: int = None) -> None:
        self.depth: int = depth
        self.parent: Node = parent

        self.leaf_class: int = leaf_class
        self.is_leaf: bool = leaf_class is not None
        self.split_value: float | int | str = None
        self.split_feature: int = None
        self.is_categorical: bool = None
        self.left_child: Node = None
        self.right_child: Node = None

        self.freezed: bool = False
        self.samples = np.array([], dtype=int)
        self.used_features = set()
        if parent:
            self.used_features = parent.used_features.copy()

    def get_samples(self) -> np.ndarray:
        return self.samples if self.samples is not None else np.array([], dtype=int)

    def get_used_features(self) -> Set[int]:
        return self.used_features

    def split(
        self,
        feat_idx: int,
        feat_value: float | int | str,
        is_categorical: bool,
        left_class: int,
        right_class: int,
    ) -> None:
        self.is_leaf = False
        self.used_features.add(feat_idx)

        self.left_child = Node(self, self.depth + 1, leaf_class=left_class)
        self.right_child = Node(self, self.depth + 1, leaf_class=right_class)
        self.split_value = feat_value
        self.split_feature = feat_idx
        self.is_categorical = is_categorical

    def unsplit(self) -> None:
        self.is_leaf = True
        self.left_child = None
        self.right_child = None
        self.split_value = None
        self.split_feature = None
        self.is_categorical = None

    def freeze(self) -> None:
        self.freezed = True
        if self.is_leaf and self.leaf_class == -1:
            logger.log("Freezing unknown class (-1) leaf node")
        else:
            logger.log(f"Freezing node with prediction: {self.leaf_class}")

    def next_node(self, feat_value: float | str | int) -> "Node":
        if self.is_leaf:
            return None
        if not self.is_categorical:
            return self.left_child if feat_value < self.split_value else self.right_child

        assert type(feat_value) == type(self.split_value)
        return self.left_child if feat_value == self.split_value else self.right_child

    def _dfs(self, on_node, callback_after_recurse=None):
        if self is None:
            return
        
        on_node(self)
        
        if not self.is_leaf:
            if self.left_child is not None:
                self.left_child._dfs(on_node, callback_after_recurse)
            if self.right_child is not None:
                self.right_child._dfs(on_node, callback_after_recurse)
        
        if callback_after_recurse:
            callback_after_recurse(self)

    def predict(self, x: np.ndarray) -> int:
        if self.is_leaf:
            return self.leaf_class
        
        feature_value = x[self.split_feature]
        if self.is_categorical:
            if feature_value == self.split_value:
                return self.left_child.predict(x) if self.left_child else -1
            return self.right_child.predict(x) if self.right_child else -1
        else:
            if feature_value < self.split_value:
                return self.left_child.predict(x) if self.left_child else -1
            return self.right_child.predict(x) if self.right_child else -1

    def to_graphviz_node(self, node_id: str, features: list[str] = None, classes: list[str] = None) -> str:
        if self.is_leaf:
            if self.leaf_class == -1:
                return f'  {node_id} [label="-1"]'
            
            label = str(self.leaf_class)
            if classes and 0 <= self.leaf_class < len(classes):
                label = classes[self.leaf_class]
            return f'  {node_id} [label="{label}"]'
        
        feature_name = str(self.split_feature)
        if features and 0 <= self.split_feature < len(features):
            feature_name = features[self.split_feature]
        
        if self.is_categorical:
            return f'  {node_id} [label="{feature_name} == {self.split_value}?"]'
        else:
            return f'  {node_id} [label="{feature_name} < {self.split_value}?"]'

    @property
    def prediction(self) -> int:
        return self.leaf_class

    @prediction.setter
    def prediction(self, value: int) -> None:
        self.leaf_class = value
        logger.log(f"Setting node prediction to: {value}")


class TreeBase:
    def predict_one(self, x: np.ndarray) -> int:
        raise NotImplementedError()

    def to_graphviz(self, features: list[str] = None, classes: list[str] = None):
        import graphviz
        return graphviz.Source(self.to_graphviz_source(features, classes))

    def to_graphviz_source(self, features: list[str] = None, 
                         classes: list[str] = None, graph_name: str = None) -> str:
        raise NotImplementedError()


class DecisionTree(TreeBase):
    def __init__(self, max_depth: int, categories_map: dict[int, set[str]]) -> None:
        self.max_depth = max_depth
        self.categories_map = categories_map
        self.root_node = Node(None, 1)
        self.train_x = None

    def set_train_data(self, train_x: np.ndarray) -> None:
        self.train_x = train_x
        self.root_node.samples = np.arange(len(train_x))

    def _go_left(self, x: np.ndarray, node: Node) -> bool:
        if node.is_leaf:
            return False
            
        feature = node.split_feature
        split_value = node.split_value
        
        if split_value is None:
            return False
        
        if node.is_categorical:
            return x[feature] == split_value
        else:
            try:
                return float(x[feature]) < float(split_value)
            except (ValueError, TypeError):
                logger.log(f"Warning: Cannot compare feature value {x[feature]} with split {split_value}")
                return False

    def next_to_split(self) -> Node:
        def _dfs(node: Node) -> Node:
            if not node or node.freezed:
                return None
                
            if node.is_leaf:
                if hasattr(self, 'train_y') and self.train_y is not None:
                    samples = node.get_samples()
                    if samples.size > 0:
                        labels = self.train_y[samples]
                        unique_labels, counts = np.unique(labels, return_counts=True)
                        logger.log(f"Leaf node samples: {dict(zip(unique_labels.astype(int), counts))}, Prediction: {node.leaf_class}")
                return node
            
            if node.split_value is None:
                if hasattr(self, 'train_y') and self.train_y is not None:
                    samples = node.get_samples()
                    if samples.size > 0:
                        labels = self.train_y[samples]
                        unique_labels, counts = np.unique(labels, return_counts=True)
                        logger.log(f"Unsplitted node samples: {dict(zip(unique_labels.astype(int), counts))}")
                return node
            
            left_samples = []
            right_samples = []
            
            for idx in node.get_samples():
                try:
                    if self._go_left(self.train_x[idx], node):
                        left_samples.append(idx)
                    else:
                        right_samples.append(idx)
                except Exception as e:
                    logger.log(f"Error assigning sample: {e}")
                    continue

            if left_samples:
                node.left_child.samples = np.array(left_samples)
            if right_samples:
                node.right_child.samples = np.array(right_samples)
            
            left_result = _dfs(node.left_child)
            if left_result:
                return left_result
            
            return _dfs(node.right_child)
        
        return _dfs(self.root_node)
    def predict_one(self, x: np.ndarray) -> int:
        node = self.root_node
        if not node:
            return -1
        
        while node and not node.is_leaf:
            if self._go_left(x, node):
                node = node.left_child
            else:
                node = node.right_child
            
            if not node:
                return -1
            
        return node.leaf_class if node else -1

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = np.zeros(x.shape[0], dtype=int)
        
        for i in range(x.shape[0]):
            node = self.root_node
            if not node:
                predictions[i] = -1
                continue
            
            while node and not node.is_leaf:
                if self._go_left(x[i], node):
                    node = node.left_child
                else:
                    node = node.right_child
                
                if not node:
                    predictions[i] = -1
                    break
                
            predictions[i] = node.leaf_class if node else -1
        
        return predictions

    def predict_raw(self, x: np.ndarray) -> list[int]:
        return self.predict(x)

    def export_paths(self) -> list[RulePath]:
        paths = []
        
        def _dfs(node: Node, conditions: list[Condition] = None) -> None:
            if not conditions:
                conditions = []
                
            if node.is_leaf:
                if path := RulePath.from_conditions(conditions, node.leaf_class):
                    paths.append(path)
                return
                
            if not node.split_feature:
                return
                
            if node.is_categorical:
                left_cond = Condition.categorical(node.split_feature, {node.split_value})
                right_cond = Condition.categorical(node.split_feature, set())
            else:
                left_cond = Condition.numerical(node.split_feature, None, node.split_value)
                right_cond = Condition.numerical(node.split_feature, node.split_value, None)
            
            if node.left_child:
                _dfs(node.left_child, conditions + [left_cond])
            if node.right_child:
                _dfs(node.right_child, conditions + [right_cond])
        
        _dfs(self.root_node)
        return paths

    def get_rules(self) -> list[RulePath]:
        rules = []
        
        def _collect_rules(node, conditions, collect_leaf):
            if not node:
                return
            
            if node.is_leaf and collect_leaf:
                if node.leaf_class is not None:
                    logger.log(f"Collected leaf node rule with label: {node.leaf_class}")
                    if rule := RulePath.from_conditions(conditions.copy(), node.leaf_class):
                        rules.append(rule)
                return
            
            if not node.split_feature:
                return
            
            if node.is_categorical:
                left_cond = Condition.categorical(node.split_feature, {node.split_value})
                right_values = self.categories_map[node.split_feature] - {node.split_value}
                right_cond = Condition.categorical(node.split_feature, right_values)
            else:
                left_cond = Condition.numerical(node.split_feature, None, node.split_value)
                right_cond = Condition.numerical(node.split_feature, node.split_value, None)
            
            conditions.append(left_cond)
            _collect_rules(node.left_child, conditions, collect_leaf)
            conditions.pop()
            
            conditions.append(right_cond)
            _collect_rules(node.right_child, conditions, collect_leaf)
            conditions.pop()
        
        _collect_rules(self.root_node, [], True)
        logger.log(f"Total rules collected: {len(rules)}")
        return rules

    def export_nodes_dict(self) -> dict:
        return self._export_nodes(self.root_node)
        
    def export(self) -> dict:
        return {
            "nodes": self._export_nodes(self.root_node),
            "args": {
                "max_depth": self.max_depth,
                "categories": {str(k): list(v) for k, v in self.categories_map.items()},
            },
        }

    def _export_nodes(self, node: Node) -> dict:
        if not node:
            return None
            
        return {
            "feature": node.split_feature if not node.is_leaf else None,
            "split_value": node.split_value if not node.is_leaf else None,
            "prediction": node.leaf_class if node.is_leaf else None,
            "left": self._export_nodes(node.left_child),
            "right": self._export_nodes(node.right_child),
        }

    @staticmethod
    def load_nodes_dict(nodes_dict: dict, max_depth: int, categories_map: dict) -> "DecisionTree":
        tree = DecisionTree(max_depth, categories_map)
        
        def _load_node(node_dict: dict) -> Node:
            if not node_dict:
                return None
                
            node = Node(None, 1)
            node.split_feature = node_dict["feature"]
            node.split_value = node_dict["split_value"]
            node.leaf_class = node_dict["prediction"]
            node.left_child = _load_node(node_dict["left"])
            node.right_child = _load_node(node_dict["right"])
            return node
            
        tree.root_node = _load_node(nodes_dict)
        return tree

    def to_graphviz_source(self, features: list[str] = None, 
                          classes: list[str] = None, graph_name: str = None) -> str:
        lines = []
        self.root_node._id = "0"

        def on_node(node: Node):
            if not node.is_leaf:
                if node.left_child:
                    node.left_child._id = node._id + "1"
                    lines.append(f"{node._id} -> {node.left_child._id} [label=yes]")
                if node.right_child:
                    node.right_child._id = node._id + "2"
                    lines.append(f"{node._id} -> {node.right_child._id} [label=no]")

            node_label = (f"{classes[node.leaf_class]}" if node.is_leaf and classes else 
                         f"{node.leaf_class}" if node.is_leaf else
                         f"{features[node.split_feature] if features else node.split_feature}"
                         f"{' == ' if node.is_categorical else ' < '}{node.split_value}?")
            lines.append(f'{node._id} [label="{node_label}"]')

        self.root_node._dfs(on_node)
        return f"digraph {graph_name or 'G'} {{\n  " + "\n  ".join(lines) + "\n}}"

    def _assign_node_label(self, node):
        samples = node.get_samples()
        if not samples.size:
            return
        
        labels = self.train_y[samples]
        label_counts = {}
        for label in labels:
            label_int = int(label)
            label_counts[label_int] = label_counts.get(label_int, 0) + 1
        
        if label_counts:
            majority_label = max(label_counts.items(), key=lambda x: x[1])[0]
            node.leaf_class = majority_label
            logger.log(f"Assigned label: {majority_label}, distribution: {label_counts}")

    def to_dict(self) -> dict:
        return {
            "nodes": self._nodes_to_dict(self.root_node),
            "num_classes": max(self._get_available_predictions()) + 1,
        }
        
    def _nodes_to_dict(self, node) -> dict:
        return {"id": id(node)} if not node else {
            "id": id(node),
            "value": node.leaf_class,
            "unknown": node.leaf_class == -1
        }

    def _get_available_predictions(self) -> list[int]:
        return [-1, *range(max(1, np.max(self.train_y) + 1) if hasattr(self, 'train_y') and self.train_y.size else 1)]

    def find_non_leaf_nodes(self):
        nodes = []
        def traverse(node):
            if node and not node.is_leaf:
                nodes.append(node)
                traverse(node.left_child)
                traverse(node.right_child)
        traverse(self.root_node)
        return nodes
class RandomForest(TreeBase):
    def __init__(
        self,
        trees: list[DecisionTree],
        feature_groups: list[list[int]],
        num_classes: int,
    ) -> None:
        self.trees = trees
        self.feature_groups = feature_groups
        self.num_classes = num_classes

    def predict_one(self, x: np.ndarray) -> int:
        votes, _ = self.predict_votes(x)
        idx = np.argmax(votes)

        if votes[idx] == 0:
            return -1
        return idx

    def predict_votes(self, x: np.ndarray) -> tuple[np.ndarray, int]:
        votes = np.zeros(self.num_classes, dtype=int)
        unknown_votes = 0
        for i, tree in enumerate(self.trees):
            vote = tree.predict_one(x[self.feature_groups[i]])
            if vote >= 0:
                votes[vote] += 1
            else:
                unknown_votes += 1
        return votes, unknown_votes

    def export_paths(self) -> list[list[RulePath]]:
        return [tree.get_rules() for tree in self.trees]

    def export_dict(self) -> dict:
        trees = [tree.export() for tree in self.trees]
        return {
            "trees": trees,
            "feature_groups": [self.feature_groups],
            "num_classes": self.num_classes,
        }

    @staticmethod
    def load_dict(
        model_dict: dict,
        categories_map: dict[str, set[str]],
    ) -> "RandomForest":
        trees = [
            # FIXME: max_depth
            DecisionTree.load_nodes_dict(tree_dict, 1000, categories_map)
            for tree_dict in model_dict["trees"]
        ]

        return RandomForest(
            trees,
            model_dict["feature_groups"],
            model_dict["num_classes"],
        )

    def to_graphviz_source(self, features: list[str] = None, classes: list[str] = None) -> str:
        return "\n".join(
            [
                tree.to_graphviz_source(features, classes, f"tree{i}")
                for i, tree in enumerate(self.trees)
            ]
        )
