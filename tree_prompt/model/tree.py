import numpy as np


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
        # the order matters!
        assert self.feature == other.feature

        # categorical
        if self.is_categorical:
            intersection = self.categories.intersection(other.categories)
            if len(intersection) == 0 or len(intersection) == len(self.categories):
                return None
            return Condition.categorical(self.feature, intersection)

        # numerical
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

        if upper is not None and lower is not None and upper <= lower:
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
            elif rule.conditions[cond.feature] is None:
                return None
            else:
                rule.conditions[cond.feature] = rule.conditions[cond.feature].merged(
                    cond
                )
                if rule.conditions[cond.feature] is None:
                    return None

        return rule

    def __repr__(self) -> str:
        return self.__dict__.__repr__()


class Node:
    def __init__(self, parent: "Node", depth: int, leaf_class: int = None) -> None:
        self.depth: Node = depth
        self.parent: Node = parent

        self.leaf_class: int = leaf_class
        self.is_leaf = leaf_class is not None
        self.split_value: float | int | str = None
        self.split_feature: int = None
        self.is_categorical: bool = None
        # < split_value (numerical) or is split_value (categorical)
        self.left_child: Node = None
        # >= split_value (numerical) or is not split_value (categorical)
        self.right_child: Node = None

        self.freezed = False

    def split(
        self,
        feat_idx: int,
        feat_value: float | int | str,
        is_categorical: bool,
        left_class: int,
        right_class: int,
    ) -> None:
        self.is_leaf = False

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

    def next_node(self, feat_value: float | str | int) -> "Node":
        if self.is_leaf:
            return None
        if not self.is_categorical:
            if feat_value < self.split_value:
                return self.left_child
            else:
                return self.right_child

        assert type(feat_value) == type(self.split_value)
        if feat_value == self.split_value:
            return self.left_child
        else:
            return self.right_child

    def _dfs(
        self,
        on_node: callable,
        on_exit: callable,
    ) -> None:
        if on_node:
            on_node(self)

        if not self.is_leaf:
            self.left_child._dfs(on_node, on_exit)
            self.right_child._dfs(on_node, on_exit)

        if on_exit:
            on_exit(self)


class TreeBase:
    def predict_one(self, x: np.ndarray) -> int:
        raise NotImplementedError()

    def to_graphviz(
        self,
        features: list[str] = None,
        classes: list[str] = None,
    ):
        import graphviz

        source = self.to_graphviz_source(features, classes)
        return graphviz.Source(source)

    def to_graphviz_source(
        self,
        features: list[str] = None,
        classes: list[str] = None,
        graph_name: str = None,
    ) -> str:
        raise NotImplementedError()


class DecisionTree(TreeBase):
    def __init__(self, max_depth: int, feature_categories: dict[int, set[str]]) -> None:
        self.root_node = Node(parent=None, depth=1, leaf_class=-1)
        self.max_depth = max_depth
        self.feature_categories = feature_categories

    def next_to_split(self) -> Node:
        # level order traversal: shallow first
        nodes = [self.root_node]
        while len(nodes) > 0:
            node = nodes.pop(0)
            if node.is_leaf and not node.freezed:
                return node
            if not node.is_leaf and node.depth < self.max_depth:
                nodes.append(node.left_child)
                nodes.append(node.right_child)
        return None

    def predict_one(self, x: np.ndarray) -> int:
        node = self.get_leaf(x)
        return node.leaf_class

    def get_leaf(self, x: np.ndarray) -> Node:
        node = self.root_node
        while not node.is_leaf:
            node = node.next_node(x[node.split_feature])
        return node

    def export_paths(self) -> list[RulePath]:
        rules: list[RulePath] = []
        conditions: list[Condition] = []

        invalid = False

        def on_node(node: Node):
            nonlocal invalid
            if node.parent is None:
                return
            if node.parent.left_child == node:
                if node.parent.is_categorical:
                    condition = Condition.categorical(
                        node.parent.split_feature, {node.parent.split_value}
                    )
                else:
                    condition = Condition.numerical(
                        node.parent.split_feature, None, node.parent.split_value
                    )
            else:
                if node.parent.is_categorical:
                    condition = Condition.categorical(
                        node.parent.split_feature,
                        self.feature_categories[node.parent.split_feature]
                        - {node.parent.split_value},
                    )
                else:
                    condition = Condition.numerical(
                        node.parent.split_feature, node.parent.split_value, None
                    )

            conditions.append(condition)

            if node.is_leaf:
                rule = RulePath.from_conditions(conditions, node.leaf_class)
                if rule is None:
                    invalid = True
                else:
                    rules.append(rule)

        def on_exit(node: Node):
            if node.parent is not None:
                conditions.pop()

        self.root_node._dfs(on_node, on_exit)
        return rules if not invalid else None

    def export_nodes_dict(self) -> dict:
        all_nodes: list[Node] = []
        current_id = 0

        def get_node(node: Node):
            nonlocal current_id
            node.id = current_id
            all_nodes.append(node)
            current_id += 1

        self.root_node._dfs(get_node, None)

        exported_dict: dict[int, dict] = {}
        for node in all_nodes:
            exported_dict[node.id] = {"leaf": node.is_leaf}
            if node.is_leaf:
                exported_dict[node.id]["class"] = int(node.leaf_class)
            else:
                exported_dict[node.id]["left"] = node.left_child.id
                exported_dict[node.id]["right"] = node.right_child.id
                exported_dict[node.id]["categorical"] = node.is_categorical
                exported_dict[node.id]["feature"] = node.split_feature
                exported_dict[node.id]["value"] = node.split_value

        return exported_dict

    @staticmethod
    def load_nodes_dict(
        nodes_dict: dict, max_depth: int, categories_map: dict[str, set[str]]
    ) -> "DecisionTree":
        tree = DecisionTree(max_depth, categories_map)
        tree.root_node = Node(parent=None, depth=0)
        node_list = [(nodes_dict["0"], tree.root_node)]

        while len(node_list) > 0:
            node_dict, node = node_list.pop(0)
            if node_dict["leaf"]:
                node.leaf_class = node_dict["class"]
                node.is_leaf = True
            else:
                node.is_categorical = node_dict.get("categorical", False)
                node.split_feature = node_dict["feature"]
                node.split_value = node_dict["value"]
                node.is_leaf = False
                node.left_child = Node(parent=node, depth=node.depth + 1)
                node.right_child = Node(parent=node, depth=node.depth + 1)
                node_list.append((nodes_dict[str(node_dict["left"])], node.left_child))
                node_list.append(
                    (nodes_dict[str(node_dict["right"])], node.right_child)
                )

        return tree

    def to_graphviz_source(
        self,
        features: list[str] = None,
        classes: list[str] = None,
        graph_name: str = None,
    ) -> str:
        lines = []
        self.root_node._id = "0"

        def on_node(node: Node):
            if not node.is_leaf:
                left = node.left_child
                right = node.right_child
                left._id = node._id + "1"
                right._id = node._id + "2"
                lines.append(f"{node._id} -> {left._id} [label=yes]")
                lines.append(f"{node._id} -> {right._id} [label=no]")

            if node.is_leaf:
                node_label = (
                    (classes[node.leaf_class] if node.leaf_class >= 0 else "Unknown")
                    if classes
                    else node.leaf_class
                )
            else:
                node_label = (
                    f"{features[node.split_feature] if features else node.split_feature}"
                    + (" = " if node.is_categorical else " < ")
                    + f"{node.split_value}?"
                )

            lines.append(f'{node._id} [label="{node_label}"]')

        self.root_node._dfs(on_node, None)

        return (
            f"digraph {graph_name if graph_name else 'G'}"
            + " {\n  "
            + "\n  ".join(lines)
            + "\n}"
        )


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
        return [tree.export_paths() for tree in self.trees]

    def export_dict(self) -> dict:
        trees = [tree.export_nodes_dict() for tree in self.trees]
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

    def to_graphviz_source(
        self, features: list[str] = None, classes: list[str] = None
    ) -> str:
        return "\n".join(
            [
                tree.to_graphviz_source(features, classes, f"tree{i}")
                for i, tree in enumerate(self.trees)
            ]
        )
