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
        """从条件列表创建规则路径"""
        rule = RulePath()
        rule.value = value
        
        for cond in conditions:
            if cond.feature not in rule.conditions:
                rule.conditions[cond.feature] = cond
            elif rule.conditions[cond.feature] is None:
                return None
            else:
                merged = rule.conditions[cond.feature].merged(cond)
                if merged is None:
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
        """获取节点样本，确保始终返回数组"""
        if self.samples is None:
            return np.array([], dtype=int)
        return self.samples

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

    def _dfs(self, on_node, callback_after_recurse=None):
        """深度优先遍历树，确保处理空节点"""
        # 检查节点是否存在
        if self is None:
            return
        
        on_node(self)
        
        # 继续递归子节点，确保它们存在
        if not self.is_leaf:
            if self.left_child is not None:
                self.left_child._dfs(on_node, callback_after_recurse)
            if self.right_child is not None:
                self.right_child._dfs(on_node, callback_after_recurse)
        
        if callback_after_recurse:
            callback_after_recurse(self)

    def predict(self, x: np.ndarray) -> int:
        """预测单个样本的类别"""
        if self.is_leaf:
            return self.leaf_class
        
        feature_value = x[self.split_feature]
        if self.is_categorical:
            if feature_value == self.split_value:
                return self.left_child.predict(x) if self.left_child else -1
            else:
                return self.right_child.predict(x) if self.right_child else -1
        else:
            if feature_value < self.split_value:
                return self.left_child.predict(x) if self.left_child else -1
            else:
                return self.right_child.predict(x) if self.right_child else -1


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
    def __init__(self, max_depth: int, categories_map: dict[int, set[str]]) -> None:
        self.max_depth = max_depth
        self.categories_map = categories_map
        self.root_node = Node(None, 1)
        self.train_x = None

    def set_train_data(self, train_x: np.ndarray) -> None:
        self.train_x = train_x
        self.root_node.samples = np.arange(len(train_x))

    def _go_left(self, x: np.ndarray, node: Node) -> bool:
        """判断样本是否应该去左子树"""
        if node.is_leaf:
            return False
            
        feature = node.split_feature
        split_value = node.split_value
        
        # 如果分裂值为None，说明节点还未完成分裂
        if split_value is None:
            return False
        
        if node.is_categorical:
            # 类别型特征
            return x[feature] == split_value
        else:
            # 数值型特征
            try:
                return float(x[feature]) < float(split_value)
            except (ValueError, TypeError):
                logger.log(f"警告: 无法比较特征值 {x[feature]} 和分裂点 {split_value}")
                return False

    def next_to_split(self) -> Node:
        """获取下一个要分裂的节点"""
        def _dfs(node: Node) -> Node:
            if node is None or node.freezed:
                return None
                
            if node.is_leaf:
                # 打印叶节点样本标签信息
                if hasattr(self, 'train_y') and self.train_y is not None:
                    samples = node.get_samples()
                    if len(samples) > 0:
                        labels = self.train_y[samples]
                        unique_labels, counts = np.unique(labels, return_counts=True)
                        label_dist = {int(label): count for label, count in zip(unique_labels, counts)}
                        logger.log(f"叶节点({id(node)})样本标签分布: {label_dist}, 预测标签: {node.leaf_class}")
                return node
            
            # 如果节点未完成分裂（split_value为None），返回该节点
            if node.split_value is None:
                if hasattr(self, 'train_y') and self.train_y is not None:
                    samples = node.get_samples()
                    if len(samples) > 0:
                        labels = self.train_y[samples]
                        unique_labels, counts = np.unique(labels, return_counts=True)
                        label_dist = {int(label): count for label, count in zip(unique_labels, counts)}
                        logger.log(f"未分裂节点({id(node)})样本标签分布: {label_dist}")
                return node
            
            # 检查左右子树的样本是否正确
            left_samples = []
            right_samples = []
            
            for idx in node.get_samples():
                try:
                    if self._go_left(self.train_x[idx], node):
                        left_samples.append(idx)
                    else:
                        right_samples.append(idx)
                except Exception as e:
                    logger.log(f"警告: 分配样本时出错: {e}")
                    continue
            
            # 打印分裂后的样本标签分布
            if hasattr(self, 'train_y') and self.train_y is not None:
                if len(left_samples) > 0:
                    left_labels = self.train_y[left_samples]
                    left_unique, left_counts = np.unique(left_labels, return_counts=True)
                    left_dist = {int(label): count for label, count in zip(left_unique, left_counts)}
                    logger.log(f"分裂后左子节点({id(node.left_child) if node.left_child else 'None'})样本标签分布: {left_dist}")
                
                if len(right_samples) > 0:
                    right_labels = self.train_y[right_samples]
                    right_unique, right_counts = np.unique(right_labels, return_counts=True)
                    right_dist = {int(label): count for label, count in zip(right_unique, right_counts)}
                    logger.log(f"分裂后右子节点({id(node.right_child) if node.right_child else 'None'})样本标签分布: {right_dist}")
            
            if len(left_samples) > 0:
                node.left_child.samples = np.array(left_samples)
            if len(right_samples) > 0:
                node.right_child.samples = np.array(right_samples)
            
            # 递归检查左右子树
            left_result = _dfs(node.left_child)
            if left_result is not None:
                return left_result
            
            return _dfs(node.right_child)
        
        return _dfs(self.root_node)

    def predict_one(self, x: np.ndarray) -> int:
        """预测单个样本的类别"""
        node = self.root_node
        # 检查根节点是否为空
        if node is None:
            return -1  # 返回未知类别
        
        while node is not None and not node.is_leaf:
            if self._go_left(x, node):
                node = node.left_child
            else:
                node = node.right_child
            
            # 检查子节点是否为空
            if node is None:
                return -1  # 子节点为空，返回未知类别
            
        # 如果节点存在并且是叶节点，返回其类别
        return node.leaf_class if node is not None else -1

    def predict(self, x: np.ndarray) -> list[int]:
        """预测多个样本的类别"""
        return [self.predict_one(sample) for sample in x]

    def predict_raw(self, x: np.ndarray) -> list[int]:
        """预测样本的原始类别（包括unknown）"""
        return self.predict(x)

    def export_paths(self) -> list[RulePath]:
        """导出所有规则路径"""
        paths = []
        
        def _dfs(node: Node, conditions: list[Condition] = None) -> None:
            if conditions is None:
                conditions = []
                
            if node.is_leaf:
                # 创建规则路径
                path = RulePath.from_conditions(conditions, node.leaf_class)
                if path is not None:
                    paths.append(path)
                return
                
            if node.split_feature is None:
                return
                
            # 创建左子树条件（小于或等于分裂值）
            if node.is_categorical:
                left_cond = Condition.categorical(node.split_feature, {node.split_value})
            else:
                left_cond = Condition.numerical(node.split_feature, None, node.split_value)
                
            # 创建右子树条件（大于分裂值）
            if node.is_categorical:
                right_cond = Condition.categorical(node.split_feature, set())  # 不等于分裂值
            else:
                right_cond = Condition.numerical(node.split_feature, node.split_value, None)
            
            # 递归遍历左右子树
            if node.left_child:
                _dfs(node.left_child, conditions + [left_cond])
            if node.right_child:
                _dfs(node.right_child, conditions + [right_cond])
        
        _dfs(self.root_node)
        return paths

    def get_rules(self) -> list[RulePath]:
        """获取所有决策路径规则"""
        rules = []
        
        def _collect_rules(node, conditions, collect_leaf):
            if node is None:
                return
            
            if node.is_leaf and collect_leaf:
                # 确保叶节点有有效的标签
                if hasattr(node, 'leaf_class') and node.leaf_class is not None:
                    # 添加调试信息
                    logger.log(f"收集到叶节点规则，标签值: {node.leaf_class}")
                    rule = RulePath.from_conditions(conditions.copy(), node.leaf_class)
                    if rule:
                        rules.append(rule)
                return
            
            # 非叶节点或不收集叶节点的情况
            feature = node.split_feature
            if feature is None:
                return
            
            # 左子树的条件
            if node.is_categorical:
                left_cond = Condition.categorical(feature, {node.split_value})
            else:
                left_cond = Condition.numerical(feature, None, node.split_value)
            
            conditions.append(left_cond)
            _collect_rules(node.left_child, conditions, collect_leaf)
            conditions.pop()
            
            # 右子树的条件
            if node.is_categorical:
                right_cond = Condition.categorical(feature, 
                    set(self._get_other_categorical_values(feature, node.split_value)))
            else:
                right_cond = Condition.numerical(feature, node.split_value, None)
            
            conditions.append(right_cond)
            _collect_rules(node.right_child, conditions, collect_leaf)
            conditions.pop()
        
        _collect_rules(self.root_node, [], True)
        logger.log(f"总共收集了 {len(rules)} 条规则")
        return rules

    def export_nodes_dict(self) -> dict:
        """导出节点字典，供序列化使用"""
        return self._export_nodes(self.root_node)
        
    def export(self) -> dict:
        """导出完整模型"""
        return {
            "nodes": self._export_nodes(self.root_node),
            "args": {
                "max_depth": self.max_depth,
                "categories": {
                    str(k): list(v) for k, v in self.categories_map.items()
                },
            },
        }

    def _export_nodes(self, node: Node) -> dict:
        """导出节点"""
        if node is None:
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
        """从字典加载模型"""
        tree = DecisionTree(max_depth, categories_map)
        
        def _load_node(node_dict: dict) -> Node:
            if node_dict is None:
                return None
                
            node = Node(None, 1)  # depth will be set later
            node.split_feature = node_dict["feature"]
            node.split_value = node_dict["split_value"]
            node.leaf_class = node_dict["prediction"]
            node.left_child = _load_node(node_dict["left"])
            node.right_child = _load_node(node_dict["right"])
            return node
            
        tree.root_node = _load_node(nodes_dict)
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
            """处理没有子节点的情况"""
            if not node.is_leaf:
                left = node.left_child
                right = node.right_child
                
                # 检查节点是否存在才添加ID和连接
                if left is not None:
                    left._id = node._id + "1"
                    lines.append(f"{node._id} -> {left._id} [label=yes]")
                
                if right is not None:
                    right._id = node._id + "2"
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

    def _assign_node_label(self, node):
        """分配节点标签（使用多数类）"""
        # 获取节点样本
        samples = node.get_samples()
        if len(samples) == 0:
            return
        
        # 获取样本标签
        labels = self.train_y[samples]
        
        # 使用字典统计每个标签的出现次数，处理不连续标签
        label_counts = {}
        for label in labels:
            label_int = int(label)  # 确保转换为整数
            if label_int not in label_counts:
                label_counts[label_int] = 0
            label_counts[label_int] += 1
        
        # 找出出现次数最多的标签
        if label_counts:
            majority_label = max(label_counts.items(), key=lambda x: x[1])[0]
            node.set_prediction(majority_label)
            logger.log(f"节点分配标签: {majority_label}, 样本标签分布: {label_counts}")
        else:
            logger.log("警告: 节点没有样本，无法分配标签")


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
