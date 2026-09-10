from .strategy import TrainStrategy
from .tree import RandomForest, DecisionTree
from .. import logger


class Classifier:
    def __init__(self, strategy: TrainStrategy) -> None:
        self.strategy = strategy

    def fit(self, x, y) -> float:
        logger.log("Train data:")
        for xx, yy in zip(x.tolist(), y.tolist()):
            logger.log(f"{xx} -> {yy}")
        self.strategy.set_train_data(x, y)
        step = 0
        result = (True, None)
        while result[0]:
            result = self.strategy.step()
            logger.log(f"Step {step}")
            logger.log("Tree:\n" + self.strategy.get_tree().to_graphviz_source())
            step += 1

        logger.log("Decision tree construction completed, starting pruning...")
        self.prune_tree(self.strategy.get_tree())
        logger.log(
            "Pruned decision tree:\n" + self.strategy.get_tree().to_graphviz_source()
        )

        token_stats = self.strategy.get_token_stats()
        build_tokens = token_stats["build_tokens"]
        logger.log(
            f"Tree construction total token usage: input={build_tokens['prompt']}, output={build_tokens['completion']}, total={build_tokens['total']}"
        )

        return

    def export(self) -> dict:
        """ """
        return self.strategy.export()

    def to_graphviz(self, features: list[str], classes: list[str]):
        return self.strategy.get_tree().to_graphviz(features, classes)

    def prune_tree(self, tree):
        if isinstance(tree, RandomForest):
            logger.log(f"Pruning {len(tree.trees)} subtrees of the random forest")
            total_merged = 0
            for i, subtree in enumerate(tree.trees):
                logger.log(f"Pruning subtree {i + 1}/{len(tree.trees)}")
                subtree_merged = self._prune_decision_tree(subtree)
                total_merged += subtree_merged
            logger.log(f"Random forest pruning completed, merged {total_merged} nodes")
            return total_merged
        elif isinstance(tree, DecisionTree):
            return self._prune_decision_tree(tree)
        else:
            logger.log(f"Cannot prune, unknown tree type: {type(tree)}")
            return 0

    def _prune_decision_tree(self, tree):
        if tree.root_node is None:
            return 0
        total_merged = 0
        while True:
            merged_count = self._prune_one_pass(tree.root_node)
            total_merged += merged_count
            if merged_count == 0:
                break
        logger.log(f"Decision tree pruning completed, merged {total_merged} nodes")
        return total_merged

    def _prune_one_pass(self, node):

        if node is None or node.is_leaf:
            return 0
        merged_left = self._prune_one_pass(node.left_child)
        merged_right = self._prune_one_pass(node.right_child)

        merged_current = 0
        if (
            node.left_child
            and node.left_child.is_leaf
            and node.right_child
            and node.right_child.is_leaf
            and node.left_child.leaf_class == node.right_child.leaf_class
        ):
            node.is_leaf = True
            node.leaf_class = node.left_child.leaf_class
            node.left_child = None
            node.right_child = None
            merged_current = 1
            logger.log(
                f"Pruning: Merging leaf nodes with the same label {node.leaf_class}"
            )
        return merged_left + merged_right + merged_current

    def predict(self, x):
        return self.strategy.predict_tree(x)
