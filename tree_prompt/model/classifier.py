from .strategy import TrainStrategy, FeatureBaggingStrategy
from .tree import RandomForest, DecisionTree
from .. import logger


class Classifier:
    def __init__(self, strategy: TrainStrategy) -> None:
        self.strategy = strategy

    def _prune_identical_leaves(self):
        tree = self.strategy.get_tree()
        nodes_pruned = 0
        
        for node in tree.find_non_leaf_nodes():
            if not hasattr(node, 'left_child') or not hasattr(node, 'right_child'):
                continue
                
            left = node.left_child
            right = node.right_child
            
            if left is not None and right is not None and left.is_leaf and right.is_leaf:
                if left.leaf_class == right.leaf_class:
                    node.is_leaf = True
                    node.leaf_class = left.leaf_class
                    node.prediction = left.leaf_class
                    
                    node.left_child = None
                    node.right_child = None
                    if hasattr(node, 'split_feature'):
                        node.split_feature = None
                    if hasattr(node, 'split_value'):
                        node.split_value = None
                    
                    if hasattr(node, 'freeze'):
                        node.freeze()
                    
                    nodes_pruned += 1
        
        return nodes_pruned

    def fit(self, x, y) -> float:
        logger.log("Train data:")
        for xx, yy in zip(x.tolist(), y.tolist()):
            logger.log(f"{xx} -> {yy}")
        self.strategy.set_train_data(x, y)
        step = 0
        result = (True, None)
        last_loss = None
        while result[0]:
            result = self.strategy.step()
            if result[1] != None:
                last_loss = result[1]
            logger.log(f"Step {step}: loss={result[1]}")
            logger.log("Tree:\n" + self.strategy.get_tree().to_graphviz_source())
            step += 1

        self.prune_tree(self.strategy.get_tree())
        
        token_stats = self.strategy.get_token_stats()
        build_tokens = token_stats["build_tokens"]
        logger.log(f"Tree Building Total Token Usage: Input={build_tokens['prompt']}, Output={build_tokens['completion']}, Total={build_tokens['total']}")

        return last_loss

    def predict(self, x) -> tuple[list[int], list[int], list[int], list[list[int]] | None]:
        if isinstance(self.strategy, FeatureBaggingStrategy):
            llm_results = self.strategy.predict_llm_with_tree(x, with_examples=True)
            tree_results = self.strategy.predict_tree(x)
            tree_raw_results = self.strategy.predict_tree_raw(x)
            return (llm_results, tree_results, tree_raw_results, None)
        else:
            return (
                self.strategy.predict_llm_with_tree(x, with_examples=True),
                self.strategy.predict_tree(x),
                self.strategy.predict_tree_raw(x),
                None
            )

    def export(self) -> dict:
        return self.strategy.export()

    def to_graphviz(self, features: list[str], classes: list[str]):
        return self.strategy.get_tree().to_graphviz(features, classes)

    def prune_tree(self, tree):
        if isinstance(tree, RandomForest):
            total_merged = 0
            for i, subtree in enumerate(tree.trees):
                subtree_merged = self._prune_decision_tree(subtree)
                total_merged += subtree_merged
            return total_merged
        elif isinstance(tree, DecisionTree):
            return self._prune_decision_tree(tree)
        else:
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
        
        return total_merged

    def _prune_one_pass(self, node):
        if node is None or node.is_leaf:
            return 0
        
        merged_left = self._prune_one_pass(node.left_child)
        merged_right = self._prune_one_pass(node.right_child)
        
        merged_current = 0
        
        if (node.left_child and node.left_child.is_leaf and 
            node.right_child and node.right_child.is_leaf and
            node.left_child.leaf_class == node.right_child.leaf_class):
            
            node.is_leaf = True
            node.leaf_class = node.left_child.leaf_class
            node.left_child = None
            node.right_child = None
            merged_current = 1
        
        return merged_left + merged_right + merged_current