from .strategy import TrainStrategy, FeatureBaggingStrategy
from .. import logger


class Classifier:
    def __init__(self, strategy: TrainStrategy) -> None:
        self.strategy = strategy

    def _prune_identical_leaves(self):
        """剪枝：合并标签相同的叶子节点对"""
        tree = self.strategy.get_tree()
        nodes_pruned = 0
        
        # 遍历所有非叶子节点
        for node in tree.find_non_leaf_nodes():
            # 检查是否有左右子节点 - 注意属性名称
            if not hasattr(node, 'left_child') or not hasattr(node, 'right_child'):
                continue
                
            left = node.left_child  # 使用left_child而非left
            right = node.right_child  # 使用right_child而非right
            
            # 检查左右子节点是否都是叶子节点
            if left is not None and right is not None and left.is_leaf and right.is_leaf:
                # 检查左右子节点标签是否相同
                if left.leaf_class == right.leaf_class:
                    # 将当前节点变为叶子节点，使用相同的标签
                    node.is_leaf = True
                    node.leaf_class = left.leaf_class
                    node.prediction = left.leaf_class
                    
                    # 删除左右子节点的引用
                    node.left_child = None
                    node.right_child = None
                    if hasattr(node, 'split_feature'):
                        node.split_feature = None
                    if hasattr(node, 'split_value'):
                        node.split_value = None
                    
                    # 冻结节点
                    if hasattr(node, 'freeze'):
                        node.freeze()
                    
                    nodes_pruned += 1
                    logger.log(f"剪枝：合并具有相同标签 {left.leaf_class} 的叶子节点对")
        
        logger.log(f"剪枝完成，合并了 {nodes_pruned} 个节点")
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

        # 树构建完成后执行剪枝操作
        logger.log("决策树构建完成，开始执行剪枝...")
        self.prune_tree(self.strategy.get_tree())
        logger.log("剪枝后的决策树:\n" + self.strategy.get_tree().to_graphviz_source())

        # 打印树构建的总token使用情况
        token_stats = self.strategy.get_token_stats()
        build_tokens = token_stats["build_tokens"]
        logger.log(f"树构建总Token使用: 输入={build_tokens['prompt']}, 输出={build_tokens['completion']}, 总计={build_tokens['total']}")

        return last_loss

    def predict(self, x) -> tuple[list[int], list[int], list[int]]:
        if type(self.strategy) == FeatureBaggingStrategy:
            sub_results = self.strategy.predict_llm_with_all_subtrees(
                x, with_examples=True
            )
        else:
            sub_results = []

        return (
            self.strategy.predict_llm_with_tree(x, with_examples=True),
            self.strategy.predict_tree(x),
            self.strategy.predict_tree_raw(x),
            sub_results,
        )

    def export(self) -> dict:
        """ """
        return self.strategy.export()

    def to_graphviz(self, features: list[str], classes: list[str]):
        return self.strategy.get_tree().to_graphviz(features, classes)

    def prune_tree(self, tree):
        """
        剪枝算法：合并具有相同标签的子树
        """
        if tree.root_node is None:
            return 0
        
        # 继续剪枝直到没有节点被合并
        total_merged = 0
        while True:
            merged_count = self._prune_one_pass(tree.root_node)
            total_merged += merged_count
            
            # 如果本次遍历没有合并任何节点，则停止
            if merged_count == 0:
                break
        
        logger.log(f"剪枝完成，合并了 {total_merged} 个节点")
        return total_merged

    def _prune_one_pass(self, node):
        """执行一次剪枝遍历，返回合并的节点数"""
        if node is None or node.is_leaf:
            return 0
        
        # 递归处理左右子树
        merged_left = self._prune_one_pass(node.left_child)
        merged_right = self._prune_one_pass(node.right_child)
        
        # 检查本节点是否可以剪枝
        merged_current = 0
        
        # 如果左右子节点都是叶子节点且标签相同
        if (node.left_child and node.left_child.is_leaf and 
            node.right_child and node.right_child.is_leaf and
            node.left_child.leaf_class == node.right_child.leaf_class):
            
            # 将该节点变为叶子节点，继承子节点的标签
            node.is_leaf = True
            node.leaf_class = node.left_child.leaf_class
            node.left_child = None
            node.right_child = None
            merged_current = 1
            logger.log(f"剪枝：合并具有相同标签 {node.leaf_class} 的叶子节点对")
        
        return merged_left + merged_right + merged_current
