import numpy as np
import jinja2
from itertools import product
from functools import reduce
import random
from tqdm import tqdm
from collections import Counter

from ..dataset import DatasetMeta, get_feature_importance_ranking
from ..runner import Runner
from ..prompt import Serializer, TabularSerializer, ListSerializer, TextSerializer
from .tree import DecisionTree, RandomForest, TreeBase, RulePath, Node
from .. import logger
from .feature_selection import calculate_gini_scores, select_best_feature, calculate_weight_factor, calculate_gini_impurity


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
                # histogram
                nums, values = np.histogram(
                    values,
                    hist_nbins,
                )
                values = values[np.where(nums > 0)[0] + 1]
                feature_values.append(values)
            else:
                if len(x) > 1:
                    feature_values.append(values[1:])  # drop the first value
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
        self.llm_feature_ranking = None  # 添加此属性

    def set_train_data(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        self.train_x = train_x
        self.train_y = train_y
        
        # 确保_meta被正确实现
        if not hasattr(self, '_meta_instance'):
            try:
                self._meta_instance = self._meta
            except NotImplementedError:
                logger.log("错误: 子类必须实现_meta属性")
                raise
                
        # 确保特征映射关系被保存并记录到日志
        self._feature_shuffle_map = getattr(self._meta, 'feature_shuffle_map', None)
        if self._feature_shuffle_map:
            logger.log(f"策略中保存特征映射关系: {self._feature_shuffle_map}")
        else:
            logger.log("警告: 未找到特征映射关系")
        
        # 获取LLM特征排序（如果适用）
        if hasattr(self, 'runner') and hasattr(self, 'get_feature_ranking'):
            self.llm_feature_ranking = self.get_feature_ranking(self._meta, self.runner)
            logger.log(f"LLM特征重要性排序: {self.llm_feature_ranking}")
        else:
            # 默认排序
            self.llm_feature_ranking = list(range(self._meta.feature_count()))
            logger.log(f"使用默认特征排序: {self.llm_feature_ranking}")
        
        self.split_values = _get_feature_values(self._meta, train_x, self.hist_nbins)
        
        self.tree = DecisionTree(self.max_depth, self._meta.categories_map)
        self.tree.set_train_data(train_x)

    def step(self) -> tuple[bool, float]:
        # 获取下一个要分裂的节点
        next_split = self.tree.next_to_split()
        if next_split is None:
            logger.log("没有可分裂的节点，训练结束")
            return False, None
        
        logger.log(f"正在处理节点，深度: {next_split.depth}")
        
        samples = next_split.get_samples()
        
        # 新增：检查是否所有样本都属于同一类别
        if samples is not None and len(samples) > 0:
            node_labels = [self.train_y[i] for i in samples]
            if len(set(node_labels)) == 1:
                # 如果是，直接将该节点标记为叶子节点
                logger.log(f"节点样本标签一致，直接设为叶子节点，标签: {node_labels[0]}")
                next_split.is_leaf = True
                next_split.prediction = node_labels[0]
                # 关键修复：确保节点被正确冻结，不再参与分裂
                next_split.freeze()
                return True, None
        
        # 确保samples不是None，并且长度检查安全
        if samples is None or len(samples) < 2 or next_split.depth >= self.max_depth:
            logger.log(f"节点无法继续分裂: 样本数={len(samples) if samples is not None else 0}, 深度={next_split.depth}, 最大深度={self.max_depth}")
            next_split.freeze()
            return True, None
        
        logger.log(f"节点样本数: {len(samples)}")
        
        # 特征选择
        logger.log("开始特征选择...")
        best_feature = self._quick_feature_selection(next_split)
        
        if best_feature is None:
            logger.log("找不到合适的特征，节点冻结")
            next_split.freeze()
            return True, None
        
        # 增加日志，输出所选特征的名称
        if hasattr(self, '_meta') and self._meta:
            feature_name = self._meta.features[best_feature].name if best_feature < len(self._meta.features) else f"未知特征({best_feature})"
            logger.log(f"选择特征: {best_feature} ({feature_name})")
        
        # 快速确定分裂点（减少LLM调用）
        logger.log("开始确定分裂点...")
        split_values = self._quick_split_values(best_feature, next_split)
        if len(split_values) == 0:
            logger.log("找不到合适的分裂点，节点冻结")
            next_split.freeze()
            return True, None
        
        # 格式化分裂点输出
        split_value = split_values[0]
        if isinstance(split_value, (float, np.float64, np.float32)):
            # 确定小数位数
            node_samples = next_split.get_samples()
            node_data = self.train_x[node_samples, best_feature]
            unique_values = np.unique(node_data)
            
            decimal_places = 1  # 默认至少保留1位小数
            for val in unique_values:
                if isinstance(val, (float, np.float64, np.float32)):
                    str_val = str(val)
                    if '.' in str_val:
                        curr_places = len(str_val.split('.')[1])
                        decimal_places = max(decimal_places, curr_places + 1)
            
            logger.log(f"分裂点: {split_value:.{decimal_places}f}")
        else:
            logger.log(f"分裂点: {split_value}")
        
        # 简单启发式选择标签
        logger.log("分配叶节点值...")
        left_class, right_class = self._assign_leaf_values(next_split, best_feature, split_value)
        logger.log(f"左子节点标签: {left_class}, 右子节点标签: {right_class}")
        
        # 执行分裂
        next_split.split(best_feature, split_value, 
                      self._meta.features[best_feature].is_categorical,
                      left_class, right_class)
        
        logger.log(f"节点已分裂，特征: {best_feature}, 分裂点: {split_value}")
        
        # 无需每次都调用LLM评估
        return True, 0.0

    def predict_tree(self, x: np.ndarray) -> list[int]:
        results = self.predict_tree_raw(x)
        for i, res in enumerate(results):
            if res < 0:
                idx = random.randint(0, self._meta.label_count() - 1)
                results[i] = self._meta.labels[idx].value
        return results

    def predict_tree_raw(self, x: np.ndarray) -> list[int]:
        """预测单个样本，返回原始标签值"""
        # 添加调试信息
        logger.log(f"开始预测，_meta.labels: {[(i, l.name, l.value) for i, l in enumerate(self._meta.labels)]}")
        
        if self.tree is None:
            logger.log("警告: 决策树尚未初始化")
            return [-1] * len(x)  # 返回未知标签
        
        # 获取决策树预测结果
        raw_predictions = self.tree.predict(x)
        
        # 添加调试信息，显示原始预测和转换过程
        logger.log(f"原始预测结果: {raw_predictions[:10]}...")  # 只显示前10个
        
        # 对每个预测结果进行转换
        results = []
        for idx in raw_predictions:
            # 打印更多调试信息
            logger.log(f"处理预测结果: {idx}")
            
            # 安全地获取标签值
            if idx >= 0:
                # 查找对应的标签值而不是使用索引直接访问
                label_value = None
                for label in self._meta.labels:
                    if label.value == idx:
                        label_value = idx
                        break
                
                if label_value is None:
                    logger.log(f"警告: 未找到标签值为 {idx} 的标签，使用原始值")
                    label_value = idx
                
                y = label_value
            else:
                y = idx
            
            results.append(y)
        
        return results

    def predict_llm_with_tree(
        self, x: np.ndarray, with_examples: bool = False
    ) -> list[int]:
        prompt = self._gen_prompt(
            x,
            (self.train_x, self.train_y) if with_examples else None,
        )
        if prompt is None:
            raise RuntimeError("Invalid tree rules!")
        results = self._predict_llm_with_tree_batched([prompt], [len(x)])
        return results[0] if results is not None else None

    def export(self) -> any:
        """导出模型，确保包含特征映射信息"""
        # 记录特征映射是否被导出
        if hasattr(self, '_feature_shuffle_map') and self._feature_shuffle_map:
            logger.log(f"导出特征映射到JSON: {self._feature_shuffle_map}")
        else:
            logger.log("警告: 导出时没有特征映射信息")
        
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

    def get_tree(self) -> TreeBase:
        raise NotImplementedError

    @staticmethod
    def load(model_dict: dict) -> "TrainStrategy":
        raise NotImplementedError

    def _get_available_predictions(self, x: np.ndarray) -> list[int]:
        raise NotImplementedError

    @property
    def _meta(self) -> DatasetMeta:
        raise NotImplementedError


    def _quick_feature_selection(self, node):
        """快速特征选择，使用缓存的LLM排序和简单统计"""
        # 获取已使用的特征
        used_features = node.get_used_features()
        logger.log(f"节点深度: {node.depth}, 已使用特征: {used_features}")
        
        # 获取节点样本
        node_samples = node.get_samples()
        if len(node_samples) == 0:
            logger.log("警告: 节点样本数为0，无法选择特征")
            return None
            
        node_x = self.train_x[node_samples]
        node_y = self.train_y[node_samples]
        
        logger.log(f"节点样本数: {len(node_samples)}, 特征数: {node_x.shape[1]}")
        
        # 计算特征的卡方检验分数
        gini_scores = calculate_gini_scores(
            node_x, node_y, self._meta
        )
        
        # 没有传入LLM排序，让我们使用初始化时设置的排序
        if not hasattr(self, 'llm_feature_ranking') or self.llm_feature_ranking is None:
            # 尝试重新获取一次
            self.llm_feature_ranking = self.get_feature_ranking(self._meta, self.runner)
            logger.log(f"重新获取LLM特征排序: {self.llm_feature_ranking}")
        
        # 使用特征选择函数
        best_feature = select_best_feature(
            self.llm_feature_ranking,
            gini_scores,
            node.depth,
            used_features
        )
        
        logger.log(f"最终选择特征: {best_feature}")
        
        if best_feature is not None and hasattr(self, '_meta') and self._meta:
            feature_name = self._meta.features[best_feature].name if best_feature < len(self._meta.features) else f"未知特征({best_feature})"
            logger.log(f"选择特征: {best_feature} ({feature_name})")
        
        return best_feature
    
    def _quick_split_values(self, feature_idx, node):
        """确定最佳分裂点，使用CART算法的方式"""
        # 添加日志：打印当前节点所有样本的标签
        node_samples = node.get_samples()
        node_labels = self.train_y[node_samples]
        unique_labels, label_counts = np.unique(node_labels, return_counts=True)
        label_dist = {int(label): count for label, count in zip(unique_labels, label_counts)}
        logger.log(f"节点({id(node)})样本标签分布: {label_dist}, 总样本数: {len(node_samples)}")
        
        # 如果是分类特征，使用专门的处理方法
        if self._meta.features[feature_idx].is_categorical:
            # 使用专门的分类特征处理方法寻找最佳分裂值
            best_value = self._process_categorical_feature(feature_idx, node)
            if best_value is not None:
                logger.log(f"分类特征 {self._meta.features[feature_idx].name} 使用优化的分裂值: {best_value}")
                return [best_value]
            else:
                # 如果专门方法失败，回退到原始方法
                logger.log(f"分类特征 {self._meta.features[feature_idx].name} 优化方法失败，使用默认分裂值")
                return self.split_values[feature_idx]
        
        # 获取节点样本数据
        node_data = self.train_x[node_samples, feature_idx]
        
        # 获取唯一值并排序
        unique_values = np.unique(node_data)
        if len(unique_values) <= 1:
            logger.log(f"特征 {self._meta.features[feature_idx].name} 的所有值都相同，跳过分裂")
            return []
        
        # 确定小数位数 - 检查特征值的小数位数，取最大值加1
        decimal_places = 1  # 默认至少保留1位小数
        for val in unique_values:
            if isinstance(val, (float, np.float64, np.float32)):
                # 将值转换为字符串，然后检查小数点后的位数
                str_val = str(val)
                if '.' in str_val:
                    curr_places = len(str_val.split('.')[1])
                    decimal_places = max(decimal_places, curr_places + 1)
        
        # 如果只有两个不同的值，使用它们的中点
        if len(unique_values) == 2:
            split_point = (unique_values[0] + unique_values[1]) / 2
            # 格式化分裂点，控制小数位数
            split_point = round(split_point, decimal_places)
            logger.log(f"只有两个不同值，使用中点 {split_point:.{decimal_places}f} 作为分裂点")
            return [split_point]
        
        # CART算法：尝试所有可能的分裂点，找到最优的
        best_split = None
        best_gain = -float('inf')
        
        # 计算父节点的基尼系数
        parent_gini = 1.0
        for label in unique_labels:
            p = np.sum(node_labels == label) / len(node_labels)
            parent_gini -= p * p
        
        # 尝试所有可能的分裂点
        for i in range(len(unique_values) - 1):
            # 计算可能的分裂点（相邻值的中点）
            split_value = (unique_values[i] + unique_values[i+1]) / 2
            
            # 格式化分裂点，控制小数位数
            split_value = round(split_value, decimal_places)
            
            # 分割样本
            left_mask = node_data <= split_value
            right_mask = ~left_mask
            
            # 计算左右子节点的基尼系数
            left_gini = 0
            right_gini = 0
            
            # 左子节点基尼系数
            if np.any(left_mask):
                left_labels = node_labels[left_mask]
                left_count = len(left_labels)
                left_gini = 1.0
                for label in unique_labels:
                    p = np.sum(left_labels == label) / left_count
                    left_gini -= p * p
            
            # 右子节点基尼系数
            if np.any(right_mask):
                right_labels = node_labels[right_mask]
                right_count = len(right_labels)
                right_gini = 1.0
                for label in unique_labels:
                    p = np.sum(right_labels == label) / right_count
                    right_gini -= p * p
            
            # 计算加权基尼系数
            left_weight = np.sum(left_mask) / len(node_labels)
            right_weight = np.sum(right_mask) / len(node_labels)
            weighted_gini = left_weight * left_gini + right_weight * right_gini
            
            # 计算基尼系数增益
            gain = parent_gini - weighted_gini
            
            # 更新最佳分裂点
            if gain > best_gain:
                best_gain = gain
                best_split = split_value
                
                # 记录详细信息
                logger.log(f"新的最佳分裂点: {best_split:.{decimal_places}f}, 增益: {best_gain:.4f}")
                logger.log(f"  左子节点: 样本数={np.sum(left_mask)}, 基尼={left_gini:.4f}")
                logger.log(f"  右子节点: 样本数={np.sum(right_mask)}, 基尼={right_gini:.4f}")
        
        if best_split is not None:
            logger.log(f"最终选择的最佳分裂点: {best_split:.{decimal_places}f}, 增益: {best_gain:.4f}")
            return [best_split]
        
        return []
    
    def _assign_leaf_values(self, node, feature_idx, split_value):
        """使用多数投票快速确定叶节点值"""
        node_samples = node.get_samples()
        node_x = self.train_x[node_samples]
        node_y = self.train_y[node_samples]
        
        # 添加完整的标签分布日志
        unique_labels, label_counts = np.unique(node_y, return_counts=True)
        label_dist = {int(label): count for label, count in zip(unique_labels, label_counts)}
        logger.log(f"分配叶值前节点({id(node)})标签分布: {label_dist}, 总样本数: {len(node_samples)}")
        
        # 根据分裂值将样本分为左右两组
        if self._meta.features[feature_idx].is_categorical:
            left_mask = node_x[:, feature_idx] == split_value
        else:
            left_mask = node_x[:, feature_idx] < split_value
        
        right_mask = ~left_mask
        
        # 获取左右子节点的标签
        left_y = node_y[left_mask] if np.any(left_mask) else []
        right_y = node_y[right_mask] if np.any(right_mask) else []
        
        # 添加左右子节点的标签分布日志
        if len(left_y) > 0:
            left_unique, left_counts = np.unique(left_y, return_counts=True)
            left_dist = {int(label): count for label, count in zip(left_unique, left_counts)}
            logger.log(f"左子节点标签分布: {left_dist}, 总样本数: {len(left_y)}")
        
        if len(right_y) > 0:
            right_unique, right_counts = np.unique(right_y, return_counts=True)
            right_dist = {int(label): count for label, count in zip(right_unique, right_counts)}
            logger.log(f"右子节点标签分布: {right_dist}, 总样本数: {len(right_y)}")
        
        # 使用多数投票确定叶节点值，确保结果在有效范围内
        valid_labels = []
        for label_info in self._meta.labels:
            valid_labels.append(label_info.value)
        logger.log(f"有效标签值列表: {valid_labels}")
        
        # 左子节点标签处理 - 修改为详细的计算过程
        if len(left_y) > 0:
            # 计算各标签出现次数并详细记录
            left_label_counts = {}
            for label in left_y:
                label_int = int(label)
                if label_int not in left_label_counts:
                    left_label_counts[label_int] = 0
                left_label_counts[label_int] += 1
            
            logger.log(f"左子节点标签计数: {left_label_counts}")
            
            # 找出多数类
            if left_label_counts:
                left_class = max(left_label_counts.items(), key=lambda x: x[1])[0]
                logger.log(f"左子节点多数类标签: {left_class}")
            else:
                left_class = valid_labels[0] if valid_labels else 0  # 默认使用第一个有效标签
                logger.log(f"左子节点无样本，使用默认标签: {left_class}")
        else:
            left_class = valid_labels[0] if valid_labels else 0
            logger.log(f"左子节点无样本，使用默认标签: {left_class}")
        
        # 右子节点标签处理 - 同样修改为详细计算过程
        if len(right_y) > 0:
            right_label_counts = {}
            for label in right_y:
                label_int = int(label)
                if label_int not in right_label_counts:
                    right_label_counts[label_int] = 0
                right_label_counts[label_int] += 1
            
            logger.log(f"右子节点标签计数: {right_label_counts}")
            
            if right_label_counts:
                right_class = max(right_label_counts.items(), key=lambda x: x[1])[0]
                logger.log(f"右子节点多数类标签: {right_class}")
            else:
                right_class = valid_labels[0] if valid_labels else 0
                logger.log(f"右子节点无样本，使用默认标签: {right_class}")
        else:
            right_class = valid_labels[0] if valid_labels else 0
            logger.log(f"右子节点无样本，使用默认标签: {right_class}")
        
        logger.log(f"最终分配标签 - 左: {left_class}, 右: {right_class}")
        return left_class, right_class

    def _process_categorical_feature(self, feature_idx, node):
        """专门处理分类特征的方法"""
        # 获取该特征在当前节点的所有数据
        node_samples = node.get_samples()
        node_data = self.train_x[node_samples, feature_idx]
        unique_values = np.unique(node_data)
        
        # 增加调试信息 - 显示特征分布
        logger.log(f"特征 {feature_idx} 的值分布: {unique_values}")
        
        # 允许处理只有一个或多个值的特征
        if len(unique_values) < 1:
            logger.log(f"特征 {feature_idx} 没有有效值")
            return None
        
        # 获取当前节点的标签
        node_labels = self.train_y[node_samples]
        unique_labels = np.unique(node_labels)
        
        # 增加调试信息 - 显示标签分布
        label_counts = {int(label): np.sum(node_labels == label) for label in unique_labels}
        logger.log(f"当前节点的标签分布: {label_counts}")
        
        # 如果只有一个类别，不需要再分裂
        if len(unique_labels) <= 1:
            logger.log(f"当前节点已经是纯净的，类别为 {unique_labels[0]}")
            return None
        
        # 计算父节点基尼系数 - 直接在这里实现，避免方法调用问题
        # 直接计算基尼系数，不调用可能出问题的方法
        parent_gini = 1.0
        total = len(node_labels)
        label_counts = {}
        for label in node_labels:
            label = int(label)
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        
        for _, count in label_counts.items():
            p = count / total
            parent_gini -= p * p
        
        logger.log(f"父节点基尼系数: {parent_gini:.4f}")
        
        # 寻找最佳分裂值
        best_gain = -float('inf')
        best_value = None
        min_samples_leaf = max(1, int(0.05 * len(node_labels)))
        
        for value in unique_values:
            # 创建掩码
            left_mask = node_data == value
            right_mask = ~left_mask
            
            # 获取样本标签
            left_labels = node_labels[left_mask]
            right_labels = node_labels[right_mask]
            
            # 确保两边都有足够样本
            if len(left_labels) < min_samples_leaf or len(right_labels) < min_samples_leaf:
                logger.log(f"  特征值 {value} 导致一侧样本数不足，跳过")
                continue
            
            # 显示分裂后每侧的标签分布
            left_counts = {int(label): np.sum(left_labels == label) for label in np.unique(left_labels)}
            right_counts = {int(label): np.sum(right_labels == label) for label in np.unique(right_labels)}
            logger.log(f"  特征值 {value}: 左侧 {len(left_labels)} 样本 {left_counts}, 右侧 {len(right_labels)} 样本 {right_counts}")
            
            try:
                # 内联计算左右子节点基尼系数，避免方法调用
                # 左子节点基尼系数
                left_gini = 1.0
                left_total = len(left_labels)
                left_label_counts = {}
                for label in left_labels:
                    label = int(label)
                    if label not in left_label_counts:
                        left_label_counts[label] = 0
                    left_label_counts[label] += 1
                
                for _, count in left_label_counts.items():
                    p = count / left_total
                    left_gini -= p * p
                    
                # 右子节点基尼系数
                right_gini = 1.0
                right_total = len(right_labels)
                right_label_counts = {}
                for label in right_labels:
                    label = int(label)
                    if label not in right_label_counts:
                        right_label_counts[label] = 0
                    right_label_counts[label] += 1
                
                for _, count in right_label_counts.items():
                    p = count / right_total
                    right_gini -= p * p
                    
                logger.log(f"  左侧基尼: {left_gini:.4f}, 右侧基尼: {right_gini:.4f}")
                
                # 计算信息增益
                n_left = len(left_labels)
                n_right = len(right_labels)
                n_total = len(node_labels)
                
                weighted_gini = (n_left/n_total)*left_gini + (n_right/n_total)*right_gini
                gain = parent_gini - weighted_gini
                logger.log(f"  信息增益: {gain:.4f}")
                
                # 选择最佳分裂值
                if gain > 0.0001 and gain > best_gain:
                    best_gain = gain
                    best_value = value
                    logger.log(f"  ✓ 当前最佳分裂值: {value}, 增益: {gain:.4f}")
            except Exception as e:
                logger.log(f"  计算特征值 {value} 的增益时出错: {str(e)}")
                continue
        
        if best_value is not None:
            logger.log(f"最终选择分裂值: {best_value}, 信息增益: {best_gain:.4f}")
        else:
            # 如果没找到最佳分裂值，使用频率最高的值
            if len(unique_values) > 1:
                values, counts = np.unique(node_data, return_counts=True)
                best_value = values[np.argmax(counts)]
                logger.log(f"未找到有效分裂值，选择频率最高的值: {best_value}")
            else:
                logger.log(f"未找到有效分裂值")
        
        return best_value

    def _split_node(self, feature_idx, split_value, node):
        """根据特征和分裂值分割节点"""
        samples = node.get_samples()
        node_data = self.train_x[samples, feature_idx]
        node_labels = self.train_y[samples]
        
        # 打印详细的分裂前样本信息
        label_counts = {}
        for label in node_labels:
            label_int = int(label)
            if label_int not in label_counts:
                label_counts[label_int] = 0
            label_counts[label_int] += 1
        logger.log(f"分裂前节点样本标签分布: {label_counts}")
        
        # 根据特征类型进行分裂
        if self._meta.features[feature_idx].is_categorical:
            left_mask = node_data == split_value
        else:
            left_mask = node_data < split_value
        right_mask = ~left_mask
        
        # 检查分裂结果
        left_samples = samples[left_mask]
        right_samples = samples[right_mask]
        left_labels = node_labels[left_mask]
        right_labels = node_labels[right_mask]
        
        # 打印详细的分裂后样本信息
        left_counts = {}
        for label in left_labels:
            label_int = int(label)
            if label_int not in left_counts:
                left_counts[label_int] = 0
            left_counts[label_int] += 1
            
        right_counts = {}
        for label in right_labels:
            label_int = int(label)
            if label_int not in right_counts:
                right_counts[label_int] = 0
            right_counts[label_int] += 1
        
        logger.log(f"分裂后左子节点样本标签分布: {left_counts}")
        logger.log(f"分裂后右子节点样本标签分布: {right_counts}")
        
        # 返回分裂结果...

    def _serialize_rule(self, rule):
        """序列化单个规则为字符串"""
        if rule is None:
            return None
        
        conditions = []
        for feature_id, condition in rule.conditions.items():
            feature_name = self._meta.features[feature_id].name
            if condition.is_categorical:
                values = list(condition.categories)
                if len(values) == 0:
                    # 空集合表示"不等于"任何分裂值
                    # 我们需要找出这个特征在决策树中的分裂值
                    
                    # 获取该特征的所有可能值
                    feature_values = []
                    if hasattr(self.tree, 'categories_map') and self.tree.categories_map and feature_id in self.tree.categories_map:
                        # 从决策树的类别映射中获取
                        feature_values = list(self.tree.categories_map[feature_id])
                    else:
                        # 从元数据中获取
                        for label in self._meta.features[feature_id].labels:
                            feature_values.append(label.name)
                    
                    # 查找该特征的分裂值
                    split_value = None
                    # 从决策树根节点开始搜索该特征的分裂节点
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
                        # 如果找到了分裂值，使用"不等于"表达式
                        conditions.append(f"{feature_name} != {split_value}")
                    else:
                        # 如果无法找到分裂值，使用通用表达式
                        conditions.append(f"{feature_name} 不等于任何分裂值")
                        logger.log(f"警告: 无法确定特征 {feature_name} 的分裂值")
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
        
        # 修复这里：通过值查找标签，而不是使用值作为索引
        label_name = None
        rule_value = rule.value
        
        # 打印调试信息
        logger.log(f"规则值: {rule_value}, 标签信息: {[f'值:{l.value},名称:{l.name}' for l in self._meta.labels]}")
        
        # 查找匹配的标签
        for label in self._meta.labels:
            if label.value == rule_value:
                label_name = label.name
                break
        
        # 如果没找到匹配的标签，使用默认名称
        if label_name is None:
            label_name = f"未知标签({rule_value})"
            logger.log(f"警告: 未找到值为 {rule_value} 的标签")
        
        if conditions:
            return f"IF {' AND '.join(conditions)} THEN {label_name}"
        else:
            return f"{label_name} (无条件)"


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
        self._meta_instance = serializer.meta  # 确保_meta_instance被设置
        self.max_depth = max_depth
        self.train_batch = train_batch
        self.hist_nbins = hist_nbins

    @property
    def _meta(self) -> DatasetMeta:
        """返回数据集元数据"""
        return self.serializer.meta  # 使用serializer中的meta

    @classmethod
    def get_feature_ranking(cls, meta: DatasetMeta, runner: Runner) -> list[int]:
        """获取数据集的特征重要性排序（类级别缓存）"""
        if not hasattr(cls, '_cached_rankings'):
            cls._cached_rankings = {}
            
        # 使用数据集名称作为缓存键
        cache_key = meta.name
        if cache_key not in cls._cached_rankings:
            # 获取LLM对特征的排序
            ranking = get_feature_importance_ranking(meta, runner)
            if ranking:
                cls._cached_rankings[cache_key] = ranking
                logger.log(f"获取数据集 {cache_key} 的LLM特征重要性排序: {ranking}")
            else:
                # 如果获取失败，使用默认顺序
                ranking = list(range(meta.feature_count()))
                logger.log(f"无法获取LLM特征排序，使用默认顺序: {ranking}")
                cls._cached_rankings[cache_key] = ranking
                
        return cls._cached_rankings[cache_key]

    def _get_available_predictions(self) -> list[int]:
        """获取可用的预测值列表"""
        return [-1, *range(self._meta.label_count())]

    def _gen_prompt(
        self, x: np.ndarray, examples: tuple[np.ndarray, np.ndarray] = None
    ) -> str:
        rules = self._get_tree_rules()
        if rules is None:
            return None

        x_test_str = [self.serializer.serialize(xx, None) for xx in x]
        if examples is not None:
            examples_str = [self.serializer.serialize(x, y) for x, y in zip(*examples)]
        else:
            examples_str = []

        prompt = self.master_template.render(
            meta=self._meta,
            examples=examples_str,
            rules=rules,
            format_desc=self.serializer.format_desc(),
            prediction_intro=self.serializer.answer_requirement(len(x_test_str)),
            tests=x_test_str,
        )

        return prompt

    def _predict_llm_with_tree_batched(
        self, prompts: list[str], expected_lens: list[int]
    ) -> list[list[int]]:
        all_results = []
        for i, resp_candidates in enumerate(self.runner.run(prompts)):
            for resp in resp_candidates:
                results = self.serializer.answer_decoder.decode(resp)
                results = [self._meta.get_label_value(r) for r in results]

                if None in results or len(results) != expected_lens[i]:
                    continue

                all_results.append(results)
                break
            else:
                logger.log(
                    "No valid response found, raw responses: {}".format(resp_candidates)
                )
                return None

        return all_results

    def _get_tree_rules(self) -> list[str]:
        rules: list[str] = []
        paths: list[RulePath] = self.tree.export_paths()
        if paths is None:
            return None
        for r in paths:
            depth = len(r.conditions)
            r = self._serialize_rule(r)
            if r is not None:
                rules.append((r, depth))

        rules.sort(key=lambda x: x[1])
        return [x[0] for x in rules]

    def export(self) -> any:
        return {
            "type": "unknown_class",
            "model": self.tree.export_nodes_dict(),
            "args": {"max_depth": self.max_depth, "categories": None},  # TODO
            "prompt": self._gen_prompt(
                [],
                (self.train_x, self.train_y),
            ),
        }

    @staticmethod
    def load(
        model_dict: dict,
        runner: Runner = None,
        master_template: jinja2.Template = None,
        serializer: Serializer = None,
    ) -> "UnknownClassStrategy":
        categories_map = model_dict["args"]["categories"]
        # list to dict
        categories_map = {int(k): set(v) for k, v in categories_map.items()}
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
        """获取可用的预测值列表（不包含unknown）"""
        return [*range(self._meta.label_count())]

    def _quick_feature_selection(self, node):
        """快速特征选择，使用缓存的LLM排序和简单统计"""
        # 获取已使用的特征
        used_features = node.get_used_features()
        logger.log(f"节点深度: {node.depth}, 已使用特征: {used_features}")
        
        # 获取节点样本
        node_samples = node.get_samples()
        if len(node_samples) == 0:
            logger.log("警告: 节点样本数为0，无法选择特征")
            return None
            
        node_x = self.train_x[node_samples]
        node_y = self.train_y[node_samples]
        
        logger.log(f"节点样本数: {len(node_samples)}, 特征数: {node_x.shape[1]}")
        
        # 计算特征的卡方检验分数
        gini_scores = calculate_gini_scores(
            node_x, node_y, self._meta
        )
        
        # 没有传入LLM排序，让我们使用初始化时设置的排序
        if not hasattr(self, 'llm_feature_ranking') or self.llm_feature_ranking is None:
            # 尝试重新获取一次
            self.llm_feature_ranking = self.get_feature_ranking(self._meta, self.runner)
            logger.log(f"重新获取LLM特征排序: {self.llm_feature_ranking}")
        
        # 使用特征选择函数
        best_feature = select_best_feature(
            self.llm_feature_ranking,
            gini_scores,
            node.depth,
            used_features
        )
        
        logger.log(f"最终选择特征: {best_feature}")
        
        if best_feature is not None and hasattr(self, '_meta') and self._meta:
            feature_name = self._meta.features[best_feature].name if best_feature < len(self._meta.features) else f"未知特征({best_feature})"
            logger.log(f"选择特征: {best_feature} ({feature_name})")
        
        return best_feature


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
        self.runner = runner
        self.template = template
        self.all_meta = all_meta
        self.max_depth = max_depth
        self.train_batch = train_batch
        self.hist_nbins = hist_nbins

        self.num_trees = min(num_trees, self.all_meta.feature_count())

        def _split(a, n):
            k, m = divmod(len(a), n)
            return (
                a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
            )

        # divide features into groups
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
        for feature_idxes in self.feature_groups:
            meta = DatasetMeta()
            meta.name = self.all_meta.name
            meta.target = self.all_meta.target
            meta.desc = self.all_meta.desc
            meta.label_meaning = self.all_meta.label_meaning
            meta.features = [self.all_meta.features[i] for i in feature_idxes]
            meta.labels = self.all_meta.labels
            self.sub_metas.append(meta)

        # TODO: Other strategies
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

    def set_train_data(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        self.train_x = train_x
        self.train_y = train_y

        feature_values = _get_feature_values(self.all_meta, train_x, self.hist_nbins)

        if len(train_x) > 1:
            for values in feature_values:
                values = values[1:]

        self.split_values = feature_values
        self.trees = []

        for i, feature_group in enumerate(self.feature_groups):
            self.sub_strategies[i].set_train_data(train_x[:, feature_group], train_y)
            self.trees.append(self.sub_strategies[i].get_tree())

        self.random_forest = RandomForest(
            self.trees, self.feature_groups, len(self.all_meta.labels)
        )

        self.trees_active = [True] * self.num_trees

    def step(self) -> tuple[bool, list[float]]:
        losses = []
        continue_step = False

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

        return continue_step, losses

    def predict_tree(self, x: np.ndarray) -> list[int]:
        results = self.predict_tree_raw(x)
        for i, res in enumerate(results):
            if res < 0:
                idx = random.randint(0, self.all_meta.label_count() - 1)
                results[i] = self.all_meta.labels[idx].value
        return results

    def predict_tree_raw(self, x: np.ndarray) -> list[int]:
        ret = []
        for xx in x:
            idx = self.random_forest.predict_one(xx)
            y = self.all_meta.labels[idx].value if idx >= 0 else idx
            ret.append(y)
        return ret

    def predict_llm_with_tree(
        self, x: np.ndarray, with_examples: bool = False
    ) -> list[int]:
        prompt = self._gen_prompt(
            x,
            self.sub_strategies,
            (self.train_x, self.train_y) if with_examples else None,
        )
        if prompt is None:
            raise RuntimeError("Invalid tree rules!")

        for resp_candidates in self.runner.run([prompt]):
            for resp in resp_candidates:
                results = self.serializer.answer_decoder.decode(resp)
                results = [self.all_meta.get_label_value(r) for r in results]

                if None in results or len(results) != len(x):
                    continue

                return results
            else:
                logger.log(
                    "No valid response found, raw responses: {}".format(resp_candidates)
                )
                return None

    def predict_llm_with_all_subtrees(
        self, x: np.ndarray, with_examples: bool = False
    ) -> list[list[int]]:
        return [
            strategy.predict_llm_with_tree(x[:, feature_group], with_examples)
            for feature_group, strategy in zip(self.feature_groups, self.sub_strategies)
        ]

    def _get_tree_rules(self, sub_strategies: list[UnknownClassStrategy]) -> list[str]:
        rules = []
        for sub_strategy in sub_strategies:
            rules += sub_strategy._get_tree_rules()
        return rules

    def _gen_prompt(
        self,
        x: np.ndarray,
        sub_strategies: list[TrainStrategy],
        examples: tuple[np.ndarray, np.ndarray] = None,
    ) -> str:
        rules = self._get_tree_rules(sub_strategies)
        if rules is None:
            return None

        x_test_str = [self.serializer.serialize(xx, None) for xx in x]
        if examples is not None:
            examples_str = [self.serializer.serialize(x, y) for x, y in zip(*examples)]
        else:
            examples_str = []

        prompt = self.template.render(
            meta=self.all_meta,
            examples=examples_str,
            rules=rules,
            format_desc=self.serializer.format_desc(),
            prediction_intro=self.serializer.answer_requirement(len(x_test_str)),
            tests=x_test_str,
        )

        return prompt

    def get_tree(self) -> TreeBase:
        return self.random_forest

    def export(self) -> any:
        """导出随机森林模型，包含特征映射信息"""
        # 记录特征映射是否被导出
        if hasattr(self, '_feature_shuffle_map') and self._feature_shuffle_map:
            logger.log(f"导出随机森林特征映射到JSON: {self._feature_shuffle_map}")
        
        return {
            "type": "random_forest",
            "model": self.random_forest.export_dict(),
            "args": {
                "max_depth": self.max_depth, 
                "num_trees": self.num_trees,
                "feature_shuffle_map": self._feature_shuffle_map if hasattr(self, '_feature_shuffle_map') else None
            },
            "prompt": self._gen_prompt(
                [], self.sub_strategies, (self.train_x, self.train_y)
            ),
        }

    def _determine_split_point(self, node, feature_idx):
        """确定特征的最佳分裂点"""
        # 获取节点样本
        node_samples = node.get_samples()
        node_x = self.train_x[node_samples]
        node_y = self.train_y[node_samples]
        
        logger.log(f"节点({id(node)})样本标签分布: {dict(Counter(node_y))}, 总样本数: {len(node_y)}")
        
        # 检查特征是否为类别型
        is_categorical = self._meta.features[feature_idx].is_categorical if feature_idx < len(self._meta.features) else False
        
        if is_categorical:
            # 类别型特征，找出最佳类别值作为分裂点
            unique_values = np.unique(node_x[:, feature_idx])
            best_gain = 0.0
            best_split = None
            
            for val in unique_values:
                left_mask = node_x[:, feature_idx] == val
                right_mask = ~left_mask
                
                # 确保两边都有样本
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # 计算基尼增益
                parent_gini = calculate_gini_impurity(node_y)
                left_gini = calculate_gini_impurity(node_y[left_mask])
                right_gini = calculate_gini_impurity(node_y[right_mask])
                
                left_weight = np.sum(left_mask) / len(node_y)
                right_weight = np.sum(right_mask) / len(node_y)
                
                weighted_gini = left_weight * left_gini + right_weight * right_gini
                gain = parent_gini - weighted_gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = val
                    logger.log(f"新的最佳分裂点: {val}, 增益: {gain:.4f}")
                    logger.log(f"  左子节点: 样本数={np.sum(left_mask)}, 基尼={left_gini:.4f}")
                    logger.log(f"  右子节点: 样本数={np.sum(right_mask)}, 基尼={right_gini:.4f}")
            
            logger.log(f"最终选择的最佳分裂点: {best_split}, 增益: {best_gain:.4f}")
            return best_split
        else:
            # 数值型特征，找出最佳阈值作为分裂点
            unique_values = np.unique(node_x[:, feature_idx])
            
            # 确定小数位数 - 检查特征值的小数位数，取最大值加1
            decimal_places = 1  # 默认至少保留1位小数
            for val in unique_values:
                if isinstance(val, (float, np.float64, np.float32)):
                    # 将值转换为字符串，然后检查小数点后的位数
                    str_val = str(val)
                    if '.' in str_val:
                        curr_places = len(str_val.split('.')[1])
                        decimal_places = max(decimal_places, curr_places + 1)
            
            # 对特征值排序
            sorted_indices = np.argsort(node_x[:, feature_idx])
            sorted_feature = node_x[sorted_indices, feature_idx]
            sorted_y = node_y[sorted_indices]
            
            best_gain = 0.0
            best_split = None
            
            # 尝试所有可能的分裂点
            for i in range(1, len(sorted_feature)):
                # 如果当前值与前一个值相同，跳过
                if sorted_feature[i] == sorted_feature[i-1]:
                    continue
                
                # 计算分裂点（相邻值的中点）
                split_value = (sorted_feature[i] + sorted_feature[i-1]) / 2
                
                # 格式化分裂点，控制小数位数
                split_value = round(split_value, decimal_places)
                
                # 创建左右子节点的掩码
                left_mask = node_x[:, feature_idx] <= split_value
                right_mask = ~left_mask
                
                # 确保两边都有样本
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # 计算基尼增益
                parent_gini = calculate_gini_impurity(node_y)
                left_gini = calculate_gini_impurity(node_y[left_mask])
                right_gini = calculate_gini_impurity(node_y[right_mask])
                
                left_weight = np.sum(left_mask) / len(node_y)
                right_weight = np.sum(right_mask) / len(node_y)
                
                weighted_gini = left_weight * left_gini + right_weight * right_gini
                gain = parent_gini - weighted_gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = split_value
                    logger.log(f"新的最佳分裂点: {split_value:.{decimal_places}f}, 增益: {gain:.4f}")
                    logger.log(f"  左子节点: 样本数={np.sum(left_mask)}, 基尼={left_gini:.4f}")
                    logger.log(f"  右子节点: 样本数={np.sum(right_mask)}, 基尼={right_gini:.4f}")
            
            logger.log(f"最终选择的最佳分裂点: {best_split:.{decimal_places}f}, 增益: {best_gain:.4f}")
            return best_split
