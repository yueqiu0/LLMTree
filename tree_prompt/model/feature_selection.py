import numpy as np
from scipy.stats import chi2_contingency
from typing import List, Dict, Set
from ..dataset import DatasetMeta
from .. import logger

def calculate_weight_factor(depth: int) -> float:
    """计算LLM排序和统计分析的权重因子
    depth: 节点深度(1,2,3,...)
    return: LLM排序的权重(统计分析的权重为1-α)
    """
    alpha = 0.8
    logger.log(f"深度 {depth} 的权重因子计算: α={alpha:.2f}")
    return alpha

# def calculate_chi_square_scores(
#     X: np.ndarray, 
#     y: np.ndarray,
#     meta: DatasetMeta
# ) -> Dict[int, float]:
#     """计算每个特征的卡方检验分数"""
#     scores = {}
#     
#     logger.log(f"计算卡方检验分数，样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
#     
#     for feat_idx in range(X.shape[1]):
#         feature_name = meta.features[feat_idx].name
#         is_categorical = meta.features[feat_idx].is_categorical
#         
#         logger.log(f"处理特征 {feat_idx}: {feature_name} (类别型: {is_categorical})")
#         
#         # 检查特征值的分布
#         unique_values = np.unique(X[:, feat_idx])
#         logger.log(f"  特征值分布: {len(unique_values)} 个不同值")
#         
#         # 检查标签分布
#         unique_labels = np.unique(y)
#         label_counts = {label: np.sum(y == label) for label in unique_labels}
#         logger.log(f"  标签分布: {label_counts}")
#         
#         if is_categorical:
#             # 类别型特征直接计算卡方值
#             contingency = np.array([
#                 [np.sum((X[:, feat_idx] == val) & (y == label)) 
#                  for label in unique_labels]
#                 for val in unique_values
#             ])
#             logger.log(f"  类别型特征列联表:\n{contingency}")
#         else:
#             # 数值型特征需要先分箱
#             if len(unique_values) <= 1:
#                 logger.log(f"  警告: 特征值都相同，无法进行有效分箱")
#                 scores[feat_idx] = 0.0
#                 continue
#                 
#             # 使用更简单的分箱策略，确保至少有2个箱
#             n_bins = min(max(2, len(unique_values) // 2), 5)
#             bins = np.linspace(np.min(X[:, feat_idx]), np.max(X[:, feat_idx]), n_bins+1)
#             binned = np.digitize(X[:, feat_idx], bins)
#             
#             logger.log(f"  数值型特征分箱: {n_bins} 个箱")
#             
#             # 计算每个箱中每个标签的样本数
#             contingency = np.array([
#                 [np.sum((binned == i) & (y == label)) 
#                  for label in unique_labels]
#                 for i in range(1, n_bins+1)
#             ])
#             logger.log(f"  数值型特征列联表:\n{contingency}")
#         
#         # 计算卡方值
#         if contingency.shape[0] <= 1 or contingency.shape[1] <= 1:
#             logger.log(f"  警告: 列联表维度不足 {contingency.shape}")
#             scores[feat_idx] = 0.0
#             continue
#             
#         # 检查是否有全0行或全0列
#         if np.any(np.sum(contingency, axis=1) == 0) or np.any(np.sum(contingency, axis=0) == 0):
#             logger.log(f"  警告: 列联表存在全0行或全0列")
#             # 移除全0行和全0列
#             contingency = contingency[np.sum(contingency, axis=1) > 0, :]
#             contingency = contingency[:, np.sum(contingency, axis=0) > 0]
#             
#             if contingency.shape[0] <= 1 or contingency.shape[1] <= 1:
#                 logger.log(f"  警告: 清理后列联表维度不足 {contingency.shape}")
#                 scores[feat_idx] = 0.0
#                 continue
#         
#         try:
#             chi2, _, _, _ = chi2_contingency(contingency)
#             logger.log(f"  卡方值: {chi2:.4f}")
#             scores[feat_idx] = chi2
#         except Exception as e:
#             logger.log(f"  卡方计算错误: {e}")
#             scores[feat_idx] = 0.0
#     
#     logger.log(f"所有特征的卡方分数: {scores}")
#     return scores

def calculate_gini_scores(
    X: np.ndarray, 
    y: np.ndarray,
    meta: DatasetMeta
) -> Dict[int, float]:
    """计算每个特征的基尼系数增益，使用CART算法的方式"""
    scores = {}
    
    logger.log(f"计算基尼系数增益，样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    
    # 计算数据集总体的基尼系数
    unique_labels = np.unique(y)
    total_samples = len(y)
    parent_gini = 1.0
    for label in unique_labels:
        p = np.sum(y == label) / total_samples
        parent_gini -= p * p
    
    logger.log(f"父节点基尼系数: {parent_gini:.4f}")
    
    for feat_idx in range(X.shape[1]):
        feature_name = meta.features[feat_idx].name if feat_idx < len(meta.features) else f"特征{feat_idx}"
        is_categorical = meta.features[feat_idx].is_categorical if feat_idx < len(meta.features) else False
        
        logger.log(f"处理特征 {feat_idx}: {feature_name} (类别型: {is_categorical})")
        
        # 检查特征值的分布
        unique_values = np.unique(X[:, feat_idx])
        logger.log(f"  特征值分布: {len(unique_values)} 个不同值")
        
        if len(unique_values) <= 1:
            logger.log(f"  警告: 特征值都相同，无法进行有效分裂")
            scores[feat_idx] = 0.0
            continue
        
        best_gain = 0.0
        
        if is_categorical:
            # 类别型特征
            for val in unique_values:
                subset_indices = X[:, feat_idx] == val
                subset_size = np.sum(subset_indices)
                
                if subset_size == 0:
                    continue
                    
                subset_labels = y[subset_indices]
                subset_gini = 1.0
                
                for label in unique_labels:
                    p = np.sum(subset_labels == label) / subset_size if subset_size > 0 else 0
                    subset_gini -= p * p
                
                # 计算加权基尼系数
                weight = subset_size / total_samples
                weighted_gini = weight * subset_gini
                
                # 计算基尼系数增益
                gain = parent_gini - weighted_gini
                
                # 更新最佳增益
                if gain > best_gain:
                    best_gain = gain
                    logger.log(f"  特征值 {val} 的基尼增益: {gain:.4f} (新的最佳)")
        else:
            # 数值型特征 - 使用CART算法尝试所有可能的分裂点
            # 对唯一值排序
            sorted_values = np.sort(unique_values)
            
            # 尝试所有可能的分裂点
            for i in range(len(sorted_values) - 1):
                # 计算可能的分裂点（相邻值的中点）
                split_value = (sorted_values[i] + sorted_values[i+1]) / 2
                
                # 分割样本
                left_mask = X[:, feat_idx] <= split_value
                right_mask = ~left_mask
                
                # 计算左右子节点的基尼系数
                left_gini = 0
                right_gini = 0
                
                # 左子节点基尼系数
                if np.any(left_mask):
                    left_labels = y[left_mask]
                    left_count = len(left_labels)
                    left_gini = 1.0
                    for label in unique_labels:
                        p = np.sum(left_labels == label) / left_count
                        left_gini -= p * p
                
                # 右子节点基尼系数
                if np.any(right_mask):
                    right_labels = y[right_mask]
                    right_count = len(right_labels)
                    right_gini = 1.0
                    for label in unique_labels:
                        p = np.sum(right_labels == label) / right_count
                        right_gini -= p * p
                
                # 计算加权基尼系数
                left_weight = np.sum(left_mask) / total_samples
                right_weight = np.sum(right_mask) / total_samples
                weighted_gini = left_weight * left_gini + right_weight * right_gini
                
                # 计算基尼系数增益
                gain = parent_gini - weighted_gini
                
                # 更新最佳增益
                if gain > best_gain:
                    best_gain = gain
                    logger.log(f"  分裂点 {split_value:.4f} 的基尼增益: {gain:.4f} (新的最佳)")
        
        scores[feat_idx] = best_gain
    
    logger.log(f"所有特征的基尼系数增益: {scores}")
    return scores

def select_best_feature(
    llm_ranking: list[int], 
    gini_scores: Dict[int, float],  # 这里改为基尼系数增益
    depth: int, 
    used_features: set[int] = None
) -> int:
    """
    根据LLM排名和基尼系数增益选择最佳特征
    """
    if used_features is None:
        used_features = set()
    
    # 计算特征权重因子
    alpha = calculate_weight_factor(depth)
    logger.log(f"深度 {depth} 的权重因子: α={alpha:.2f} (LLM权重)")
    
    # 找出最大基尼增益用于归一化
    max_gini = max(gini_scores.values()) if gini_scores else 0
    
    # 合并LLM排名和基尼系数增益
    combined_scores = []
    for i, feature in enumerate(llm_ranking):
        if feature in used_features:
            continue
            
        # 归一化LLM排名 (倒排，越靠前分数越高)
        llm_score = 1.0 - (i / len(llm_ranking)) if len(llm_ranking) > 0 else 0
        
        # 获取特征的基尼增益并归一化
        gini_score = 0
        if feature in gini_scores:
            gini_score = gini_scores[feature] / max_gini if max_gini > 0 else 0
        
        # 合并分数
        combined_score = alpha * llm_score + (1 - alpha) * gini_score
        combined_scores.append((feature, combined_score, llm_score, gini_score))
    
    if not combined_scores:
        return None
    
    # 按合并分数排序
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 输出详细的特征选择信息
    logger.log("特征选择详情:")
    for feature, score, llm_score, gini_score in combined_scores[:5]:  # 仅输出前5个特征
        logger.log(f"  特征 {feature}: 合并分数 {score:.4f} (LLM: {llm_score:.4f}, 基尼: {gini_score:.4f})")
    
    # 返回得分最高的特征
    return combined_scores[0][0] 