import numpy as np
from scipy.stats import chi2_contingency
from typing import List, Dict, Set
from ..dataset import DatasetMeta
from .. import logger

def calculate_weight_factor(depth: int) -> float:
    """计算LLM排序和统计分析的权重因子
    depth: 节点深度(1,2,3)
    return: LLM排序的权重(统计分析的权重为1-α)
    """
    return max(0.1, 0.5 - (depth - 1) * 0.2)

def calculate_chi_square_scores(
    X: np.ndarray, 
    y: np.ndarray,
    meta: DatasetMeta
) -> Dict[int, float]:
    """计算每个特征的卡方检验分数"""
    scores = {}
    
    logger.log(f"计算卡方检验分数，样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    
    for feat_idx in range(X.shape[1]):
        feature_name = meta.features[feat_idx].name
        is_categorical = meta.features[feat_idx].is_categorical
        
        logger.log(f"处理特征 {feat_idx}: {feature_name} (类别型: {is_categorical})")
        
        # 检查特征值的分布
        unique_values = np.unique(X[:, feat_idx])
        logger.log(f"  特征值分布: {len(unique_values)} 个不同值")
        
        # 检查标签分布
        unique_labels = np.unique(y)
        label_counts = {label: np.sum(y == label) for label in unique_labels}
        logger.log(f"  标签分布: {label_counts}")
        
        if is_categorical:
            # 类别型特征直接计算卡方值
            contingency = np.array([
                [np.sum((X[:, feat_idx] == val) & (y == label)) 
                 for label in unique_labels]
                for val in unique_values
            ])
            logger.log(f"  类别型特征列联表:\n{contingency}")
        else:
            # 数值型特征需要先分箱
            if len(unique_values) <= 1:
                logger.log(f"  警告: 特征值都相同，无法进行有效分箱")
                scores[feat_idx] = 0.0
                continue
                
            # 使用更简单的分箱策略，确保至少有2个箱
            n_bins = min(max(2, len(unique_values) // 2), 5)
            bins = np.linspace(np.min(X[:, feat_idx]), np.max(X[:, feat_idx]), n_bins+1)
            binned = np.digitize(X[:, feat_idx], bins)
            
            logger.log(f"  数值型特征分箱: {n_bins} 个箱")
            
            # 计算每个箱中每个标签的样本数
            contingency = np.array([
                [np.sum((binned == i) & (y == label)) 
                 for label in unique_labels]
                for i in range(1, n_bins+1)
            ])
            logger.log(f"  数值型特征列联表:\n{contingency}")
        
        # 计算卡方值
        if contingency.shape[0] <= 1 or contingency.shape[1] <= 1:
            logger.log(f"  警告: 列联表维度不足 {contingency.shape}")
            scores[feat_idx] = 0.0
            continue
            
        # 检查是否有全0行或全0列
        if np.any(np.sum(contingency, axis=1) == 0) or np.any(np.sum(contingency, axis=0) == 0):
            logger.log(f"  警告: 列联表存在全0行或全0列")
            # 移除全0行和全0列
            contingency = contingency[np.sum(contingency, axis=1) > 0, :]
            contingency = contingency[:, np.sum(contingency, axis=0) > 0]
            
            if contingency.shape[0] <= 1 or contingency.shape[1] <= 1:
                logger.log(f"  警告: 清理后列联表维度不足 {contingency.shape}")
                scores[feat_idx] = 0.0
                continue
        
        try:
            chi2, _, _, _ = chi2_contingency(contingency)
            logger.log(f"  卡方值: {chi2:.4f}")
            scores[feat_idx] = chi2
        except Exception as e:
            logger.log(f"  卡方计算错误: {e}")
            scores[feat_idx] = 0.0
    
    logger.log(f"所有特征的卡方分数: {scores}")
    return scores

def select_best_feature(
    llm_ranking: List[int],
    chi_square_scores: Dict[int, float],
    depth: int,
    used_features: Set[int]
) -> int:
    """选择最佳分裂特征"""
    # 过滤掉已使用的特征
    available_features = [f for f in llm_ranking if f not in used_features]
    if not available_features:
        return None
        
    alpha = calculate_weight_factor(depth)
    
    # 计算综合得分
    feature_scores = {}
    max_chi = max(chi_square_scores.values())
    
    # 记录分数
    logger.log(f"\nNode depth={depth}, α={alpha:.2f}")
    logger.log("Feature scores (combined = α * llm_score + (1-α) * chi_score):")
    
    if max_chi == 0:
        # 如果所有特征的卡方值都为0，只使用LLM得分
        logger.log("All chi-square scores are 0, using only LLM scores")
        for feat_idx in available_features:
            llm_score = 1.0 - llm_ranking.index(feat_idx) / len(llm_ranking)
            feature_scores[feat_idx] = llm_score
            logger.log(f"Feature {feat_idx}: {llm_score:.4f} (LLM score only)")
    else:
        # 正常计算综合得分
        for feat_idx in available_features:
            llm_score = 1.0 - llm_ranking.index(feat_idx) / len(llm_ranking)
            chi_score = chi_square_scores[feat_idx] / max_chi
            combined_score = alpha * llm_score + (1 - alpha) * chi_score
            feature_scores[feat_idx] = combined_score
            logger.log(f"Feature {feat_idx}: {combined_score:.4f}")
    
    return max(feature_scores.items(), key=lambda x: x[1])[0] 