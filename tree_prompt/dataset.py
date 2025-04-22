import yaml
import sklearn.datasets
import pandas as pd
import numpy as np
from typing import List
import re

from .common_args import DatasetArgs
from .runner import Runner
from . import logger


class DatasetMeta:
    class Feature:
        name: str
        desc: str
        type: str
        categories: dict[str, str]

        @property
        def is_categorical(self) -> bool:
            return self.type == "categorical"

        def __repr__(self) -> str:
            return str(self.__dict__)

    class Label:
        name: str
        value: float
        desc: str

        def __repr__(self) -> str:
            return str(self.__dict__)

    def __init__(self) -> None:
        self.features: list[DatasetMeta.Feature] = []
        self.labels: list[DatasetMeta.Label] = []
        self.name: str = ""
        self.target: str = ""
        self.desc: str = ""
        self.labal_meaning: str = ""
        self.feature_shuffle_map: dict[int, int] = {}

    def get_label(self, id: int) -> Label:
        return self.labels[id]

    def find_label(self, value: float) -> Label:
        for label in self.labels:
            if label.value == value:
                return label
        return None

    def get_label_value(self, name: str) -> float:
        for label in self.labels:
            if label.name == name:
                return label.value
        return None

    def feature_count(self) -> int:
        return len(self.features)

    def label_names(self) -> str:
        return [label.name for label in self.labels]

    def label_count(self) -> int:
        return len(self.labels)

    def shuffle_features(self, indices: list[int]):
        self.features = [self.features[i] for i in indices]

    def value_repr(self, feat_idx: int, val) -> str:
        feat = self.features[feat_idx]
        if feat.type == "int":
            repr = int(val)
        elif feat.type == "categorical":
            if type(val) != str:
                repr = feat.categories[int(val)]
            else:
                repr = feat.categories[val]
        elif feat.type == "float":
            repr = f"{val:.3f}"
        else:
            repr = f"{val}"
        return repr

    @property
    def categories_map(self) -> dict[int, set[str]]:
        categories_map: dict[str, int[str]] = {}
        for i, feature in enumerate(self.features):
            if feature.is_categorical:
                categories_map[i] = set(feature.categories.keys())
        return categories_map

    def __repr__(self) -> str:
        return str(self.__dict__)


def load_meta(path: str) -> DatasetMeta:
    with open(path, "r") as f:
        data: dict = yaml.safe_load(f)

    meta = DatasetMeta()
    meta.name = data.get("name")
    meta.desc = data.get("desc")
    meta.target = data.get("target")
    meta.labal_meaning = data.get("label_meaning")

    features: list[dict] = data.get("features")
    for feat in features:
        feature = meta.Feature()
        feature.name = feat["name"]
        feature.desc = feat["desc"]
        feature.type = feat["type"]
        feature.categories = feat.get("categories")
        meta.features.append(feature)

    labels: list[dict] = data.get("labels")
    for l in labels:
        label = meta.Label()
        label.name = l["name"]
        label.value = l["value"]
        label.desc = l["desc"]
        meta.labels.append(label)

    logger.log(f"Loaded dataset metadata from {path}")
    logger.log(f"Dataset: {meta.name}, Features: {len(meta.features)}, Labels: {len(meta.labels)}")
    return meta


def _dummy(num_features: int) -> DatasetMeta:
    meta = DatasetMeta()
    meta.name = "dummy"
    meta.labal_meaning = "result"

    for i in range(num_features):
        feature = DatasetMeta.Feature()
        feature.name = "feature_{}".format(i + 1)
        feature.desc = ""
        feature.type = "float"
        meta.features.append(feature)

    for i, name in enumerate(["no", "yes"]):
        label = DatasetMeta.Label()
        label.name = name
        label.value = i
        label.desc = ""
        meta.labels.append(label)

    return meta


def load_dataset(
    args: DatasetArgs,
) -> tuple[DatasetMeta, np.ndarray, np.ndarray]:
    meta = load_meta(args.meta_file)
    if args.format == "libsvm":
        x, y = sklearn.datasets.load_svmlight_file(args.data_file)
        x = x.toarray()
    elif args.format == "csv":
        df = pd.read_csv(args.data_file)
        x = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
    else:
        raise ValueError("Unknown dataset format: {}".format(args.format))

    logger.log(f"Dataset loaded - Samples: {x.shape[0]}, Features: {x.shape[1]}")
    logger.log(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # 在打乱前先收集原始特征名称
    original_feature_names = [feature.name for feature in meta.features]
    
    # shuffle
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    logger.log("Data shuffled")

    if args.shuffle_column:
        # ===== 特征打乱和映射日志 =====
        indices = np.arange(x.shape[1])
        np.random.shuffle(indices)
        x = x[:, indices]
        
        # 创建明确的分隔线，使日志更易识别
        logger.log("\n" + "="*50)
        logger.log("【特征随机打乱映射】")
        
        # 保存并输出详细的映射关系
        feature_shuffle_map = {new_idx: int(old_idx) for new_idx, old_idx in enumerate(indices)}
        logger.log(f"特征随机打乱映射字典: {feature_shuffle_map}")
        
        # 详细的名称对应表，便于可视化调试
        logger.log("详细特征映射关系:")
        for new_idx, old_idx in feature_shuffle_map.items():
            old_feature_name = original_feature_names[old_idx] if old_idx < len(original_feature_names) else f"未知特征({old_idx})"
            logger.log(f"  训练特征 {new_idx} ← 原始特征 {old_idx} ({old_feature_name})")
        
        logger.log("="*50 + "\n")
        
        # 更新元数据中的特征顺序和映射信息
        meta.shuffle_features(indices)
        meta.feature_shuffle_map = feature_shuffle_map
    
    return meta, x, y


def sample_balanced(
    x: np.ndarray,
    y: np.ndarray,
    num_groups: int,
    num_samples_per_group: int,
    random_seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    logger.log(f"Starting balanced sampling - Groups: {num_groups}, Samples/group: {num_samples_per_group}, Seed: {random_seed}")
    all_samples = []
    classes = np.unique(y)
    logger.log(f"Class distribution: {dict(zip(classes, np.bincount(y.astype(int))))}")

    if num_samples_per_group != 1:
        assert num_samples_per_group % len(classes) == 0

    for group in range(num_groups):
        random_state = np.random.RandomState(random_seed + group)
        mask = np.hstack(
            [
                random_state.choice(
                    np.where(y == l)[0],
                    num_samples_per_group // len(classes)
                    if num_samples_per_group != 1
                    else 1,
                    replace=False,
                )
                for l in classes
            ]
        )
        if num_samples_per_group == 1:
            samples_x, samples_y = x[[mask[group % 2]]], y[[mask[group % 2]]]
        else:
            samples_x, samples_y = x[mask], y[mask]

        all_samples.append((samples_x, samples_y))

    logger.log(f"Completed balanced sampling - Total samples generated: {len(all_samples)}")
    return all_samples


def create_feature_ranking_prompt(meta: DatasetMeta) -> str:
    """创建用于特征重要性排序的提示"""
    prompt = f"""As a data analyst, you need to determine which features are most important for predicting {meta.labal_meaning or "the target variable"}.

Dataset Information:
The dataset consists of the following features:
"""
    for i, feat in enumerate(meta.features):
        prompt += f"{i+1}. {feat.name}: {feat.desc or 'No description available'}\n"
    
    prompt += f"\nTarget Variable: {meta.labal_meaning or 'The output'}\n"
    prompt += "Possible values: " + ", ".join([f"{label.name}" for label in meta.labels]) + "\n\n"
    prompt += """Based on common knowledge and intuition about this kind of data, rank the features from most important to least important for predicting the target variable.

Please return your answer as a comma-separated list of feature indices, ordered from most important to least important. For example: 2,4,1,3

Feature Ranking: """
    return prompt


def get_feature_importance_ranking(meta: DatasetMeta, runner: Runner) -> list[int]:
    """获取LLM对特征重要性的排序"""
    prompt = create_feature_ranking_prompt(meta)
    logger.log(f"请求特征重要性排序，提示词:\n{prompt}")
    
    for responses in runner.run([prompt]):
        for response in responses:
            logger.log(f"收到LLM响应:\n{response}")
            
            # 尝试从回复中提取特征排序
            try:
                # 首先尝试直接查找逗号分隔的数字列表
                number_lists = re.findall(r'(\d+(?:\s*,\s*\d+)*)', response)
                for number_list in number_lists:
                    # 分割并转换为整数列表
                    ranking = [int(num.strip()) - 1 for num in number_list.split(',')]
                    
                    # 验证排序的有效性
                    if (len(ranking) == meta.feature_count() and 
                        all(0 <= x < meta.feature_count() for x in ranking) and
                        len(set(ranking)) == len(ranking)):
                        logger.log(f"成功解析特征排序: {ranking}")
                        return ranking
                    else:
                        logger.log(f"跳过无效的特征排序: {ranking} (长度={len(ranking)}, 期望长度={meta.feature_count()})")
                
                # 如果没有找到有效的排序，尝试查找单独的数字
                numbers = re.findall(r'\b(\d+)\b', response)
                if numbers:
                    ranking = [int(num) - 1 for num in numbers]
                    if (len(ranking) == meta.feature_count() and 
                        all(0 <= x < meta.feature_count() for x in ranking) and
                        len(set(ranking)) == len(ranking)):
                        logger.log(f"通过单独数字解析特征排序: {ranking}")
                        return ranking
                    else:
                        logger.log(f"跳过无效的单独数字排序: {ranking}")
                
            except Exception as e:
                logger.log(f"解析特征排序时出错: {str(e)}")
                continue
    
    # 如果无法获取有效的排序，返回默认顺序
    default_order = list(range(meta.feature_count()))
    logger.log(f"无法获取有效的特征排序，使用默认顺序: {default_order}")
    return default_order



def generate_CoT_tree_prompt(
    meta: DatasetMeta,
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: int = 3,  # 强制设置为3层深度
    num_examples: int = 5
    ) -> str:
        """
        生成决策树构建提示词（完整优化版）
        修改重点：
        1. 预计算所有特征统计信息避免重复调用
        2. 增强维度匹配校验
        3. 统一统计信息访问接口
        4. 强制决策树深度为3层
        """
        # ================= 输入校验 =================
        if not hasattr(meta, 'features') or len(meta.features) == 0:
            logger.error("特征元数据为空，请检查meta文件")
            return ""
        
        if x_train.shape[1] != len(meta.features):
            logger.error(f"特征数量不匹配：数据有{x_train.shape[1]}列，元数据定义{len(meta.features)}个特征")
            return ""

        # ============== 预计算特征统计信息 ==============
        feature_stats = []
        for col_idx in range(len(meta.features)):
            try:
                col = x_train[:, col_idx]
                feat = meta.features[col_idx]
                stats = {}

                if feat.type in ["int", "float"]:
                    # 数值型特征处理
                    clean = col[~np.isnan(col.astype(float))]
                    if len(clean) == 0:
                        logger.log(f"特征 {feat.name} 无有效数值数据")
                        stats = {"min": None, "max": None, "mean": None, "std": 0.0}
                    else:
                        stats = {
                            "min": float(np.min(clean)),
                            "max": float(np.max(clean)),
                            "mean": float(np.mean(clean)),
                            "std": float(np.std(clean)) if len(clean) > 1 else 0.0
                        }
                else:
                    # 类别型特征处理
                    unique, counts = np.unique(col, return_counts=True)
                    stats = {
                        "unique_values": len(unique),
                        "most_common": dict(zip(unique.astype(str), counts.astype(int)))
                    }
                
                feature_stats.append(stats)
            except Exception as e:
                logger.log(f"特征统计计算失败(第{col_idx+1}列): {str(e)}")
                feature_stats.append({})

        # ============== 标签分布计算 ==============
        label_dist = {}
        try:
            unique_labels, label_counts = np.unique(y_train, return_counts=True)
            for val, count in zip(unique_labels, label_counts):
                label = meta.find_label(float(val))
                label_name = label.name if label else f"未知({val})"
                label_dist[label_name] = int(count)
            logger.log(f"标签分布计算完成: {label_dist}")
        except Exception as e:
            logger.log(f"标签分布计算失败: {str(e)}")
            return ""

        # ============== 示例数据生成 ==============
        examples = []
        try:
            indices = np.random.choice(len(x_train), min(num_examples, len(x_train)), replace=False)
            for idx in indices:
                features = []
                for i, feat in enumerate(meta.features):
                    val = x_train[idx][i]
                    # 处理特殊值
                    if feat.is_categorical:
                        val = feat.categories.get(str(val), f"未知({val})")
                    elif feat.type == "float":
                        val = f"{float(val):.4f}"
                    features.append(f"{feat.name}={val}")
                
                label_val = float(y_train[idx])
                label = meta.find_label(label_val)
                label_name = label.name if label else "未知"
                examples.append(f"样本 {idx+1}: {', '.join(features)} → {label_name}")
            logger.log(f"已生成{len(examples)}条示例数据")
        except Exception as e:
            logger.log(f"示例数据生成失败: {str(e)}")
            examples = []

        # ============== 构建提示词 ==============
        prompt_parts = []
        
        # ------ 元数据部分 ------
        prompt_parts.append("# Dataset Metadata")
        prompt_parts.append(f"Dataset Name: {getattr(meta, 'name', 'Unnamed Dataset')}")
        prompt_parts.append(f"Target Variable: {getattr(meta, 'target', 'Not Specified')}")
        prompt_parts.append(f"Sample Count: {x_train.shape[0]}")
        prompt_parts.append(f"Feature Count: {len(meta.features)}")
        prompt_parts.append("Label Distribution: " + ", ".join(
            [f"{k}({v})" for k, v in label_dist.items()]
        ))
        prompt_parts.append("")

        # ------ 特征详情 ------
        prompt_parts.append("# Feature Details")
        feature_details = []
        for i, feat in enumerate(meta.features):
            stats = feature_stats[i]
            feat_lines = []
            
            # 基础信息
            feat_lines.append(f"Feature {i+1}: {feat.name}")
            feat_lines.append(f"- Type: {feat.type}{' (Categorical)' if feat.is_categorical else ''}")
            
            # 处理描述信息
            desc = getattr(feat, 'desc', 'No description').replace('"', '\\"')
            feat_lines.append(f"- Description: {desc}")

            # 类型特定信息
            if feat.is_categorical:
                categories = []
                for k, v in feat.categories.items():
                    safe_k = k.replace('"', '\\"')
                    categories.append(f"{safe_k}({v})")
                feat_lines.append(f"- Categories: {', '.join(categories)}")
            else:
                stats_str = []
                for stat in ['min', 'max', 'mean']:
                    value = stats.get(stat)
                    stats_str.append(
                        f"{stat.capitalize()}={value:.4f}" 
                        if value is not None 
                        else f"{stat.capitalize()}=N/A"
                    )
                feat_lines.append("- Stats: " + ", ".join(stats_str))
            
            feature_details.append("\n".join(feat_lines))
        
        prompt_parts.append("\n\n".join(feature_details))
        prompt_parts.append("")

        # ------ 标签定义 ------
        prompt_parts.append("# Label Definitions")
        label_defs = []
        for label in meta.labels:
            desc = getattr(label, 'desc', 'No description').replace('"', '\\"')
            label_defs.append(f"{label.name} (Value={label.value}): {desc}")
        prompt_parts.append("\n".join(label_defs))
        prompt_parts.append("")

        # ------ 决策树要求 ------
        prompt_parts.append("# Decision Tree Requirements")
        prompt_parts.append("1. The tree must have exactly 3 levels (max_depth=3)")
        prompt_parts.append("2. Each level must split on a different feature")
        prompt_parts.append("3. Rules must be mutually exclusive")
        prompt_parts.append("4. You must leverage your knowledge about the domain when creating rules")
        prompt_parts.append("5. All rules must include both THEN and ELSE branches")
        prompt_parts.append("")

        # ------ 决策树示例 ------
        prompt_parts.append("# Decision Tree Examples (3 levels)")
        # 生成3个示例规则，使用特征统计信息
        rule_examples = []
        for i in range(3):
            # 确保示例展示3层深度结构
            # 选择信息增益高的特征
            feat_idx = i % len(meta.features)
            feat = meta.features[feat_idx]
            label1 = meta.labels[i%len(meta.labels)].name
            label2 = meta.labels[(i+1)%len(meta.labels)].name
            
            logger.log(f"生成规则 {i+1} - 特征: {feat.name}, 类型: {feat.type}")
            if feat.is_categorical:
                # 对于分类特征，使用最常见的类别
                common_cat = max(feat.categories.items(), key=lambda x: x[1])[0]
                rule = f"Rule Group {i+1}:\n  Main Rule:\n    IF {feat.name} = {common_cat}\n    THEN 0\n    ELSE 1"
                logger.log(f"分类规则: {rule}")
            else:
                # 对于数值特征，使用均值作为阈值
                threshold = feature_stats[feat_idx].get('mean', 0.5)
                rule = f"Rule Group {i+1}:\n  Main Rule:\n    IF {feat.name} {'<=' if i%2 else '>'} {threshold:.4f}\n    THEN 0\n    ELSE 1"
                logger.log(f"数值规则: {rule}")
            
            rule_examples.append(rule)
        logger.log(f"完成规则生成，共生成{len(rule_examples)}条规则")
    
        prompt_parts.append("# Decision Tree Requirements")
        prompt_parts.append("1. The tree must have exactly 2 levels (max_depth=3)")
        prompt_parts.append("2. Each level must split on a different feature")
        prompt_parts.append("3. Rules must be mutually exclusive")
        prompt_parts.append("4. You must leverage your knowledge about the domain when creating rules")
        prompt_parts.append("5. All rules must include both THEN and ELSE branches")
        prompt_parts.append("")

        # ++++ NEW REASONING GUIDANCE SECTION ++++
        prompt_parts.append("# Step-by-Step Construction Guidance")
        prompt_parts.append("Let's think step by step. First, we need to select the best root node feature. Consider the following:")
        prompt_parts.append("1. Which feature provides the highest information gain for the initial split?")
        prompt_parts.append("2. How do the statistical properties (mean, distribution, categories) influence this choice?")
        prompt_parts.append("3. After selecting the root feature:")
        prompt_parts.append("   a. For numerical features: What threshold optimally separates the classes?")
        prompt_parts.append("   b. For categorical features: Which category provides the purest subgroup?")
        prompt_parts.append("4. Repeat this process for subsequent levels, ensuring:")
        prompt_parts.append("   a. Different features are used at each level")
        prompt_parts.append("   b. Splits maximize class separation at each node")
        prompt_parts.append("")


        # ============== 最终组装 ==============
        full_prompt = "\n".join(prompt_parts)

    
        # 统计信息验证
        invalid_stats = sum(1 for s in feature_stats if not s)
        if invalid_stats > 0:
            logger.log(f"检测到{invalid_stats}个特征的统计信息异常，可能影响规则质量")

        # 调试日志
        logger.log("=============== 完整提示词内容 ===============")
        logger.log(full_prompt)
        logger.log(f"提示词长度: {len(full_prompt)} 字符")
        logger.log("=============================================")
        
        return full_prompt.strip()

