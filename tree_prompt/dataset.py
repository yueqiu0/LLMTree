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



def generate_LLM_tree_prompt(
    meta: DatasetMeta,
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: int = 3,  # 强制设置为3层深度
    num_examples: int = 5
) -> dict:
    """
    生成决策树构建提示词
    返回值：dict，包括prompt、label_dist、num_examples等
    """
    # ================= 输入校验 =================
    if not hasattr(meta, 'features') or len(meta.features) == 0:
        logger.error("特征元数据为空，请检查meta文件")
        return {"prompt": "", "label_dist": {}, "num_examples": 0}
        
    if x_train.shape[1] != len(meta.features):
        logger.error(f"特征数量不匹配：数据有{x_train.shape[1]}列，元数据定义{len(meta.features)}个特征")
        return {"prompt": "", "label_dist": {}, "num_examples": 0}

    # ============== 标签分布计算 =============
    label_dist = {}
    try:
        unique_labels, label_counts = np.unique(y_train, return_counts=True)
        for val, count in zip(unique_labels, label_counts):
            label = meta.find_label(float(val))
            label_name = label.name if label else f"未知({val})"
            label_dist[label_name] = int(count)
    except Exception as e:
        return {"prompt": "", "label_dist": {}, "num_examples": 0}

    # ============== 示例数据生成 ==============
    examples = []
    try:
        indices = np.random.choice(len(x_train), min(num_examples, len(x_train)), replace=False)
        for idx in indices:
            features = []
            for i, feat in enumerate(meta.features):
                val = x_train[idx][i]
                if feat.is_categorical:
                    val = feat.categories.get(str(val), f"未知({val})")
                elif feat.type == "float":
                    val = f"{float(val):.4f}"
                features.append(f"{feat.name}={val}")
            
            label_val = float(y_train[idx])
            label = meta.find_label(label_val)
            label_name = label.name if label else "未知"
            examples.append(f"{', '.join(features)} → {label_name}")
    except Exception as e:
        examples = []

    # ============== 构建提示词 ==============
    prompt_parts = []
    
    # ------ 元数据部分 ------
    prompt_parts.append("# Dataset Information")
    prompt_parts.append(f"Dataset: {getattr(meta, 'name', 'Unnamed Dataset')}")
    prompt_parts.append(f"Target: {getattr(meta, 'target', 'Not Specified')}")
    prompt_parts.append(f"Samples: {x_train.shape[0]}")
    prompt_parts.append(f"Features: {len(meta.features)}")
    prompt_parts.append("")

    # ------ 特征定义 ------
    prompt_parts.append("# Feature Definitions")
    for feat in meta.features:
        # 自动提取单位（如desc中有括号）
        desc = getattr(feat, 'desc', 'No description')
        unit = ''
        import re
        m = re.search(r'\(([^)]+)\)', desc)
        if m:
            unit = m.group(1)
        if unit:
            prompt_parts.append(f"- {feat.name}: {desc} [unit: {unit}]")
        else:
            prompt_parts.append(f"- {feat.name}: {desc}")
        if feat.is_categorical:
            categories = [f"{k}({v})" for k, v in feat.categories.items()]
            prompt_parts.append(f"  Categories: {', '.join(categories)}")
    prompt_parts.append("")

    # ------ 标签定义 ------
    prompt_parts.append("# Label Definitions")
    for label in meta.labels:
        desc = getattr(label, 'desc', 'No description')
        prompt_parts.append(f"- {label.name}: {desc}")
    prompt_parts.append("")
    
    # ------ 训练样本展示 ------
    prompt_parts.append("# Training Samples")
    prompt_parts.append("Here are some examples from the training dataset:")
    
    # 添加特征顺序说明
    feature_names = [feat.name for feat in meta.features]
    prompt_parts.append(f"For each line:")
    prompt_parts.append(f"<{', '.join(feature_names)}> <RESULT>")
    prompt_parts.append("")
    
    # 确定要展示的样本数量（最多10个）
    num_samples_to_show = min(10, len(x_train))
    
    # 随机选择样本索引，以确保样本具有代表性
    if num_samples_to_show < len(x_train):
        import random
        sample_indices = random.sample(range(len(x_train)), num_samples_to_show)
    else:
        sample_indices = range(len(x_train))
    
    # 遍历选定的样本，使用更简洁的格式展示
    for i, idx in enumerate(sample_indices):
        # 构建特征值字符串
        feature_values = []
        for feat_idx, feature in enumerate(meta.features):
            value = x_train[idx, feat_idx]
            # 根据特征类型决定格式化方式
            if feature.type == "float" and isinstance(value, (float, np.float32, np.float64)):
                # 浮点型特征保留3位小数
                value_repr = f"{value:.3f}"
            elif feature.type == "int" and isinstance(value, (float, np.float32, np.float64)):
                # 整数型特征保持整数形式
                if value.is_integer():
                    value_repr = str(int(value))
                else:
                    value_repr = str(value)
            else:
                # 分类特征或其他类型直接转字符串
                value_repr = str(value)
            feature_values.append(value_repr)
        
        # 获取标签
        label_value = y_train[idx]
        label_name = "Unknown"
        for label in meta.labels:
            if label.value == label_value:
                label_name = label.name
                break
        
        # 使用简洁的一行格式展示样本
        sample_str = f"({i+1}) {', '.join(feature_values)} {label_name}"
        prompt_parts.append(sample_str)
    
    prompt_parts.append("")
    
    # ------ 决策树要求 ------
    prompt_parts.append("# Decision Tree Requirements")
    prompt_parts.append(f"please generate a decision tree with a maximum depth of {max_depth} (which means {max_depth-1} levels because the max_depth includes the root node).")
    prompt_parts.append("1. Type Matching: Use integers for int features and floating-point numbers for float features.")
    prompt_parts.append("2. Decimal Precision: Use precision suited to each feature's scale - typically up to 3 decimals. Avoid overly precise thresholds (e.g., 0.165) when simpler ones (e.g., 0.22) better match value ranges.")
    prompt_parts.append("3. Use different features for each split.")
    prompt_parts.append("4. Rules must follow this exact format:")
    prompt_parts.append(f"4.1 Mathematical Formalism: Let depth d ∈ [1, {max_depth}], for any rule r, |r.conditions| = d-1. Therefore when d=3: ∀r, |r.conditions|=2") 
    prompt_parts.append("   (N) IF condition [AND condition] THEN label_1")
    prompt_parts.append(f"MOST IMPORTANT: The decision tree MUST satisfy: ∀rule∈Tree, len(conditions) = {max_depth-1}. Mathematical proof required: depth={max_depth} ⇒ each path has exactly {max_depth-1} splits ⇒ {max_depth-1} conditions per rule")
    prompt_parts.append("5. If you are highly confident (e.g. >0.95) that a simple rule or shallow tree is sufficient, you may generate a tree with only 1 level. Otherwise, try to use 2 levels and make the splits as full as possible.")
    prompt_parts.append("6. You must analyse the features about wether they have categories and use the proper way to generate the rules.")
    prompt_parts.append("7. For features without Categories:you can only use '>=' and '<' conditions,Example: persons >= 5 ")
    prompt_parts.append("8. For features that provides you Categories:you can only use '=' and '!=' conditions and only use the categories name that I provided,Example: maintenance = high")
    prompt_parts.append("9. For categorical features: Every '=' condition must be followed by a complementary '!=' condition to ensure exhaustive coverage of all category possibilities. ")
    prompt_parts.append("10. After splitting on a categorical feature (using '=' in any rule), all subsequent rules for that feature must use '!=' conditions. Example: If rule (1) uses 'size = big', then other rules  cannot use 'size = small' - they must use 'size != big' for further splits.")
    prompt_parts.append("11. A categorical feature can only be assigned ONE specific '=' value in the ENTIRE tree. Once a value is chosen (e.g. price = vhigh), other rules must use '!=' for this value instead of creating new '=' conditions with different values.")
    prompt_parts.append("12. IMPORTANT: If the rules are already able to cover all the labels,please stop generating rules and do not generate any other rules.Wrong examples:(1) IF shell_weight >= 0.5 THEN older(2) IF shell_weight < 0.5 THEN younger(3) IF length >= 0.5 THEN older(4) IF length < 0.5 THEN younger(the later two rules are redundant)")
    prompt_parts.append("")

    # ------ 决策树大师引导与英文推理要求 ------
    prompt_parts.append("# Instructions for LLM")
    prompt_parts.append("You are a Decision Tree Generation Master. Your task is to analyze the following data features and generate a decision tree for classification.")
    prompt_parts.append("For each decision tree you construct, you must leverage your domain knowledge and expertise in this field to guide the feature selection, splitting, and rule generation process.")
    prompt_parts.append("Please directly generate the decision tree rules in the format specified below. Do not include any additional explanations or comments.")
    prompt_parts.append("")
    prompt_parts.append("BEGIN_TREE")
    prompt_parts.append("(1) IF condition1 AND condition2 THEN no")
    prompt_parts.append("END_TREE")
    prompt_parts.append("Each rule must be a single line, start with a number in parentheses, and follow the format: (N) IF ... THEN ... Only use this format. Do NOT use any format like 'Rule N: ...' or with ELSE or jumps. All rules must be complete and mutually exclusive if needed.")
    prompt_parts.append("")
    
    # Example Rules
    prompt_parts.append("# Example: 1 level of split (only one condition per rule)")
    prompt_parts.append("(1) IF temperature >= 38 THEN high_fever")
    prompt_parts.append("(2) IF temperature < 38 THEN normal")
    prompt_parts.append("...")

    # Two levels of split (depth = 2)
    prompt_parts.append("# Example: 2 levels of split (two conditions per rule)")
    prompt_parts.append("(1) IF income >= 50000.50 AND credit_score >= 700 THEN approve")
    prompt_parts.append("(2) IF income >= 50000.50 AND credit_score >= 600 THEN reject")
    prompt_parts.append("(3) IF income < 50000.50 AND credit_score < 600 THEN reject")
    prompt_parts.append("(4) IF income < 50000.50 AND credit_score >= 600 THEN reject")
    
    prompt_parts.append("NOTE: This is a 2 levels of split, since some rules use 2 features:")
    prompt_parts.append("(1) IF income >= 50000.50 THEN approve")
    prompt_parts.append("(2) IF income < 50000.50 AND credit_score < 600 THEN approve")
    prompt_parts.append("(3) IF income < 50000.50 AND credit_score >= 600 THEN reject")
    prompt_parts.append("...")
    
    prompt_parts.append(f"Please generate corrected tree rules for a tree of max depth of {max_depth} (i.e., {max_depth - 1} levels of splits)")
    prompt_parts.append(f"Remember you can use {max_depth - 1} features at most for one rule in the tree, and you should generate {2**(max_depth - 1)} rules at most.")

    # ============== 最终组装 ==============
    full_prompt = "\n".join(prompt_parts)

    if not hasattr(generate_LLM_tree_prompt, "_printed"):
        logger.log("=============== Complete Prompt ===============")
        logger.log(full_prompt)
        logger.log(f"Prompt length: {len(full_prompt)} characters")
        generate_LLM_tree_prompt._printed = True
    
    return {
        "prompt": full_prompt.strip(),
        "label_dist": label_dist,
        "num_examples": len(examples),
        "examples": examples
    }
