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
    """获取ToT对特征重要性的排序"""
    prompt = create_feature_ranking_prompt(meta)
    logger.log(f"请求特征重要性排序，提示词:\n{prompt}")
    
    for responses in runner.run([prompt]):
        for response in responses:
            logger.log(f"收到ToT响应:\n{response}")
            
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



def generate_ToT_tree_prompt(
meta: DatasetMeta,
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: int = 3,  
    num_examples: int = 5,
    current_path: str = "Root",
    applied_rules: str = "None"
) -> dict:
    """
    生成决策树构建提示词
    返回值：dict，包括prompt、label_dist、num_examples等
    """
    # ================= 输入校验 =================
    if not hasattr(meta, 'features') or len(meta.features) == 0:
        logger.log("特征元数据为空，请检查meta文件")
        return {"prompt": "", "label_dist": {}, "num_examples": 0}
        
    if x_train.shape[1] != len(meta.features):
        logger.log(f"特征数量不匹配：数据有{x_train.shape[1]}列，元数据定义{len(meta.features)}个特征")
        return {"prompt": "", "label_dist": {}, "num_examples": 0}




    # ============== 构建新版英文任务描述和提示词 ==============
    prompt_parts = []
    prompt_parts.append("#Task Description")
    prompt_parts.append("You are a decision tree generator. Your task is to generate candidate split rules for the current node. Please strictly follow the requirements below.Please don't explain.")
    prompt_parts.append("#Core Requirements")
    prompt_parts.append("Generate mutually exclusive rule pairs forming complete branches")
    prompt_parts.append("Directly assign final class (younger/older) to leaf nodes")
    prompt_parts.append("Mark intermediate nodes with [NODE]")
    prompt_parts.append("Always inherit parent path conditions")
    prompt_parts.append(f"When parent's used features reach 1, generate leaf nodes instead of new nodes")
    prompt_parts.append("#Structural Constraints")
    prompt_parts.append("Each split must form complete complementary conditions")
    prompt_parts.append("Continuous features use single-threshold splits")
    prompt_parts.append("Categorical features use explicit category combinations")
    prompt_parts.append("Path conditions are automatically inherited without repetition")
    prompt_parts.append("Final class labels must appear in leaf node rules")
    prompt_parts.append(f"Next split produces only leaves when parent's feature count ≥ 1")
    prompt_parts.append("## Input Context")
    prompt_parts.append("**Current Path:** ")
    prompt_parts.append(f"{current_path}")
    prompt_parts.append("]\n")

    # # ============== 数据集介绍部分 ==============
    feature_lines = []
    for feat in meta.features:
        desc = getattr(feat, 'desc', 'No description')
        unit = ''
        m = re.search(r'\(([^)]+)\)', desc)
        if m:
            unit = m.group(1)
        line = f"{feat.name}: {desc}"
        if unit:
            line += f" [unit: {unit}]"
        if feat.is_categorical:
            categories = [f"{k}({v})" for k, v in feat.categories.items()]
            line += f" | Categories: {', '.join(categories)}"
        feature_lines.append(line)
    prompt_parts.append("# Feature Definitions")
    prompt_parts.append("; ".join(feature_lines))  # 用分号拼接，紧凑排列

    label_lines = [f"{label.name}: {getattr(label, 'desc', 'No description')}" for label in meta.labels]
    prompt_parts.append("# Label Definitions")
    prompt_parts.append("; ".join(label_lines))  # 同样紧凑排列

    prompt_parts.append("## Output Specifications")
    prompt_parts.append("Generate one rule pairs that must:")
    prompt_parts.append("- Use identical feature combinations per pair")
    prompt_parts.append("- Cover all possible value ranges")
    prompt_parts.append("- Prioritize features with maximum information gain")
    prompt_parts.append("- Maintain tree structure compatibility")

    prompt_parts.append("## Examples")
    prompt_parts.append("parent is Root->generate one")
    prompt_parts.append("IF size = big THEN yes")
    prompt_parts.append("IF size != big THEN [NODE]")
    prompt_parts.append("")
    prompt_parts.append("Have parents->Add AND condition")
    prompt_parts.append("IF age >= 9 AND height < 1.3 THEN [NODE]")
    prompt_parts.append("IF age >= 9 AND height < 1.3 THEN older")
    prompt_parts.append("")
    

    # ============== 最终组装 ==============
    full_prompt = "\n".join(prompt_parts)

    return {
        "prompt": full_prompt.strip(),
    }
