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

        def __init__(self, name="", desc="", type="", categories=None):
            self.name = name
            self.desc = desc
            self.type = type
            self.categories = categories or {}

        def __repr__(self) -> str:
            return str(self.__dict__)

        @classmethod
        def from_dict(cls, data):
            """从字典创建Feature对象"""
            feature = cls()
            feature.name = data.get("name", "")
            feature.desc = data.get("desc", "")
            feature.type = data.get("type", "")
            feature.categories = data.get("categories", {})
            return feature

    class Label:
        name: str
        value: float
        desc: str

        def __init__(self, name="", value=0, desc=""):
            self.name = name
            self.value = value
            self.desc = desc

        def __repr__(self) -> str:
            return str(self.__dict__)

        @classmethod
        def from_dict(cls, data):
            """从字典创建Label对象"""
            label = cls()
            label.name = data.get("name", "")
            label.value = data.get("value", 0)
            label.desc = data.get("desc", "")
            return label

    def __init__(self, name, desc='', target='', label_meaning='', features=None, labels=None):
        self.name = name
        self.desc = desc
        self.target = target
        self.label_meaning = label_meaning  
        self.features = features or []
        self.labels = labels or []
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

    @classmethod
    def from_dict(cls, meta_dict):
        """从字典创建数据集元数据"""
        name = meta_dict.get('name', '')
        desc = meta_dict.get('desc', '')
        target = meta_dict.get('target', '')
        label_meaning = meta_dict.get('label_meaning', '') 
    
        return cls(
            name=name,
            desc=desc,
            target=target,
            label_meaning=label_meaning,
            features=[DatasetMeta.Feature.from_dict(f) for f in meta_dict.get('features', [])],
            labels=[DatasetMeta.Label.from_dict(l) for l in meta_dict.get('labels', [])]
        )


def load_meta(path: str) -> DatasetMeta:
    with open(path, "r") as f:
        data: dict = yaml.safe_load(f)

    meta = DatasetMeta.from_dict(data)
    return meta


def _dummy(num_features: int) -> DatasetMeta:
    meta = DatasetMeta(
        name="dummy",
        desc="",
        target="",
        label_meaning="result",
        features=[DatasetMeta.Feature(name=f"feature_{i+1}", desc="", type="float") for i in range(num_features)],
        labels=[DatasetMeta.Label(name="no", value=0, desc=""), DatasetMeta.Label(name="yes", value=1, desc="")]
    )
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

    original_feature_names = [feature.name for feature in meta.features]
    
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    if args.shuffle_column:
        indices = np.arange(x.shape[1])
        np.random.shuffle(indices)
        x = x[:, indices]
        
        feature_shuffle_map = {new_idx: int(old_idx) for new_idx, old_idx in enumerate(indices)}
        
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
    all_samples = []
    classes = np.unique(y)

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

    return all_samples


def create_feature_ranking_prompt(meta: DatasetMeta) -> str:
    prompt = f"""As a data analyst, you need to determine which features are most important for predicting {meta.label_meaning or "the target variable"}.

Dataset Information:
"""
    if meta.target:
        prompt += f"Task: {meta.target}\n\n"
    
    prompt += "The dataset consists of the following features:\n"
    for i, feat in enumerate(meta.features):
        prompt += f"{i+1}. {feat.name}: {feat.desc or 'No description available'}\n"
        
        if feat.is_categorical and feat.categories:
            prompt += "   Possible values:\n"
            for cat_value, cat_desc in feat.categories.items():
                prompt += f"   - {cat_value}: {cat_desc}\n"
    
    prompt += f"\nTarget Variable: {meta.label_meaning or 'The output'}\n"
    
    prompt += "Possible values:\n"
    for label in meta.labels:
        desc = f" ({label.desc})" if label.desc else ""
        prompt += f"- {label.name}{desc}\n"
    
    prompt += """
Based on common knowledge and intuition about this kind of data, rank the features from most important to least important for predicting the target variable.

Please return your answer as a comma-separated list of feature indices, ordered from most important to least important. For example: 2,4,1,3

Feature Ranking: """
    return prompt
