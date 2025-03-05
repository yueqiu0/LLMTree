import yaml
import sklearn.datasets
import pandas as pd
import numpy as np

from .common_args import DatasetArgs


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

    # shuffle
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    if args.shuffle_column:
        indices = np.arange(x.shape[1])
        np.random.shuffle(indices)
        x = x[:, indices]
        meta.shuffle_features(indices)

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
