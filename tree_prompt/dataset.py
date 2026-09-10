import yaml
import sklearn.datasets
import pandas as pd
import numpy as np
from pathlib import Path

from .common_args import DatasetArgs
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
        self.label_meaning: str = ""
        self.feature_shuffle_map: dict[int, int] = {}

    def get_label(self, id: int) -> Label:
        return self.labels[id]

    def find_label(self, value: float) -> Label:
        for label in self.labels:

            if label.value == value:
                return label

            try:
                if float(label.value) == float(value):
                    return label
            except (ValueError, TypeError):
                pass
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
    meta.label_meaning = data.get("label_meaning")

    features: list[dict] = data.get("features", [])
    for i, feat in enumerate(features):
        if not isinstance(feat, dict):
            raise ValueError(f"Feature at index {i} in {path} is not a dictionary: {feat}")
        if "name" not in feat:
            raise ValueError(f"Feature at index {i} in {path} is missing 'name' field. Feature data: {feat}")
        feature = meta.Feature()
        feature.name = feat["name"]
        feature.desc = feat.get("desc", "")
        feature.type = feat.get("type", "float")
        feature.categories = feat.get("categories")
        meta.features.append(feature)

    labels: list[dict] = data.get("labels", [])
    for i, l in enumerate(labels):
        if not isinstance(l, dict):
            raise ValueError(f"Label at index {i} in {path} is not a dictionary: {l}")
        if "name" not in l:
            raise ValueError(f"Label at index {i} in {path} is missing 'name' field. Label data: {l}")
        label = meta.Label()
        label.name = str(l["name"])  #  name
        label.value = l.get("value", i)
        label.desc = l.get("desc", "")
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
        # Read CSV with string dtype to prevent automatic boolean conversion
        df = pd.read_csv(args.data_file, dtype=str)
        x = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
    else:
        raise ValueError("Unknown dataset format: {}".format(args.format))


    original_feature_names = [feature.name for feature in meta.features]
    
    # shuffle
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    if args.shuffle_column:
        # =====  =====
        indices = np.arange(x.shape[1])
        np.random.shuffle(indices)
        x = x[:, indices]
        

        logger.log("\n" + "="*50)
        logger.log("")
        

        feature_shuffle_map = {new_idx: int(old_idx) for new_idx, old_idx in enumerate(indices)}
        logger.log(f": {feature_shuffle_map}")
        

        logger.log(":")
        for new_idx, old_idx in feature_shuffle_map.items():
            old_feature_name = original_feature_names[old_idx] if old_idx < len(original_feature_names) else f"({old_idx})"
            logger.log(f"   {new_idx} ←  {old_idx} ({old_feature_name})")
        
        logger.log("="*50 + "\n")
        

        meta.shuffle_features(indices)
        meta.feature_shuffle_map = feature_shuffle_map
    
    return meta, x, y


def _fixed_split_paths(data_file: str, random_seed: int = 0) -> tuple[Path, Path]:
    """Return the precomputed partition paths for the selected seed."""
    source = Path(data_file)
    dataset_name = source.parent.name
    split_dir = source.parent / f"cv{random_seed}"
    return (
        split_dir / f"{dataset_name}_cv{random_seed}train_all.csv",
        split_dir / f"{dataset_name}_cv{random_seed}test.csv",
    )


def _shuffle_fixed_split_columns(
    meta: DatasetMeta, train_x: np.ndarray, test_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the legacy feature permutation consistently to both partitions."""
    indices = np.arange(train_x.shape[1])
    np.random.shuffle(indices)
    train_x = train_x[:, indices]
    test_x = test_x[:, indices]

    original_feature_names = [feature.name for feature in meta.features]
    feature_shuffle_map = {new_idx: int(old_idx) for new_idx, old_idx in enumerate(indices)}
    logger.log("\n" + "=" * 50)
    logger.log("")
    logger.log(f": {feature_shuffle_map}")
    logger.log(":")
    for new_idx, old_idx in feature_shuffle_map.items():
        logger.log(f"   {new_idx} ←  {old_idx} ({original_feature_names[old_idx]})")
    logger.log("=" * 50 + "\n")

    meta.shuffle_features(indices)
    meta.feature_shuffle_map = feature_shuffle_map
    return train_x, test_x


def load_dataset_split(
    args: DatasetArgs, test_size: int, random_seed: int = 0
) -> tuple[DatasetMeta, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load cv0--cv4 for seeds 0--4; otherwise split the full dataset.

    The fixed train/test row order is intentional.  It reproduces the published
    selected split while keeping feature-column shuffling compatible with the
    previous loader. The caller initializes NumPy's RNG with random_seed.
    """
    if random_seed not in range(5):
        meta, x, y = load_dataset(args)
        return meta, x[test_size:], y[test_size:], x[:test_size], y[:test_size]
    train_path, test_path = _fixed_split_paths(args.data_file, random_seed)
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Fixed split must contain both train and test files: "
            f"{train_path}, {test_path}"
        )
    if args.format != "csv":
        raise ValueError("Precomputed splits are supported only for CSV datasets")

    train_df = pd.read_csv(train_path, dtype=str)
    test_df = pd.read_csv(test_path, dtype=str)
    if list(train_df.columns) != list(test_df.columns):
        raise ValueError(f"Fixed split columns differ for {args.data_file}")
    if len(test_df) != test_size:
        raise ValueError(
            f"Configured test_size={test_size} does not match fixed split size "
            f"{len(test_df)} for {args.data_file}"
        )

    meta = load_meta(args.meta_file)
    # Metadata may use display names such as "Blood Pressure" for the CSV
    # column "BloodPressure". Compare in order, ignoring whitespace only.
    csv_names = ["".join(name.split()) for name in train_df.columns[:-1]]
    meta_names = ["".join(feature.name.split()) for feature in meta.features]
    if csv_names != meta_names or len(set(csv_names)) != len(csv_names):
        raise ValueError(f"Fixed split features do not match metadata for {args.data_file}")

    train_x = train_df.iloc[:, :-1].to_numpy()
    train_y = train_df.iloc[:, -1].to_numpy()
    test_x = test_df.iloc[:, :-1].to_numpy()
    test_y = test_df.iloc[:, -1].to_numpy()

    # Previously every invocation shuffled rows before columns.  Consume the
    # same RNG state so fixed partitions retain the seed-zero column mapping.
    np.random.shuffle(np.arange(len(train_df) + len(test_df)))
    if args.shuffle_column:
        train_x, test_x = _shuffle_fixed_split_columns(meta, train_x, test_x)

    return meta, train_x, train_y, test_x, test_y


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
                    min(
                        num_samples_per_group // len(classes)
                        if num_samples_per_group != 1
                        else 1,
                        len(np.where(y == l)[0])  # Don't exceed available samples
                    ),
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
