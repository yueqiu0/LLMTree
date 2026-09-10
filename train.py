from argparse import ArgumentParser
import yaml
from pathlib import Path
import random
import numpy as np
from sklearn.metrics import roc_auc_score
import time
import json
import os

import tree_prompt.logger as logger
from tree_prompt.model.strategy import (
    TrainStrategy,
    SingleStrategy,
    LLMTForestStrategy,
)
from tree_prompt.prompt import (
    TabularSerializer,
    ListSerializer,
    TextSerializer,
)
from tree_prompt.model import Classifier
from tree_prompt.common_args import (
    DatasetArgs,
    OpenAIAPIArgs,
    HuggingChatArgs,
    TogetherAPIArgs,
    LLMTForestStrategyArgs,
)
from tree_prompt.dataset import load_dataset_split, sample_balanced


def _get_missing_fields(instance: any, prefix: str = None) -> list[str]:
    missing_fields = []
    for k, v in instance.__dict__.items():
        if isinstance(instance, LLMTForestStrategyArgs) and k == "tree_parallelism":
            continue
        if v is None:
            missing_fields.append(k if not prefix else f"{prefix}.{k}")
    return missing_fields


def _load_from_dict(instance: any, config: dict):
    for k in instance.__dict__.keys():
        if k in config:
            setattr(instance, k, config[k])


def _merge_dict(a: dict, b: dict):
    for k in b.keys():
        if k in a and isinstance(a[k], dict) and isinstance(b[k], dict):  # noqa
            _merge_dict(a[k], b[k])
        else:
            a[k] = b[k]


class Repr:
    def __repr__(self):
        return self.__dict__.__repr__()


class SingleStrategyArgs(Repr):
    def __init__(self) -> None:
        self.max_depth: int = None
        self.tau: float = 0.70


class TrainArgs(Repr):
    def __init__(self) -> None:
        self.exp_name: str = None
        self.strategy: str = None
        self.strategy_args: SingleStrategyArgs | LLMTForestStrategyArgs = None
        self.runner: str = None
        self.runner_args: OpenAIAPIArgs | HuggingChatArgs = None
        self.dataset_args: DatasetArgs = None
        self.output_dir: str = None
        self.random_seed: int = None
        self.train_sizes: list[int] = None
        self.train_batch: int = None
        self.num_tests_per_set: int = None
        self.test_size: int = None
        self.test_batch: int = None
        self.serializer_type: str = None
        self.exp_id: str = ""

    def get_missing_fields(self) -> list[str]:
        missing_fields = _get_missing_fields(self)

        if (
            (
                self.runner == "openai_api"
                and not isinstance(self.runner_args, OpenAIAPIArgs)
            )
            or (
                self.runner == "huggingchat"
                and not isinstance(self.runner_args, HuggingChatArgs)
            )
            or (
                self.runner == "together_api"
                and not isinstance(self.runner_args, TogetherAPIArgs)
            )
        ):
            missing_fields.append("runner_args")

        if (
            self.runner == "openai_api"
            and self.runner_args.api_base == ""
            and self.runner_args.openai_api_key == ""
        ):
            missing_fields.append("runner_args.openai_api_key")

        if (
            self.runner == "together_api"
            and self.runner_args.api_base == ""
            and self.runner_args.openai_api_key == ""
        ):
            missing_fields.append("runner_args.together_api_key")

        if (
            self.strategy == "single"
            and not isinstance(self.strategy_args, SingleStrategyArgs)
            or self.strategy == "llmt_forest"
            and not isinstance(self.strategy_args, LLMTForestStrategyArgs)
        ):
            missing_fields.append("strategy_args")

        if self.runner_args:
            missing_fields += _get_missing_fields(self.runner_args, "runner_args")

        if self.dataset_args:
            missing_fields += _get_missing_fields(self.dataset_args, "dataset")

        if self.strategy_args:
            missing_fields += _get_missing_fields(self.strategy_args, "strategy_args")

        return missing_fields

    def load_sub_args(self):
        assert isinstance(self.runner_args, dict)
        assert isinstance(self.strategy_args, dict)
        assert isinstance(self.dataset_args, dict)

        runner_args_dict = self.runner_args
        strategy_args_dict = self.strategy_args
        dataset_args_dict = self.dataset_args

        if self.runner == "openai_api":
            self.runner_args = OpenAIAPIArgs()
        elif self.runner == "huggingchat":
            self.runner_args = HuggingChatArgs()
        elif self.runner == "together_api":
            self.runner_args = TogetherAPIArgs()
        else:
            raise ValueError("Unknown or missing runner_type: {}".format(self.runner))

        _load_from_dict(self.runner_args, runner_args_dict)

        if self.strategy == "single":
            self.strategy_args = SingleStrategyArgs()
        elif self.strategy == "llmt_forest":
            self.strategy_args = LLMTForestStrategyArgs()
        else:
            raise ValueError(
                "Unknown or missing strategy_type: {}".format(self.strategy)
            )

        _load_from_dict(self.strategy_args, strategy_args_dict)

        self.dataset_args = DatasetArgs()
        _load_from_dict(self.dataset_args, dataset_args_dict)


def parse_args() -> TrainArgs:
    """
    Parse the arguments from, with descending priority:
      1. command line
      2. config file
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config file (yaml)")

    parser.add_argument("--exp-name", type=str, help="experiment name")

    # strategy
    parser.add_argument(
        "--strategy",
        type=str,
        choices=("single", "llmt_forest"),
        help="strategy type (single or llmt_forest)",
    )
    parser.add_argument("--max-depth", type=int, help="max depth of the tree")
    parser.add_argument("--num-trees", type=int, help="number of trees in the forest")
    parser.add_argument(
        "--tree-parallelism",
        type=int,
        help="maximum concurrent LLMT Forest tree builders (capped by API parallel_batch_size)",
    )
    parser.add_argument("--train-batch", type=int, help="train batch size")
    parser.add_argument("--tau", type=float, help="supervision confidence threshold")

    # runner
    parser.add_argument(
        "--runner", type=str, help="runner type (open_api, huggingchat)"
    )
    parser.add_argument("--openai-api-base", type=str, help="openai api base url")
    parser.add_argument("--openai-api-key", type=str, help="openai api key")
    parser.add_argument("--hf-username", type=str, help="huggingface username")
    parser.add_argument("--hf-password", type=str, help="huggingface password")
    parser.add_argument("--hf-cookie-dir", type=str, help="huggingface cookie dir")
    parser.add_argument("--together-api-key", type=str, help="together api key")
    parser.add_argument("--together-api-base", type=str, help="together api base url")
    parser.add_argument("--model-name", type=str, help="model name")

    # dataset
    parser.add_argument(
        "--dataset-data-file",
        type=str,
        help="path to dataset file (libsvm or csv format)",
    )
    parser.add_argument(
        "--dataset-meta-file", type=str, help="path to dataset meta file (yaml)"
    )
    parser.add_argument(
        "--dataset-format", type=str, help="dataset format (libsvm or csv)"
    )
    parser.add_argument(
        "--shuffle-column", type=int, help="whether to shuffle feature order"
    )

    parser.add_argument("--output-dir", type=str, help="output directory")
    parser.add_argument("--random-seed", type=int, help="random seed for the tree")
    parser.add_argument("--serializer", type=str, help="serializer type")

    parser.add_argument("--train-sizes", type=int, nargs="+", help="train set sizes")
    parser.add_argument(
        "--num-tests-per-set", type=int, help="number of tests per training set size"
    )
    parser.add_argument("--test-size", type=int, help="test set size")
    parser.add_argument(
        "--test-batch",
        type=int,
        help="number of tests presented to the model per request",
    )
    parser.add_argument("--timeout", type=int, help="timeout per request, in seconds")
    parser.add_argument(
        "--request-interval", type=float, help="interval between requests, in seconds"
    )
    parser.add_argument("--parallel-batch-size", type=int, help="parallel batch size")

    parser.add_argument("--exp-id", type=str, help="experiment id for display")

    cml_args = parser.parse_args()

    args = TrainArgs()

    sup_config_file_path = cml_args.config
    config_file_paths = []
    if sup_config_file_path or config_file_paths:
        config = {}
        config_sup = {}
        if sup_config_file_path:
            with open(sup_config_file_path) as f:
                config_sup = yaml.safe_load(f)

            sup_config_dir_path = Path(sup_config_file_path).parent
            if "base_configs" in config_sup:
                config_file_paths = [
                    sup_config_dir_path / p for p in config_sup["base_configs"]
                ]

        for config_file_path in config_file_paths:
            with open(config_file_path) as f:
                config_part: dict = yaml.safe_load(f)
                _merge_dict(config, config_part)

        _merge_dict(config, config_sup)

        config_config: dict = config.get("config")
        if config_config:
            args.exp_name = config_config.get("exp_name")

            args.strategy = config_config.get("strategy")
            args.strategy_args = config_config.get("strategy_args")
            args.runner = config_config.get("runner")
            args.runner_args = config_config.get("runner_args")
            args.dataset_args = config_config.get("dataset_args")
            args.output_dir = config_config.get("output_dir")
            args.random_seed = config_config.get("random_seed")
            args.train_sizes = config_config.get("train_sizes")
            args.train_batch = config_config.get("train_batch")
            args.num_tests_per_set = config_config.get("num_tests_per_set")
            args.test_size = config_config.get("test_size")
            args.test_batch = config_config.get("test_batch")
            args.serializer_type = config_config.get("serializer")

    runner_args_dict = {}
    strategy_args_dict = {}
    dataset_args_dict = {}

    if cml_args.exp_name is not None:
        args.exp_name = cml_args.exp_name
    if cml_args.runner is not None:
        args.runner = cml_args.runner
    if cml_args.openai_api_base is not None:
        runner_args_dict["api_base"] = cml_args.openai_api_base
    if cml_args.openai_api_key is not None:
        runner_args_dict["openai_api_key"] = cml_args.openai_api_key
    if cml_args.hf_username is not None:
        runner_args_dict["hf_username"] = cml_args.hf_username
    if cml_args.hf_password is not None:
        runner_args_dict["hf_password"] = cml_args.hf_password
    if cml_args.hf_cookie_dir is not None:
        runner_args_dict["hf_cookie_dir"] = cml_args.hf_cookie_dir
    if cml_args.together_api_key is not None:
        runner_args_dict["together_api_key"] = cml_args.together_api_key
    if cml_args.together_api_base is not None:
        runner_args_dict["together_api_base"] = cml_args.together_api_base
    if cml_args.model_name is not None:
        runner_args_dict["model_name"] = cml_args.model_name
    if cml_args.timeout is not None:
        runner_args_dict["timeout"] = cml_args.timeout
    if cml_args.request_interval is not None:
        runner_args_dict["request_interval"] = cml_args.request_interval
    if cml_args.parallel_batch_size is not None:
        runner_args_dict["parallel_batch_size"] = cml_args.parallel_batch_size

    if cml_args.strategy is not None:
        args.strategy = cml_args.strategy
    if cml_args.max_depth is not None:
        strategy_args_dict["max_depth"] = cml_args.max_depth
    if cml_args.num_trees is not None:
        strategy_args_dict["num_trees"] = cml_args.num_trees
    if cml_args.tree_parallelism is not None:
        strategy_args_dict["tree_parallelism"] = cml_args.tree_parallelism
    if cml_args.train_batch is not None:
        args.train_batch = cml_args.train_batch
    if cml_args.tau is not None:
        strategy_args_dict["tau"] = cml_args.tau

    if cml_args.output_dir is not None:
        args.output_dir = cml_args.output_dir
    if cml_args.random_seed is not None:
        args.random_seed = cml_args.random_seed

    if cml_args.dataset_data_file is not None:
        dataset_args_dict["data_file"] = cml_args.dataset_data_file
    if cml_args.dataset_meta_file is not None:
        dataset_args_dict["meta_file"] = cml_args.dataset_meta_file
    if cml_args.dataset_format is not None:
        dataset_args_dict["format"] = cml_args.dataset_format
    if cml_args.shuffle_column is not None:
        dataset_args_dict["shuffle_column"] = not (cml_args.shuffle_column == 0)

    if cml_args.serializer is not None:
        args.serializer_type = cml_args.serializer
    if cml_args.train_sizes is not None:
        args.train_sizes = cml_args.train_sizes
    if cml_args.num_tests_per_set is not None:
        args.num_tests_per_set = cml_args.num_tests_per_set
    if cml_args.test_size is not None:
        args.test_size = cml_args.test_size
    if cml_args.test_batch is not None:
        args.test_batch = cml_args.test_batch

    if cml_args.exp_id is not None:
        args.exp_id = cml_args.exp_id
    # read openai api key from env
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if (
        openai_api_key
        and not runner_args_dict.get("openai_api_key")
        and not (args.runner_args or {}).get("openai_api_key")
    ):
        runner_args_dict["openai_api_key"] = openai_api_key

    if args.runner_args is None:
        args.runner_args = {}
    if args.strategy_args is None:
        args.strategy_args = {}
    if args.dataset_args is None:
        args.dataset_args = {}
    _merge_dict(args.runner_args, runner_args_dict)
    _merge_dict(args.strategy_args, strategy_args_dict)
    _merge_dict(args.dataset_args, dataset_args_dict)
    args.load_sub_args()
    missing_fields = args.get_missing_fields()

    if len(missing_fields) > 0:
        raise ValueError("Incomplete arguments: missing {}".format(missing_fields))

    return args


def evaluate(x_train, y_train, x_test, y_test, model, test_batch):
    start = time.time()
    model.fit(x_train, y_train)
    elapsed = time.time() - start
    predictions = []
    for start in range(0, len(x_test), test_batch):
        predictions.extend(model.predict(x_test[start : start + test_batch]))
    labels = np.asarray(y_test).astype(int)
    accuracy = float(np.mean(labels == predictions)) if len(labels) else None
    auc = None
    try:
        classes = np.unique(labels)
        if len(classes) == 2:
            auc = float(
                roc_auc_score(
                    labels == classes[1], np.asarray(predictions) == classes[1]
                )
            )
        elif len(classes) > 2:
            from sklearn.preprocessing import label_binarize

            targets = label_binarize(labels, classes=classes)
            scores = label_binarize(predictions, classes=classes)
            auc = {
                average: float(roc_auc_score(targets, scores, average=average))
                for average in ("macro", "micro")
            }
    except ValueError:
        pass
    return {
        "tree_acc": accuracy,
        "tree_auc": auc,
        "tree_results": predictions,
        "labels": labels.tolist(),
        "model": model.export(),
        "train_elapsed": elapsed,
        "token_stats": model.strategy.get_token_stats(),
    }


def load_args(
    args: TrainArgs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TrainStrategy]:
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    meta, avail_x, avail_y, test_x, test_y = load_dataset_split(
        args.dataset_args, args.test_size, random_seed=args.random_seed
    )

    if args.serializer_type == "tabular":
        serializer = TabularSerializer(meta)
    elif args.serializer_type == "list":
        serializer = ListSerializer(meta)
    elif args.serializer_type == "text":
        serializer = TextSerializer(meta)
    else:
        raise ValueError("Unknown serializer type: {}".format(args.serializer_type))

    if args.runner == "openai_api":
        from tree_prompt.runner.openai_api import OpenAIAPIParallelRunner

        runner = OpenAIAPIParallelRunner(
            args.runner_args.api_base,
            args.runner_args.model_name,
            args.runner_args.openai_api_key,
            args.runner_args.request_interval,
            args.runner_args.timeout,
            args.runner_args.parallel_batch_size,
        )

    elif args.runner == "huggingchat":
        from tree_prompt.runner.huggingchat import HuggingChatParallelRunner

        runner = HuggingChatParallelRunner(
            args.runner_args.hf_username,
            args.runner_args.hf_password,
            args.runner_args.hf_cookie_dir,
            args.runner_args.request_interval,
            args.runner_args.timeout,
            args.runner_args.parallel_batch_size,
        )
    else:
        raise ValueError("Unknown runner type: {}".format(args.runner))
    if args.strategy == "single":
        strategy = SingleStrategy(
            runner,
            serializer,
            args.strategy_args.max_depth,
            args.train_batch,
            tau=args.strategy_args.tau,
        )
    elif args.strategy == "llmt_forest":
        strategy = LLMTForestStrategy(
            meta,
            runner,
            args.serializer_type,
            args.strategy_args.num_trees,
            args.strategy_args.max_depth,
            args.train_batch,
            args.strategy_args.tree_parallelism,
            args.random_seed,
            tau=args.strategy_args.tau,
        )

    else:
        raise ValueError("Unknown strategy type: {}".format(args.strategy))

    return avail_x, avail_y, test_x, test_y, strategy


def main():
    args = parse_args()
    avail_x, avail_y, test_x, test_y, strategy = load_args(args)
    output_file = Path(args.output_dir) / (args.exp_name + ".json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        raise FileExistsError(output_file)
    results = {}
    for train_size in args.train_sizes:
        records = results.setdefault(train_size, [])
        for train_x, train_y in sample_balanced(
            avail_x, avail_y, args.num_tests_per_set, train_size, args.random_seed
        ):
            records.append(
                evaluate(
                    train_x,
                    train_y,
                    test_x,
                    test_y,
                    Classifier(strategy),
                    args.test_batch,
                )
            )

            def encode(value):
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
                return value.__dict__

            saved_args = dict(args.__dict__)
            saved_args["runner_args"] = {
                k: v
                for k, v in vars(args.runner_args).items()
                if not any(word in k for word in ("key", "password"))
            }
            output_file.write_text(
                json.dumps(
                    {"args": saved_args, "results": results}, indent=2, default=encode
                )
            )
        scores = [r["tree_acc"] for r in records]
        logger.log(
            f"train_size={train_size}: accuracy={np.mean(scores):.4f} ± {np.std(scores):.4f}"
        )


if __name__ == "__main__":
    main()
