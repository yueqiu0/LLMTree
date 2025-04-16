import sklearn.datasets
import sklearn.metrics
from pathlib import Path
import numpy as np
from datetime import datetime
import json
import random
import jinja2
from tqdm import tqdm
from argparse import ArgumentParser
import yaml
from pathlib import Path
import os
import tree_prompt.prompt as prompt
import tree_prompt.dataset as dataset
from tree_prompt.external.tree import (
    DecisionTree,
    SimpleDecisionTree,
    XGBoostDecisionTree,
    RandomForestDecisionTree,
    FederatedDecisionTree,
    LLMDecisionTree,
)
from tree_prompt.prompt import (
    Serializer,
    TabularSerializer,
    ListSerializer,
    TextSerializer,
)
from tree_prompt.runner import Runner
import tree_prompt.logger as logger
from tree_prompt.common_args import (
    DatasetArgs,
    OpenAIAPIArgs,
    HuggingChatArgs,
    TogetherAPIArgs,
)
from tree_prompt.dataset import DatasetMeta, load_dataset, sample_balanced


def _get_missing_fields(instance: any, prefix: str = None) -> list[str]:
    missing_fields = []
    for k, v in instance.__dict__.items():
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


class SimpleTreeArgs:
    def __init__(self) -> None:
        self.max_depth: int = None

    def __repr__(self):
        return str(self.__dict__)


class XGBoostArgs:
    def __init__(self) -> None:
        self.max_depth: int = None
        self.num_trees: int = None

    def __repr__(self):
        return str(self.__dict__)

class LLMTreeArgs:
    def __init__(self) -> None:
        self.model_name: str = "togethercomputer/llama-2-70b-chat"
        self.temperature: float = 0
        self.max_retry: int = 3
        self.max_depth: int = 3
    
    def __repr__(self):
        return str(self.__dict__)
    
RandomForestArgs = XGBoostArgs
FederatedTreeArgs = XGBoostArgs


class EvaluateArgs:
    def __init__(self) -> None:
        self.exp_name: str = None
        self.runner: str = None
        self.runner_args: OpenAIAPIArgs | HuggingChatArgs | TogetherAPIArgs = None
        self.tree_type: str = None
        self.tree_args: SimpleTreeArgs | XGBoostArgs = None
        self.tree_only: bool = False
        self.output_dir: str = None
        self.random_seed: int = None
        self.dataset_args: DatasetArgs = None
        self.train_sizes: list[int] = None
        self.num_tests_per_set: int = None
        self.test_size: float = None
        self.test_batch: int = None
        self.use_tree_rules: bool = None
        self.template: str = None
        self.serializer_type: str = None
        self.exp_id: str = ""
        self.print_only: bool = False
        self.llm_gen_tree: bool = False
        self.llm_tree_args: LLMTreeArgs = LLMTreeArgs()  

    def get_missing_fields(self) -> list[str]:
        missing_fields = _get_missing_fields(self)

        # Runner参数检查（保持不变）
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
        elif self.runner_args:
            missing_fields += _get_missing_fields(self.runner_args, "runner_args")

        # Tree参数检查（添加LLM支持）
        if (
            (
                self.tree_type == "simple"
                and not isinstance(self.tree_args, SimpleTreeArgs)
            )
            or (
                self.tree_type == "xgboost"
                and not isinstance(self.tree_args, XGBoostArgs)
            )
            or (
                (self.tree_type == "random_forest" or self.tree_type == "federated")
                and not isinstance(self.tree_args, RandomForestArgs)
            )
            or (  # 新增LLM树类型检查
                self.tree_type == "llm"
                and not isinstance(self.tree_args, LLMTreeArgs)
            )
        ):
            missing_fields.append("tree_args")
        elif self.tree_args:
            missing_fields += _get_missing_fields(self.tree_args, "tree_args")

        # Dataset参数检查（保持不变）
        if self.dataset_args:
            missing_fields += _get_missing_fields(self.dataset_args, "dataset")

        return missing_fields

    def get_missing_fields_tree_only(self) -> list[str]:
        missing_fields = self.get_missing_fields()
        required_fields = [
            "exp_name",
            "tree_type",
            "tree_args",
            "output_dir",
            "random_seed",
            "dataset_args",
            "train_sizes",
            "num_tests_per_set",
            "test_size",
            "test_batch",
            "shuffle",
            "shuffle_column",
        ]
        tree_only_missing_fields = []

        for field in required_fields:
            for missing_field in missing_fields:
                if missing_field.startswith(field):
                    tree_only_missing_fields.append(missing_field)

        return tree_only_missing_fields

    def load_sub_args(self):
        assert isinstance(self.tree_args, dict)
        assert isinstance(self.dataset_args, dict)

        if not self.tree_only:
            assert isinstance(self.runner_args, dict)
            runner_args_dict = self.runner_args

            if self.runner == "openai_api":
                self.runner_args = OpenAIAPIArgs()
            elif self.runner == "huggingchat":
                self.runner_args = HuggingChatArgs()
            elif self.runner == "together_api":
                self.runner_args = TogetherAPIArgs()
            else:
                raise ValueError("Unknown runner type: {}".format(self.runner))

            _load_from_dict(self.runner_args, runner_args_dict)

        tree_args_dict = self.tree_args
        dataset_args_dict = self.dataset_args

        if self.tree_type == "simple":
            self.tree_args = SimpleTreeArgs()
        elif self.tree_type == "xgboost":
                self.tree_args = XGBoostArgs()
        elif self.tree_type == "random_forest" or self.tree_type == "federated":
                self.tree_args = RandomForestArgs()
        elif self.tree_type == "llm":  # 新增LLM树类型支持
                self.tree_args = LLMTreeArgs()
        else:
                raise ValueError("Unknown tree type: {}".format(self.tree_type))

        _load_from_dict(self.tree_args, tree_args_dict)

        self.dataset_args = DatasetArgs()
        _load_from_dict(self.dataset_args, dataset_args_dict)

    def __repr__(self):
        return str(self.__dict__)


def parse_args() -> EvaluateArgs:
    """
    Parse the arguments from, with descending priority:
      1. command line
      2. config file
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="path to super config file (yaml)")

    parser.add_argument("--exp-name", type=str, help="experiment name")
    parser.add_argument("--runner", type=str, help="runner type (chatgpt or llama)")

    parser.add_argument("--openai-api-base", type=str, help="openai api base url")
    parser.add_argument("--openai-api-key", type=str, help="openai api key")
    parser.add_argument("--hf-username", type=str, help="huggingface username")
    parser.add_argument("--hf-password", type=str, help="huggingface password")
    parser.add_argument("--hf-cookie-dir", type=str, help="huggingface cookie dir")
    parser.add_argument("--together-api-key", type=str, help="together api key")
    parser.add_argument("--together-api-base", type=str, help="together api base url")
    parser.add_argument("--model-name", type=str, help="model name")

    parser.add_argument(
        "--tree-type", type=str, help="tree model type (simple, xgboost)"
    )
    parser.add_argument("--tree-only", type=int, help="only evaluate the tree")
    parser.add_argument("--max-depth", type=int, help="max depth of the tree")
    parser.add_argument("--num-trees", type=int, help="number of trees")

    parser.add_argument("--output-dir", type=str, help="output directory")
    parser.add_argument("--random-seed", type=int, help="random seed for the tree")
    parser.add_argument(
        "--use-tree-rules", type=int, help="if the tree rules should given in prompts"
    )
    parser.add_argument("--template", type=str, help="path to template file (jinja2)")
    parser.add_argument("--serializer", type=str, help="serializer type")

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

    parser.add_argument("--train-sizes", type=int, nargs="+", help="train set sizes")
    parser.add_argument(
        "--num-tests-per-set", type=int, help="number of tests per training set size"
    )
    parser.add_argument("--test-size", type=float, help="test set size")
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

    parser.add_argument("--llm-gen-tree", action="store_true", help="Enable LLM-generated decision tree")
    parser.add_argument("--llm-max-depth", type=int, help="Max depth for LLM tree")
    parser.add_argument("--llm-temperature", type=float, help="Temperature for LLM generation")
    
    cml_args = parser.parse_args()

    args = EvaluateArgs()

    sup_config_file_path = cml_args.config
    config_file_paths = []
    if sup_config_file_path or config_file_paths:
        config = {}
        config_sup = {}
        if sup_config_file_path:
            with open(sup_config_file_path, encoding='utf-8') as f:
                config_sup = yaml.safe_load(f)

            sup_config_dir_path = Path(sup_config_file_path).parent
            if "base_configs" in config_sup:
                config_file_paths = [
                    sup_config_dir_path / p for p in config_sup["base_configs"]
                ]

        for config_file_path in config_file_paths:
            with open(config_file_path, encoding='utf-8') as f:
                config_part: dict = yaml.safe_load(f)
                _merge_dict(config, config_part)

        _merge_dict(config, config_sup)

        config_config: dict = config.get("config")
        if config_config:
            args.exp_name = config_config.get("exp_name")

            args.runner = config_config.get("runner")
            args.runner_args = config_config.get("runner_args")
            args.dataset_args = config_config.get("dataset_args")
            args.tree_type = config_config.get("tree_type")
            args.tree_args = config_config.get("tree_args")
            args.tree_only = config_config.get("tree_only", args.tree_only)
            args.output_dir = config_config.get("output_dir")
            args.random_seed = config_config.get("random_seed")
            args.use_tree_rules = config_config.get("use_tree_rules")
            args.template = config_config.get("template")
            args.serializer_type = config_config.get("serializer")
            args.train_sizes = config_config.get("train_sizes")
            args.num_tests_per_set = config_config.get("num_tests_per_set")
            args.test_size = config_config.get("test_size")
            args.test_batch = config_config.get("test_batch")

    runner_args_dict = {}
    tree_args_dict = {}
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
    if cml_args.tree_type is not None:
        args.tree_type = cml_args.tree_type
    if cml_args.tree_only is not None:
        args.tree_only = not (cml_args.tree_only == 0)
    if cml_args.max_depth is not None:
        tree_args_dict["max_depth"] = cml_args.max_depth
    if cml_args.num_trees is not None:
        tree_args_dict["num_trees"] = cml_args.num_trees
    if cml_args.output_dir is not None:
        args.output_dir = cml_args.output_dir
    if cml_args.random_seed is not None:
        args.random_seed = cml_args.random_seed
    if cml_args.use_tree_rules is not None:
        args.use_tree_rules = not (cml_args.use_tree_rules == 0)
    if cml_args.template is not None:
        args.template = cml_args.template

    if cml_args.serializer is not None:
        args.serializer_type = cml_args.serializer

    if cml_args.dataset_data_file is not None:
        dataset_args_dict["data_file"] = cml_args.dataset_data_file
    if cml_args.dataset_meta_file is not None:
        dataset_args_dict["meta_file"] = cml_args.dataset_meta_file
    if cml_args.dataset_format is not None:
        dataset_args_dict["format"] = cml_args.dataset_format
    if cml_args.shuffle_column is not None:
        dataset_args_dict["shuffle_column"] = not (cml_args.shuffle_column == 0)

    if cml_args.train_sizes is not None:
        args.train_sizes = cml_args.train_sizes
    if cml_args.num_tests_per_set is not None:
        args.num_tests_per_set = cml_args.num_tests_per_set
    if cml_args.test_size is not None:
        try:
            args.test_size = int(cml_args.test_size)
        except ValueError:
            raise ValueError("test_size must be an integer")
    if cml_args.test_batch is not None:
        args.test_batch = cml_args.test_batch
    if cml_args.shuffle_column is not None:
        args.shuffle_column = not (cml_args.shuffle_column == 0)

    if cml_args.timeout is not None:
        runner_args_dict["timeout"] = cml_args.timeout
    if cml_args.request_interval is not None:
        runner_args_dict["request_interval"] = cml_args.request_interval
    if cml_args.parallel_batch_size is not None:
        runner_args_dict["parallel_batch_size"] = cml_args.parallel_batch_size

    if cml_args.exp_id is not None:
        args.exp_id = cml_args.exp_id
    if args.template is not None:
        # 处理模板路径，支持绝对路径和相对路径
        template_path = Path(args.template)
        if not template_path.exists():
            # 如果是相对路径，尝试从config文件所在目录查找
            if sup_config_file_path:
                config_dir = Path(sup_config_file_path).parent
                template_path = config_dir / args.template
                if not template_path.exists():
                    raise FileNotFoundError(f"Template file not found: {args.template}")
            else:
                raise FileNotFoundError(f"Template file not found: {args.template}")
        args.template = str(template_path)
    if cml_args.llm_gen_tree:
        args.llm_gen_tree = True
        args.tree_type = "llm"
        args.use_tree_rules = True
        
        if cml_args.llm_max_depth:
            args.llm_tree_args.max_depth = cml_args.llm_max_depth
        if cml_args.llm_temperature:
            args.llm_tree_args.temperature = cml_args.llm_temperature
        if cml_args.model_name:
            args.llm_tree_args.model_name = cml_args.model_name
    # read openai api key from env
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if (
        openai_api_key
        and not runner_args_dict.get("openai_api_key")
        and not args.runner_args.get("openai_api_key")
    ):
        runner_args_dict["openai_api_key"] = openai_api_key

    if args.runner_args is None:
        args.runner_args = {}
    if args.tree_args is None:
        args.tree_args = {}
    if args.dataset_args is None:
        args.dataset_args = {}
    _merge_dict(args.runner_args, runner_args_dict)
    _merge_dict(args.tree_args, tree_args_dict)
    _merge_dict(args.dataset_args, dataset_args_dict)
    args.load_sub_args()

    if args.tree_only:
        missing_fields = args.get_missing_fields_tree_only()
    else:
        missing_fields = args.get_missing_fields()

    if len(missing_fields) > 0:
        raise ValueError("Incomplete arguments: missing {}".format(missing_fields))

    return args


def gen_prompt(
    meta: dataset.DatasetMeta,
    master_template: jinja2.Template,
    serializer: Serializer,
    x_train,
    y_train,
    x_test,
    y_test,
    tree_rules,
    num_tests_per_round,
    template_params=None,
) -> tuple[list[str], list[tuple[int, int]], list[str]]:
    # 获取必要的模板参数
    feature_names = meta.feature_names()
    if not feature_names:
        raise ValueError("Dataset feature names are not defined")
    
    label_names = meta.label_names()
    if not label_names:
        raise ValueError("Dataset label names are not defined")

    # 构建完整的模板上下文
    base_context = {
        "meta": meta,
        "features": feature_names,
        "num_features": len(feature_names),
        "labels": label_names,
        "max_depth": 3  # 或从参数获取实际值
    }
    if template_params:
        base_context.update(template_params)

    # 生成基础提示语
    prompts, test_splits = prompt.gen_prompt(
        master_template,
        serializer,
        x_train,
        y_train,
        x_test,
        tree_rules,
        num_tests_per_round,
        template_params=base_context  # 传递完整参数
    )

    # 生成测试标签（添加空值保护）
    test_labels = []
    for y in y_test:
        label_obj = meta.find_label(y)
        if label_obj and hasattr(label_obj, 'name'):
            test_labels.append(label_obj.name)
        else:
            test_labels.append(f"unknown_{y}")  # 防止无效标签

    # 动态渲染模板时使用完整上下文
    try:
        final_prompt = master_template.render(**base_context)
        prompts.append(final_prompt)
    except jinja2.UndefinedError as e:
        error_msg = f"Template rendering failed: Missing variable '{e.message}'"
        raise ValueError(error_msg) from e

    return prompts, test_splits, test_labels


def calc_accuracy(labels: list, results: list) -> float:
    assert len(labels) == len(results)
    correct_count = 0
    for i, result in enumerate(results):
        if isinstance(result, str):
            if result.lower() == labels[i].lower():
                correct_count += 1
        else:
            if result == labels[i]:
                correct_count += 1

    return correct_count / len(labels)


def evaluate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    runner: Runner,
    master_template: jinja2.Template,
    serializer: Serializer,
    meta: DatasetMeta,
    tree_model: DecisionTree,
    use_tree_rules: bool,
    tree_only: bool,
    num_tests_per_round: int,
):
    # get tree's prediction rules & results
    if use_tree_rules or tree_only:
        if type(tree_model) == FederatedDecisionTree:
            all_tree_predict, _ = tree_model.predict(x_train, y_train, x_test)
            tree_aucs, tree_accuracies = [], []
            for tree_predict in all_tree_predict:
                tree_auc = sklearn.metrics.roc_auc_score(y_test, tree_predict)
                tree_accuracy = calc_accuracy(y_test, tree_predict)
                tree_aucs.append(tree_auc)
                tree_accuracies.append(tree_accuracy)
            tree_auc = tree_aucs
            tree_accuracy = tree_accuracies
        else:
            tree_predict, rules = tree_model.predict(
                x_test, export_rules=True
            )
            tree_auc = sklearn.metrics.roc_auc_score(y_test, tree_predict)
            tree_accuracy = calc_accuracy(y_test, tree_predict)

    if tree_only:
        return {
            "tree_auc": tree_auc,
            "tree_accuracy": tree_accuracy,
            "tree_results": tree_predict,
        }

    prompts, test_splits, labels = gen_prompt(
        meta,
        master_template,
        serializer,
        x_train,
        y_train,
        x_test,
        y_test,
        rules if use_tree_rules else [],
        num_tests_per_round,
    )

    raw_results = []
    results = []

    for idx, responses in enumerate(runner.run(prompts)):
        expected_len = test_splits[idx][1] - test_splits[idx][0]
        found = False
        for response in responses:
            results_batch = serializer.answer_decoder.decode(response)
            if len(results_batch) == expected_len:
                found = True
                results += results_batch
                raw_results.append(response)
                break
            else:
                logger.log(
                    "Length of labels and results do not match (expected: {}, actual: {}), response: {}".format(
                        expected_len, len(results_batch), response
                    )
                )
        if not found:
            logger.log("Failed to find any valid response, skipping...")
            result_dict = {
                "record": prompts[0],
                "failed_raw_output": responses,
            }
            return result_dict

    acc = calc_accuracy(labels, results)
    auc = sklearn.metrics.roc_auc_score(
        y_test,
        [meta.get_label_value(r) for r in results],
    )

    logger.log("Accuracy/AUC: {}/{}".format(acc, auc))

    if use_tree_rules:
        logger.log("Tree accuracy/AUC: {}/{}".format(tree_accuracy, tree_auc))

    result_dict = {}
    result_dict["record"] = {"prompt": prompts[0]}
    result_dict["labels"] = [int(y) for y in y_test]
    result_dict["results"] = [meta.get_label_value(r) for r in results]
    result_dict["auc"] = auc
    result_dict["accuracy"] = acc

    if use_tree_rules:
        result_dict["tree_auc"] = tree_auc
        result_dict["tree_accuracy"] = tree_accuracy
        result_dict["tree_results"] = tree_predict

    return result_dict


def main():
    args = parse_args()
    train_size = 0.8  # 默认值初始化
    # 后续条件分支可修改该值


    if args.exp_id:
        logger.DEFAULT_LOGGERS[0].prefix = "[{}] ".format(args.exp_id)

    logger.log("Arguments:")
    for k, v in args.__dict__.items():
        logger.log(" - {}: {}".format(k, v))

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    
        
    if args.test_size <= 0:
        raise ValueError(f"Invalid test_size: {args.test_size} (must be positive)")
    # 在main函数中修改数据集划分部分：
    meta, x, y = load_dataset(args.dataset_args)
    print("Feature Names:", meta.feature_names())  # 必须返回非空列表
    print("Label Names:", meta.label_names())     # 必须返回非空列表
    total_samples = len(y)

    test_size = int(len(x) * args.test_size)
    # 确保test_size不超过可用样本量
    test_size = min(test_size, len(x)-1) if len(x) > 1 else 0
    test_x, test_y = x[:test_size], y[:test_size]
    avail_x, avail_y = x[test_size:], y[test_size:]
    if args.print_only:
        raise NotImplementedError("Not implemented yet")

    
    
    runner, serializer, master_template = None, None, None
    
    if not args.tree_only:
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
            )
        elif args.runner == "together_api":
            from tree_prompt.runner.together_api import TogetherAPIParallelRunner
            runner = TogetherAPIParallelRunner(
                args.runner_args.api_base,
                args.runner_args.model_name,
                args.runner_args.together_api_key,
                args.runner_args.request_interval,
                args.runner_args.timeout,
                args.runner_args.parallel_batch_size,
            )
        else:
            raise ValueError("Unknown runner type: {}".format(args.runner))

    # 统一模板加载方式，所有树类型都从这里加载
    master_template_path = Path(args.template)
    try:
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(master_template_path.parent),
            undefined=jinja2.StrictUndefined  # 开启严格模式
        )
        master_template = env.get_template(master_template_path.name)
    except jinja2.TemplateNotFound as e:
        raise FileNotFoundError(
            f"Template file not found: {e}"
        ) from e
    

  

    if args.serializer_type == "tabular":
        serializer = TabularSerializer(meta)
    elif args.serializer_type == "list":
        serializer = ListSerializer(meta)
    elif args.serializer_type == "text":
        serializer = TextSerializer(meta)
    else:
        raise ValueError("Unknown serializer type: {}".format(args.serializer_type))

    if args.tree_type == "llm":
        tree_model = LLMDecisionTree(
            meta,
            args.llm_tree_args.model_name,
            args.llm_tree_args.temperature,
            args.llm_tree_args.max_depth
        )
        num_features = len(meta.feature_names())  # 从 meta 
        # 准备训练数据
        train_cases = sample_balanced(
    avail_x, 
    avail_y,
    num_groups=5,
    num_samples_per_group=args.num_tests_per_set,
    random_seed=args.random_seed
)
        x_train, y_train = train_cases[0]
        
        feature_names = meta.feature_names() or []  # 防止返回None
        label = meta.find_label(y) or "unknown"     # 无效标签处理

        if not tree_model.build_tree(master_template, runner, x_train, y_train,template_params={  # 新增参数
            'num_features': num_features,
            'features': feature_names,
            'labels': meta.label_names()
        }):
            raise ValueError("Failed to generate valid decision tree from LLM")
        
        logger.log(f"Template Loaded: {master_template.filename}")
        logger.log(f"Available Variables: {master_template.environment.globals.keys()}")
        # # 使用统一的模板渲染方式
        # prompt = master_template.render(
        #     meta=meta,
        #     max_depth=args.llm_tree_args.max_depth,
        #     features=meta.feature_names(),
        #     labels=meta.label_names(),
        #     num_features=len(meta.feature_names()),
        #     num_labels=len(meta.label_names()),
        #     feature_descriptions=meta.feature_descriptions()
        # )

        # if runner is None:
        #     raise ValueError("Runner must be initialized for LLM tree generation")

        # if not tree_model.build_tree(prompt, runner):
        #     raise ValueError("Failed to generate valid decision tree from LLM")
    else:
        if args.tree_type == "simple":
            tree_model = SimpleDecisionTree(meta, args.tree_args.max_depth)
        elif args.tree_type == "xgboost":
            tree_model = XGBoostDecisionTree(
                meta,
                args.tree_args.max_depth,
                args.tree_args.num_trees,
                args.random_seed,
            )
        elif args.tree_type == "random_forest":
            if not args.tree_only:
                raise ValueError("Random forest is only supported in tree only mode")
            tree_model = RandomForestDecisionTree(
                meta,
                args.tree_args.num_trees,
                args.tree_args.max_depth,
            )
        elif args.tree_type == "federated":
            if not args.tree_only:
                raise ValueError("Federated tree is only supported in tree only mode")
            tree_model = FederatedDecisionTree(
                meta,
                args.tree_args.num_trees,
                args.tree_args.max_depth,
            )
        else:
            raise ValueError("Unknown tree type: {}".format(args.tree_type))

    results: dict[int, list[dict]] = {}



    bar = tqdm(desc="Total", total=len(args.train_sizes) * args.num_tests_per_set)

    for train_size in args.train_sizes:
        train_cases = sample_balanced(
            avail_x,
            avail_y,
            args.num_tests_per_set,
            train_size,
            args.random_seed,
        )

        for train_x, train_y in train_cases:
            result = evaluate(
                train_x,
                train_y,
                test_x,
                test_y,
                runner,
                master_template,
                serializer,
                meta,
                tree_model,
                args.use_tree_rules,
                args.tree_only,
                args.test_batch,
            )

            results.setdefault(train_size, []).append(result)
            bar.update(1)

    file_name = args.exp_name + ".json"
    output_file = Path(args.output_dir) / file_name

    logger.log("Saving results to {}...".format(output_file))

    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)

    if output_file.exists():
        date = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        target = output_file.rename(
            output_file.parent / (output_file.stem + "-" + date + output_file.suffix)
        )
        logger.log("Output file already exists, renamed to {}".format(target))

    def json_default_decode(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.__dict__

    with open(output_file, "w") as output_file:
        output = {"args": args.__dict__, "results": results}
        json.dump(output, output_file, indent=2, default=json_default_decode)


if __name__ == "__main__":
    main()
