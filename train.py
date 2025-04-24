import jinja2
from argparse import ArgumentParser
import yaml
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from datetime import datetime
import time
import json
import os
import tree_prompt

import tree_prompt.logger as logger
from tree_prompt.model.strategy import (
    TrainStrategy,
    UnknownClassStrategy,
    KnownClassStrategy,
    FeatureBaggingStrategy,
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
    UnknownClassStrategyArgs,
    KnownClassStrategyArgs,
    FeatureBaggingStrategyArgs,
)
from tree_prompt.dataset import load_dataset, sample_balanced


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


class Repr:
    def __repr__(self):
        return self.__dict__.__repr__()


class UnknownClassStrategyArgs(Repr):
    def __init__(self) -> None:
        self.max_depth: int = None
        self.hist_nbins: int = 10

    def __repr__(self) -> str:
        return self.__dict__.__repr__()


class KnownClassStrategyArgs(Repr):
    def __init__(self) -> None:
        self.max_depth: int = None
        self.hist_nbins: int = 10


class TrainArgs(Repr):
    def __init__(self) -> None:
        self.exp_name: str = None
        self.strategy: str = None
        self.strategy_args: UnknownClassStrategyArgs | KnownClassStrategyArgs | FeatureBaggingStrategyArgs = (
            None
        )
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
        self.template: str = None
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
            self.strategy == "unknown_class"
            and not isinstance(self.strategy_args, UnknownClassStrategyArgs)
            or self.strategy == "known_class"
            and not isinstance(self.strategy_args, KnownClassStrategyArgs)
            or self.strategy == "feature_bagging"
            and not isinstance(self.strategy_args, FeatureBaggingStrategyArgs)
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

        if self.strategy == "unknown_class":
            self.strategy_args = UnknownClassStrategyArgs()
        elif self.strategy == "known_class":
            self.strategy_args = KnownClassStrategyArgs()
        elif self.strategy == "feature_bagging":
            self.strategy_args = FeatureBaggingStrategyArgs()
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
        help="strategy type (unknown_class, known_class, feature_bagging)",
    )
    parser.add_argument("--max-depth", type=int, help="max depth of the tree")
    parser.add_argument("--num-trees", type=int, help="number of trees in the forest")
    parser.add_argument("--train-batch", type=int, help="train batch size")
    parser.add_argument("--hist-nbins", type=int, help="number of bins of histogram")

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
    parser.add_argument(
        "--template", type=str, help="path to prompt template file (jinja2)"
    )
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
    parser.add_argument('--delta', type=int, default=2, help='delta for max offset of confidence of meta rule selection')
    parser.add_argument("--timeout", type=int, help="timeout per request, in seconds")
    parser.add_argument(
        "--request-interval", type=float, help="interval between requests, in seconds"
    )
    parser.add_argument("--parallel-batch-size", type=int, help="parallel batch size")

    parser.add_argument("--exp-id", type=str, help="experiment id for display")

    parser.add_argument('--threshold', type=float, default=0.70, help='threshold of modifying labels for leaf supervision')

   
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
            args.template = config_config.get("template")
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
    if cml_args.train_batch is not None:
        args.train_batch = cml_args.train_batch
    if cml_args.hist_nbins is not None:
        strategy_args_dict["hist_nbins"] = cml_args.hist_nbins

    if cml_args.output_dir is not None:
        args.output_dir = cml_args.output_dir
    if cml_args.random_seed is not None:
        args.random_seed = cml_args.random_seed
    if cml_args.template is not None:
        args.template = cml_args.template

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
    if cml_args.delta is not None:
        args.delta = cml_args.delta
    if cml_args.threshold is not None:
        args.threshold = cml_args.threshold

    if cml_args.exp_id is not None:
        args.exp_id = cml_args.exp_id

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


def evaluate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model: Classifier,
    test_batch: int,
):
    start = time.time()
    model.fit(x_train, y_train)
    elapsed = time.time() - start

    if len(x_test) == 0:
        return None
    llm_with_tree_results, llm_with_tree_subresults = [], None
    tree_results, tree_raw_results = [], []

    for test_start in tqdm(range(0, len(x_test), test_batch), desc="Test"):
        test_end_batch = min(test_start + test_batch, len(x_test))
        results = model.predict(x_test[test_start:test_end_batch])
        if results[0] is None:
            logger.log("Model prediction failed")
        else:
            llm_with_tree_results += results[0]
            if llm_with_tree_subresults is None:
                llm_with_tree_subresults = results[3]
            else:
                for i in range(len(llm_with_tree_subresults)):
                    llm_with_tree_subresults[i] += results[3][i]

        tree_results += results[1]
        tree_raw_results += results[2]

    if len(llm_with_tree_results) == len(tree_results):
        llm_with_tree_auc = roc_auc_score(
            y_test, llm_with_tree_results, multi_class="ovr"
        )
    else:
        llm_with_tree_auc = None

    # 分析节点样本分布
    train_node_stats, label_names = analyze_node_samples(model, x_train, y_train)
    logger.log("\n=== 训练集节点样本分布 ===")
    
    for node_id, counts in train_node_stats.items():
        logger.log(f"{node_id}: {counts}")
    
    # 修改后的测试集分析
    test_node_stats = None
    for test_start in tqdm(range(0, len(x_test), test_batch)):
        test_end = min(test_start + test_batch, len(x_test))
        batch_stats, _ = analyze_node_samples(model, x_test[test_start:test_end], y_test[test_start:test_end])
        
        if test_node_stats is None:
            test_node_stats = batch_stats
        else:
            for node_id, counts in batch_stats.items():
                for cls in counts:
                    test_node_stats[node_id][cls] += counts[cls]
    
    logger.log("\n=== 测试集节点样本分布 ===")
    for node_id, counts in test_node_stats.items():
        logger.log(f"{node_id}: {counts}")

    llm_with_sub_tree_aucs = []
    if llm_with_tree_subresults is not None:
        for sub_result in llm_with_tree_subresults:
            llm_with_sub_tree_aucs.append(
                roc_auc_score(y_test, sub_result, multi_class="ovr")
            )

    tree_auc = roc_auc_score(y_test, tree_results, multi_class="ovr")
    logger.log(
        "llm + tree AUC: {}, tree AUC: {}, llm + subtree AUC: {}".format(
            llm_with_tree_auc, tree_auc, llm_with_sub_tree_aucs
        )
    )

    return (
        llm_with_tree_auc,
        tree_auc,
        llm_with_sub_tree_aucs,
        None,
        llm_with_tree_results,
        tree_results,
        tree_raw_results,
        llm_with_tree_subresults,
        elapsed,
        train_node_stats,  # 新增返回项
        test_node_stats,   # 新增返回项
    )



def load_args(
    args: TrainArgs,
) -> tuple[np.ndarray, np.ndarray, TrainStrategy]:
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    meta, x, y = load_dataset(args.dataset_args)

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
    # 在创建runner后添加监控装饰器（新增代码开始）
    def add_monitoring(runner_instance):
        original_run = runner_instance.__class__.run
        
        def monitored_run(self, messages):
            for prompt in messages:
                logger.log(f"[LLM Request] {datetime.now().isoformat()}\nPrompt: {prompt}")
                responses = original_run(self, [prompt])  # 保持单条处理
                for response in responses:
                    logger.log(f"[LLM Response] {datetime.now().isoformat()}\nResponse: {response[0]}")
                    yield response
                    
        runner_instance.__class__.run = monitored_run
        return runner_instance
    
    runner = add_monitoring(runner)  # 应用监控装饰器
    master_template_path = Path(args.template)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(master_template_path.parent),
    )
    master_template = env.get_template(master_template_path.name)

    if args.strategy == "unknown_class" or args.strategy == "known_class":
        StrategyClass = (
            UnknownClassStrategy
            if args.strategy == "unknown_class"
            else KnownClassStrategy
        )
        strategy = StrategyClass(
            runner,
            master_template,
            serializer,
            args.strategy_args.max_depth,
            args.train_batch,
            args.strategy_args.hist_nbins,
        )
    elif args.strategy == "feature_bagging":
        strategy = FeatureBaggingStrategy(
            meta,
            runner,
            master_template,
            args.serializer_type,
            args.strategy_args.num_trees,
            args.strategy_args.max_depth,
            args.train_batch,
            args.strategy_args.hist_nbins,
        )

    else:
        raise ValueError("Unknown strategy type: {}".format(args.strategy))

    return x, y, strategy

def analyze_node_samples(model: Classifier, X: np.ndarray, y: np.ndarray):
    """动态分析决策树节点样本分布（兼容多标签）"""
    if not hasattr(model.strategy, 'root'):
        return {}, None
    
    from collections import defaultdict
    node_samples = defaultdict(list)
    
    # 自动检测所有可能的标签类别
    unique_labels = np.unique(y)
    label_names = {label: f"class_{label}" for label in unique_labels}
    
    # 追踪每个样本经过的路径
    for sample_idx, (sample, label) in enumerate(zip(X, y)):
        path = []
        node = model.strategy.root
        while node is not None and not node.is_leaf:
            path.append(node)
            # 修复特征索引访问 - 使用split_feature而不是feature_idx
            if not hasattr(node, 'split_feature') or node.split_feature is None:
                break
                
            # 检查是否是分类特征
            if hasattr(node, 'is_categorical') and node.is_categorical:
                if sample[node.split_feature] == node.split_value:
                    node = node.left_child
                else:
                    node = node.right_child
            else:
                # 数值特征
                if sample[node.split_feature] < node.split_value:
                    node = node.left_child
                else:
                    node = node.right_child
                    
        if node is not None:
            path.append(node)  # 添加叶子节点
        
        # 记录样本标签到所有经过的节点
        for node in path:
            node_samples[node].append(label)
    
    # 转换为标签统计（自动适应多分类）
    node_stats = {}
    sample_paths = {}  # 记录每个样本的路径
    
    for node, labels in node_samples.items():
        # 创建标签计数字典
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[int(label)] += 1
        
        # 使用节点ID作为键
        node_id = f"node_{id(node)}"
        node_stats[node_id] = dict(label_counts)
        
        # 添加节点类型信息
        if hasattr(node, 'is_leaf') and node.is_leaf:
            node_stats[node_id]['_type'] = 'leaf'
            if hasattr(node, 'leaf_class'):
                node_stats[node_id]['_prediction'] = node.leaf_class
        else:
            node_stats[node_id]['_type'] = 'internal'
            if hasattr(node, 'split_feature'):
                node_stats[node_id]['_feature'] = node.split_feature
                node_stats[node_id]['_value'] = node.split_value
    
    # 处理叶子节点的预测值
    if hasattr(node, 'prediction') and node.prediction is not None:
        pred_value = node.prediction
        # 处理特殊的-1值（unknown）
        if pred_value < 0:
            node_stats[node_id]['_prediction'] = "unknown"
        else:
            node_stats[node_id]['_prediction'] = pred_value
    
    return node_stats, label_names

def main():
    start_time = time.time()
    args = parse_args()
    
    if args.exp_id:
        logger.DEFAULT_LOGGERS[0].prefix = "[{}] ".format(args.exp_id)

    log_path = (
        Path(args.output_dir)
        / "log"
        / (datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + "-" + args.exp_name + ".log")
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    train_logger = logger.Logger(open(log_path, "w"))
    logger.add_logger(train_logger)
    
    # 增加调试信息
    logger.log("=== 训练开始 ===")
    logger.log("日志输出路径: " + str(log_path))

    logger.log("Arguments:")
    for k, v in args.__dict__.items():
        logger.log(" - {}: {}".format(k, v))

    # 检查配置文件中的strategy_args是否存在
    if args.strategy_args is None:
        # 根据strategy类型设置默认值
        if args.strategy == "unknown_class":
            args.strategy_args = UnknownClassStrategyArgs()
        elif args.strategy == "known_class":
            args.strategy_args = KnownClassStrategyArgs()
        elif args.strategy == "feature_bagging":
            args.strategy_args = FeatureBaggingStrategyArgs()
    
    # 确保必要的参数存在
    if args.strategy_args.max_depth is None:
        args.strategy_args.max_depth = 3  # 设置默认值
    
    # 设置delta参数
    if args.delta is not None:
        from tree_prompt.model.strategy import set_delta
        set_delta(args.delta)
        logger.log(f"set the delta as: {args.delta}")


    
    # 设置threshold参数 (叶子节点标签修改阈值)
    if args.threshold is not None:
        # 导入和设置阈值
        if hasattr(tree_prompt.model.strategy, 'set_threshold'):
            tree_prompt.model.strategy.set_threshold(args.threshold)
            logger.log(f"set the leaf node label modification threshold (threshold) as: {args.threshold}")
        else:
            logger.log(f"warning: cannot set threshold, the set_threshold function does not exist in the module")


    


    x, y, strategy = load_args(args)

    file_name = args.exp_name + ".json"
    output_file = Path(args.output_dir) / file_name
    logger.log("The result is saving to {}...".format(output_file))

    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)

    if output_file.exists():
        exit()
        date = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        target = output_file.rename(
            output_file.parent / (output_file.stem + "-" + date + output_file.suffix)
        )
        logger.log("Output file already exists, renamed to {}".format(target))

    results: dict[int, list[dict]] = {}

    test_x, test_y = x[: args.test_size], y[: args.test_size]
    avail_x, avail_y = x[args.test_size :], y[args.test_size :]

    bar = tqdm(desc="Total", total=len(args.train_sizes) * args.num_tests_per_set)
    
    def json_default_decode(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.__dict__
    
    for train_size in args.train_sizes:
        train_cases = sample_balanced(
            avail_x, avail_y, args.num_tests_per_set, train_size, args.random_seed
        )
        logger.log(f"实际生成的训练案例数量: {len(train_cases)}")
        if len(train_cases) > 0:
            logger.log(f"首例训练集形状: {train_cases[0][0].shape}")
        for train_x, train_y in train_cases:
            model = Classifier(strategy)
            result = evaluate(train_x, train_y, test_x, test_y, model, args.test_batch)
            if result is None:
                result_dict = {
                    "model": model.export(),
                }
            else:
                (
                    llm_with_tree_auc,
                    tree_auc,
                    llm_with_sub_tree_aucs,
                    final_loss,
                    llm_with_tree_results,
                    tree_results,
                    tree_raw_results,
                    llm_with_tree_subresults,
                    elapsed,
                    train_node_stats,  # 新增返回项
                    test_node_stats,   # 新增返回项
                    
                ) = result

                logger.log("Training time: {}".format(elapsed))

                result_dict = {
                    "llm_tree": llm_with_tree_auc,
                    "tree": tree_auc,
                    "sub_trees": llm_with_sub_tree_aucs,
                    "llm_tree_results": llm_with_tree_results,
                    "tree_results": tree_results,
                    "tree_raw_results": tree_raw_results,
                    "llm_tree_subresults": llm_with_tree_subresults,
                    "labels": test_y.tolist(),
                    "model": model.export(),
                    "train_elapsed": elapsed,
                    'train_node_stats': train_node_stats,
                    'test_node_stats': test_node_stats
                }

            # 在保存结果到JSON文件之前
            if hasattr(strategy, '_feature_shuffle_map') and strategy._feature_shuffle_map:
                logger.log(f"将特征映射保存到结果JSON: {strategy._feature_shuffle_map}")
                
                # 确保结果中包含特征映射
                if 'args' not in result_dict:
                    result_dict['args'] = {}
                result_dict['args']['feature_shuffle_map'] = strategy._feature_shuffle_map

            results.setdefault(train_size, []).append(result_dict)
            current_count = len(results[train_size])
            logger.log(f"当前train_size={train_size}已保存结果数: {current_count}/{args.num_tests_per_set}")
            if current_count >= args.num_tests_per_set:
                logger.log(f"!! 已达到预期测试次数 {args.num_tests_per_set}")
            bar.update(1)
            
        total_llm_auc = 0.0
        total_tree_auc = 0.0
        valid_counts = 0
        for train_size in results:
            for test_result in results[train_size]:
                llm_auc = test_result.get('llm_tree')
                tree_auc = test_result.get('tree')
                
                if llm_auc is not None and tree_auc is not None:
                    total_llm_auc += llm_auc
                    total_tree_auc += tree_auc
                    valid_counts += 1
        avg_llm_auc = total_llm_auc / valid_counts if valid_counts > 0 else 0.0
        avg_tree_auc = total_tree_auc / valid_counts if valid_counts > 0 else 0.0
            # Store results each round to avoid losing data
        with open(output_file, "w") as f:
                output = {"args": args.__dict__, "results": results}
                json.dump(output, f, indent=2, default=json_default_decode)

    logger.log("\n=== Experiment Summary ===")
    total_tests = 0
    for size, tests in results.items():
        logger.log(f"训练集大小 {size}: {len(tests)}次测试")
        total_tests += len(tests)
    logger.log(f"总计测试次数: {total_tests}")
    logger.log(f"Average metrics: average llm+tree AUC={avg_llm_auc:.6f},average tree AUC={avg_tree_auc:.6f}")  # 平均值
    logger.log(f"Elapsed time: {time.time()-start_time:.2f}s")
if __name__ == "__main__":
    main()
