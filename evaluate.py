from tree_prompt.logger import log as global_log
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
from jinja2 import Environment, FileSystemLoader
import tree_prompt.prompt as prompt
import tree_prompt.dataset as dataset
from tree_prompt.external.tree import (
    DecisionTree,
    SimpleDecisionTree,
    XGBoostDecisionTree,
    RandomForestDecisionTree,
    FederatedDecisionTree,
    LLMDecisionTree,
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
from sklearn.metrics import roc_auc_score
import time
import sys
import scipy.stats

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


class LLMTreeArgs:
    def __init__(self):
        self.max_depth = 3  # 默认最大深度

        self.temperature = 0.7
        #      self.num_rules = 10

    def __repr__(self):
        return str(self.__dict__)


class XGBoostArgs:
    def __init__(self) -> None:
        self.max_depth: int = None
        self.num_trees: int = None

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
        self.test_size: int = None
        self.test_batch: int = None
        self.use_tree_rules: bool = None
        self.template: str = None
        self.serializer_type: str = None
        self.exp_id: str = ""
        self.print_only: bool = False
        self.with_llm: bool = True

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

        elif self.runner_args:
            missing_fields += _get_missing_fields(self.runner_args, "runner_args")

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
        ):
            missing_fields.append("tree_args")
        elif self.tree_args:
            missing_fields += _get_missing_fields(self.tree_args, "tree_args")

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
        elif self.tree_type == "CoT":
            self.tree_args = LLMTreeArgs()
        elif self.tree_type == "LLM":
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

    parser.add_argument('--tree-type', choices=['simple', 'xgboost', 'random_forest', 'LLM','CoT'], 
                       default='simple')
    parser.add_argument('--with-llm', type=int, choices=[0, 1], default=1,
                       help='Use LLM for final prediction (1) or use rules directly (0)')
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

    args = EvaluateArgs()

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
        args.test_size = cml_args.test_size
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
    if cml_args.with_llm is not None:
        args.with_llm = bool(cml_args.with_llm)  # 将参数值赋给args.with_llm
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
    
) -> tuple[list[str], list[tuple[int, int]], list[str]]:
    prompts, test_splits = prompt.gen_prompt(
        master_template,
        serializer,
        x_train,
        y_train,
        x_test,
        tree_rules,
        num_tests_per_round,
        
    )

    test_labels = []
    for y in y_test:
        test_labels.append(meta.find_label(y).name)

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


def format_auc_result(auc_result, is_binary=None):
    """格式化AUC结果，二分类情况下显示单个值，多分类显示字典"""
    if is_binary is None:
        # 尝试自动检测是否为二分类结果
        is_binary = isinstance(auc_result, dict) and 'macro' in auc_result and 'micro' in auc_result and auc_result['macro'] == auc_result['micro']
    
    if is_binary and isinstance(auc_result, dict):
        return auc_result.get('macro', float('nan'))
    return auc_result


def cot_calc_accuracy_auc(y_true, y_pred, meta, force_dict=False):
    """
    专用于CoT字符串标签输出的准确率和AUC计算。
    y_true, y_pred: 均为字符串标签或数值标签
    meta: DatasetMeta, 用于标签到数值的映射
    force_dict: 是否强制返回字典格式的AUC结果，即使是二分类情况
    返回: (accuracy, auc_value_or_dict)，二分类时返回单值，多分类时返回字典
    """
    global_log(f"准确率计算 - 标签类型: {type(y_true[0])}, 预测类型: {type(y_pred[0])}")
    
    # 创建标签值到标签名称的映射
    label_value_to_name = {}
    for label in meta.labels:
        label_value_to_name[label.value] = label.name
    
    global_log(f"标签映射关系: {label_value_to_name}")
    
    # 准确率
    correct = 0
    matches = []
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        match_type = "未匹配"
        is_match = False
        
        if isinstance(yt, str) and isinstance(yp, str):
            if yt.lower() == yp.lower():
                correct += 1
                match_type = "字符串比较"
                is_match = True
        elif isinstance(yt, (int, float, np.integer, np.floating)) and isinstance(yp, str):
            # 数值标签与字符串预测的比较
            if yt in label_value_to_name and label_value_to_name[yt].lower() == yp.lower():
                correct += 1
                match_type = "数值-字符串映射比较"
                is_match = True
        elif isinstance(yt, str) and isinstance(yp, (int, float, np.integer, np.floating)):
            # 字符串标签与数值预测的比较
            label_value = meta.get_label_value(yt)
            if label_value == yp:
                correct += 1
                match_type = "字符串-数值映射比较"
                is_match = True
        else:
            # 两者都是数值
            try:
                if float(yt) == float(yp):
                    correct += 1
                    match_type = "数值比较"
                    is_match = True
            except (ValueError, TypeError):
                # 不可比较的类型，输出警告
                global_log(f"警告: 无法比较的标签类型 - 真实标签: {type(yt)}({yt}), 预测标签: {type(yp)}({yp})")
        
        if is_match:
            matches.append(f"样本{i}: {yt} == {yp} [{match_type}]")
    

            
    accuracy = correct / len(y_true)
    global_log(f"准确率计算结果: {correct}/{len(y_true)} = {accuracy:.4f}")
    
    # AUC计算前的详细日志
    global_log(f"准备计算AUC - 标签值分布: {dict(zip(*np.unique(y_true, return_counts=True)))}")
    global_log(f"预测值分布: {dict(zip(*np.unique(y_pred, return_counts=True)))}")
    
    # 转换标签为数值（如果需要）
    if isinstance(y_true[0], str):
        global_log("将字符串标签转换为数值...")
        y_true_num = [meta.get_label_value(y) for y in y_true]
    else:
        y_true_num = y_true
        
    if isinstance(y_pred[0], str):
        y_pred_num = [meta.get_label_value(y) for y in y_pred]
    else:
        y_pred_num = y_pred
    
    # 确保所有值都是有效数值
    y_true_num = np.array([float(y) if y is not None else 0.0 for y in y_true_num])
    y_pred_num = np.array([float(y) if y is not None else 0.0 for y in y_pred_num])
    
    global_log(f"转换后 - 标签数值: {y_true_num[:5]}... (截断显示)")
    global_log(f"转换后 - 预测数值: {y_pred_num[:5]}... (截断显示)")
    
    # 检查数据格式和类别数
    unique_true = np.unique(y_true_num)
    unique_pred = np.unique(y_pred_num)
    
    global_log(f"唯一标签值: {unique_true}, 唯一预测值: {unique_pred}")
    
    # 初始化AUC结果字典
    auc_dict = {"macro": float('nan'), "micro": float('nan')}
    
    # 如果标签或预测只有一个类别，无法计算AUC
    if len(unique_true) < 2 or len(unique_pred) < 2:
        global_log(f"警告: 标签或预测只有一个类别，无法计算AUC (标签类别数={len(unique_true)}, 预测类别数={len(unique_pred)})")
        return accuracy, auc_dict if force_dict else float('nan')
    
    try:
        # 多分类情况
        is_multiclass = len(unique_true) > 2
        if is_multiclass:
            global_log(f"检测到多分类情况 (类别数={len(unique_true)}), 使用one-vs-rest方法计算AUC")
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_auc_score
            
            # 确保classes包含所有可能的类别
            all_classes = sorted(np.union1d(unique_true, unique_pred))
            global_log(f"用于二值化的所有类别: {all_classes}")
            
            # 二进制化处理
            y_true_bin = label_binarize(y_true_num, classes=all_classes)
            y_pred_bin = label_binarize(y_pred_num, classes=all_classes)
            
            global_log(f"二值化后形状: y_true_bin={y_true_bin.shape}, y_pred_bin={y_pred_bin.shape}")
            
            try:
                # 计算宏平均AUC (macro)
                auc_macro = roc_auc_score(y_true_bin, y_pred_bin, multi_class='ovr', average='macro')
                auc_dict["macro"] = auc_macro
                global_log(f"多分类宏平均AUC计算成功: {auc_macro}")
                
                # 计算微平均AUC (micro)
                auc_micro = roc_auc_score(y_true_bin, y_pred_bin, multi_class='ovr', average='micro')
                auc_dict["micro"] = auc_micro
                global_log(f"多分类微平均AUC计算成功: {auc_micro}")
                
                # 多分类情况总是返回字典
                return accuracy, auc_dict
            except Exception as e:
                global_log(f"多分类AUC计算失败: {e}")
                if 'samples are not positive and negative' in str(e):
                    global_log("可能是某些类别在二值化后没有正样本或负样本")
                return accuracy, auc_dict if force_dict else float('nan')
        else:
            # 二分类情况
            global_log("检测到二分类情况，直接计算AUC")
            from sklearn.metrics import roc_auc_score
            
            # 对于二分类，确保预测值是概率或决策值
            if np.all(np.isin(y_pred_num, [0, 1])) or np.all(np.isin(y_pred_num, unique_true)):
                global_log("预测值不是概率值，尝试转换为决策值...")
                # 将类别映射到0和1（如果需要）
                unique_classes = sorted(unique_true)
                if not np.array_equal(unique_classes, [0, 1]):
                    global_log(f"将类别 {unique_classes} 映射到 [0, 1]")
                    y_true_mapped = np.where(y_true_num == unique_classes[0], 0, 1)
                    y_pred_mapped = np.where(y_pred_num == unique_classes[0], 0, 1)
                    
                    auc = roc_auc_score(y_true_mapped, y_pred_mapped)
                    # 二分类情况下，macro=micro=标准AUC
                    auc_dict["macro"] = auc
                    auc_dict["micro"] = auc
                    global_log(f"使用映射后的类别计算AUC成功: {auc}")
                    return accuracy, auc_dict if force_dict else auc
            
            # 直接计算AUC
            try:
                auc = roc_auc_score(y_true_num, y_pred_num)
                # 二分类情况下，macro=micro=标准AUC
                auc_dict["macro"] = auc
                auc_dict["micro"] = auc
                global_log(f"AUC计算成功: {auc}")
                return accuracy, auc_dict if force_dict else auc
            except Exception as e:
                global_log(f"AUC计算失败: {e}")
                import warnings
                warnings.warn(f"roc_auc_score failed: {e}. Returning NaN.")
                return accuracy, auc_dict if force_dict else float('nan')
    except Exception as e:
        global_log(f"AUC计算过程中发生意外错误: {e}")
        return accuracy, auc_dict if force_dict else float('nan')


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
    with_llm: bool = True,
):
    # 初始化token统计
    token_stats = {
        "tree_building": {"prompt": 0, "completion": 0, "total": 0},
        "evaluation": [],
        "total_tokens": 0
    }
    # 如果是CoTDecisionTree，每次评估都创建一个新实例，确保token统计独立
    if isinstance(tree_model, LLMDecisionTree):
        tree_model = LLMDecisionTree(
            meta=tree_model.meta,
            max_depth=tree_model.max_depth,
            runner=tree_model.runner,
            log_file=tree_model.log_file
        )
        # 确保新实例有新的token_stats
        tree_model.token_stats = {
            "tree_building": {"prompt": 0, "completion": 0, "total": 0},
            "evaluation": [],
            "total_tokens": 0
        }
        global_log("[DEBUG] 为当前评估创建了新的LLMDecisionTree实例，确保token统计独立")
    
    rules = []  # 保证所有分支下rules都已定义
    # get tree's prediction rules & results
    # --- 强制CoTDecisionTree二次LLM交互始终被执行 ---
    if isinstance(tree_model, LLMDecisionTree) and with_llm:
        global_log(f"[DEBUG][LLM] >>> 进入LLMDecisionTree二次LLM推理分支 <<< use_tree_rules={use_tree_rules}, tree_only={tree_only}, with_llm={with_llm}")
        # 第一次：用LLM prompt生成规则
        global_log("[DEBUG][LLM] === 第一次LLM交互：生成规则 ===")
        
        # 添加时间统计
        fit_start_time = time.time()
        tree_model.fit(x_train, y_train)
        fit_elapsed_time = time.time() - fit_start_time
        global_log(f"[DEBUG][LLM] 树模型训练耗时: {fit_elapsed_time:.2f}秒")
        
        # 获取LLMDecisionTree的token统计
        if hasattr(tree_model, 'get_token_stats'):
            tree_token_stats = tree_model.get_token_stats()
            global_log(f"[DEBUG][LLM] 树模型训练token统计: {tree_token_stats}")
            # 更新总token统计
            if 'tree_building' in tree_token_stats:
                token_stats['tree_building'] = tree_token_stats['tree_building']
                token_stats['total_tokens'] += tree_token_stats['tree_building']['total']
            if 'evaluation' in tree_token_stats and tree_token_stats['evaluation']:
                token_stats['evaluation'].extend(tree_token_stats['evaluation'])
                for eval_token in tree_token_stats['evaluation']:
                    token_stats['total_tokens'] += eval_token.get('total_tokens', 0)
        
        rules = tree_model.get_rules()
        tree_model.rules = rules
        global_log("[DEBUG][LLM] 规则生成完毕，规则内容如下：")
        if isinstance(tree_model.rules, list):
            rules_str = "\n".join(str(rule) for rule in tree_model.rules)
            global_log(rules_str)
        else:
            global_log(str(tree_model.rules))
        global_log("[DEBUG][LLM] === 进入第二次LLM交互：用basic.jinja渲染规则并让LLM预测 ===")
        prompts, test_splits, labels = gen_prompt(
            meta,
            master_template,
            serializer,
            x_train,
            y_train,
            x_test,
            y_test,
            rules,
            num_tests_per_round,
        )
        global_log(f"[DEBUG][LLM] basic.jinja渲染后prompt数量: {len(prompts)}，test_splits: {test_splits}")
        if prompts and len(prompts) > 0:
            global_log("[DEBUG][LLM] basic.jinja prompt内容如下：")
            global_log(prompts[0])
            global_log("[DEBUG][LLM] ===== end of basic.jinja prompt =====")
        if with_llm and (not prompts or len(prompts) == 0 or not prompts[0].strip()):
            global_log("[ERROR][LLM] basic.jinja prompt未能成功生成或内容为空！")
            global_log(f"with_llm={with_llm}, rules类型={type(rules)}, rules内容示例={str(rules)[:200]}")
            global_log(f"x_train shape: {getattr(x_train, 'shape', None)}, y_train shape: {getattr(y_train, 'shape', None)}")
            global_log(f"x_test shape: {getattr(x_test, 'shape', None)}, y_test shape: {getattr(y_test, 'shape', None)}")
            global_log(f"meta: {getattr(meta, 'name', None)} features: {getattr(meta, 'features', None)}")
            raise RuntimeError("basic.jinja prompt生成失败，请检查模板、数据、规则格式和gen_prompt调用参数！")
        raw_results = []
        results = []
        for idx, (responses, token_info) in enumerate(runner.run(prompts)):
            # 记录token信息
            tree_model.token_stats['evaluation'].append(token_info)
            tree_model.token_stats['total_tokens'] += token_info["total_tokens"]
            global_log(f"[DEBUG][LLM] 第{idx+1}个prompt，期望labels数: {test_splits[idx][1] - test_splits[idx][0]}")
            global_log(f"[DEBUG][LLM] LLM返回response candidates: {responses}")
            global_log(f"[DEBUG][LLM] Token使用: prompt={token_info['prompt_tokens']}, completion={token_info['completion_tokens']}, total={token_info['total_tokens']}")
            expected_len = test_splits[idx][1] - test_splits[idx][0]
            found = False
            for response in responses:
                global_log(f"[DEBUG][LLM] LLM response内容如下:\n{response}")
                results_batch = serializer.answer_decoder.decode(response)
                global_log(f"[DEBUG][LLM] decode后结果: {results_batch}")
                if len(results_batch) == expected_len:
                    found = True
                    results += results_batch
                    raw_results.append(response)
                    break
                else:
                    global_log(
                        "[DEBUG][LLM] Length of labels and results do not match (expected: {}, actual: {}), response: {}".format(
                            expected_len, len(results_batch), response
                        )
                    )
            if not found:
                global_log("[ERROR][LLM] Failed to find any valid response, skipping...")
                result_dict = {
                    "record": prompts[0],
                    "failed_raw_output": responses,
                    "token_stats": tree_model.token_stats,  # 添加token统计
                }
                return result_dict
        global_log(f"[DEBUG][LLM] 二次LLM推理最终labels: {labels}")
        global_log(f"[DEBUG][LLM] 二次LLM推理最终results: {results}")
        tree_accuracy, tree_auc = cot_calc_accuracy_auc(y_test, results, meta, force_dict=True)
        
        # 优化二分类情况下的AUC输出格式
        is_binary = len(np.unique(y_test)) <= 2
        formatted_auc = format_auc_result(tree_auc, is_binary)
        global_log(f"[DEBUG][LLM] 二次LLM推理结果 - 准确率: {tree_accuracy:.4f}, AUC: {formatted_auc}")
        
        tree_results = results
        acc = tree_accuracy
        auc = tree_auc
        global_log(f"[DEBUG][LLM] <<< 结束LLMDecisionTree二次LLM推理分支，已完成全部流程 >>>")
    else:
        # 其它树类型和本地推理逻辑保持不变
        if use_tree_rules or tree_only:
            if isinstance(tree_model, FederatedDecisionTree):
                # 添加时间统计
                fit_start_time = time.time()
                all_tree_predict, _ = tree_model.predict(x_train, y_train, x_test)
                fit_elapsed_time = time.time() - fit_start_time
                global_log(f"[DEBUG] 树模型预测耗时: {fit_elapsed_time:.2f}秒")
                
                tree_aucs, tree_accuracies = [], []
                for tree_predict in all_tree_predict:
                    tree_auc = sklearn.metrics.roc_auc_score(y_test, tree_predict)
                    tree_accuracy = calc_accuracy(y_test, tree_predict)
                    tree_aucs.append(tree_auc)
                    tree_accuracies.append(tree_accuracy)
                tree_auc = tree_aucs
                tree_accuracy = tree_accuracies
            elif isinstance(tree_model, LLMDecisionTree):
                # 确保在调用 predict 之前调用 fit 方法
                if with_llm:
                    # 第一次：用LLM prompt生成规则
                    global_log("[DEBUG][LLM] === 第一次LLM交互：生成规则 ===")
                    
                    # 添加时间统计
                    fit_start_time = time.time()
                    tree_model.fit(x_train, y_train)
                    fit_elapsed_time = time.time() - fit_start_time
                    global_log(f"[DEBUG][LLM] 树模型训练耗时: {fit_elapsed_time:.2f}秒")
                    
                    # 获取LLMDecisionTree的token统计
                    if hasattr(tree_model, 'get_token_stats'):
                        tree_token_stats = tree_model.get_token_stats()
                        global_log(f"[DEBUG][LLM] 树模型训练token统计: {tree_token_stats}")
                        # 更新总token统计
                        if 'tree_building' in tree_token_stats:
                            tree_model.token_stats['tree_building'] = tree_token_stats['tree_building']
                            tree_model.token_stats['total_tokens'] += tree_token_stats['tree_building']['total']
                        if 'evaluation' in tree_token_stats and tree_token_stats['evaluation']:
                            tree_model.token_stats['evaluation'].extend(tree_token_stats['evaluation'])
                            for eval_token in tree_token_stats['evaluation']:
                                tree_model.token_stats['total_tokens'] += eval_token.get('total_tokens', 0)
                    
                    rules = tree_model.get_rules()
                    tree_model.rules = rules
                    global_log("[DEBUG][LLM] 规则生成完毕，规则内容如下：")
                    if isinstance(tree_model.rules, list):
                        rules_str = "\n".join(str(rule) for rule in tree_model.rules)
                        global_log(rules_str)
                    else:
                        global_log(str(tree_model.rules))
                    global_log("[DEBUG][LLM] === 进入第二次LLM交互：用basic.jinja渲染规则并让LLM预测 ===")
                    prompts, test_splits, labels = gen_prompt(
                        meta,
                        master_template,
                        serializer,
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        rules,
                        num_tests_per_round,
                
                    )
                    global_log(f"[DEBUG][LLM] basic.jinja渲染后prompt数量: {len(prompts)}，test_splits: {test_splits}")
                    if prompts and len(prompts) > 0:
                        global_log("[DEBUG][LLM] basic.jinja prompt内容如下：")
                        global_log(prompts[0])
                        global_log("[DEBUG][LLM] ===== end of basic.jinja prompt =====")
                    if with_llm and (not prompts or len(prompts) == 0 or not prompts[0].strip()):
                        global_log("[ERROR][LLM] basic.jinja prompt未能成功生成或内容为空！")
                        global_log(f"with_llm={with_llm}, rules类型={type(rules)}, rules内容示例={str(rules)[:200]}")
                        global_log(f"x_train shape: {getattr(x_train, 'shape', None)}, y_train shape: {getattr(y_train, 'shape', None)}")
                        global_log(f"x_test shape: {getattr(x_test, 'shape', None)}, y_test shape: {getattr(y_test, 'shape', None)}")
                        global_log(f"meta: {getattr(meta, 'name', None)} features: {getattr(meta, 'features', None)}")
                        raise RuntimeError("basic.jinja prompt生成失败，请检查模板、数据、规则格式和gen_prompt调用参数！")
                    raw_results = []
                    results = []
                    for idx, (responses, token_info) in enumerate(runner.run(prompts)):
                        # 记录token信息
                        token_stats["evaluation"].append(token_info)
                        token_stats["total_tokens"] += token_info["total_tokens"]
                        global_log(f"[DEBUG][LLM] 第{idx+1}个prompt，期望labels数: {test_splits[idx][1] - test_splits[idx][0]}")
                        global_log(f"[DEBUG][LLM] LLM返回response candidates: {responses}")
                        global_log(f"[DEBUG][LLM] Token使用: prompt={token_info['prompt_tokens']}, completion={token_info['completion_tokens']}, total={token_info['total_tokens']}")
                        expected_len = test_splits[idx][1] - test_splits[idx][0]
                        found = False
                        for response in responses:
                            global_log(f"[DEBUG][LLM] LLM response内容如下:\n{response}")
                            results_batch = serializer.answer_decoder.decode(response)
                            global_log(f"[DEBUG][LLM] decode后结果: {results_batch}")
                            if len(results_batch) == expected_len:
                                found = True
                                results += results_batch
                                raw_results.append(response)
                                break
                            else:
                                global_log(
                                    "[DEBUG][LLM] Length of labels and results do not match (expected: {}, actual: {}), response: {}".format(
                                        expected_len, len(results_batch), response
                                    )
                                )
                        if not found:
                            global_log("[ERROR][LLM] Failed to find any valid response, skipping...")
                            result_dict = {
                                "record": prompts[0],
                                "failed_raw_output": responses,
                                "token_stats": token_stats,  # 添加token统计
                            }
                            return result_dict
                    global_log(f"[DEBUG][LLM] 二次LLM推理最终labels: {labels}")
                    global_log(f"[DEBUG][LLM] 二次LLM推理最终results: {results}")
                    tree_accuracy, tree_auc = cot_calc_accuracy_auc(y_test, results, meta, force_dict=True)
                    
                    # 优化二分类情况下的AUC输出格式
                    is_binary = len(np.unique(y_test)) <= 2
                    formatted_auc = format_auc_result(tree_auc, is_binary)
                    global_log(f"[DEBUG][LLM] 二次LLM推理结果 - 准确率: {tree_accuracy:.4f}, AUC: {formatted_auc}")
                    
                    tree_results = results
                    acc = tree_accuracy
                    auc = tree_auc
                    global_log(f"[DEBUG][LLM] <<< 结束LLMDecisionTree二次LLM推理分支，已完成全部流程 >>>")
                else:
                    # 直接应用规则进行预测（with_llm=0时）
                    global_log("[DEBUG][LLM] 直接用规则本地推理，无LLM参与")
                    
                    # 添加时间统计
                    fit_start_time = time.time()
                    tree_model.fit(x_train, y_train)  # 先生成规则，防止predict报错
                    fit_elapsed_time = time.time() - fit_start_time
                    global_log(f"[DEBUG][LLM] 树模型训练耗时: {fit_elapsed_time:.2f}秒")
                    
                    # 获取LLMDecisionTree的token统计
                    if hasattr(tree_model, 'get_token_stats'):
                        tree_token_stats = tree_model.get_token_stats()
                        global_log(f"[DEBUG][LLM] 树模型训练token统计: {tree_token_stats}")
                        if 'tree_building' in tree_token_stats:
                            tree_model.token_stats['tree_building'] = tree_token_stats['tree_building']
                            tree_model.token_stats['total_tokens'] += tree_token_stats['tree_building']['total']
                        if 'evaluation' in tree_token_stats and tree_token_stats['evaluation']:
                            tree_model.token_stats['evaluation'].extend(tree_token_stats['evaluation'])
                            for eval_token in tree_token_stats['evaluation']:
                                tree_model.token_stats['total_tokens'] += eval_token.get('total_tokens', 0)
                    
                    tree_results = tree_model.predict(x_test)
                    
                    # 新增：检查并转换预测结果类型
                    if isinstance(tree_results[0], str):
                        global_log("[DEBUG] LLMDecisionTree返回的预测结果是字符串类型，转换为数值...")
                        # 将字符串标签转换为对应的数值标签
                        tree_results_numeric = []
                        for label in tree_results:
                            label_value = meta.get_label_value(label)
                            if label_value is None:
                                global_log(f"[WARNING] 无法找到标签 '{label}' 对应的数值")
                                # 使用默认值
                                label_value = meta.labels[0].value
                            tree_results_numeric.append(label_value)
                        tree_results = np.array(tree_results_numeric)
                        global_log(f"[DEBUG] 转换后的预测结果: {tree_results[:5]}...")
                    
                    # LLM输出为字符串，专用评测函数
                    tree_accuracy, tree_auc_result = cot_calc_accuracy_auc(y_test, tree_results, meta, force_dict=True)
                    global_log("[LLMDecisionTree] 直接用规则推理（未调用LLM）")
                    global_log(f"规则数量: {len(tree_model.rules)}")
                    
                    # 优化二分类情况下的AUC输出格式
                    is_binary = len(np.unique(y_test)) <= 2
                    if is_binary and isinstance(tree_auc_result, dict):
                        # 二分类情况下，简化输出为单个值
                        auc_value = tree_auc_result.get('macro', float('nan'))
                        global_log(f"tree_accuracy: {tree_accuracy}, tree_auc: {auc_value}")
                    else:
                        global_log(f"tree_accuracy: {tree_accuracy}, tree_auc: {tree_auc_result}")
                        
                    results = tree_results
                    acc = tree_accuracy
                    auc = tree_auc_result
            else:
                # 其他类型的决策树
                tree_results, rules = tree_model.predict(
                    x_train, y_train, x_test, export_rules=True
                )
                # 修改这里的直接AUC计算，使用cot_calc_accuracy_auc函数
                if len(set(tree_results)) < 2:
                    global_log("[WARNING] 预测结果只有一个类别，AUC无法计算，返回NaN")
                    tree_auc = {"macro": float('nan'), "micro": float('nan')}
                else:
                    tree_accuracy, tree_auc = cot_calc_accuracy_auc(y_test, tree_results, meta, force_dict=True)

    if tree_only:
        # 分类树在训练时已经被训练好了，直接预测。
        # 这些预测结果可能是浮点数或整数。
        results, rule_trace = tree_model.predict(x_test)
        y_test_num = [float(y) for y in y_test]
        
        # 如果预测结果为字符串，则转换为标签值
        if isinstance(results[0], str):
            global_log("树模型返回字符串标签，转换为数值...")
            results_num = []
            for r in results:
                label_val = meta.get_label_value(r)
                if label_val is None:
                    # 如果找不到对应的标签值，使用默认值0
                    global_log(f"警告：找不到标签'{r}'对应的值，使用默认值0")
                    results_num.append(0)
                else:
                    results_num.append(label_val)
        else:
            results_num = results
            
        # 计算准确率
        tree_accuracy = calc_accuracy(y_test, results)
        
        # 计算AUC (修改为使用cot_calc_accuracy_auc函数)
        if len(set(results_num)) < 2:
            global_log("[WARNING] 预测结果只有一个类别，AUC无法计算，返回NaN")
            tree_auc = {"macro": float('nan'), "micro": float('nan')}
        else:
            tree_accuracy, tree_auc = cot_calc_accuracy_auc(y_test_num, results_num, meta, force_dict=True)
            
        tree_results = results_num
        acc = tree_accuracy
        auc = tree_auc
        
        return {
            "acc": acc,
            "auc": auc,  # 现在是字典格式
            "llm_tree_results": None,
            "labels": y_test.tolist(),
            "rule_trace": rule_trace,
            "tree_results": tree_results,
            "samples": [],
            "match_by_samples": [],
            "prompts": [],
            "trees": 1,
            "token_stats": tree_model.token_stats,  # 添加token统计信息
            "train_elapsed": fit_elapsed_time  # 添加模型训练时间
        }

    # 如果不是tree_only模式且不是LLMDecisionTree的with_llm模式
    if not (isinstance(tree_model, LLMDecisionTree) and with_llm):
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

        for idx, (responses, token_info) in enumerate(runner.run(prompts)):
            # 记录token信息
            tree_model.token_stats['evaluation'].append(token_info)
            tree_model.token_stats['total_tokens'] += token_info["total_tokens"]
            global_log(f"[DEBUG] Token使用: prompt={token_info['prompt_tokens']}, completion={token_info['completion_tokens']}, total={token_info['total_tokens']}")
            expected_len = test_splits[idx][1] - test_splits[idx][0]
            found = False
            for response in responses:
                global_log(f"Full LLM response:\n{response}")
                results_batch = serializer.answer_decoder.decode(response)
                if len(results_batch) == expected_len:
                    found = True
                    results += results_batch
                    raw_results.append(response)
                    break
                else:
                    global_log(
                        "Length of labels and results do not match (expected: {}, actual: {}), response: {}".format(
                            expected_len, len(results_batch), response
                        )
                    )
            if not found:
                global_log("Failed to find any valid response, skipping...")
                result_dict = {
                    "record": prompts[0],
                    "failed_raw_output": responses,
                     "token_stats": tree_model.token_stats,  # 添加token统计
                }
                return result_dict

        acc = calc_accuracy(labels, results)
        # 使用专用函数计算AUC，将结果改为字典形式
        tree_accuracy, tree_auc = cot_calc_accuracy_auc(y_test, [meta.get_label_value(r) for r in results], meta, force_dict=True)

        # 优化二分类情况下的AUC输出格式
        formatted_auc = format_auc_result(tree_auc)
        global_log(f"Accuracy/AUC: {acc}/{formatted_auc}")

    # 增加修改：始终保存tree的预测结果，即使AUC计算失败
    result = {
        "tree_results": tree_results,
        "labels": y_test.tolist(),
        "token_stats": tree_model.token_stats,  # 添加token统计信息
        "train_elapsed": fit_elapsed_time  # 添加模型训练时间
    }
    
    # 计算基础准确率和AUC（用于决策树）
    global_log("计算决策树准确率和AUC...")
    tree_acc, tree_auc = cot_calc_accuracy_auc(
        y_test, tree_results, meta, force_dict=True
    )
    is_binary = len(np.unique(y_test)) <= 2
    formatted_auc = format_auc_result(tree_auc, is_binary)
    global_log(f"决策树准确率: {tree_acc:.4f}, AUC: {formatted_auc}")
    
    # 确保tree_auc被保存，即使值为NaN
    result["tree_acc"] = tree_acc
    result["tree_auc"] = tree_auc
    
    # 处理LLM预测（如果有）
    if with_llm:
        # 计算LLM+树的准确率和AUC
        llm_tree_acc, llm_tree_auc = cot_calc_accuracy_auc(
            y_test, results, meta, force_dict=True
        )
        formatted_llm_auc = format_auc_result(llm_tree_auc, is_binary)
        global_log(f"LLM+树准确率: {llm_tree_acc:.4f}, AUC: {formatted_llm_auc}")
        
        # 增加LLM相关结果
        result["llm_tree_acc"] = llm_tree_acc
        result["llm_tree_auc"] = llm_tree_auc
        result["llm_tree_results"] = results
        result["prompts"] = prompts
    
    # 增加模型导出
    try:
        global_log("导出模型...")
        result["model"] = tree_model.export_dict()
        global_log("模型导出成功")
    except Exception as e:
        global_log(f"模型导出失败: {e}")
        result["model_error"] = str(e)
    
    # 增加标签分布信息
    try:
        y_dist = dict(zip(*np.unique(y_test, return_counts=True)))
        result["label_distribution"] = {str(k): int(v) for k, v in y_dist.items()}
    except Exception as e:
        global_log(f"无法计算标签分布: {e}")
    
    return result


def main():
    args = parse_args()
    log_dir = Path("output") / "evaluate" / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 统一文件名主干
    file_name = f"{args.exp_name}_with_llm_{timestamp}" if getattr(args, 'with_llm', False) else f"{args.exp_name}_{timestamp}"
    log_file_path = log_dir / (file_name + ".log")
    output_file = Path(args.output_dir) / (file_name + ".json")
    from tree_prompt.logger import Logger, add_logger
    add_logger(Logger(open(log_file_path, "a", encoding="utf-8", buffering=1)))

    global_log("Arguments:")
    for k, v in args.__dict__.items():
        global_log(" - {}: {}".format(k, v))

    global_log(f"最终 max_depth: {args.tree_args.max_depth}")

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # 添加总评估时间统计
    eval_start_time = time.time()

    results = []
    meta, x, y = load_dataset(args.dataset_args)

    if args.print_only:
        raise NotImplementedError("Not implemented yet")

    results: dict[int, list] = {}
    runner,serializer, master_template = None, None,None
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
            global_log(f"OpenAIAPIParallelRunner initialized with model={args.runner_args.model_name}, batch_size={args.runner_args.parallel_batch_size}")

        elif args.runner == "huggingchat":
            from tree_prompt.runner.huggingchat import HuggingChatParallelRunner

            runner = HuggingChatParallelRunner(
                args.runner_args.hf_username,
                args.runner_args.hf_password,
                args.runner_args.hf_cookie_dir,
                args.runner_args.request_interval,
                args.runner_args.timeout,
            )
            global_log("HuggingChatParallelRunner initialized")
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
            global_log(f"TogetherAPIParallelRunner initialized with model={args.runner_args.model_name}, batch_size={args.runner_args.parallel_batch_size}")
        else:
            global_log(f"Unknown runner type: {args.runner}")
            raise ValueError("Unknown runner type: {}".format(args.runner))

        if args.serializer_type == "tabular":
            serializer = TabularSerializer(meta)
            global_log("Using TabularSerializer")
        elif args.serializer_type == "list":
            serializer = ListSerializer(meta)
            global_log("Using ListSerializer")
        elif args.serializer_type == "text":
            serializer = TextSerializer(meta)
            global_log("Using TextSerializer")
        else:
            global_log(f"Unknown serializer type: {args.serializer_type}")
            raise ValueError("Unknown serializer type: {}".format(args.serializer_type))

        
        master_template_path = Path(args.template)
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(master_template_path.parent),
        )
        master_template = env.get_template(master_template_path.name)
        global_log(f"Template loaded from: {master_template_path}")
    if args.tree_type == "LLM":
        template_dir = Path("C:/Users/chenx/git/tree/template")
        tree_template_path = template_dir / "basic.jinja"
        # 这里log_file_path和主日志文件一致
        tree_model = LLMDecisionTree(
            meta=meta,
            max_depth=args.tree_args.max_depth,
            runner=runner,
            log_file=log_file_path,
        )
        global_log(f"LLMDecisionTree initialized with max_depth={args.tree_args.max_depth}")
    

    results: dict[int, list[dict]] = {}

    test_x, test_y = x[: args.test_size], y[: args.test_size]
    avail_x, avail_y = x[args.test_size :], y[args.test_size :]
    global_log(f"Test set size: {len(test_x)}, Available set size: {len(avail_x)}")

    bar = tqdm(desc="Total", total=len(args.train_sizes) * args.num_tests_per_set)
    global_log(f"Starting evaluation with train sizes: {args.train_sizes}")

    for train_size in args.train_sizes:
        global_log(f"Processing train size: {train_size}")
        train_cases = sample_balanced(
            avail_x,
            avail_y,
            args.num_tests_per_set,
            train_size,
            args.random_seed,
        )
        global_log(f"Generated {len(train_cases)} balanced training cases")
        for train_x, train_y in train_cases:
            # 添加开始时间统计
            eval_start_time = time.time()
            if args.tree_type == "LLM":
                current_tree_model = LLMDecisionTree(
                    meta=meta,
                    max_depth=args.tree_args.max_depth,
                    runner=runner,
                    log_file=log_file_path,
                )
                global_log(f"[DEBUG] 为当前训练样本创建了新的LLMDecisionTree实例")
            else:
                current_tree_model = tree_model  # 其他类型的树直接使用原始实例
            result = evaluate(
                train_x,
                train_y,
                test_x,
                test_y,
                runner,
                master_template,
                serializer,
                meta,
                current_tree_model,  # 使用新创建的树模型实例
                args.use_tree_rules,
                args.tree_only,
                args.test_batch,
                with_llm=args.with_llm,
            )
            
            # 计算总耗时
            eval_elapsed_time = time.time() - eval_start_time
            global_log(f"评估总耗时: {eval_elapsed_time:.2f}秒")
            
            # 确保结果中包含评估耗时
            if isinstance(result, dict):
                result["eval_elapsed_time"] = eval_elapsed_time
            
            # 处理二分类结果，将AUC从字典转换为单个值
            if len(np.unique(test_y)) <= 2:
                global_log("检测到二分类数据集，将AUC从字典格式转换为单个值")
                
                # 转换tree_auc (如果存在且是字典格式)
                if "tree_auc" in result and isinstance(result["tree_auc"], dict):
                    result["tree_auc"] = format_auc_result(result["tree_auc"], is_binary=True)
                    global_log(f"转换后的tree_auc: {result['tree_auc']}")
                
                # 转换llm_tree_auc (如果存在且是字典格式)
                if "llm_tree_auc" in result and isinstance(result["llm_tree_auc"], dict):
                    result["llm_tree_auc"] = format_auc_result(result["llm_tree_auc"], is_binary=True)
                    global_log(f"转换后的llm_tree_auc: {result['llm_tree_auc']}")
                
                # 转换auc (如果存在且是字典格式)
                if "auc" in result and isinstance(result["auc"], dict):
                    result["auc"] = format_auc_result(result["auc"], is_binary=True)
                    global_log(f"转换后的auc: {result['auc']}")
            
            results.setdefault(train_size, []).append(result)
            bar.update(1)

    # 计算总评估时间
    eval_elapsed_time = time.time() - eval_start_time
    global_log(f"总评估耗时: {eval_elapsed_time:.2f}秒")
    
    # 打印每个训练集大小的平均训练耗时
    for train_size, train_results in results.items():
        total_time = 0.0
        count = 0
        for result in train_results:
            if 'train_elapsed' in result:
                total_time += result['train_elapsed']
                count += 1
        if count > 0:
            global_log(f"训练样本数 {train_size} 的平均训练耗时: {total_time/count:.2f}秒 (共{count}次)")
    
    global_log("Saving results to {}...".format(output_file))

    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)

    if output_file.exists():
        date = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        target = output_file.rename(
            output_file.parent / (output_file.stem + "-" + date + output_file.suffix)
        )
        global_log("Output file already exists, renamed to {}".format(target))
    
    # 处理所有结果中的二元分类AUC格式
    for train_size, train_results in results.items():
        for res in train_results:
            # 检查是否为二元分类数据
            if 'labels' in res and isinstance(res['labels'], list) and len(np.unique(res['labels'])) <= 2:
                global_log(f"检测到二元分类数据，正在格式化AUC结果")
                
                # 处理tree_auc
                if 'tree_auc' in res and isinstance(res['tree_auc'], dict):
                    # 格式化二元分类的AUC结果
                    try:
                        if res['tree_auc'].get('macro') == res['tree_auc'].get('micro'):
                            global_log(f"将tree_auc从字典格式转换为单值: {res['tree_auc']}")
                            res['tree_auc'] = res['tree_auc']['macro']
                    except Exception as e:
                        global_log(f"转换tree_auc时出错: {e}")
                
                # 处理llm_tree_auc
                if 'llm_tree_auc' in res and isinstance(res['llm_tree_auc'], dict):
                    try:
                        if res['llm_tree_auc'].get('macro') == res['llm_tree_auc'].get('micro'):
                            global_log(f"将llm_tree_auc从字典格式转换为单值: {res['llm_tree_auc']}")
                            res['llm_tree_auc'] = res['llm_tree_auc']['macro']
                    except Exception as e:
                        global_log(f"转换llm_tree_auc时出错: {e}")
                
                # 处理auc（如果存在）
                if 'auc' in res and isinstance(res['auc'], dict):
                    try:
                        if res['auc'].get('macro') == res['auc'].get('micro'):
                            global_log(f"将auc从字典格式转换为单值: {res['auc']}")
                            res['auc'] = res['auc']['macro']
                    except Exception as e:
                        global_log(f"转换auc时出错: {e}")
    
    output_file = Path(args.output_dir) / (file_name + ".json")
    global_log(f"Saving results to {output_file}...")
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)
    def json_default_decode(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.__dict__
    with open(output_file, "w") as output_file_fp:
        output = {
            "args": args.__dict__, 
            "results": results
        }
        json.dump(output, output_file_fp, indent=2, default=json_default_decode)


if __name__ == "__main__":

    main()
