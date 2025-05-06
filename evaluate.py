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
from tree_prompt.model.strategy import clean_llm_response


def _get_missing_fields(instance: any, prefix: str = None) -> list[str]:
    missing_fields = []
    for k, v in instance.__dict__.items():
        # 检查是否为tree_only模式，如果是则跳过检查特定字段
        if hasattr(instance, 'tree_only') and instance.tree_only and k in ["runner", "use_tree_rules", "template", "serializer_type"]:
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

    def get_missing_fields(self) -> list[str]:
        # 先获取所有缺失字段
        missing_fields = []
        for k, v in self.__dict__.items():
            # 只有在非tree_only模式下才检查runner和use_tree_rules字段
            if v is None and not (self.tree_only and k in ["runner", "use_tree_rules", "template", "serializer_type"]):
                missing_fields.append(k)
        
        # 只有在非tree_only模式下才检查runner_args
        if not self.tree_only:
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

            elif self.runner_args and not isinstance(self.runner_args, dict):
                missing_fields += _get_missing_fields(self.runner_args, "runner_args")

        # 检查tree_args
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
        elif self.tree_args and not isinstance(self.tree_args, dict):
            missing_fields += _get_missing_fields(self.tree_args, "tree_args")

        # 检查dataset_args
        if not isinstance(self.dataset_args, DatasetArgs) and not isinstance(self.dataset_args, dict):
            missing_fields.append("dataset_args")
        elif self.dataset_args and not isinstance(self.dataset_args, dict):
            missing_fields += _get_missing_fields(self.dataset_args, "dataset_args")

        return missing_fields

    def get_missing_fields_tree_only(self) -> list[str]:
        # 避免检查runner_args
        self_copy = EvaluateArgs()
        self_copy.__dict__ = self.__dict__.copy()
        self_copy.tree_only = True  # 强制设置为tree_only模式
        
        return self_copy.get_missing_fields()

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

    # read openai api key from env
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key and not runner_args_dict.get("openai_api_key"):
        if args.runner_args is None or not args.runner_args.get("openai_api_key"):
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


def evaluate(
    train_x,
    train_y,
    test_x,
    test_y,
    runner,
    master_template,
    serializer,
    meta,
    tree_model,
    use_tree_rules,
    tree_only,
    test_batch,
):
    # 检查模型类型，只有在简单决策树时才尝试导出规则
    export_rules = use_tree_rules
    if isinstance(tree_model, (RandomForestDecisionTree, XGBoostDecisionTree, FederatedDecisionTree)):
        export_rules = False
    
    # 调用预测时捕获NotImplementedError
    try:
        tree_predict, rules = tree_model.predict(
            train_x, train_y, test_x, export_rules
        )
        # 获取树的训练时间
        train_elapsed = getattr(tree_model, "train_elapsed", 0.0)
    except NotImplementedError:
        # 如果规则导出不被支持，重新调用但禁用规则导出
        tree_predict, rules = tree_model.predict(
            train_x, train_y, test_x, False
        )
        # 获取树的训练时间
        train_elapsed = getattr(tree_model, "train_elapsed", 0.0)
        rules = []  # 提供空规则列表
    
    tree_accuracy = calc_accuracy(test_y, tree_predict)
    tree_auc = None
    
    # 只在需要使用树规则或只使用树时计算树模型的AUC
    if use_tree_rules or tree_only:
        n_classes = len(np.unique(test_y))
        tree_predict_values = np.array(tree_predict)  # 确保预测值是numpy数组
        
        # 处理树模型预测的AUC计算
        try:
            if n_classes > 2:
                from sklearn.preprocessing import label_binarize
                classes = np.unique(test_y)
                
                # 二进制化真实标签
                y_test_bin = label_binarize(test_y, classes=classes)
                
                # 检查树模型预测是否需要二进制化
                if tree_predict_values.ndim == 1:
                    # 如果是一维预测（类别标签），则二进制化
                    tree_predict_bin = label_binarize(tree_predict_values, classes=classes)
                    tree_auc = {
                        'macro': sklearn.metrics.roc_auc_score(y_test_bin, tree_predict_bin, average='macro'),
                        'micro': sklearn.metrics.roc_auc_score(y_test_bin, tree_predict_bin, average='micro')
                    }
                else:
                    # 如果已经是概率矩阵，则直接使用
                    tree_auc = {
                        'macro': sklearn.metrics.roc_auc_score(y_test_bin, tree_predict_values, average='macro'),
                        'micro': sklearn.metrics.roc_auc_score(y_test_bin, tree_predict_values, average='micro')
                    }
                
                # 将计算结果记录到日志
                logger.log(f"Tree AUC (macro/micro): {tree_auc['macro']:.4f}/{tree_auc['micro']:.4f}")
            else:
                # 二分类情况，直接计算AUC
                tree_auc = sklearn.metrics.roc_auc_score(test_y, tree_predict_values)
                logger.log(f"Tree AUC: {tree_auc:.4f}")
        except Exception as e:
            # 处理AUC计算中可能出现的错误（例如所有预测都是同一个类别）
            logger.log(f"计算树模型AUC时出错: {e}")
            if n_classes > 2:
                tree_auc = {'macro': 0.5, 'micro': 0.5, 'error': str(e)}
            else:
                tree_auc = 0.5
                logger.log("使用默认AUC值: 0.5")

    if tree_only:
        logger.log(f"树模型训练时间: {train_elapsed:.4f}秒")
        return {
            "tree_auc": tree_auc,
            "tree_accuracy": tree_accuracy,
            "tree_results": tree_predict.tolist() if isinstance(tree_predict, np.ndarray) else tree_predict,
            "train_elapsed": train_elapsed  # 添加树训练时间
        }

    prompts, test_splits, labels = gen_prompt(
        meta,
        master_template,
        serializer,
        train_x,
        train_y,
        test_x,
        test_y,
        rules if use_tree_rules else [],
        test_batch,
    )

    raw_results = []
    results = []
    # 添加token统计
    eval_tokens = []

    for idx, response_data in enumerate(runner.run(prompts)):
        responses, token_info = response_data  # 解包响应和token信息
        # 记录token信息
        eval_tokens.append(token_info)
        logger.log(f"评估 #{idx+1} Token使用: 输入={token_info['prompt_tokens']}, 输出={token_info['completion_tokens']}, 总计={token_info['total_tokens']}")
        
        expected_len = test_splits[idx][1] - test_splits[idx][0]
        found = False
        for response in responses:
            # 清理响应中的<think>标签
            response = clean_llm_response(response)
            
            # 预处理响应 - 分割每行并清理空白
            try:
                # 尝试不同的响应格式处理方法
                # 方法1: 按行分割，清理每行的空白
                lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
                if len(lines) == expected_len:
                    found = True
                    results += lines
                    raw_results.append(response)
                    break
                
                # 方法2: 使用原有解码器
                results_batch = serializer.answer_decoder.decode(response)
                if len(results_batch) == expected_len:
                    found = True
                    results += results_batch
                    raw_results.append(response)
                    break
                
            except Exception as e:
                logger.log(f"解析响应时出错: {e}")
                logger.log(f"原始响应: {response}")
        if not found:
            logger.log("Failed to find any valid response, skipping...")
            result_dict = {
                "record": prompts[0],
                "failed_raw_output": responses,
                "eval_tokens": eval_tokens  # 添加token统计
            }
            return result_dict

    acc = calc_accuracy(labels, results)
    
    # 收集并清理预测结果值，确保没有None值
    results_values = []
    for r in results:
        try:
            value = meta.get_label_value(r)
            if value is not None:  # 确保没有None值
                results_values.append(value)
            else:
                # 如果找不到对应的标签值，使用0作为默认值（或者其他合适的默认值）
                logger.log(f"警告: 无法为'{r}'找到有效的标签值，使用默认值0")
                results_values.append(0)
        except Exception as e:
            logger.log(f"警告: 处理预测结果'{r}'时出错: {e}，使用默认值0")
            results_values.append(0)
    
    # 确保结果长度与测试标签相同
    if len(results_values) != len(test_y):
        logger.log(f"警告: 预测结果数量({len(results_values)})与测试标签数量({len(test_y)})不匹配")
        # 如果长度不匹配，可能需要调整
        
    # 计算AUC，添加异常处理
    try:
        n_classes = len(meta.labels)
        if n_classes > 2:
            from sklearn.preprocessing import label_binarize
            classes = np.unique(test_y)
            
            # 二进制化处理
            y_test_bin = label_binarize(test_y, classes=classes)
            if np.array(results_values).ndim == 1:  # 如果预测是一维的
                results_bin = label_binarize(results_values, classes=classes)
                # 计算宏平均AUC和微平均AUC
                auc = {
                    'macro': sklearn.metrics.roc_auc_score(y_test_bin, results_bin, average='macro'),
                    'micro': sklearn.metrics.roc_auc_score(y_test_bin, results_bin, average='micro')
                }
            else:  # 如果已经是二维的
                auc = {
                    'macro': sklearn.metrics.roc_auc_score(y_test_bin, results_values, average='macro'),
                    'micro': sklearn.metrics.roc_auc_score(y_test_bin, results_values, average='micro')
                }
        else:
            auc = sklearn.metrics.roc_auc_score(test_y, results_values)
    except Exception as e:
        logger.log(f"计算LLM预测AUC时出错: {e}")
        # 提供默认的AUC值
        if n_classes > 2:
            auc = {'macro': 0.5, 'micro': 0.5, 'error': str(e)}
        else:
            auc = 0.5
        logger.log("使用默认AUC值: 0.5")

    if isinstance(auc, dict):
        logger.log("LLM Accuracy/AUC (macro/micro): {:.4f}/{:.4f}/{:.4f}".format(acc, auc['macro'], auc['micro']))
    else:
        logger.log("LLM Accuracy/AUC: {:.4f}/{:.4f}".format(acc, auc))

    # 只在使用树规则时输出树模型的结果
    if use_tree_rules and tree_auc is not None:
        if isinstance(tree_auc, dict):
            logger.log("Tree Accuracy/AUC (macro/micro): {:.4f}/{:.4f}/{:.4f}".format(
                tree_accuracy, tree_auc['macro'], tree_auc['micro']
            ))
        else:
            logger.log("Tree Accuracy/AUC: {:.4f}/{:.4f}".format(tree_accuracy, tree_auc))

    # 计算总token使用量
    total_tokens = {"prompt": 0, "completion": 0, "total": 0}
    for token_info in eval_tokens:
        total_tokens["prompt"] += token_info.get("prompt_tokens", 0)
        total_tokens["completion"] += token_info.get("completion_tokens", 0)
        total_tokens["total"] += token_info.get("total_tokens", 0)
    
    logger.log(f"总评估Token使用: 输入={total_tokens['prompt']}, 输出={total_tokens['completion']}, 总计={total_tokens['total']}")

    result_dict = {}
    result_dict["record"] = {"prompt": prompts[0]}
    result_dict["labels"] = [int(y) for y in test_y]
    result_dict["results"] = [meta.get_label_value(r) for r in results]
    result_dict["auc"] = auc
    result_dict["accuracy"] = acc
    result_dict["eval_tokens"] = eval_tokens
    result_dict["total_tokens"] = total_tokens

    # 只在使用树规则时添加树相关结果
    if use_tree_rules or tree_only:
        result_dict["tree_auc"] = tree_auc
        result_dict["tree_accuracy"] = tree_accuracy
        result_dict["tree_results"] = tree_predict.tolist() if isinstance(tree_predict, np.ndarray) else tree_predict
        result_dict["train_elapsed"] = train_elapsed  # 添加树训练时间
        logger.log(f"最终决策树训练时间: {train_elapsed:.4f}秒")

    return result_dict


def main():
    args = parse_args()

    if args.exp_id:
        logger.DEFAULT_LOGGERS[0].prefix = "[{}] ".format(args.exp_id)

    logger.log("Arguments:")
    for k, v in args.__dict__.items():
        logger.log(" - {}: {}".format(k, v))

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    results = []
    meta, x, y = load_dataset(args.dataset_args)

    if args.print_only:
        raise NotImplementedError("Not implemented yet")

    results: dict[int, list] = {}

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

        if args.serializer_type == "tabular":
            serializer = TabularSerializer(meta)
        elif args.serializer_type == "list":
            serializer = ListSerializer(meta)
        elif args.serializer_type == "text":
            serializer = TextSerializer(meta)
        else:
            raise ValueError("Unknown serializer type: {}".format(args.serializer_type))

        master_template_path = Path(args.template)
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(master_template_path.parent),
        )
        master_template = env.get_template(master_template_path.name)

    results: dict[int, list[dict]] = {}

    test_x, test_y = x[: args.test_size], y[: args.test_size]
    avail_x, avail_y = x[args.test_size :], y[args.test_size :]

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

    # 添加总的token统计数据
    total_eval_tokens = {"prompt": 0, "completion": 0, "total": 0}
    eval_records_count = 0
    
    # 计算所有评估的总token用量
    for train_size, train_results in results.items():
        for result in train_results:
            if "total_tokens" in result:
                total_eval_tokens["prompt"] += result["total_tokens"].get("prompt", 0)
                total_eval_tokens["completion"] += result["total_tokens"].get("completion", 0) 
                total_eval_tokens["total"] += result["total_tokens"].get("total", 0)
                eval_records_count += 1
    
    # 记录总token使用情况
    if eval_records_count > 0:
        logger.log(f"实验总评估Token使用: 输入={total_eval_tokens['prompt']}, 输出={total_eval_tokens['completion']}, 总计={total_eval_tokens['total']}")

    with open(output_file, "w") as output_file:
        output = {
            "args": args.__dict__, 
            "results": results,
            "total_eval_tokens": total_eval_tokens,
            "eval_records_count": eval_records_count
        }
        json.dump(output, output_file, indent=2, default=json_default_decode)


if __name__ == "__main__":
    main()
