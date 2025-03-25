import numpy as np
import jinja2
from itertools import product
from functools import reduce
import random
from tqdm import tqdm

from ..dataset import DatasetMeta, get_feature_importance_ranking
from ..runner import Runner
from ..prompt import Serializer, TabularSerializer, ListSerializer, TextSerializer
from .tree import DecisionTree, RandomForest, TreeBase, RulePath, Node
from .. import logger
from .loss import LossFunction
from .feature_selection import calculate_chi_square_scores, select_best_feature, calculate_weight_factor


def _get_feature_values(
    meta: DatasetMeta, x: np.ndarray, hist_nbins: int
) -> list[list]:
    feature_values = []

    for i in range(meta.feature_count()):
        feature = meta.features[i]
        if feature.is_categorical:
            feature_values.append(np.unique(x[:, i]))
        else:
            values = np.sort(np.unique(x[:, i]))
            if len(values) > hist_nbins:
                # histogram
                nums, values = np.histogram(
                    values,
                    hist_nbins,
                )
                values = values[np.where(nums > 0)[0] + 1]
                feature_values.append(values)
            else:
                if len(x) > 1:
                    feature_values.append(values[1:])  # drop the first value
                else:
                    feature_values.append(values)

    return feature_values


class TrainStrategy:
    def set_train_data(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        raise NotImplementedError

    def step(self) -> tuple[bool, float]:
        raise NotImplementedError

    def predict_tree(self, x: np.ndarray) -> list[int]:
        raise NotImplementedError

    def predict_tree_raw(self, x: np.ndarray) -> list[int]:
        raise NotImplementedError

    def predict_llm_with_tree(
        self, x: np.ndarray, with_examples: bool = False
    ) -> list[int]:
        raise NotImplementedError

    def export(self) -> dict:
        raise NotImplementedError

    def get_tree(self) -> TreeBase:
        raise NotImplementedError

    @staticmethod
    def load(model_dict: dict) -> "TrainStrategy":
        raise NotImplementedError

    def _get_available_predictions(self, x: np.ndarray) -> list[int]:
        raise NotImplementedError


class UnknownClassStrategy(TrainStrategy):
    LOSS_THRESHOLD = 1e-3

    def __init__(
        self,
        runner: Runner,
        template: jinja2.Template,
        serializer: Serializer,
        loss_f: LossFunction,
        max_depth: int,
        train_batch: int,
        hist_nbins: int,
    ) -> None:
        self.runner = runner
        self.template = template
        self.serializer = serializer
        self.max_depth = max_depth
        self.train_batch = train_batch
        self.hist_nbins = hist_nbins
        self.loss_f = loss_f

    @classmethod
    def get_feature_ranking(cls, meta: DatasetMeta, runner: Runner) -> list[int]:
        """获取数据集的特征重要性排序（类级别缓存）"""
        if not hasattr(cls, '_cached_rankings'):
            cls._cached_rankings = {}
            
        # 使用数据集名称作为缓存键
        cache_key = meta.name
        if cache_key not in cls._cached_rankings:
            # 获取LLM对特征的排序
            ranking = get_feature_importance_ranking(meta, runner)
            if ranking:
                cls._cached_rankings[cache_key] = ranking
                logger.log(f"获取数据集 {cache_key} 的LLM特征重要性排序: {ranking}")
            else:
                # 如果获取失败，使用默认顺序
                ranking = list(range(meta.feature_count()))
                logger.log(f"无法获取LLM特征排序，使用默认顺序: {ranking}")
                cls._cached_rankings[cache_key] = ranking
                
        return cls._cached_rankings[cache_key]

    def set_train_data(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        self.train_x = train_x
        self.train_y = train_y
        
        self.split_values = _get_feature_values(self._meta, train_x, self.hist_nbins)
        
        self.tree = DecisionTree(self.max_depth, self._meta.categories_map)
        self.tree.set_train_data(train_x)
        self.last_loss = None
        self.lass_loss_components = None
        
        # 获取数据集的特征排序（使用缓存）
        self.llm_feature_ranking = self.get_feature_ranking(self._meta, self.runner)

    def _get_available_predictions(self) -> list[int]:
        """获取可用的预测值列表"""
        return [-1, *range(self._meta.label_count())]

    def step(self) -> tuple[bool, float]:
        """执行一步训练，返回是否继续训练和当前损失"""
        # 获取下一个要分裂的节点
        next_split = self.tree.next_to_split()
        if next_split is None:
            return False, None
        
        samples = next_split.get_samples()
        # 确保samples不是None，并且长度检查安全
        if samples is None or len(samples) < 2 or next_split.depth >= self.max_depth:
            next_split.freeze()
            return True, None
        
        # 批量处理：根据LLM排序一次性选择特征和分裂点
        node_samples = samples
        
        # 使用缓存的特征重要性排序
        best_feature = self._quick_feature_selection(next_split)
        if best_feature is None:
            next_split.freeze()
            return True, None
        
        # 快速确定分裂点（减少LLM调用）
        split_values = self._quick_split_values(best_feature, next_split)
        if len(split_values) == 0:
            next_split.freeze()
            return True, None
        
        # 简单启发式选择标签
        left_class, right_class = self._assign_leaf_values(next_split, best_feature, split_values[0])
        
        # 执行分裂
        next_split.split(best_feature, split_values[0], 
                        self._meta.features[best_feature].is_categorical,
                        left_class, right_class)
        
        # 无需每次都调用LLM评估
        return True, 0.0

    def prune_tree(self) -> None:
        """
        Freeze leaf nodes that is certain & contains only one
        class (and same as leaf node class) from the training set
        """

        nodes_map: dict[Node, set[int]] = {}
        for x, y in zip(self.train_x, self.train_y):
            node = self.tree.get_leaf(x)
            if node.leaf_class >= 0:
                nodes_map.setdefault(node, set()).add(y)

        for node, classes in nodes_map.items():
            if len(classes) == 1 and node.leaf_class == classes.pop():
                node.freeze()

    def predict_tree(self, x: np.ndarray) -> list[int]:
        results = self.predict_tree_raw(x)
        for i, res in enumerate(results):
            if res < 0:
                idx = random.randint(0, self._meta.label_count() - 1)
                results[i] = self._meta.labels[idx].value
        return results

    def predict_tree_raw(self, x: np.ndarray) -> list[int]:
        ret = []
        for xx in x:
            idx = self.tree.predict_one(xx)
            y = self._meta.labels[idx].value if idx >= 0 else idx
            ret.append(y)
        return ret

    def _gen_prompt(
        self, x: np.ndarray, examples: tuple[np.ndarray, np.ndarray] = None
    ) -> str:
        rules = self._get_tree_rules()
        if rules is None:
            return None

        x_test_str = [self.serializer.serialize(xx, None) for xx in x]
        if examples is not None:
            examples_str = [self.serializer.serialize(x, y) for x, y in zip(*examples)]
        else:
            examples_str = []

        prompt = self.template.render(
            meta=self._meta,
            examples=examples_str,
            rules=rules,
            format_desc=self.serializer.format_desc(),
            prediction_intro=self.serializer.answer_requirement(len(x_test_str)),
            tests=x_test_str,
        )

        return prompt

    def _predict_llm_with_tree_batched(
        self, prompts: list[str], expected_lens: list[int]
    ) -> list[list[int]]:
        all_results = []
        for i, resp_candidates in enumerate(self.runner.run(prompts)):
            for resp in resp_candidates:
                results = self.serializer.answer_decoder.decode(resp)
                results = [self._meta.get_label_value(r) for r in results]

                if None in results or len(results) != expected_lens[i]:
                    continue

                all_results.append(results)
                break
            else:
                logger.log(
                    "No valid response found, raw responses: {}".format(resp_candidates)
                )
                return None

        return all_results

    def predict_llm_with_tree(
        self, x: np.ndarray, with_examples: bool = False
    ) -> list[int]:
        prompt = self._gen_prompt(
            x,
            (self.train_x, self.train_y) if with_examples else None,
        )
        if prompt is None:
            raise RuntimeError("Invalid tree rules!")
        results = self._predict_llm_with_tree_batched([prompt], [len(x)])
        return results[0] if results is not None else None

    def _get_tree_rules(self) -> list[str]:
        rules: list[str] = []
        paths: list[RulePath] = self.tree.export_paths()
        if paths is None:
            return None
        for r in paths:
            depth = len(r.conditions)
            r = self._serialize_rule(r)
            if r is not None:
                rules.append((r, depth))

        rules.sort(key=lambda x: x[1])
        return [x[0] for x in rules]

    def _serialize_rule(self, rule: RulePath):
        if len(rule.conditions) == 0:
            return None
        if rule.value < 0:
            return None

        label_name = self._meta.labels[rule.value].name

        conds = []
        for feat_idx, cond in rule.conditions.items():
            feat_name = self._meta.features[feat_idx].name
            if cond.is_categorical:
                desc = (
                    feat_name
                    + " is "
                    + " or ".join(
                        f'"{self._meta.value_repr(feat_idx, cat)}"'
                        for cat in cond.categories
                    )
                )
            else:
                if cond.lower is None:
                    desc = (
                        f"{feat_name} < {self._meta.value_repr(feat_idx, cond.upper)}"
                    )
                elif cond.upper is None:
                    desc = (
                        f"{feat_name} >= {self._meta.value_repr(feat_idx, cond.lower)}"
                    )
                else:
                    desc = (
                        f"{self._meta.value_repr(feat_idx, cond.lower)} <= "
                        + f"{feat_name} < "
                        + f"{self._meta.value_repr(feat_idx, cond.upper)}"
                    )

            conds.append(desc)

        return label_name + ": " + " and ".join(conds)

    def export(self) -> any:
        return {
            "type": "unknown_class",
            "model": self.tree.export_nodes_dict(),
            "args": {"max_depth": self.max_depth, "categories": None},  # TODO
            "prompt": self._gen_prompt(
                [],
                (self.train_x, self.train_y),
            ),
        }

    @staticmethod
    def load(
        model_dict: dict,
        runner: Runner = None,
        template: jinja2.Template = None,
        serializer: Serializer = None,
        loss_f: LossFunction = None,
    ) -> "UnknownClassStrategy":
        categories_map = model_dict["args"]["categories"]
        # list to dict
        categories_map = {int(k): set(v) for k, v in categories_map.items()}
        tree = DecisionTree.load_nodes_dict(
            model_dict["model"], model_dict["args"]["max_depth"], categories_map
        )
        strategy = UnknownClassStrategy(
            runner=runner,
            template=template,
            serializer=serializer,
            loss_f=loss_f,
            max_depth=tree.max_depth,
            train_batch=1024,
            hist_nbins=1024,
        )
        strategy.tree = tree

        return strategy

    def get_tree(self) -> DecisionTree:
        return self.tree

    @property
    def _meta(self) -> DatasetMeta:
        return self.serializer.meta

    def _evaluate_split(self, node: Node) -> float:
        """评估当前分裂的损失值"""
        #我们不再使用LLM评估分裂
        pass

    def _quick_feature_selection(self, node):
        """快速特征选择，使用缓存的LLM排序和简单统计"""
        # 获取已使用的特征
        used_features = node.get_used_features()
        
        # 获取节点样本
        node_samples = node.get_samples()
        node_x = self.train_x[node_samples]
        node_y = self.train_y[node_samples]
        
        # 计算特征的卡方检验分数
        chi_square_scores = calculate_chi_square_scores(
            node_x, node_y, self._meta
        )
        
        # 使用缓存的LLM排序和卡方分数选择最佳特征
        depth = node.depth
        best_feature = select_best_feature(
            self.llm_feature_ranking,
            chi_square_scores,
            depth,
            used_features
        )
        
        return best_feature
    
    def _quick_split_values(self, feature_idx, node):
        """快速确定分裂点，不调用LLM"""
        if self._meta.features[feature_idx].is_categorical:
            return self.split_values[feature_idx]
        
        # 获取节点样本数据
        node_samples = node.get_samples()
        node_data = self.train_x[node_samples, feature_idx]
        
        # 获取唯一值
        unique_values = np.unique(node_data)
        if len(unique_values) <= 1:
            logger.log(f"特征 {self._meta.features[feature_idx].name} 的所有值都相同，跳过分裂")
            return []
        
        # 如果只有两个不同的值，使用它们的中点
        if len(unique_values) == 2:
            split_point = (unique_values[0] + unique_values[1]) / 2
            logger.log(f"使用中点 {split_point} 作为分裂点")
            return [split_point]
        
        # 使用统计分位数作为分裂点
        split_points = np.percentile(unique_values, [50])  # 只使用中位数，减少分裂数
        split_points = np.unique(split_points)
        logger.log(f"使用分位数作为分裂点: {split_points}")
        return split_points.tolist()
    
    def _assign_leaf_values(self, node, feature_idx, split_value):
        """使用多数投票快速确定叶节点值"""
        node_samples = node.get_samples()
        node_x = self.train_x[node_samples]
        node_y = self.train_y[node_samples]
        
        # 根据分裂值将样本分为左右两组
        if self._meta.features[feature_idx].is_categorical:
            left_mask = node_x[:, feature_idx] == split_value
        else:
            left_mask = node_x[:, feature_idx] < split_value
        
        right_mask = ~left_mask
        
        # 获取左右子节点的标签
        left_y = node_y[left_mask] if np.any(left_mask) else []
        right_y = node_y[right_mask] if np.any(right_mask) else []
        
        # 使用多数投票确定叶节点值，确保结果在有效范围内
        valid_labels = list(range(self._meta.label_count()))
        
        # 左子节点标签处理
        if len(left_y) > 0:
            # 计算各标签出现次数
            counts = np.bincount(left_y)
            # 限制为有效标签范围
            valid_counts = [counts[i] if i < len(counts) else 0 for i in valid_labels]
            left_class = valid_labels[np.argmax(valid_counts)]
        else:
            left_class = 0  # 默认标签
        
        # 右子节点标签处理
        if len(right_y) > 0:
            counts = np.bincount(right_y)
            valid_counts = [counts[i] if i < len(counts) else 0 for i in valid_labels]
            right_class = valid_labels[np.argmax(valid_counts)]
        else:
            right_class = 0  # 默认标签
        
        return left_class, right_class


class KnownClassStrategy(UnknownClassStrategy):
    def _get_available_predictions(self) -> list[int]:
        """获取可用的预测值列表（不包含unknown）"""
        return [*range(self._meta.label_count())]


class FeatureBaggingStrategy(TrainStrategy):
    def __init__(
        self,
        all_meta: DatasetMeta,
        runner: Runner,
        template: jinja2.Template,
        serializer_type: str,
        loss_f: LossFunction,
        num_trees: int,
        max_depth: int,
        train_batch: int,
        hist_nbins: int,
    ) -> None:
        self.runner = runner
        self.template = template
        self.all_meta = all_meta
        self.loss_f = loss_f
        self.max_depth = max_depth
        self.train_batch = train_batch
        self.hist_nbins = hist_nbins

        self.num_trees = min(num_trees, self.all_meta.feature_count())

        def _split(a, n):
            k, m = divmod(len(a), n)
            return (
                a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
            )

        # divide features into groups
        self.feature_groups = list(
            _split(range(self.all_meta.feature_count()), self.num_trees)
        )
        self.feature_groups = [list(x) for x in self.feature_groups]

        if serializer_type == "tabular":
            self.serializer = TabularSerializer(all_meta)
        elif serializer_type == "list":
            self.serializer = ListSerializer(all_meta)
        elif serializer_type == "text":
            self.serializer = TextSerializer(all_meta)

        self.sub_metas = []
        for feature_idxes in self.feature_groups:
            meta = DatasetMeta()
            meta.name = self.all_meta.name
            meta.target = self.all_meta.target
            meta.desc = self.all_meta.desc
            meta.labal_meaning = self.all_meta.labal_meaning
            meta.features = [self.all_meta.features[i] for i in feature_idxes]
            meta.labels = self.all_meta.labels
            self.sub_metas.append(meta)

        # TODO: Other strategies
        self.sub_strategies: list[UnknownClassStrategy] = []
        for i in range(self.num_trees):
            if serializer_type == "tabular":
                serializer = TabularSerializer(self.sub_metas[i])
            elif serializer_type == "list":
                serializer = ListSerializer(self.sub_metas[i])
            elif serializer_type == "text":
                serializer = TextSerializer(self.sub_metas[i])

            self.sub_strategies.append(
                UnknownClassStrategy(
                    runner=runner,
                    template=template,
                    serializer=serializer,
                    loss_f=loss_f,
                    max_depth=max_depth,
                    train_batch=train_batch,
                    hist_nbins=hist_nbins,
                )
            )

    def set_train_data(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        self.train_x = train_x
        self.train_y = train_y

        feature_values = _get_feature_values(self.all_meta, train_x, self.hist_nbins)

        if len(train_x) > 1:
            for values in feature_values:
                values = values[1:]

        self.split_values = feature_values
        self.trees = []

        for i, feature_group in enumerate(self.feature_groups):
            self.sub_strategies[i].set_train_data(train_x[:, feature_group], train_y)
            self.trees.append(self.sub_strategies[i].get_tree())

        self.random_forest = RandomForest(
            self.trees, self.feature_groups, len(self.all_meta.labels)
        )

        self.trees_active = [True] * self.num_trees

    def step(self) -> tuple[bool, list[float]]:
        losses = []
        continue_step = False

        for i, sub_strategy in enumerate(tqdm(self.sub_strategies, desc="Trees")):
            if not self.trees_active[i]:
                continue
            logger.log(
                f"Training tree {i + 1}/{self.num_trees} (features: {self.feature_groups[i]})"
            )
            this_continue_step, loss = sub_strategy.step()
            self.trees_active[i] = this_continue_step
            continue_step = continue_step or this_continue_step
            losses.append(loss)

        return continue_step, losses

    def predict_tree(self, x: np.ndarray) -> list[int]:
        results = self.predict_tree_raw(x)
        for i, res in enumerate(results):
            if res < 0:
                idx = random.randint(0, self.all_meta.label_count() - 1)
                results[i] = self.all_meta.labels[idx].value
        return results

    def predict_tree_raw(self, x: np.ndarray) -> list[int]:
        ret = []
        for xx in x:
            idx = self.random_forest.predict_one(xx)
            y = self.all_meta.labels[idx].value if idx >= 0 else idx
            ret.append(y)
        return ret

    def predict_llm_with_tree(
        self, x: np.ndarray, with_examples: bool = False
    ) -> list[int]:
        prompt = self._gen_prompt(
            x,
            self.sub_strategies,
            (self.train_x, self.train_y) if with_examples else None,
        )
        if prompt is None:
            raise RuntimeError("Invalid tree rules!")

        for resp_candidates in self.runner.run([prompt]):
            for resp in resp_candidates:
                results = self.serializer.answer_decoder.decode(resp)
                results = [self.all_meta.get_label_value(r) for r in results]

                if None in results or len(results) != len(x):
                    continue

                return results
            else:
                logger.log(
                    "No valid response found, raw responses: {}".format(resp_candidates)
                )
                return None

    def predict_llm_with_all_subtrees(
        self, x: np.ndarray, with_examples: bool = False
    ) -> list[list[int]]:
        return [
            strategy.predict_llm_with_tree(x[:, feature_group], with_examples)
            for feature_group, strategy in zip(self.feature_groups, self.sub_strategies)
        ]

    def _get_tree_rules(self, sub_strategies: list[UnknownClassStrategy]) -> list[str]:
        rules = []
        for sub_strategy in sub_strategies:
            rules += sub_strategy._get_tree_rules()
        return rules

    def _gen_prompt(
        self,
        x: np.ndarray,
        sub_strategies: list[TrainStrategy],
        examples: tuple[np.ndarray, np.ndarray] = None,
    ) -> str:
        rules = self._get_tree_rules(sub_strategies)
        if rules is None:
            return None

        x_test_str = [self.serializer.serialize(xx, None) for xx in x]
        if examples is not None:
            examples_str = [self.serializer.serialize(x, y) for x, y in zip(*examples)]
        else:
            examples_str = []

        prompt = self.template.render(
            meta=self.all_meta,
            examples=examples_str,
            rules=rules,
            format_desc=self.serializer.format_desc(),
            prediction_intro=self.serializer.answer_requirement(len(x_test_str)),
            tests=x_test_str,
        )

        return prompt

    def get_tree(self) -> TreeBase:
        return self.random_forest

    def export(self) -> any:
        return {
            "type": "random_forest",
            "model": self.random_forest.export_dict(),
            "args": {"max_depth": self.max_depth, "num_trees": self.num_trees},  # TODO
            "prompt": self._gen_prompt(
                [], self.sub_strategies, (self.train_x, self.train_y)
            ),
        }
