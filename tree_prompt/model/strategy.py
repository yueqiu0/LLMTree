import numpy as np
import jinja2
from itertools import product
from functools import reduce
import random
from tqdm import tqdm

from .tree import DecisionTree
from ..runner import Runner
from ..prompt import Serializer, TabularSerializer, ListSerializer, TextSerializer
from .tree import DecisionTree, RandomForest, TreeBase, RulePath, Node
from ..dataset import DatasetMeta
from .. import logger
from .loss import LossFunction


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

    def step(self) -> tuple[bool, float | list[float]]:
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

    def set_train_data(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        self.train_x = train_x
        self.train_y = train_y

        self.split_values = _get_feature_values(self._meta, train_x, self.hist_nbins)

        self.tree = DecisionTree(self.max_depth, self._meta.categories_map)
        self.last_loss = None
        self.lass_loss_components = None

    def _get_available_predictions(self) -> list[int]:
        return [-1, *range(self._meta.label_count())]

    def step(self) -> tuple[bool, float]:
        next_split = self.tree.next_to_split()
        if next_split is None:
            return False, None

        avail_predictions = self._get_available_predictions()
        best_loss, best_loss_components = (
            self.last_loss,
            self.lass_loss_components,
        )

        best_split: tuple[int, float, bool, int, int] = None

        # split train data into batches (self.train_batch)
        x_train_splits = []
        x_train_split_lens = []

        for split_start in range(0, len(self.train_x), self.train_batch):
            end = min(split_start + self.train_batch, len(self.train_x))
            x_train_splits.append(self.train_x[split_start:end])
            x_train_split_lens.append(len(x_train_splits[-1]))

        all_train_data = []
        for feat_idx, left_class, right_class in product(
            range(self._meta.feature_count()), avail_predictions, avail_predictions
        ):
            if left_class == right_class:
                continue
            for split_value in self.split_values[feat_idx]:
                next_split.split(
                    feat_idx,
                    split_value,
                    self._meta.features[feat_idx].is_categorical,
                    left_class,
                    right_class,
                )

                prompts = []
                for split_x in x_train_splits:
                    prompt = self._gen_prompt(split_x)
                    if prompt is None:
                        break
                    prompts.append(prompt)

                else:
                    tree_results_raw = self.predict_tree_raw(self.train_x)
                    all_train_data.append(
                        (
                            prompts,
                            feat_idx,
                            self._meta.features[feat_idx].is_categorical,
                            split_value,
                            left_class,
                            right_class,
                            tree_results_raw,
                        )
                    )

        for train_data in tqdm(all_train_data, desc="Split points"):
            (
                prompts,
                feat_idx,
                is_categorical,
                split_value,
                left_class,
                right_class,
                tree_results_raw,
            ) = train_data

            results = self._predict_llm_with_tree_batched(prompts, x_train_split_lens)

            if results is None:
                logger.log(
                    "Invalid response for split (feature={}, split={}, left={}, right={}), skip...".format(
                        feat_idx, split_value, left_class, right_class
                    )
                )
                continue

            results = reduce(lambda x, y: x + y, results, [])

            loss, loss_components = self.loss_f(
                self.train_y, np.array(results), np.array(tree_results_raw)
            )

            if best_loss is None or loss < best_loss:
                best_split = (
                    feat_idx,
                    split_value,
                    is_categorical,
                    left_class,
                    right_class,
                )
                logger.log(
                    (
                        "Found better split (feature={}, split={}, left={}, right={}), "
                        + "loss: {} -> {:3f} (loss components: [{}])"
                    ).format(
                        feat_idx,
                        split_value,
                        left_class,
                        right_class,
                        f"{'' if best_loss is None else f'{best_loss:.3f}'}",
                        loss,
                        ", ".join(
                            [
                                f"{'' if x1 is None else f'{x1:.3f}'} -> {x2:.3f}"
                                for x1, x2 in zip(
                                    best_loss_components
                                    if best_loss_components is not None
                                    else [None] * len(loss_components),
                                    loss_components,
                                )
                            ]
                        ),
                    )
                )
                best_loss, best_loss_components = loss, loss_components

                if best_loss < self.LOSS_THRESHOLD:
                    logger.log("Loss is small enough, stop training")
                    break
            else:
                logger.log(
                    "Split (feature={}, split={}, left={}, right={}) loss: {:.3f} (loss components: [{}])".format(
                        feat_idx,
                        split_value,
                        left_class,
                        right_class,
                        loss,
                        ", ".join([f"{x:.3f}" for x in loss_components]),
                    )
                )

        if best_split is None:
            next_split.unsplit()
            next_split.freeze()
            logger.log("No better split found, freeze node")
            return True, best_loss

        next_split.split(*best_split)
        self.prune_tree()
        self.last_loss = best_loss
        self.lass_loss_components = best_loss_components
        logger.log(
            "New split: feature={}, split={} (categorical={}), left={}, right={}".format(
                *best_split
            )
        )

        return (best_loss >= self.LOSS_THRESHOLD), best_loss

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


class KnownClassStrategy(UnknownClassStrategy):
    def _get_available_predictions(self) -> list[int]:
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
