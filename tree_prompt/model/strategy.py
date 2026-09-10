import copy
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import jinja2
import numpy as np

from ..dataset import DatasetMeta
from ..runner import Runner
from ..prompt import TabularSerializer, ListSerializer, TextSerializer
from .. import logger
from .tree import DecisionTree, RandomForest, TreeBase, RulePath
from .feature_selection import calculate_meta_rule_gini
from .meta_rule import MetaRule

TEMPLATE_DIRECTORY = Path(__file__).resolve().parents[2] / "template"


def load_template(name):
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_DIRECTORY),
        undefined=jinja2.StrictUndefined,
    ).get_template(name)


def _clone_tree_runner(runner: Runner | None) -> Runner | None:
    """Create an isolated runner for one forest tree.

    Tree construction sends at most one request per worker at a time.  Giving
    every cloned runner a capacity of one prevents N tree workers from each
    opening an additional N-request batch and therefore keeps total provider
    concurrency bounded by the forest scheduler.
    """
    if runner is None:
        return None

    try:
        cloned_runner = copy.deepcopy(runner)
    except Exception as exc:
        raise TypeError("LLMT Forest requires a copyable runner per tree") from exc

    if hasattr(cloned_runner, "parallel_batch_size"):
        cloned_runner.parallel_batch_size = 1
    if hasattr(cloned_runner, "num_parallel"):
        cloned_runner.num_parallel = 1
    return cloned_runner


class _SharedRequestScheduler:
    """Reserve globally spaced request starts for one LLMT Forest build."""

    def __init__(self, request_interval: float) -> None:
        self.request_interval = max(0.0, float(request_interval))
        self._lock = threading.Lock()
        self._next_request_at = 0.0

    def wait_turn(self) -> None:
        with self._lock:
            now = time.monotonic()
            scheduled_start = max(now, self._next_request_at)
            self._next_request_at = scheduled_start + self.request_interval
        delay = scheduled_start - now
        if delay > 0:
            time.sleep(delay)


class _RateLimitedTreeRunner(Runner):
    """Isolated runner instance that shares only the forest request schedule."""

    def __init__(self, runner: Runner, scheduler: _SharedRequestScheduler) -> None:
        self._runner = runner
        self._scheduler = scheduler

    def __getattr__(self, name):
        return getattr(self._runner, name)

    def run(self, messages: list[str]):
        for message in messages:
            self._scheduler.wait_turn()
            yield from self._runner.run([message])


class TrainStrategy:
    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.tree = None
        self.meta_rules = []
        self.tau = 0.70
        self.reset_token_stats()

    def set_train_data(self, train_x, train_y):
        self.reset_token_stats()
        self.train_x = np.asarray(train_x, dtype=object).copy()
        for i, feature in enumerate(self._meta.features):
            if not feature.is_categorical:
                self.train_x[:, i] = self.train_x[:, i].astype(float)
        self.train_y = np.asarray(train_y).astype(int)
        self._feature_shuffle_map = self._meta.feature_shuffle_map
        self.tree = DecisionTree(self.max_depth, self._meta.categories_map)
        self.tree.set_train_data(self.train_x)

    def step(self) -> tuple[bool, float]:
        next_split = self.tree.next_to_split()

        if next_split is None:
            logger.log("No splitable nodes, training finished")
            return False, None

        depth = next_split.depth
        samples = next_split.get_samples()

        if depth >= self.max_depth:
            logger.log(
                f"Node depth ({depth}) has reached the maximum depth ({self.max_depth}), automatically set to a leaf node"
            )

            if samples is not None and len(samples) > 0:
                node_y = self.train_y[samples]
                most_common_label = np.argmax(np.bincount(node_y))
                logger.log(
                    f"Based on sample label distribution, predict: {most_common_label}"
                )
            else:
                most_common_label = -1
                logger.log("Node has no samples, initial set to unknown class (-1)")

            verified_prediction = self._llm_verify_leaf_node(
                next_split, most_common_label
            )
            logger.log(f"LLM verification result: {verified_prediction}")

            next_split.is_leaf = True
            next_split.leaf_class = verified_prediction
            next_split.prediction = verified_prediction
            next_split.freeze()

            logger.log(f"Maximum depth leaf node set completed: {verified_prediction}")
            return True, None

        if samples is None or len(samples) == 0:
            logger.log("Node has no samples, initial set to unknown class (-1)")

            initial_prediction = -1
            verified_prediction = self._llm_verify_leaf_node(
                next_split, initial_prediction
            )

            logger.log(f"LLM verification result (empty node): {verified_prediction}")

            next_split.is_leaf = True
            next_split.leaf_class = verified_prediction
            next_split.prediction = verified_prediction
            next_split.freeze()

            logger.log(f"Empty node set completed, final class: {verified_prediction}")
            return True, None

        node_y = self.train_y[samples]

        if len(set(node_y)) == 1:
            original_prediction = node_y[0]
            logger.log(
                f"Node samples have the same label, directly set to a leaf node, label: {original_prediction}"
            )

            verified_prediction = self._llm_verify_leaf_node(
                next_split, original_prediction
            )

            logger.log(f"LLM verification result: {verified_prediction}")

            next_split.is_leaf = True
            next_split.leaf_class = verified_prediction
            next_split.prediction = verified_prediction

            next_split.freeze()

            logger.log(
                f"Leaf node set completed: leaf_class={next_split.leaf_class}, is_leaf={next_split.is_leaf}"
            )

            return True, None

        unique_labels = np.unique(node_y)
        if len(unique_labels) == 1:
            logger.log(
                f"Node samples have the same label, directly set to a leaf node, label: {unique_labels[0]}"
            )
            next_split.is_leaf = True
            next_split.prediction = unique_labels[0]
            next_split.freeze()
            return True, None

        best_meta_rule, best_gain = self._select_meta_rule(next_split, self.meta_rules)

        shallow_depth = depth <= 1
        is_small_sample = len(samples) <= 5

        if best_meta_rule is None:
            logger.log("No available meta rules, node frozen")

            most_common_label = np.argmax(np.bincount(node_y))
            logger.log(f"Node initial prediction: {most_common_label}")

            verified_prediction = self._llm_verify_leaf_node(
                next_split, most_common_label
            )
            logger.log(f"LLM verification result: {verified_prediction}")

            next_split.prediction = verified_prediction
            next_split.leaf_class = verified_prediction
            next_split.is_leaf = True
            next_split.freeze()

            logger.log(f"Node final set to: {verified_prediction}")
            return True, None
        elif best_gain <= 0:
            if shallow_depth and is_small_sample:
                logger.log(
                    f"Shallow node (depth={depth}), small sample case ({len(samples)} samples), even with a gain of 0, use rule: {best_meta_rule}"
                )
            else:
                logger.log(
                    f"Gain is 0 and non-shallow small sample case (depth={depth}, samples={len(samples)}), node frozen"
                )

                most_common_label = np.argmax(np.bincount(node_y))
                logger.log(f"Node initial prediction: {most_common_label}")

                verified_prediction = self._llm_verify_leaf_node(
                    next_split, most_common_label
                )
                logger.log(f"LLM verification result: {verified_prediction}")

                next_split.prediction = verified_prediction
                next_split.leaf_class = verified_prediction
                next_split.is_leaf = True
                next_split.freeze()

                logger.log(f"Node final set to: {verified_prediction}")
                return True, None

        logger.log(f"Using rule: {best_meta_rule}, Gini gain: {best_gain:.4f}")
        best_feature = best_meta_rule.feature_idx
        split_value = best_meta_rule.split_value
        is_categorical = best_meta_rule.is_categorical

        if hasattr(self, "_meta") and self._meta:
            feature_name = (
                self._meta.features[best_feature].name
                if best_feature < len(self._meta.features)
                else f"Unknown feature ({best_feature})"
            )
            logger.log(
                f"Selected feature: {best_feature} ({feature_name}), split value: {split_value}, is categorical: {is_categorical}"
            )

        node_X = self.train_x[samples]
        if is_categorical:
            left_mask = node_X[:, best_feature] == split_value
        else:
            left_mask = node_X[:, best_feature] < split_value
        right_mask = ~left_mask

        left_y = node_y[left_mask]
        right_y = node_y[right_mask]

        left_class = np.argmax(np.bincount(left_y)) if len(left_y) > 0 else -1
        right_class = np.argmax(np.bincount(right_y)) if len(right_y) > 0 else -1

        logger.log(
            f"Left child node label: {left_class}, right child node label: {right_class}"
        )

        next_split.split(
            best_feature, split_value, is_categorical, left_class, right_class
        )

        logger.log(
            f"Node has been split, feature: {best_feature}, split value: {split_value}"
        )

        return True, 0.0

    def predict_tree(self, x: np.ndarray) -> list[int]:
        if hasattr(self, "random_forest"):
            if self.random_forest is None:
                logger.log(
                    "Warning: Random forest not initialized, returning default prediction"
                )
                default_value = 0
                return [default_value] * len(x)
        elif hasattr(self, "tree"):
            if self.tree is None:
                logger.log(
                    "Warning: Decision tree not initialized, returning default prediction"
                )
                default_value = 0
                return [default_value] * len(x)

        results = self.predict_tree_raw(x)
        for i, res in enumerate(results):
            if res < 0:
                valid_labels = [label.value for label in self._meta.labels]
                selected_label = random.choice(valid_labels)
                logger.log(
                    f"Encountered a node with a prediction value of unknown(-1), randomly selected label: {selected_label}"
                )
                results[i] = selected_label
        return results

    def predict_tree_raw(self, x: np.ndarray) -> list[int]:
        if hasattr(self, "random_forest"):
            if self.random_forest is None:
                logger.log(
                    "Warning: Random forest not initialized, returning default prediction"
                )
                return [-1] * len(x)
        elif hasattr(self, "tree"):
            if self.tree is None:
                logger.log(
                    "Warning: Decision tree not initialized, returning default prediction"
                )
                return [-1] * len(x)

        ret = []
        if hasattr(self, "random_forest") and self.random_forest is not None:
            for xx in x:
                predicted_value = self.random_forest.predict_one(xx)
                ret.append(predicted_value)
        elif hasattr(self, "tree") and self.tree is not None:
            for xx in x:
                predicted_value = self.tree.predict_one(xx)
                ret.append(predicted_value)
        else:
            return [-1] * len(x)

        return ret

    def _serialize_rule(self, rule):
        if rule is None:
            return None

        conditions = []
        for feature_id, condition in rule.conditions.items():
            feature_name = self._meta.features[feature_id].name
            if condition.is_categorical:
                values = list(condition.categories)
                if len(values) == 0:
                    feature_values = []
                    if (
                        hasattr(self.tree, "categories_map")
                        and self.tree.categories_map
                        and feature_id in self.tree.categories_map
                    ):
                        feature_values = list(self.tree.categories_map[feature_id])
                    else:
                        for label in self._meta.features[feature_id].labels:
                            feature_values.append(label.name)

                    split_value = None

                    def find_split_value(node):
                        if node is None or node.is_leaf:
                            return None
                        if node.split_feature == feature_id:
                            return node.split_value
                        left_result = find_split_value(node.left_child)
                        if left_result is not None:
                            return left_result
                        return find_split_value(node.right_child)

                    split_value = find_split_value(self.tree.root_node)

                    if split_value is not None:
                        conditions.append(f"{feature_name} != {split_value}")
                    else:
                        conditions.append(
                            f"{feature_name} is not equal to any split value"
                        )
                        logger.log(
                            f"Warning: Unable to determine the split value for feature {feature_name}"
                        )
                elif len(values) == 1:
                    conditions.append(f"{feature_name} = {values[0]}")
                else:
                    values_str = ", ".join([str(v) for v in values])
                    conditions.append(f"{feature_name} in [{values_str}]")
            else:
                lower_bound = condition.lower
                upper_bound = condition.upper

                if lower_bound is not None and upper_bound is not None:
                    conditions.append(f"{lower_bound} ≤ {feature_name} < {upper_bound}")
                elif lower_bound is not None:
                    conditions.append(f"{feature_name} ≥ {lower_bound}")
                elif upper_bound is not None:
                    conditions.append(f"{feature_name} < {upper_bound}")

        label_name = None
        rule_value = rule.value

        if rule_value == -1:
            label_name = "unknown"
        else:
            for label in self._meta.labels:
                if label.value == rule_value:
                    label_name = label.name
                    break

        if label_name is None:
            label_name = f"unknown"
            logger.log(f"Warning: No label found for value {rule_value}")

        if conditions:
            return f"IF {' AND '.join(conditions)} THEN {label_name}"
        else:
            return f"{label_name}"

    def _get_path_to_node(self, node):
        path = []
        current = node

        while hasattr(current, "parent") and current.parent is not None:
            parent = current.parent
            if not hasattr(parent, "split_feature") or parent.split_feature is None:
                break

            feature_idx = parent.split_feature
            split_value = parent.split_value

            is_left = parent.left_child == current

            feature_name = f"Feature {feature_idx}"
            unit_info = ""

            if (
                hasattr(self, "_meta")
                and self._meta
                and feature_idx < len(self._meta.features)
            ):
                feature = self._meta.features[feature_idx]
                feature_name = feature.name

                if hasattr(feature, "desc") and feature.desc:
                    import re

                    unit_match = re.search(r"\((.*?)\)$", feature.desc.strip())
                    if unit_match:
                        unit = unit_match.group(1)
                        unit_info = f" ({unit})"
                    else:
                        unit_match = re.search(r"\((.*?)\)", feature.desc)
                        if unit_match:
                            unit = unit_match.group(1)
                            unit_info = f" ({unit})"
                        else:
                            unit_info = ""

            if hasattr(parent, "is_categorical") and parent.is_categorical:
                if is_left:
                    rule = f"{feature_name} = {split_value}"
                else:
                    rule = f"{feature_name} != {split_value}"
            else:
                if is_left:
                    rule = f"{feature_name} < {split_value}{unit_info}"
                else:
                    rule = f"{feature_name} >= {split_value}{unit_info}"

            path.append(rule)
            current = parent

        path.reverse()
        return path

    def _llm_verify_leaf_node(self, node_or_rules, prediction):
        path_rules = None
        if isinstance(node_or_rules, list):
            path_rules = node_or_rules
        else:
            path_rules = self._get_path_to_node(node_or_rules)

        if not path_rules:
            logger.log("Unable to get node path rules, set to unknown class (-1)")
            return -1

        feature_descriptions = []
        if hasattr(self, "_meta") and self._meta:
            for i, feature in enumerate(self._meta.features):
                feature_type = "Categorical" if feature.is_categorical else "Numerical"
                desc = feature.desc if hasattr(feature, "desc") and feature.desc else ""
                feature_desc = (
                    f"Feature {i}: {feature.name} (Type: {feature_type}) - {desc}"
                )

                if (
                    feature.is_categorical
                    and hasattr(feature, "categories")
                    and feature.categories
                ):
                    feature_desc += "\n    Possible values:"
                    for cat_value, cat_desc in feature.categories.items():
                        feature_desc += f"\n    - {cat_value}: {cat_desc}"

                feature_descriptions.append(feature_desc)

        label_descriptions = []
        if hasattr(self, "_meta") and self._meta:
            for label in self._meta.labels:
                description = ""
                if hasattr(label, "meaning") and label.meaning:
                    description = label.meaning
                elif hasattr(label, "desc") and label.desc:
                    description = label.desc

                label_descriptions.append(
                    f"Label {label.value}: {label.name} - {description}"
                )

        prompt = self.supervision_template.render(
            domain_expertise=self._get_domain_expertise(),
            label_meaning=self._meta.label_meaning
            if hasattr(self._meta, "label_meaning")
            else "Classification evaluation",
            feature_descriptions=feature_descriptions,
            label_descriptions=label_descriptions,
            path_rules=path_rules,
            labels=self._meta.labels,
        )

        logger.log(
            f"Sending LLM verification request, path rules: {' AND '.join(path_rules)}"
        )

        if not hasattr(self, "llm_runner") or self.llm_runner is None:
            from ..runner import Runner

            if hasattr(self, "_runner") and isinstance(self._runner, Runner):
                self.llm_runner = self._runner
            else:
                logger.log("LLM runner not set, skipping verification")
                return prediction

        try:
            response_gen = self.llm_runner.run([prompt])
            response_data, token_info = next(response_gen)
            response = response_data[0]

            logger.log(f"LLM response: {response}")

            confidence_scores = {}
            highest_confidence = 0
            best_label = prediction

            for line in response.split("\n"):
                if line.startswith("Label "):
                    parts = line.split(":")
                    if len(parts) >= 2:
                        label_str = parts[0].strip().replace("Label ", "")
                        try:
                            label = int(label_str)
                            confidence_str = parts[1].strip()
                            import re

                            confidence_match = re.search(
                                r"(\d+\.\d+|\d+)", confidence_str
                            )
                            if confidence_match:
                                confidence = float(confidence_match.group(1))
                                confidence_scores[label] = confidence

                                if confidence > highest_confidence:
                                    highest_confidence = confidence
                                    best_label = label
                        except ValueError:
                            continue
            self.supervision_tokens["prompt"] += token_info["prompt_tokens"]
            self.supervision_tokens["completion"] += token_info["completion_tokens"]
            self.supervision_tokens["total"] += token_info["total_tokens"]
            tree_index = getattr(self, "_tree_index", -1)
            tree_info = f"Tree {tree_index}: " if tree_index >= 0 else ""
            logger.log(
                f"{tree_info}Supervision Prompt Token usage: input={token_info['prompt_tokens']}, output={token_info['completion_tokens']}, total={token_info['total_tokens']}"
            )

            if highest_confidence >= self.tau and best_label != prediction:
                logger.log(
                    f"LLM suggests replacing label: {prediction} -> {best_label} (confidence: {highest_confidence})"
                )
                return best_label
            else:
                return prediction

        except Exception as e:
            logger.log(f"LLM verification process error: {e}")
            return prediction

    def _get_domain_expertise(self):
        if (
            hasattr(self, "_meta")
            and hasattr(self._meta, "target")
            and self._meta.target
        ):
            return self._meta.target

        if hasattr(self, "_meta") and hasattr(self._meta, "name") and self._meta.name:
            return f"{self._meta.name} classification"

        return "data analysis"

    def get_token_stats(self):
        build_tokens = {
            "prompt": self.supervision_tokens["prompt"]
            + self.meta_rule_tokens["prompt"],
            "completion": self.supervision_tokens["completion"]
            + self.meta_rule_tokens["completion"],
            "total": self.supervision_tokens["total"] + self.meta_rule_tokens["total"],
        }
        return {
            "build_tokens": build_tokens,
            "supervision_tokens": self.supervision_tokens,
            "meta_rule_tokens": self.meta_rule_tokens,
        }

    def reset_token_stats(self):
        self.supervision_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self.meta_rule_tokens = {"prompt": 0, "completion": 0, "total": 0}
        logger.log("Token statistics reset")


class SingleStrategy(TrainStrategy):
    def __init__(self, runner, serializer, max_depth, train_batch=8, tau=0.70):
        super().__init__()
        if not 0 <= tau <= 1:
            raise ValueError("tau must be in [0, 1]")
        self.runner = runner
        self.llm_runner = runner
        self.serializer = serializer
        self.max_depth = max_depth
        self.train_batch = train_batch
        self.tau = tau
        self.rule_template = load_template("meta_rule.jinja")
        self.supervision_template = load_template("supervision.jinja")

    @property
    def _meta(self):
        return self.serializer.meta

    def _create_meta_rules_prompt(self, max_depth):
        return self.rule_template.render(
            meta=self._meta, num_rules_required=max(10, 2**max_depth - 1)
        )

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

    def export(self) -> any:
        return {
            "type": "single",
            "model": self.tree.export_nodes_dict(),
            "args": {"max_depth": self.max_depth, "categories": None},  # TODO
        }

    def get_tree(self) -> DecisionTree:
        return self.tree

    def _get_meta_rules(self, max_depth: int, runner: Runner = None) -> list[MetaRule]:
        tree_index = -1
        if hasattr(self, "_tree_index"):
            tree_index = self._tree_index

        if runner is None:
            runner = self.runner

        prompt = self._create_meta_rules_prompt(max_depth)

        meta_rules = []
        for responses, token_info in runner.run([prompt]):
            for response in responses:
                logger.log(f"Received LLM response:\n{response}")

                for line in response.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue

                    meta_rule = MetaRule.parse_rule(line, self._meta)
                    if meta_rule:
                        meta_rules.append(meta_rule)

        logger.log(f"Successfully parsed {len(meta_rules)} meta rules")

        meta_rules.sort(key=lambda x: x.confidence, reverse=True)

        self.meta_rule_tokens["prompt"] += token_info["prompt_tokens"]
        self.meta_rule_tokens["completion"] += token_info["completion_tokens"]
        self.meta_rule_tokens["total"] += token_info["total_tokens"]

        tree_info = f"Tree {tree_index}: " if tree_index >= 0 else ""
        logger.log(
            f"{tree_info}Meta Rule Prompt Token usage: input={token_info['prompt_tokens']}, output={token_info['completion_tokens']}, total={token_info['total_tokens']}"
        )

        return meta_rules

    def _select_meta_rule(
        self, node, meta_rules: list[MetaRule]
    ) -> tuple[MetaRule, float]:
        samples = node.get_samples()
        is_small_sample = len(samples) <= 5

        used_features = set()
        current = node
        while hasattr(current, "parent") and current.parent is not None:
            parent = current.parent
            if hasattr(parent, "split_feature") and parent.split_feature is not None:
                used_features.add(parent.split_feature)
            current = parent

        if len(meta_rules) == 0:
            return None, 0.0

        best_gain = -1
        best_rule = None
        best_confidence = -1

        best_small_sample_rule = None
        best_small_sample_confidence = -1

        confidence_levels = sorted(
            {rule.confidence for rule in meta_rules if rule.confidence >= 0},
            reverse=True,
        )
        for confidence in confidence_levels:
            logger.log(f"Searching rules with confidence {confidence}")

            usable_rules = []
            for rule in meta_rules:
                if rule.feature_idx in used_features:
                    continue

                if rule.confidence == confidence:
                    usable_rules.append(rule)

            if not usable_rules:
                continue

            logger.log(
                f"Found {len(usable_rules)} usable rules (confidence: {confidence})"
            )

            has_non_zero_gain = False

            for rule in usable_rules:
                gain, left_gini, right_gini, left_mask, right_mask = (
                    calculate_meta_rule_gini(rule, self.train_x, self.train_y, samples)
                )

                logger.log(f"Rule '{rule}' Gini gain: {gain:.4f}")
                logger.log(
                    f"   Left child: sample count={np.sum(left_mask)}, Gini={left_gini:.4f}"
                )
                logger.log(
                    f"   Right child: sample count={np.sum(right_mask)}, Gini={right_gini:.4f}"
                )

                if is_small_sample and (
                    np.sum(left_mask) == 0 or np.sum(right_mask) == 0
                ):
                    if (
                        best_small_sample_rule is None
                        or rule.confidence > best_small_sample_confidence
                    ):
                        best_small_sample_rule = rule
                        best_small_sample_confidence = rule.confidence

                if not is_small_sample and (
                    np.sum(left_mask) == 0 or np.sum(right_mask) == 0
                ):
                    continue

                if gain > 0:
                    has_non_zero_gain = True

                if gain > best_gain or (
                    gain == best_gain and rule.confidence > best_confidence
                ):
                    best_gain = gain
                    best_rule = rule
                    best_confidence = rule.confidence

            if has_non_zero_gain:
                logger.log(
                    f"Found a rule with non-zero gain at confidence {confidence}"
                )
                break

        if is_small_sample and best_rule is None and best_small_sample_rule is not None:
            logger.log(
                f"Small sample case ({len(samples)}≤5), using the rule with the highest confidence: {best_small_sample_rule}, even if one side is empty"
            )
            return best_small_sample_rule, 0.001

        if best_rule:
            logger.log(f"Best meta rule: {best_rule}, Gini gain: {best_gain:.4f}")
        else:
            logger.log("No valid meta rules found")

        return best_rule, best_gain

    def set_train_data(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        super().set_train_data(train_x, train_y)

        self.meta_rules = self._get_meta_rules(self.max_depth, self.runner)
        logger.log(f"Number of meta rules obtained: {len(self.meta_rules)}")
        for rule in self.meta_rules[:10]:
            logger.log(f"Rule: {rule}")


class LLMTForestStrategy(TrainStrategy):
    """Feature-bagged LLMT ensemble with isolated, concurrent tree builders."""

    def __init__(
        self,
        all_meta: DatasetMeta,
        runner: Runner,
        serializer_type: str,
        num_trees: int,
        max_depth: int,
        train_batch: int,
        tree_parallelism: int | None = None,
        random_seed: int | None = 0,
        tau: float = 0.70,
    ) -> None:
        super().__init__()
        self.runner = runner
        self.all_meta = all_meta
        self.max_depth = max_depth
        self.train_batch = train_batch

        self.num_trees = int(num_trees)
        self.random_seed = random_seed
        if self.all_meta.feature_count() < 1 or self.num_trees < 1:
            raise ValueError("LLMT Forest requires at least one feature and one tree")

        runner_capacity = getattr(runner, "parallel_batch_size", None)
        if runner_capacity is None:
            runner_capacity = getattr(runner, "num_parallel", 1)
        try:
            runner_capacity = max(1, int(runner_capacity))
        except (TypeError, ValueError):
            runner_capacity = 1
        if tree_parallelism is not None and tree_parallelism < 1:
            raise ValueError("tree_parallelism must be at least one")
        requested_parallelism = tree_parallelism or runner_capacity
        self.tree_parallelism = min(
            self.num_trees, int(requested_parallelism), runner_capacity
        )
        request_interval = getattr(runner, "request_interval", None)
        if request_interval is None:
            request_interval = getattr(runner, "interval", 0.0)
        self._request_scheduler = _SharedRequestScheduler(request_interval)

        # Match the server's random-feature-bagging protocol: every tree gets
        # an independently sampled sqrt-sized feature subspace before it asks
        # the LLM for rules.  Bags may overlap across trees.
        feature_count = self.all_meta.feature_count()
        group_size = max(1, int(np.ceil(np.sqrt(feature_count))))
        rng = np.random.default_rng(self.random_seed)
        self.feature_groups = [
            rng.choice(feature_count, size=group_size, replace=False).tolist()
            for _ in range(self.num_trees)
        ]

        if serializer_type == "tabular":
            self.serializer = TabularSerializer(all_meta)
        elif serializer_type == "list":
            self.serializer = ListSerializer(all_meta)
        elif serializer_type == "text":
            self.serializer = TextSerializer(all_meta)

        self.sub_metas: list[DatasetMeta] = []
        for feature_idxes in self.feature_groups:
            meta = copy.deepcopy(self.all_meta)
            meta.features = [
                copy.deepcopy(self.all_meta.features[i]) for i in feature_idxes
            ]
            original_feature_map = getattr(self.all_meta, "feature_shuffle_map", {})
            meta.feature_shuffle_map = {
                local_index: original_feature_map.get(global_index, global_index)
                for local_index, global_index in enumerate(feature_idxes)
            }
            self.sub_metas.append(meta)

        self.sub_strategies: list[SingleStrategy] = []
        for i in range(self.num_trees):
            if serializer_type == "tabular":
                serializer = TabularSerializer(self.sub_metas[i])
            elif serializer_type == "list":
                serializer = ListSerializer(self.sub_metas[i])
            elif serializer_type == "text":
                serializer = TextSerializer(self.sub_metas[i])

            tree_runner = _clone_tree_runner(runner)
            if tree_runner is not None:
                tree_runner = _RateLimitedTreeRunner(
                    tree_runner, self._request_scheduler
                )
            self.sub_strategies.append(
                SingleStrategy(
                    runner=tree_runner,
                    serializer=serializer,
                    max_depth=max_depth,
                    train_batch=train_batch,
                    tau=tau,
                )
            )

    def set_train_data(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        self.train_x = train_x
        self.train_y = train_y
        if len(train_x) < 1:
            raise ValueError("LLMT Forest requires at least one training row")
        rng = np.random.default_rng(self.random_seed)
        self.bootstrap_indices = [
            rng.integers(0, len(train_x), size=len(train_x))
            for _ in range(self.num_trees)
        ]

        self.trees = [None] * self.num_trees

        def initialise_tree(index: int) -> tuple[int, DecisionTree]:
            feature_group = self.feature_groups[index]
            bootstrap_indices = self.bootstrap_indices[index]
            self.sub_strategies[index].set_train_data(
                train_x[bootstrap_indices][:, feature_group], train_y[bootstrap_indices]
            )
            return index, self.sub_strategies[index].get_tree()

        with ThreadPoolExecutor(max_workers=self.tree_parallelism) as executor:
            futures = {
                executor.submit(initialise_tree, index): index
                for index in range(self.num_trees)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    tree_index, tree = future.result()
                except BaseException as exc:
                    raise RuntimeError(
                        f"LLMT Forest tree {index + 1} failed during initialisation"
                    ) from exc
                self.trees[tree_index] = tree

        self.random_forest = RandomForest(
            self.trees, self.feature_groups, len(self.all_meta.labels)
        )

        self.trees_active = [True] * self.num_trees

    def step(self) -> tuple[bool, list[float]]:
        losses = []
        continue_step = False

        active_indices = [
            index for index, is_active in enumerate(self.trees_active) if is_active
        ]
        if not active_indices:
            return False, losses

        def step_tree(index: int) -> tuple[int, bool, float]:
            logger.log(
                f"Training LLMT Forest tree {index + 1}/{self.num_trees} "
                f"(features: {self.feature_groups[index]})"
            )
            this_continue_step, loss = self.sub_strategies[index].step()
            return index, this_continue_step, loss

        with ThreadPoolExecutor(max_workers=self.tree_parallelism) as executor:
            futures = {
                executor.submit(step_tree, index): index for index in active_indices
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    tree_index, this_continue_step, loss = future.result()
                except BaseException as exc:
                    raise RuntimeError(
                        f"LLMT Forest tree {index + 1} failed during construction"
                    ) from exc
                self.trees_active[tree_index] = this_continue_step
                continue_step = continue_step or this_continue_step
                losses.append(loss)

        return continue_step, losses

    def get_tree(self) -> TreeBase:
        return self.random_forest

    @property
    def _meta(self):
        return self.all_meta

    def predict_tree_raw(self, x):
        # Leaf values are dataset labels, which need not start at zero.
        predictions = []
        labels = [label.value for label in self.all_meta.labels]
        for row in x:
            votes = [
                tree.predict_one(row[group])
                for tree, group in zip(self.trees, self.feature_groups)
            ]
            counts = [votes.count(label) for label in labels]
            predictions.append(labels[int(np.argmax(counts))] if max(counts) else -1)
        return predictions

    def get_token_stats(self):
        stats = [s.get_token_stats() for s in self.sub_strategies]
        return {
            stage: {
                key: sum(s[stage][key] for s in stats)
                for key in ("prompt", "completion", "total")
            }
            for stage in ("build_tokens", "supervision_tokens", "meta_rule_tokens")
        }

    def export(self):
        return {
            "type": "random_forest",
            "model": self.random_forest.export_dict(),
            "args": {"max_depth": self.max_depth, "num_trees": self.num_trees},
        }


def clean_llm_response(response):
    if response is None:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    cleaned = cleaned.lstrip("\n")
    return cleaned
