from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
import sklearn.tree
import sklearn.ensemble
import numpy as np
import jinja2
from pathlib import Path
from .. import dataset
from ..dataset import DatasetMeta
from jinja2 import Environment, FileSystemLoader
from sklearn.utils import check_array
import warnings
from ..logger import Logger, add_logger  
from ..dataset import generate_ToT_tree_prompt
import random

def _encode_one_hot(x_all: np.ndarray, meta: DatasetMeta) -> tuple[np.ndarray, list]:
    feat_stack = []
    all_categories = []
    for i in range(x_all.shape[1]):
        if meta.features[i].is_categorical:
            encoder = OneHotEncoder()
            feat_stack.append(
                encoder.fit_transform(x_all[:, i].reshape(-1, 1)).toarray()
            )
            all_categories.append(encoder.categories_[0])
        else:
            feat_stack.append(x_all[:, i].reshape(-1, 1))
            all_categories.append(None)

    new_x = np.hstack(feat_stack)

    new_meta = DatasetMeta()
    new_meta.labels = meta.labels
    new_meta.name = meta.name
    new_meta.target = meta.target
    new_meta.desc = meta.desc
    new_meta.labal_meaning = meta.labal_meaning

    for feat_idx, ori_feat in enumerate(meta.features):
        if ori_feat.is_categorical:
            for cat in all_categories[feat_idx]:
                if cat not in x_all[:, feat_idx]:
                    continue
                cat_desc = ori_feat.categories[cat]
                new_feat = DatasetMeta.Feature()
                new_feat.name = ori_feat.name + " == " + cat_desc
                new_feat.desc = ori_feat.desc
                new_feat.type = "int"
                new_meta.features.append(new_feat)
        else:
            new_meta.features.append(ori_feat)

    return new_x, new_meta


class DecisionTree:
    def __init__(self, meta: dataset.DatasetMeta) -> None:
        self.meta = meta

    def predict(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        export_rules: bool,
    ) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError()


class ToTDecisionTree(DecisionTree):
    def __init__(self, meta, max_depth, runner, log_file, tree_num=5, best_num=1):
        super().__init__(meta)  # Initialize the parent class
        self.max_depth = max_depth  # Max tree depth
        self.runner = runner      # LLM runner
        self.log_file = log_file  # Log file path
        self.rules = []           # List to store parsed rules
        self.feature_names = [f.name for f in meta.features]  # Get feature names from metadata features
        self.num_rule_candidates = tree_num  # Keep internal name but accept external name
        self.select_top_k = best_num  # Keep internal name but accept external name
        # Open log file in append mode with line buffering for real-time writing
        log_file_obj = open(self.log_file, 'a', encoding='utf-8', buffering=1)  # 1 means line buffering
        self.logger = Logger(log_file_obj)  # Initialize logger with file object
        # Ensure first log message marks the start
        self.logger.log(f"=== ToTDecision initialized at {datetime.now().isoformat()} ===")

    def _build_rule_prompt(self, x_train, y_train, rules, depth, used_features=None):
        """Build prompt for generating rules in DOT format"""
        from ..dataset import generate_ToT_tree_prompt
        meta = self.meta
        num_examples = min(5, x_train.shape[0])
        prompt_info = generate_ToT_tree_prompt(meta, x_train, y_train, max_depth=self.max_depth, num_examples=num_examples)
        prompt_head = prompt_info["prompt"]
        
        # Generate current tree in DOT format
        tree_str = "empty"
        if rules:
            tree_lines = ["node0 [label=\"ROOT\"]"]
            node_id = 1
            parent_map = {}
            
            for i, rule in enumerate(rules, 1):
                condition = rule['condition']
                label = rule['label']
                path = rule.get('path', [])
                
                # Build tree structure
                current_parent = "node0"
                for cond in path[:-1]:
                    key = f"{current_parent}-{cond}"
                    if key not in parent_map:
                        parent_map[key] = f"node{node_id}"
                        tree_lines.append(f"{parent_map[key]} [label=\"{cond}\"]")
                        tree_lines.append(f"{current_parent} -> {parent_map[key]}")
                        node_id += 1
                    current_parent = parent_map[key]
                
                # Add leaf node
                leaf_node = f"node{node_id}"
                tree_lines.append(f"{leaf_node} [label=\"{path[-1] if path else condition}\\n[{label}]\"]")
                tree_lines.append(f"{current_parent} -> {leaf_node}")
                node_id += 1
            
            tree_str = "\n".join(tree_lines)
        
        # Forbidden features note
        forbid_str = ""
        if used_features:
            forbid_str = f"\n# Note: The following features cannot be used in this layer: {', '.join(used_features)}"
        
        # Prompt template in English
        prompt_tail = f"""
# Current decision :
{tree_str}
{forbid_str}

Please propose the optimal classification rule for layer {depth+1}.
"""
        return prompt_head + "\n" + prompt_tail

    def _parse_llm_rule_response(self, response):
        """解析LLM生成的单条规则"""
        return self.parse_dot_tree(response)[0] if response else None

    @staticmethod
    def parse_dot_tree(dot_str):
        """Parse DOT format tree into decision rules
        
        Example input:
        node0 [label="ROOT"]
        node1 [label="price = vhigh"]
        node0 -> node1
        node2 [label="safety = high\n[good]"]
        node1 -> node2
        node3 [label="safety != high\n[unacceptable]"]
        node1 -> node3
        
        Returns list of rules in format:
        [{
            'condition': 'price = vhigh AND safety = high',
            'label': 'good',
            'path': ['price = vhigh', 'safety = high']
        }]
        """
        import re
        from collections import defaultdict
        
        # Parse nodes
        nodes = {}
        node_pattern = re.compile(r'node(\d+)\s*\[label="([^"]+)"\]')
        for node_id, label in node_pattern.findall(dot_str):
            # Extract condition and label (if any)
            parts = label.split('\n')
            condition = parts[0].strip()
            label = parts[1][1:-1] if len(parts) > 1 else None  # Remove brackets
            nodes[node_id] = {
                'condition': condition,
                'label': label,
                'children': []
            }
        
        # Parse edges and build tree
        edge_pattern = re.compile(r'node(\d+)\s*->\s*node(\d+)')
        for parent_id, child_id in edge_pattern.findall(dot_str):
            nodes[parent_id]['children'].append(child_id)
        
        # Build rules by traversing from root
        rules = []
        root_id = '0'  # Assuming root is always node0

    def _build_select_prompt(self, rule_candidates, rules, depth):
        """构造规则筛选prompt，让LLM在候选规则中选最优k个，要求输出标准格式"""
        prompt = f"""
你正在构建一棵决策树，当前已生成的规则如下：\n{rules}\n以下是本层候选规则：\n"
"""
        for idx, rule in enumerate(rule_candidates):
            prompt += f"规则{idx+1}: {rule}\n"
        prompt += f"\n请你从中选出最优的{self.select_top_k}个规则，并只输出被选中的规则，保持如下格式：\nRule N: IF ... THEN ..."
        return prompt

    def _parse_llm_select_response(self, response, rule_candidates, k):
        """解析LLM筛选返回的最优k个规则"""
        # 直接用parse_dot_tree解析，返回前k条
        rules = self.parse_dot_tree(response)
        return rules[:k] if rules else []

    def generate_candidate_rules(self, x_train, y_train, current_rules, depth, used_features=None):
        """让LLM生成候选分类规则"""
        prompt_info = generate_ToT_tree_prompt(self.meta, x_train, y_train, 
                                             max_depth=self.max_depth,
                                             current_rules=current_rules)
        prompt = prompt_info["prompt"]
        
        # 记录发送给LLM的prompt
        self.logger.log(f"\n=== Sending to LLM at depth {depth} ===")
        self.logger.log(f"Prompt:\n{prompt}")
        
        # 处理生成器响应
        response_generator = self.runner.run(prompt)
        try:
            response = next(response_generator)
            # 记录LLM原始响应
            self.logger.log(f"\nLLM Raw Response:")
            if isinstance(response, list):
                self.logger.log("\n".join(response))
                response = '\n'.join(response)
            else:
                self.logger.log(str(response))
                
            parsed_rules = self.parse_dot_tree(response)
            # 记录解析后的规则
            self.logger.log(f"\nParsed Rules:")
            for i, rule in enumerate(parsed_rules):
                self.logger.log(f"Rule {i+1}: {rule}")
                
            return parsed_rules
        except StopIteration:
            self.logger.log(f"No response from LLM at depth {depth}")
            return []

    def evaluate_rules(self, rule_candidates, current_rules, depth):
        """让LLM评估并选择最优规则"""
        prompt = self._build_select_prompt(rule_candidates, current_rules, depth)
        
        # 记录发送给LLM的prompt
        self.logger.log(f"\n=== Evaluating Rules at depth {depth} ===")
        self.logger.log(f"Prompt:\n{prompt}")
        
        response_generator = self.runner.run(prompt)
        try:
            response = next(response_generator)
            # 记录LLM原始响应
            self.logger.log(f"\nLLM Raw Response:")
            if isinstance(response, list):
                self.logger.log("\n".join(response))
                response = '\n'.join(response)
            else:
                self.logger.log(str(response))
                
            selected_rules = self._parse_llm_select_response(response, rule_candidates, self.select_top_k)
            # 记录选择的规则
            self.logger.log(f"\nSelected Rules:")
            for i, rule in enumerate(selected_rules):
                self.logger.log(f"Rule {i+1}: {rule}")
                
            return selected_rules
        except StopIteration:
            self.logger.log(f"No response from LLM during evaluation at depth {depth}")
            return []
        """让LLM评估并选择最优规则"""
        prompt = self._build_select_prompt(rule_candidates, current_rules, depth)
        response = self.runner.run(prompt)
        return self._parse_llm_select_response(response, rule_candidates, self.select_top_k)

    def convert_to_if_rules(self, dot_rules):
        """将DOT格式规则转换为IF-THEN规则列表"""
        if_rules = []
        for rule in dot_rules:
            conditions = []
            for cond in rule['conditions']:
                if cond['is_categorical']:
                    conditions.append(f"{self.feature_names[cond['feature']]} == {cond['value']}")
                else:
                    op = ">=" if cond['is_left'] else "<"
                    conditions.append(f"{self.feature_names[cond['feature']]} {op} {cond['value']}")
            if_condition = " AND ".join(conditions)
            if_rules.append(f"IF {if_condition} THEN {rule['label']}")
        return if_rules

    def parse_rules_to_functions(self, dot_rules):
        """将规则解析为可执行Python函数"""
        funcs = []
        for i, rule in enumerate(dot_rules):
            conditions = []
            for cond in rule['conditions']:
                if cond['is_categorical']:
                    conditions.append(f"x[{cond['feature']}] == {cond['value']}")
                else:
                    op = ">=" if cond['is_left'] else "<"
                    conditions.append(f"x[{cond['feature']}] {op} {cond['value']}")
            func_code = f"def rule_{i}(x):\n    return {' and '.join(conditions)}"
            funcs.append(func_code)
        return funcs

    def render_jinja_template(self, rules, with_llm=True):
        """渲染监控模板"""
        env = Environment(loader=FileSystemLoader('template'))
        template = env.get_template('basic.jinja')
        
        if with_llm:
            # For LLM parsing
            return template.render(
                meta=self.meta,
                rules=rules,
                prediction_intro="Please predict the following cases:"
            )
        else:
            # For function parsing
            return template.render(
                meta=self.meta,
                rules=rules,
                prediction_intro=""
            )

    def build_decision_tree(self, x_train, y_train, max_depth=None):
        """递归构建决策树"""
        if max_depth is None:
            max_depth = self.max_depth
            
        if max_depth <= 0 or len(np.unique(y_train)) == 1:
            # Base case: create leaf node
            leaf_class = np.argmax(np.bincount(y_train))
            return {'is_leaf': True, 'class': leaf_class}
            
        # Generate candidate rules
        current_rules = self.rules.copy()
        rule_candidates = self.generate_candidate_rules(
            x_train, y_train, 
            current_rules, 
            self.max_depth - max_depth
        )
        
        if not rule_candidates:
            # No valid rules found
            leaf_class = np.argmax(np.bincount(y_train))
            return {'is_leaf': True, 'class': leaf_class}
            
        # Evaluate and select best rule
        best_rules = self.evaluate_rules(
            rule_candidates, 
            current_rules,
            self.max_depth - max_depth
        )
        
        if not best_rules:
            # No rules selected
            leaf_class = np.argmax(np.bincount(y_train))
            return {'is_leaf': True, 'class': leaf_class}
            
        best_rule = best_rules[0]
        self.rules.append(best_rule)
        
        # Split data based on rule
        left_mask = self._apply_rule(x_train, best_rule)
        right_mask = ~left_mask
        
        # Recursively build subtrees
        left_subtree = self.build_decision_tree(
            x_train[left_mask], 
            y_train[left_mask],
            max_depth - 1
        )
        right_subtree = self.build_decision_tree(
            x_train[right_mask],
            y_train[right_mask],
            max_depth - 1
        )
        
        return {
            'is_leaf': False,
            'rule': best_rule,
            'left': left_subtree,
            'right': right_subtree
        }

    def _apply_rule(self, x, rule):
        """应用规则生成掩码"""
        mask = np.ones(len(x), dtype=bool)
        for cond in rule['conditions']:
            if cond['is_categorical']:
                mask &= (x[:, cond['feature']] == cond['value'])
            else:
                if cond['is_left']:
                    mask &= (x[:, cond['feature']] >= cond['value'])
                else:
                    mask &= (x[:, cond['feature']] < cond['value'])
        return mask

    def get_rules(self):
        """获取当前决策树的所有规则"""
        return self.rules

    def fit(self, x_train, y_train, with_llm=True):
        """Complete tree building pipeline including:
        1. Rule generation and evaluation
        2. Dataset splitting
        3. Termination condition checking
        4. Format conversion
        5. Output selection based on with_llm flag
        
        Args:
            x_train: Training data features
            y_train: Training data labels
            with_llm: Whether to use LLM for output rendering
            
        Returns:
            self: Returns the trained model instance
        """
        # Build the complete decision tree
        tree_structure = self.build_decision_tree(x_train, y_train)
        
        # Convert tree to appropriate format
        if with_llm:
            # Render monitoring template for LLM
            rules = self.get_rules()
            self.render_jinja_template(rules, with_llm=True)
        else:
            # Generate executable decision functions
            rules = self.get_rules()
            self.parse_rules_to_functions(rules)
            
        return self


class SimpleDecisionTree(DecisionTree):
    def __init__(self, meta: dataset.DatasetMeta, max_depth: int) -> None:
        super().__init__(meta)
        self.max_depth = max_depth
        self.clf = None

    def predict(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        export_rules: bool = True,
    ):
        from sklearn.tree import DecisionTreeClassifier

        if len(np.unique(y_train)) == 1:
            return np.full(x_test.shape[0], y_train[0]), []

        old_meta = self.meta
        has_categorical = any(f.type == "categorical" for f in self.meta.features)

        # encode one-hot
        if has_categorical:
            new_x, new_meta = _encode_one_hot(
                np.concatenate([x_train, x_test]), self.meta
            )

            self.meta = new_meta
            train_size = len(x_train)
            x_train = new_x[:train_size]
            x_test = new_x[train_size:]

        self.clf = DecisionTreeClassifier(max_depth=self.max_depth)
        self.clf.fit(x_train, y_train)

        if not export_rules:
            self.meta = old_meta
            return self.clf.predict(x_test), []

        x_names = [f.name for f in self.meta.features]
        desc = sklearn.tree.export_text(self.clf, feature_names=x_names)
        rules = self._build_rules(desc)
        self.meta = old_meta
        return self.clf.predict(x_test), rules

    def _build_rules(self, desc: str) -> list[str]:
        lines = desc.split("\n")
        conditions = []
        current_level = 0
        outputs = []

        for line in lines:
            level = line.count("|")
            cond = line[4 * level :].strip()

            # for categorical features (e.g. price == high)
            if "==" in cond:
                is_gt = ">" in cond
                cond = cond.replace("<=", ">")
                gt_idx = cond.find(">")
                cond = cond[: gt_idx - 1].replace("==", "is" if is_gt else "is not")

            if level > current_level:
                conditions.append(cond)
            elif level < current_level:
                # previously leaf
                rule_desc = self._build_one_rule(conditions)
                if rule_desc is not None:
                    outputs.append((rule_desc, current_level))
                conditions = conditions[: level - 1]
                conditions.append(cond)
            else:
                assert False

            current_level = level

        # sort by depth
        outputs.sort(key=lambda x: x[1])

        return [x[0] for x in outputs]

    def _build_one_rule(self, conditions: list[str]) -> str:
        if len(conditions) <= 1:
            return None
        res = " and ".join(conditions[:-1])
        class_components = conditions[-1].split("class: ")
        if len(class_components) < 2:
            return None
        class_name = self.meta.find_label(float(class_components[1].strip())).name
        res = class_name + ": " + res
        return res


class XGBoostDecisionTree(DecisionTree):
    def __init__(
        self,
        meta: dataset.DatasetMeta,
        max_depth: int,
        num_trees: int,
        random_state: int,
    ) -> None:
        super().__init__(meta)
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.random_state = random_state

    def predict(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        export_rules: bool,
    ) -> tuple[np.ndarray, list[str]]:
        if export_rules:
            raise NotImplementedError("Rule export is not supported yet for XGBoost")

        from xgboost import XGBClassifier

        if len(np.unique(y_train)) == 1:
            return np.full(x_test.shape[0], y_train[0]), []

        old_meta = self.meta
        has_categorical = any(f.type == "categorical" for f in self.meta.features)

        # encode one-hot
        if has_categorical:
            new_x, new_meta = _encode_one_hot(
                np.concatenate([x_train, x_test]), self.meta
            )

            self.meta = new_meta
            train_size = len(x_train)
            x_train = new_x[:train_size]
            x_test = new_x[train_size:]

        clf = XGBClassifier(
            max_depth=self.max_depth,
            n_estimators=self.num_trees,
            random_state=self.random_state,
        )
        # transform y_train to 0/1
        y_train = np.array(
            [self.meta.labels.index(self.meta.find_label(y)) for y in y_train]
        )
        clf.fit(x_train, y_train)
        y_test = clf.predict(x_test)
        y_test = np.array([self.meta.labels[y].value for y in y_test])

        self.meta = old_meta
        return y_test, []


class RandomForestDecisionTree(DecisionTree):
    def __init__(
        self, meta: dataset.DatasetMeta, num_trees: int, max_depth: int
    ) -> None:
        super().__init__(meta)
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.clf = None

    def predict(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        export_rules: bool = False,
    ) -> tuple[np.ndarray, list[str]]:
        if export_rules:
            raise NotImplementedError(
                "Rule export is not supported yet for RandomForest"
            )
        from sklearn.ensemble import RandomForestClassifier

        if len(np.unique(y_train)) == 1:
            return np.full(x_test.shape[0], y_train[0]), []

        old_meta = self.meta
        has_categorical = any(f.type == "categorical" for f in self.meta.features)

        # encode one-hot
        if has_categorical:
            new_x, new_meta = _encode_one_hot(
                np.concatenate([x_train, x_test]), self.meta
            )

            self.meta = new_meta
            train_size = len(x_train)
            x_train = new_x[:train_size]
            x_test = new_x[train_size:]

        self.clf = RandomForestClassifier(
            n_estimators=self.num_trees, max_depth=self.max_depth
        )
        self.clf.fit(x_train, y_train)

        self.meta = old_meta
        return self.clf.predict(x_test), []


class FederatedDecisionTree(DecisionTree):
    def __init__(self, meta: dataset.DatasetMeta, num_trees: int, max_depth: int) -> None:
        super().__init__(meta)
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.sub_metas = []

        def _split(a, n):
            k, m = divmod(len(a), n)
            return (
                a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
            )

        self.feature_groups = list(
            _split(range(self.meta.feature_count()), self.num_trees)
        )
        self.feature_groups = [list(x) for x in self.feature_groups]

        for feature_idxes in self.feature_groups:
            meta = DatasetMeta()
            meta.name = self.meta.name
            meta.target = self.meta.target
            meta.desc = self.meta.desc
            meta.labal_meaning = self.meta.labal_meaning
            meta.features = [self.meta.features[i] for i in feature_idxes]
            meta.labels = self.meta.labels
            self.sub_metas.append(meta)

        self.sub_trees: list[SimpleDecisionTree] = [
            SimpleDecisionTree(sub_meta, self.max_depth) for sub_meta in self.sub_metas
        ]

    def predict(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        export_rules: bool = False,
    ) -> tuple[np.ndarray, list[str]]:
        if export_rules:
            raise NotImplementedError(
                "Rule export is not supported yet for FederatedDecisionTree"
            )
        all_results = []

        for sub_tree, feature_group in zip(self.sub_trees, self.feature_groups):
            sub_x_train = x_train[:, feature_group]
            sub_x_test = x_test[:, feature_group]
            result, _ = sub_tree.predict(sub_x_train, y_train, sub_x_test)
            all_results.append(result)

        return all_results, []


    