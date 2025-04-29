from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
import sklearn.tree
import sklearn.ensemble
import numpy as np
import jinja2
from pathlib import Path
from tree_prompt import dataset
from tree_prompt.dataset import DatasetMeta
from jinja2 import Environment, FileSystemLoader
from sklearn.utils import check_array
import warnings
from ..logger import Logger, add_logger  
from ..dataset import generate_CoT_tree_prompt
import re

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


class CoTDecisionTree:
    def __init__(self, meta, max_depth, runner, log_file):
        self.meta = meta          # Dataset metadata
        self.max_depth = max_depth  # Max tree depth
        self.runner = runner      # LLM runner
        self.log_file = log_file  # Log file path
        self.rules = []           # List to store parsed rules
        self.feature_names = [f.name for f in meta.features]  # Get feature names from metadata features
        # 新增：原始特征名到当前数据列idx的映射
        self.feature_name_to_col = {f.name: idx for idx, f in enumerate(meta.features)}
        # Open log file in append mode with line buffering for real-time writing
        log_file_obj = open(self.log_file, 'a', encoding='utf-8', buffering=1)  # 1 means line buffering
        self.logger = Logger(log_file_obj)  # Initialize logger with file object
        # Ensure first log message marks the start
        self.logger.log(f"=== CoTDecision initialized at {datetime.now().isoformat()} ===")

    def fit(self, x_train, y_train, with_llm=True):
        # 移除强制限制最大深度为3，直接使用self.max_depth
        self.logger.log(f"[INFO] 决策树最大深度为{self.max_depth}")
        self.logger.log(f"[DEBUG] fit() called with with_llm={with_llm}")
        prompt_dict = generate_CoT_tree_prompt(
            meta=self.meta,
            x_train=x_train,
            y_train=y_train,
            max_depth=self.max_depth
        )
        prompt = prompt_dict["prompt"]
        # 明确区分 CoT 推理 prompt 日志
        self.logger.log(f"[DEBUG] CoT tree generation prompt type: {type(prompt)} length: {len(prompt)}")
        self.logger.log(f"[DEBUG] CoT tree generation prompt preview (first 500 chars): {prompt[:500]}")
        self.logger.log("=============== CoT tree generation prompt ===============")
        self.logger.log(prompt)  # 直接完整写入一条，logger.py已保证不截断
        self.logger.file.flush()  # 立即刷新，确保日志完整
        self.logger.log("=============== end of CoT tree generation prompt ===============")
        # 强制检测prompt是否已写入日志
        if not prompt or len(str(prompt).strip()) < 10:
            self.logger.log("[FATAL] CoT tree generation prompt内容异常，流程终止。请检查模板和数据！")
            raise RuntimeError("CoT tree generation prompt内容异常，流程终止。请检查模板和数据！")

        # Get first response from generator
        response_generator = self.runner.run([prompt])
        first_response = next(response_generator)
        self.logger.log("=== Starting LLM response logging ===")
        self.logger.log(f"Prompt sent to LLM:\n{prompt}")
        self.logger.file.flush()
        if isinstance(first_response, (list, tuple)):
            first_response = "\n".join(str(r) for r in first_response)
        else:
            if not isinstance(first_response, str):
                first_response = str(first_response)
        self.logger.log("=== LLM Response (full, for debug) ===")
        self.logger.log(str(first_response))
        self.logger.log("=== End of LLM Response ===")
        self.rules = self._parse_llm_response(first_response)
        self.logger.log(f"Generated {len(self.rules)} rules:")
        formatted_rules = self.get_rules()
        self.logger.log(f"[DEBUG] get_rules() returned {len(formatted_rules)} rules.")
        if not formatted_rules:
            self.logger.log("[ERROR] get_rules() 结果为空，说明规则解析或格式化有严重问题！")
        for i, rule in enumerate(formatted_rules, 1):
            if 'IF  THEN' in rule or rule.strip().endswith('IF THEN'):
                self.logger.log(f"[ERROR] 规则{rule} 条件为空，说明解析或格式化有bug！")
        self.logger.log("=== Final Decision Tree Rules (for evaluation) ===")
        for rule in formatted_rules:
            self.logger.log(rule)
        self.logger.log("=== End of Decision Tree Rules ===")
        dot_source = self.to_graphviz_source_from_rules(formatted_rules)
        if 'node0 [label=\"ROOT\"]' in dot_source and dot_source.count('node') <= 2:
            self.logger.log("[ERROR] 生成的决策树只有ROOT节点，说明规则分支未生效！")
        self.logger.log("===============生成的决策树 ===============")
        self.logger.log(dot_source)
        print("=============== 生成的决策树 ===============")
        print(dot_source)

    def to_graphviz_source_from_rules(self, rules: list[str]) -> str:
        """根据路径规则生成结构化决策树的graphviz源码"""
        import re
        class Node:
            def __init__(self, name=None):
                self.name = name
                self.children = dict()
                self.label = None
        root = Node("root")
        for rule in rules:
            # 兼容各种编号格式
            m = re.match(r"^\s*(?:\(?\d+\)?[\.)]?)?\s*IF (.+) THEN (.+)$", rule.strip(), re.IGNORECASE)
            if not m:
                self.logger.log(f"[ERROR] to_graphviz_source_from_rules: 跳过无法解析的rule: {rule}")
                continue
            try:
                # 更健壮地分割AND
                conds = [c.strip() for c in re.split(r'\s+AND\s+', m.group(1), flags=re.IGNORECASE)]
                label = m.group(2)
            except IndexError as e:
                self.logger.log(f"[FATAL] 正则分组失败: {e}, rule: {rule}, m.groups: {m.groups()}")
                continue
            self.logger.log(f"[DEBUG] 解析rule: conds={conds}, label={label}")
            node = root
            for cond in conds:
                if cond == '':
                    continue
                cond_key = cond.replace(' ', '')
                if cond_key not in node.children:
                    node.children[cond_key] = Node(cond)
                node = node.children[cond_key]
            node.label = label
        # 2. 递归生成DOT
        lines = ["digraph DecisionTree {"]
        node_id = 0
        def dfs(node, parent_id=None):
            nonlocal node_id
            my_id = node_id
            node_id += 1
            if node is root:
                label = "ROOT"
            elif node.label is not None:
                label = f"{node.name}\\n[{node.label}]"
            else:
                label = node.name
            lines.append(f'  node{my_id} [label="{label}"]')
            if parent_id is not None:
                lines.append(f'  node{parent_id} -> node{my_id}')
            for child in node.children.values():
                dfs(child, my_id)
        dfs(root)
        lines.append("}")
        return "\n".join(lines)

    def _parse_llm_response(self, response: str) -> list[dict]:
        """解析LLM返回的决策树规则文本，支持BEGIN_TREE/END_TREE块和标准规则格式。
        Args:
            response: LLM返回的原始文本响应
        Returns:
            解析后的规则列表，每条规则是一个字典，包含conditions和label字段
        """
        import re
        self.logger.log(f"Raw LLM response (first 500 chars): {str(response)[:500]}")
        
        if not response or not isinstance(response, str):
            self.logger.log("Error: Invalid or empty LLM response")
            return []
            
        # 提取BEGIN_TREE/END_TREE之间的内容
        begin_pat = re.compile(r'BEGIN[_ ]?TREE', re.IGNORECASE)
        end_pat = re.compile(r'END[_ ]?TREE', re.IGNORECASE)
        lines = response.split('\n')
        begin_idx = end_idx = None
        
        for i, line in enumerate(lines):
            if begin_pat.search(line):
                begin_idx = i
            if end_pat.search(line):
                end_idx = i
                break
                
        if begin_idx is not None and end_idx is not None and end_idx > begin_idx:
            tree_block = '\n'.join(lines[begin_idx+1:end_idx])
            self.logger.log(f"Found BEGIN_TREE/END_TREE block (lines {begin_idx+1}-{end_idx})")
        else:
            tree_block = response
            self.logger.log("No BEGIN_TREE/END_TREE block found, using full response")
            
        rules = []
        rule_pattern = re.compile(r'^\s*(?:\(?\d+\)?[\.)]?)?\s*IF\s+(.+?)\s+THEN\s+(.+?)\s*$', re.IGNORECASE)
        
        for line in tree_block.split('\n'):
            line = line.strip()
            if not line or line.startswith('#') or line.lower().startswith('note:'):
                continue
                
            match = rule_pattern.match(line)
            if not match:
                continue
                
            condition, label = match.groups()
            conditions = []
            
            # 解析条件(支持AND连接的多个条件)
            for cond in re.split(r'\s+AND\s+', condition, flags=re.IGNORECASE):
                cond = cond.strip()
                cond_match = re.match(r'^([\w_][\w\s\-_]*)\s*(<=|>=|<|>|=)\s*([\-]?[\d.]+|\w+)$', cond)
                if not cond_match:
                    self.logger.log(f"[DEBUG] Invalid condition format: '{cond}'")
                    continue
                    
                feature, operator, value = cond_match.groups()
                feature = feature.strip()
                if feature not in self.feature_name_to_col:
                    self.logger.log(f"[ERROR] Unknown feature '{feature}'")
                    continue
                    
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
                    
                conditions.append({
                    'feature': feature,
                    'operator': operator,
                    'value': value
                })
                
            if conditions:  # 只添加有效的规则
                rules.append({
                    'conditions': conditions,
                    'label': label.strip()
                })
                
        self.logger.log(f"Successfully parsed {len(rules)} rules")
        if not rules:
            self.logger.log("[ERROR] Failed to parse any valid rules!")
            
        return rules

    def _expand_jump_tree_rules(self, lines):
        """将(编号) IF ... THEN (编号/类别) ELSE (编号/类别)格式的规则，递归展开为所有路径规则，避免重复和漏掉分支"""
        node_map = {}
        jump_pattern = re.compile(r'^\((\d+)\)\s*IF\s+(.+?)\s*THEN\s+(.+?)\s*ELSE\s+(.+?)$', re.IGNORECASE)
        for line in lines:
            m = jump_pattern.match(line.strip())
            if m:
                idx, cond, then_, else_ = m.groups()
                node_map[idx] = {'cond': cond.strip(), 'then': then_.strip(), 'else': else_.strip()}
        results = []
        def dfs(idx, conds):
            node = node_map.get(idx)
            if not node:
                return
            # THEN分支
            if node['then'].isdigit() and node['then'] in node_map:
                dfs(node['then'], conds + [node['cond']])
            else:
                results.append((conds + [node['cond']], node['then']))
            # ELSE分支
            
            if node['else'].isdigit() and node['else'] in node_map:
                dfs(node['else'], conds + [f'NOT ({node["cond"]})'])
            else:
                results.append((conds + [f'NOT ({node["cond"]})'], node['else']))
        if node_map:
            dfs('1', [])
        # 去重，保证每条路径唯一
        unique = []
        seen = set()
        for conds, label in results:
            key = (tuple(conds), label)
            if key not in seen:
                seen.add(key)
                unique.append((conds, label))
        formatted = []
        for i, (conds, label) in enumerate(unique, 1):
            cond_str = ' AND '.join(conds)
            formatted.append(f'({i}) IF {cond_str} THEN {label}')
        return formatted

    def get_rules(self) -> list[str]:
        """返回所有路径规则，自动支持编号跳转树格式"""
        if not self.rules:
            self.logger.log("[ERROR] get_rules: self.rules为空！")
            return []
        # 检查是否为编号跳转树格式
        if isinstance(self.rules[0], str) and re.match(r'^\(\d+\)\s*IF.+THEN.+ELSE.+$', self.rules[0], re.IGNORECASE):
            return self._expand_jump_tree_rules(self.rules)
        # 支持多条件结构的格式化
        formatted_rules = []
        for idx, rule in enumerate(self.rules, 1):
            conditions = []
            if 'conditions' in rule and isinstance(rule['conditions'], list):
                for cond in rule['conditions']:
                    cond_str = f"{cond['feature']} {cond['operator']} {cond['value']}"
                    conditions.append(cond_str)
            elif 'condition' in rule:
                conditions.append(rule['condition'])
            if 'sub_rules' in rule and rule['sub_rules']:
                for sub_rule in rule['sub_rules']:
                    if 'condition' in sub_rule:
                        conditions.append(sub_rule['condition'])
            then_label = rule.get('label', '')
            rule_text = f"({idx}) IF {' AND '.join(conditions)} THEN {then_label}"
            if not conditions:
                self.logger.log(f"[ERROR] 规则{idx}条件为空，rule内容: {rule}")
            formatted_rules.append(rule_text)
        if not formatted_rules:
            self.logger.log("[ERROR] get_rules: formatted_rules为空！")
        return formatted_rules

    def predict(self, x_test):
        """Predict using the generated rules, 并统计每条规则命中样本数"""
        if not self.rules:
            raise ValueError("No rules available. Call fit() first.")
        predictions = []
        rule_hit_count = [0 for _ in self.rules]
        for sample in x_test:
            matched_rule_idx = None
            prediction = None
            for idx, rule in enumerate(self.rules):
                if self._apply_rules(sample, rule=rule):
                    matched_rule_idx = idx
                    prediction = rule['label']
                    break
            if matched_rule_idx is not None:
                rule_hit_count[matched_rule_idx] += 1
                predictions.append(prediction)
            else:
                predictions.append(self.meta.get_default_label())
        # 输出每条规则命中数
        for idx, count in enumerate(rule_hit_count):
            self.logger.log(f"[规则{idx+1}] 命中样本数: {count}")
        return np.array(predictions)

    def _apply_rules(self, sample, rule=None):
        """应用规则时支持多条件 AND 结构，修正特征名到列的映射"""
        if not self.rules:
            self.logger.log("[ERROR] _apply_rules: self.rules为空，直接返回默认label！")
            return False
        # 用映射保证特征名和数据列一致
        sample_dict = {name: sample[idx] for name, idx in self.feature_name_to_col.items()}
        rules_to_check = [rule] if rule is not None else self.rules
        for rule in rules_to_check:
            if not isinstance(rule, dict):
                continue
            if 'conditions' in rule and isinstance(rule['conditions'], list):
                all_match = True
                for cond in rule['conditions']:
                    feature = cond['feature']
                    operator = cond['operator']
                    value = cond['value']
                    sample_value = sample_dict.get(feature, None)
                    if sample_value is None:
                        all_match = False
                        break
                    try:
                        num_sample = float(sample_value)
                        num_value = float(value)
                        comparison_type = 'numeric'
                    except (ValueError, TypeError):
                        str_sample = str(sample_value).lower()
                        str_value = str(value).lower()
                        comparison_type = 'string'
                    match = False
                    if comparison_type == 'numeric':
                        if operator == '>=': match = num_sample >= num_value
                        elif operator == '<=': match = num_sample <= num_value
                        elif operator == '>': match = num_sample > num_value
                        elif operator == '<': match = num_sample < num_value
                        elif operator == '=': match = abs(num_sample - num_value) < 1e-6
                    else:
                        if operator == '=': match = str_sample == str_value
                        elif operator in ['>', '<']:
                            match = str_sample == str_value
                    if not match:
                        all_match = False
                        break
                if all_match:
                    return True
        return False


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
    def __init__(
        self, meta: dataset.DatasetMeta, num_trees: int, max_depth: int
    ) -> None:
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

class TreeModel:
    def __init__(self):
        self.rules = []

    def fit(self, data):
        # 原始代码
        # self.rules = self._parse_llm_response(data)
        
        # 修改后代码：增加对解析结果的检查
        parsed_rules = self._parse_llm_response(data)
        if not parsed_rules:
            raise ValueError("Failed to parse rules from LLM response.")
        self.rules = parsed_rules

    def _parse_llm_response(self, response):
        """解析LLM返回的决策树规则文本，支持BEGIN_TREE/END_TREE壳子和标准规则格式"""
        import re
        self.logger.log(f"Raw LLM response (first 500 chars): {str(response)[:500]}")
        rules = []
        if not response:
            self.logger.log("Error: Empty LLM response received")
            return []
        response = str(response).strip()
        if not response:
            self.logger.log("Error: Response is empty after conversion")
            return []
        # 提取BEGIN_TREE/END_TREE之间内容
        begin_pat = re.compile(r'BEGIN[_ ]?TREE', re.IGNORECASE)
        end_pat = re.compile(r'END[_ ]?TREE', re.IGNORECASE)
        begin_idx = end_idx = None
        lines = response.split('\n')
        for i, line in enumerate(lines):
            if begin_pat.search(line):
                begin_idx = i
            if end_pat.search(line):
                end_idx = i
                break
        if begin_idx is not None and end_idx is not None and end_idx > begin_idx:
            tree_block = '\n'.join(lines[begin_idx+1:end_idx])
            self.logger.log(f"Detected BEGIN_TREE/END_TREE block, extracting rules from block (lines {begin_idx+1}-{end_idx-1})")
        else:
            tree_block = response
            self.logger.log("No BEGIN_TREE/END_TREE block found, using full response for rule extraction.")
        # 只解析 N. IF ... THEN ... 格式的规则
        rule_pattern = re.compile(r'^\s*(?:\(?\d+\)?[\.)]?)?\s*IF\s+(.+?)\s+THEN\s+(.+?)\s*$', re.IGNORECASE)
        for line in tree_block.split('\n'):
            line = line.strip()
            if not line or line.startswith('#') or line.lower().startswith('note:'):
                continue
            match = rule_pattern.match(line)
            if match:
                condition = match.group(1)
                consequence = match.group(2)
                # 多条件分割
                conds = [c.strip() for c in re.split(r'\s+AND\s+', condition, flags=re.IGNORECASE)]
                parsed_conditions = []
                for cond in conds:
                    cond_match = re.match(r'^([\w_][\w\s\-_]*)\s*(<=|>=|<|>)\s*([\-]?[\d.]+|\w+)$', cond)
                    if not cond_match:
                        self.logger.log(f"[DEBUG] Invalid condition format: '{cond}'")
                        continue
                    feature, operator, value = cond_match.groups()
                    feature = ' '.join(feature.split())
                    parsed_conditions.append({'feature': feature, 'operator': operator, 'value': value})
                if parsed_conditions:
                    rules.append({'conditions': parsed_conditions, 'label': consequence})
            else:
                self.logger.log(f"[DEBUG] Skipped non-rule line: {line}")
        self.logger.log(f"Successfully parsed {len(rules)} valid rules")
        if not rules:
            self.logger.log("LLM规则解析失败，未能提取到任何有效规则，请检查LLM输出格式！")
        return rules

    def _apply_rules(self, sample):
        """应用规则时支持多条件 AND 结构"""
        if not self.rules:
            self.logger.log("[ERROR] _apply_rules: self.rules为空，直接返回默认label！")
            return self.meta.get_default_label()
        sample_dict = {self.feature_names[i]: val for i, val in enumerate(sample)}
        for rule in self.rules:
            # 类型安全检查
            if not isinstance(rule, dict):
                if not hasattr(self, '_reported_invalid_rule_types'):
                    self._reported_invalid_rule_types = set()
                rule_type = str(type(rule))
                if rule_type not in self._reported_invalid_rule_types:
                    self.logger.log(f"发现无效规则类型: {rule_type}")
                    self.logger.log(f"示例无效规则内容: {str(rule)[:100]}...")
                    self.logger.log("规则应包含condition, feature, operator, value等键")
                    self._reported_invalid_rule_types.add(rule_type)
                continue
            # 支持多条件结构
            if 'conditions' in rule and isinstance(rule['conditions'], list):
                all_match = True
                for cond in rule['conditions']:
                    feature = cond['feature']
                    operator = cond['operator']
                    value = cond['value']
                    sample_value = sample_dict.get(feature, None)
                    if sample_value is None:
                        all_match = False
                        break
                    try:
                        num_sample = float(sample_value)
                        num_value = float(value)
                        comparison_type = 'numeric'
                    except (ValueError, TypeError):
                        str_sample = str(sample_value).lower()
                        str_value = str(value).lower()
                        comparison_type = 'string'
                    match = False
                    if comparison_type == 'numeric':
                        if operator == '>=': match = num_sample >= num_value
                        elif operator == '<=': match = num_sample <= num_value
                        elif operator == '>': match = num_sample > num_value
                        elif operator == '<': match = num_sample < num_value
                        elif operator == '=': match = abs(num_sample - num_value) < 1e-6
                    else:
                        if operator == '=': match = str_sample == str_value
                        elif operator in ['>', '<']:
                            match = str_sample == str_value
                    if not match:
                        all_match = False
                        break
                if all_match:
                    return rule['label']
                continue
            # 兼容旧结构
            required_keys = ['feature', 'operator', 'value', 'label']
            if all(k in rule for k in required_keys):
                feature = rule['feature']
                operator = rule['operator']
                value = rule['value']
                label = rule['label']
                sample_value = sample_dict.get(feature, None)
                if sample_value is None:
                    continue
                try:
                    num_sample = float(sample_value)
                    num_value = float(value)
                    comparison_type = 'numeric'
                except (ValueError, TypeError):
                    str_sample = str(sample_value).lower()
                    str_value = str(value).lower()
                    comparison_type = 'string'
                match = False
                if comparison_type == 'numeric':
                    if operator == '>=': match = num_sample >= num_value
                    elif operator == '<=': match = num_sample <= num_value
                    elif operator == '>': match = num_sample > num_value
                    elif operator == '<': match = num_sample < num_value
                    elif operator == '=': match = abs(num_sample - num_value) < 1e-6
                else:
                    if operator == '=': match = str_sample == str_value
                    elif operator in ['>', '<']:
                        match = str_sample == str_value
                if match:
                    return label
        self.logger.log("[ERROR] _apply_rules: 所有规则都未命中，返回默认label！")
        return self.meta.labels[0].value