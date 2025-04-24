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


class ToTDecisionTree:
    def __init__(self, meta, max_depth, runner, log_file, num_rule_candidates=5, select_top_k=1):
        self.meta = meta          # Dataset metadata
        self.max_depth = max_depth  # Max tree depth
        self.runner = runner      # LLM runner
        self.log_file = log_file  # Log file path
        self.rules = []           # List to store parsed rules
        self.feature_names = [f.name for f in meta.features]  # Get feature names from metadata features
        self.num_rule_candidates = num_rule_candidates
        self.select_top_k = select_top_k
        # Open log file in append mode with line buffering for real-time writing
        log_file_obj = open(self.log_file, 'a', encoding='utf-8', buffering=1)  # 1 means line buffering
        self.logger = Logger(log_file_obj)  # Initialize logger with file object
        # Ensure first log message marks the start
        self.logger.log(f"=== ToTDecision initialized at {datetime.now().isoformat()} ===")

    def fit(self, x_train, y_train):
        rules = []  # 存储已生成的树结构
        all_llm_responses = []
        used_features = set()
        for depth in range(self.max_depth):
            rule_candidates = []
            layer_used_features = set()  # 本层已用特征
            for i in range(self.num_rule_candidates):
                try:
                    # 生成新的随机种子
                    seed = random.randint(0, 2**32 - 1)
                    random.seed(seed)
                    self.logger.log(f"[LLM PROMPT SEED] {seed}")
                    prompt = self._build_rule_prompt(x_train, y_train, rules, depth, used_features)
                    response = next(self.runner.run([prompt]))
                    self.logger.log(f"[LLM RESPONSE SEED] {seed}")
                    parsed_rules = self._parse_llm_response(response)
                    # 只保留本层未用过的特征，且和前面层不重复
                    valid_rules = []
                    for idx, rule in enumerate(parsed_rules or []):
                        # 找到该规则用到的第一个特征
                        used_this_rule = None
                        for fname in self.feature_names:
                            if fname in rule['condition']:
                                used_this_rule = fname
                                break
                        if not used_this_rule:
                            continue
                        if used_this_rule in used_features or used_this_rule in layer_used_features:
                            self.logger.log(f"[第{depth+1}层-候选{i+1}-{idx+1}](跳过已用特征): IF {rule['condition']} THEN {rule['label']}")
                            continue
                        self.logger.log(f"[第{depth+1}层-候选{i+1}-{idx+1}]: IF {rule['condition']} THEN {rule['label']}")
                        valid_rules.append(rule)
                        layer_used_features.add(used_this_rule)
                        break  # 只取每次LLM返回的第一个有效特征规则
                    if valid_rules:
                        rule_candidates.append(valid_rules[0])
                except Exception as e:
                    self.logger.log(f"Exception in LLM rule generation at depth {depth+1} round {i+1}: {e}")
            if not rule_candidates:
                self.logger.log(f"No rule candidates generated at depth {depth+1}, aborting fit.")
                break
            # 2. 规则筛选
            try:
                # 生成新的随机种子
                select_seed = random.randint(0, 2**32 - 1)
                random.seed(select_seed)
                self.logger.log(f"[LLM SELECT SEED] {select_seed}")
                select_prompt = self._build_select_prompt(rule_candidates, rules, depth)
                select_response = next(self.runner.run([select_prompt]))
                self.logger.log(f"[LLM SELECT RESPONSE SEED] {select_seed}")
                best_rules = self._parse_llm_select_response(select_response, rule_candidates, self.select_top_k)
                # 打印本层最终选中规则
                for idx, rule in enumerate(best_rules):
                    self.logger.log(f"[第{depth+1}层-最终规则{idx+1}]: IF {rule['condition']} THEN {rule['label']}")
                    # 记录本层用到的特征
                    for fname in self.feature_names:
                        if fname in rule['condition']:
                            used_features.add(fname)
            except Exception as e:
                self.logger.log(f"Exception in LLM rule selection at depth {depth+1}: {e}")
                best_rules = []
            rules.extend(best_rules)
        self.rules = rules
        # self.logger.log("=== 最终生成的决策树规则 ===")
        # self.logger.log(self.get_rules_text())
        if not self.rules:
            self.logger.log("[ERROR] No rules generated by LLM! Dumping all LLM responses for debug:")
            for tag, resp in all_llm_responses:
                self.logger.log(f"[{tag}]\n{resp}")
            raise ValueError("No rules generated by LLM. Please check LLM output format and prompt.")

        # 打印最终用于大模型检验的规则字符串
        self.logger.log("=== The rules for LLM evaluation ===")
        self.logger.log(self.get_rules_str())
        self.logger.log("=== End of LLM evaluation rules ===")

    def _build_rule_prompt(self, x_train, y_train, rules, depth, used_features=None):
        """构造生成规则的prompt，包含数据集基本信息、特征渲染、示例、已生成树结构和当前层信息，要求输出标准格式"""
        from ..dataset import generate_ToT_tree_prompt
        meta = self.meta
        num_examples = min(5, x_train.shape[0])
        prompt_info = generate_ToT_tree_prompt(meta, x_train, y_train, max_depth=self.max_depth, num_examples=num_examples)
        prompt_head = prompt_info["prompt"]
        # 2. 已生成规则
        rules_str = "无" if not rules else "\n".join([f"Rule: {r['condition']} THEN {r['label']}" for r in rules])
        # 新增：本层禁用特征提示
        forbid_str = ""
        if used_features:
            forbid_str = f"\n# 注意：本层不能再用以下特征：{', '.join(used_features)}"
        # 3. 当前层说明和格式要求
        prompt_tail = f"""
# 当前决策树已生成规则：
{rules_str}
{forbid_str}

请为第{depth+1}层提出最优的分类规则。
请严格按照如下格式输出（不要输出多余内容、注释或空行）：
Rule 1: IF <feature> <operator> <value> THEN <label>
Rule 2: IF <feature> <operator> <value> AND <feature2> <operator2> <value2> THEN <label>
...（如有多条规则，依次编号）
"""
        return prompt_head + "\n" + prompt_tail

    def _parse_llm_rule_response(self, response):
        """解析LLM生成的单条规则"""
        # 这里直接复用原有的_parse_llm_response逻辑，取第一条规则
        rules = self._parse_llm_response(response)
        return rules[0] if rules else None

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
        # 直接用_parse_llm_response解析，返回前k条
        rules = self._parse_llm_response(response)
        return rules[:k] if rules else []

    def get_rules(self):
        # 返回所有已生成的规则（list），兼容原有接口
        return self.rules

    def predict(self, x_test):
        """Predict using the generated rules"""
        if not self.rules:
            raise ValueError("No rules available. Call fit() first.")
            
        predictions = []
        for sample in x_test:
            prediction = self._apply_rules(sample)
            predictions.append(prediction)
        return np.array(predictions)

    def _parse_llm_response(self, response):
        """只解析标准格式Rule N: IF ... THEN ...的规则，忽略其它内容，兼容LLM返回list/tuple的情况"""
        import re
        rules = []
        if not response:
            self.logger.log("Error: Empty LLM response received")
            return []
        # 新增：如果是list/tuple，拼接为字符串
        if isinstance(response, (list, tuple)):
            response = "\n".join(str(r) for r in response)
        response = str(response).strip()
        if not response:
            self.logger.log("Error: Response is empty after conversion")
            return []
        # 只保留标准格式的规则
        rule_pattern = re.compile(r'^Rule\s+\d+:\s*IF\s+(.+?)\s+THEN\s+(.+)$', re.IGNORECASE | re.MULTILINE)
        for match in rule_pattern.finditer(response):
            condition, label = match.groups()
            rules.append({
                'condition': condition.strip(),
                'label': label.strip()
            })
        self.logger.log(f"Parsed {len(rules)} rules from LLM response.")
        return rules

    def _apply_rules(self, sample):
            """应用规则时添加类型安全检查"""
            if not self.rules:
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
                    
                required_keys = ['feature', 'operator', 'value', 'label']
                if not all(k in rule for k in required_keys):
                    if not hasattr(self, '_reported_invalid_rule_structure'):
                        self.logger.log(f"不完整的规则结构，缺少必要字段: {required_keys}")
                        self._reported_invalid_rule_structure = True
                    continue
                
            
                feature = rule['feature']
                operator = rule['operator']
                value = rule['value']
                label = rule['label']
                
                # 获取特征值
                sample_value = sample_dict.get(feature, None)
                if sample_value is None:
                    continue
                    
                # 类型一致性处理
                try:
                    # 尝试数值比较
                    num_sample = float(sample_value)
                    num_value = float(value)
                    comparison_type = 'numeric'
                except (ValueError, TypeError):
                    # 字符串比较
                    str_sample = str(sample_value).lower()
                    str_value = str(value).lower()
                    comparison_type = 'string'
                
                # 执行比较
                match = False
                if comparison_type == 'numeric':
                    if operator == '>=': match = num_sample >= num_value
                    elif operator == '<=': match = num_sample <= num_value
                    elif operator == '>': match = num_sample > num_value
                    elif operator == '<': match = num_sample < num_value
                    elif operator == '=': match = abs(num_sample - num_value) < 1e-6
                else:
                    if operator == '=': match = str_sample == str_value
                    elif operator in ['>', '<']:  # 对分类特征支持排序操作
                        match = str_sample == str_value  # 这里需要根据实际需求调整
                
                if match and 'sub_rules' in rule:
                    # 验证所有子规则
                    all_sub_matched = True
                    for sub_rule in rule['sub_rules']:
                        try:
                            # 同样的安全检查
                            if not isinstance(sub_rule, dict):
                                all_sub_matched = False
                                break
                                
                            # 子规则验证逻辑
                            sub_feature = sub_rule.get('feature')
                            sub_operator = sub_rule.get('operator') 
                            sub_value = sub_rule.get('value')
                            
                            if None in [sub_feature, sub_operator, sub_value]:
                                all_sub_matched = False
                                break
                                
                            # 获取子规则特征值
                            sub_sample_value = sample_dict.get(sub_feature)
                            if sub_sample_value is None:
                                all_sub_matched = False
                                break
                                
                            # 类型一致性处理
                            try:
                                if isinstance(sub_value, (int, float)):
                                    sub_num_sample = float(sub_sample_value)
                                    sub_num_value = float(sub_value)
                                    if sub_operator == '>=': sub_match = sub_num_sample >= sub_num_value
                                    elif sub_operator == '<=': sub_match = sub_num_sample <= sub_num_value
                                    elif sub_operator == '>': sub_match = sub_num_sample > sub_num_value
                                    elif sub_operator == '<': sub_match = sub_num_sample < sub_num_value
                                    elif sub_operator == '=': sub_match = abs(sub_num_sample - sub_num_value) < 1e-6
                                    else: sub_match = False
                                else:
                                    sub_str_sample = str(sub_sample_value).lower()
                                    sub_str_value = str(sub_value).lower()
                                    sub_match = (sub_operator == '=' and sub_str_sample == sub_str_value)
                                    
                                if not sub_match:
                                    all_sub_matched = False
                                    break
                                    
                            except Exception as e:
                                self.logger.log(f"子规则验证错误: {str(e)}")
                                all_sub_matched = False
                                break
                                
                        except Exception as e:
                            self.logger.log(f"子规则处理异常: {str(e)}")
                            all_sub_matched = False
                            break
                            
                    if all_sub_matched:
                        return label
                elif match:
                    return label
                
            return self.meta.labels[0].value

    def get_rules_str(self) -> str:
        """返回评估用的规则字符串，格式为The rules are as follows:(1) IF ... THEN ..."""
        rules = self.get_rules()
        rules_str = "The rules are as follows:\n"
        for i, rule in enumerate(rules, 1):
            # 如果已经有(1)前缀则直接用，否则加编号
            if rule.strip().startswith(f"({i})"):
                rules_str += rule + "\n"
            else:
                rules_str += f"({i}) {rule}\n"
        return rules_str.strip()


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
        """只解析标准格式Rule N: IF ... THEN ...的规则，忽略其它内容，兼容LLM返回list/tuple的情况"""
        import re
        rules = []
        if not response:
            self.logger.log("Error: Empty LLM response received")
            return []
        # 新增：如果是list/tuple，拼接为字符串
        if isinstance(response, (list, tuple)):
            response = "\n".join(str(r) for r in response)
        response = str(response).strip()
        if not response:
            self.logger.log("Error: Response is empty after conversion")
            return []
        # 只保留标准格式的规则
        rule_pattern = re.compile(r'^Rule\s+\d+:\s*IF\s+(.+?)\s+THEN\s+(.+)$', re.IGNORECASE | re.MULTILINE)
        for match in rule_pattern.finditer(response):
            condition, label = match.groups()
            rules.append({
                'condition': condition.strip(),
                'label': label.strip()
            })
        self.logger.log(f"Parsed {len(rules)} rules from LLM response.")
        return rules

    def _apply_rules(self, sample):
            """应用规则时添加类型安全检查"""
            if not self.rules:
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
                    
                required_keys = ['feature', 'operator', 'value', 'label']
                if not all(k in rule for k in required_keys):
                    if not hasattr(self, '_reported_invalid_rule_structure'):
                        self.logger.log(f"不完整的规则结构，缺少必要字段: {required_keys}")
                        self._reported_invalid_rule_structure = True
                    continue
                
            
                feature = rule['feature']
                operator = rule['operator']
                value = rule['value']
                label = rule['label']
                
                # 获取特征值
                sample_value = sample_dict.get(feature, None)
                if sample_value is None:
                    continue
                    
                # 类型一致性处理
                try:
                    # 尝试数值比较
                    num_sample = float(sample_value)
                    num_value = float(value)
                    comparison_type = 'numeric'
                except (ValueError, TypeError):
                    # 字符串比较
                    str_sample = str(sample_value).lower()
                    str_value = str(value).lower()
                    comparison_type = 'string'
                
                # 执行比较
                match = False
                if comparison_type == 'numeric':
                    if operator == '>=': match = num_sample >= num_value
                    elif operator == '<=': match = num_sample <= num_value
                    elif operator == '>': match = num_sample > num_value
                    elif operator == '<': match = num_sample < num_value
                    elif operator == '=': match = abs(num_sample - num_value) < 1e-6
                else:
                    if operator == '=': match = str_sample == str_value
                    elif operator in ['>', '<']:  # 对分类特征支持排序操作
                        match = str_sample == str_value  # 这里需要根据实际需求调整
                
                if match and 'sub_rules' in rule:
                    # 验证所有子规则
                    all_sub_matched = True
                    for sub_rule in rule['sub_rules']:
                        try:
                            # 同样的安全检查
                            if not isinstance(sub_rule, dict):
                                all_sub_matched = False
                                break
                                
                            # 子规则验证逻辑
                            sub_feature = sub_rule.get('feature')
                            sub_operator = sub_rule.get('operator') 
                            sub_value = sub_rule.get('value')
                            
                            if None in [sub_feature, sub_operator, sub_value]:
                                all_sub_matched = False
                                break
                                
                            # 获取子规则特征值
                            sub_sample_value = sample_dict.get(sub_feature)
                            if sub_sample_value is None:
                                all_sub_matched = False
                                break
                                
                            # 类型一致性处理
                            try:
                                if isinstance(sub_value, (int, float)):
                                    sub_num_sample = float(sub_sample_value)
                                    sub_num_value = float(sub_value)
                                    if sub_operator == '>=': sub_match = sub_num_sample >= sub_num_value
                                    elif sub_operator == '<=': sub_match = sub_num_sample <= sub_num_value
                                    elif sub_operator == '>': sub_match = sub_num_sample > sub_num_value
                                    elif sub_operator == '<': sub_match = sub_num_sample < sub_num_value
                                    elif sub_operator == '=': sub_match = abs(sub_num_sample - sub_num_value) < 1e-6
                                    else: sub_match = False
                                else:
                                    sub_str_sample = str(sub_sample_value).lower()
                                    sub_str_value = str(sub_value).lower()
                                    sub_match = (sub_operator == '=' and sub_str_sample == sub_str_value)
                                    
                                if not sub_match:
                                    all_sub_matched = False
                                    break
                                    
                            except Exception as e:
                                self.logger.log(f"子规则验证错误: {str(e)}")
                                all_sub_matched = False
                                break
                                
                        except Exception as e:
                            self.logger.log(f"子规则处理异常: {str(e)}")
                            all_sub_matched = False
                            break
                            
                    if all_sub_matched:
                        return label
                elif match:
                    return label
                
            return self.meta.labels[0].value

    def get_rules_str(self) -> str:
        """返回评估用的规则字符串，格式为The rules are as follows:(1) IF ... THEN ..."""
        rules = self.get_rules()
        rules_str = "The rules are as follows:\n"
        for i, rule in enumerate(rules, 1):
            # 如果已经有(1)前缀则直接用，否则加编号
            if rule.strip().startswith(f"({i})"):
                rules_str += rule + "\n"
            else:
                rules_str += f"({i}) {rule}\n"
        return rules_str.strip()