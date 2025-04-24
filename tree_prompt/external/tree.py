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
        # Open log file in append mode with line buffering for real-time writing
        log_file_obj = open(self.log_file, 'a', encoding='utf-8', buffering=1)  # 1 means line buffering
        self.logger = Logger(log_file_obj)  # Initialize logger with file object
        # Ensure first log message marks the start
        self.logger.log(f"=== CoTDecision initialized at {datetime.now().isoformat()} ===")

    def fit(self, x_train, y_train):
        """Generate decision tree rules using LLM"""
        prompt_dict = generate_CoT_tree_prompt(
            meta=self.meta,
            x_train=x_train,
            y_train=y_train,
            max_depth=self.max_depth
        )
        prompt = prompt_dict["prompt"]  # 只取字符串部分
        # Get first response from generator
        response_generator = self.runner.run([prompt])
        first_response = next(response_generator)
        # 实时记录完整响应
        self.logger.log("=== Starting LLM response logging ===")
        self.logger.log(f"Prompt sent to LLM:\n{prompt}")
        self.logger.file.flush()
        if isinstance(first_response, (list, tuple)):
            for i, resp in enumerate(first_response):
                self.logger.log(f"LLM Response Part {i+1} (raw):\n{str(resp)}")
                self.logger.file.flush()
            first_response = "\n".join(str(r) for r in first_response)
        else:
            self.logger.log("LLM Response (raw):")
            self.logger.log(str(first_response))
            self.logger.file.flush()
            if not isinstance(first_response, str):
                first_response = str(first_response)
        self.logger.log("=== LLM Response (full, for debug) ===")
        self.logger.log(str(first_response))
        self.logger.log("=== End of LLM Response ===")
        self.rules = self._parse_llm_response(first_response)
        self.logger.log(f"Generated {len(self.rules)} rules:")
        # 打印最终格式化后的规则（编号格式，自动路径展开）
        formatted_rules = self.get_rules()
        self.logger.log("=== Final Decision Tree Rules (for evaluation) ===")
        for rule in formatted_rules:
            self.logger.log(rule)
        self.logger.log("=== End of Decision Tree Rules ===")

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
            return []
        # 检查是否为编号跳转树格式
        if isinstance(self.rules[0], str) and re.match(r'^\(\d+\)\s*IF.+THEN.+ELSE.+$', self.rules[0], re.IGNORECASE):
            return self._expand_jump_tree_rules(self.rules)
        # ...原有路径规则格式化逻辑...
        formatted_rules = []
        for idx, rule in enumerate(self.rules, 1):
            conditions = []
            if 'condition' in rule:
                conditions.append(rule['condition'])
            if 'sub_rules' in rule and rule['sub_rules']:
                for sub_rule in rule['sub_rules']:
                    if 'condition' in sub_rule:
                        conditions.append(sub_rule['condition'])
            then_label = rule.get('label', '')
            rule_text = f"({idx}) IF {' AND '.join(conditions)} THEN {then_label}"
            formatted_rules.append(rule_text)
        return formatted_rules

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
        """支持嵌套规则结构的解析方法，增强对 LLM 输出格式的健壮性，支持BEGIN_TREE/END_TREE壳子"""
        self.logger.log(f"Raw LLM response (first 500 chars): {str(response)[:500]}")
        import re
        rules = []
        rule_map = {}
        terminal_rules = []
        rule_groups = {}
        group_id = 1

        if not response:
            self.logger.log("Error: Empty LLM response received")
            return []
        response = str(response).strip()
        if not response:
            self.logger.log("Error: Response is empty after conversion")
            return []

        # 1. 提取BEGIN_TREE/END_TREE壳子内容
        tree_block = None
        begin_pat = re.compile(r'BEGIN[_ ]?TREE', re.IGNORECASE)
        end_pat = re.compile(r'END[_ ]?TREE', re.IGNORECASE)
        begin_idx = None
        end_idx = None
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

        # 2. 解析tree_block内容
        lines = tree_block.split('\n')
        line_count = len(lines)
        rule_prefix_count = len(re.findall(r'^rule\s+\d+:', tree_block, re.MULTILINE | re.IGNORECASE))
        self.logger.log(f"Response diagnostics - Lines: {line_count}, Rule prefixes found: {rule_prefix_count}")

        # 支持 (N) 或 Rule N: 开头的规则编号
        rule_pattern = re.compile(
            r'^\s*(?:\((\d+)\)|rule\s*(\d+))?\s*:?'  # 支持 (N) 或 Rule N: 开头
            r'\s*if\s+(.+?)'
            r'\s*then\s+(.+?)'
            r'(?:\s*else\s+(.+?))?'
            r'\s*$',
            re.IGNORECASE
        )
        fallback_pattern = re.compile(
            r'^\s*if\s+(.+?)\s*then\s+(.+?)\s*else\s+(.+?)\s*$',
            re.IGNORECASE
        )

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.lower().startswith('note:'):
                continue  # 跳过空行和注释
            match = rule_pattern.match(line)
            if match:
                rule_id = match.group(1) or match.group(2)
                condition = match.group(3)
                consequence = match.group(4)
                else_consequence = match.group(5)
                if not isinstance(condition, str) or not isinstance(consequence, str):
                    self.logger.log(f"Invalid rule type - condition: {type(condition)}, consequence: {type(consequence)}")
                    continue
                condition = condition.strip()
                consequence = consequence.strip()
                if not condition or not consequence:
                    self.logger.log(f"Empty rule condition or consequence")
                    continue
                rule_map[rule_id or str(len(rule_map)+1)] = {
                    'condition': condition,
                    'consequence': consequence,
                    'is_terminal': False
                }
                if else_consequence:
                    rule_map[f"{rule_id or str(len(rule_map)+1)}_else"] = {
                        'condition': f"NOT ({condition})",
                        'consequence': else_consequence.strip(),
                        'is_terminal': False
                    }
            else:
                fallback = fallback_pattern.match(line)
                if fallback:
                    condition, consequence, else_consequence = fallback.groups()
                    rule_map[str(len(rule_map)+1)] = {
                        'condition': condition.strip(),
                        'consequence': consequence.strip(),
                        'is_terminal': False
                    }
                    rule_map[f"{len(rule_map)+1}_else"] = {
                        'condition': f"NOT ({condition.strip()})",
                        'consequence': else_consequence.strip(),
                        'is_terminal': False
                    }
                else:
                    self.logger.log(f"Skipped non-rule line: {line}")

        # 终端规则识别
        label_names = [label.name.lower() for label in self.meta.labels]
        for rule_id, rule in rule_map.items():
            if rule['consequence'].lower() in label_names:
                rule['is_terminal'] = True
                terminal_rules.append(rule_id)

        # 构建规则链（保持原有逻辑）
        for terminal_rule_id in terminal_rules:
            current_rule_id = terminal_rule_id
            rule_chain = []
            while current_rule_id in rule_map:
                current_rule = rule_map[current_rule_id]
                rule_chain.insert(0, {
                    'condition': current_rule['condition'],
                    'label': current_rule['consequence'] if current_rule['is_terminal'] else None
                })
                # 查找前驱规则
                current_rule_id = None
                for rule_id, rule in rule_map.items():
                    if rule['consequence'] == current_rule_id:
                        current_rule_id = rule_id
                        break
            if rule_chain:
                # 支持 AND 多条件
                condition_pattern = re.compile(
                    r'^\s*([\w\s]+?)\s*(<=|>=|<|>|=)\s*([\d.]+|\".+?\"|\'.+?\'|\w+)\s*$'
                )
                parsed_conditions = []
                valid = True
                for step in rule_chain:
                    for cond in step['condition'].split(' AND '):
                        cond = cond.strip()
                        cond_match = condition_pattern.match(cond)
                        if not cond_match:
                            self.logger.log(f"Invalid condition format: {cond}")
                            valid = False
                            break
                        feature, operator, value = cond_match.groups()
                        feature = ' '.join(feature.split())
                        matched_feature = next((f for f in self.feature_names if f.lower() == feature.lower()), None)
                        if not matched_feature:
                            self.logger.log(f"Unknown feature '{feature}', skipping condition")
                            valid = False
                            break
                        try:
                            value = value.strip('\'\"')
                            numeric_value = float(value) if '.' in value else int(value)
                            value = numeric_value
                        except:
                            pass
                        parsed_conditions.append({
                            'feature': matched_feature,
                            'operator': operator,
                            'value': value,
                            'label': step['label']
                        })
                    if not valid:
                        break
                if valid and parsed_conditions:
                    main_condition = parsed_conditions[0]
                    rule_label = rule_chain[-1]['label'] or main_condition['label']
                    rule_entry = {
                        'condition': main_condition['feature'] + ' ' + main_condition['operator'] + ' ' + str(main_condition['value']),
                        'feature': main_condition['feature'],
                        'operator': main_condition['operator'],
                        'value': main_condition['value'],
                        'label': rule_label,
                        'sub_rules': []
                    }
                    if len(parsed_conditions) > 1:
                        rule_entry['sub_rules'] = [
                            {
                                'condition': c['feature'] + ' ' + c['operator'] + ' ' + str(c['value']),
                                'feature': c['feature'],
                                'operator': c['operator'],
                                'value': c['value'],
                                'label': rule_label
                            } for c in parsed_conditions[1:]
                        ]
                    group_key = f"{rule_entry['feature']}_{rule_entry['operator']}_{rule_entry['value']}"
                    if group_key not in rule_groups:
                        rule_groups[group_key] = {
                            'main_rule': rule_entry,
                            'group_id': group_id
                        }
                        group_id += 1
                        self.logger.log(f"Added new rule group for feature: {rule_entry['feature']}")
                    else:
                        existing = rule_groups[group_key]['main_rule']
                        existing['sub_rules'].extend(rule_entry['sub_rules'])
        for group in rule_groups.values():
            main_rule = group['main_rule']
            if main_rule['sub_rules']:
                seen = set()
                unique_subrules = []
                for r in main_rule['sub_rules']:
                    key = (r['feature'], r['operator'], r['value'])
                    if key not in seen:
                        seen.add(key)
                        unique_subrules.append(r)
                main_rule['sub_rules'] = unique_subrules
            rules.append(main_rule)
        self.logger.log(f"Successfully parsed {len(rules)} valid rules")
        if not rules:
            self.logger.log("LLM规则解析失败，未能提取到任何有效规则，请检查LLM输出格式！")
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
        """支持嵌套规则结构的解析方法，增强对 LLM 输出格式的健壮性，支持BEGIN_TREE/END_TREE壳子"""
        self.logger.log(f"Raw LLM response (first 500 chars): {str(response)[:500]}")
        import re
        rules = []
        rule_map = {}
        terminal_rules = []
        rule_groups = {}
        group_id = 1

        if not response:
            self.logger.log("Error: Empty LLM response received")
            return []
        response = str(response).strip()
        if not response:
            self.logger.log("Error: Response is empty after conversion")
            return []

        # 1. 提取BEGIN_TREE/END_TREE壳子内容
        tree_block = None
        begin_pat = re.compile(r'BEGIN[_ ]?TREE', re.IGNORECASE)
        end_pat = re.compile(r'END[_ ]?TREE', re.IGNORECASE)
        begin_idx = None
        end_idx = None
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

        # 2. 解析tree_block内容
        lines = tree_block.split('\n')
        line_count = len(lines)
        rule_prefix_count = len(re.findall(r'^rule\s+\d+:', tree_block, re.MULTILINE | re.IGNORECASE))
        self.logger.log(f"Response diagnostics - Lines: {line_count}, Rule prefixes found: {rule_prefix_count}")

        # 支持 (N) 或 Rule N: 开头的规则编号
        rule_pattern = re.compile(
            r'^\s*(?:\((\d+)\)|rule\s*(\d+))?\s*:?'  # 支持 (N) 或 Rule N: 开头
            r'\s*if\s+(.+?)'
            r'\s*then\s+(.+?)'
            r'(?:\s*else\s+(.+?))?'
            r'\s*$',
            re.IGNORECASE
        )
        fallback_pattern = re.compile(
            r'^\s*if\s+(.+?)\s*then\s+(.+?)\s*else\s+(.+?)\s*$',
            re.IGNORECASE
        )

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.lower().startswith('note:'):
                continue  # 跳过空行和注释
            match = rule_pattern.match(line)
            if match:
                rule_id = match.group(1) or match.group(2)
                condition = match.group(3)
                consequence = match.group(4)
                else_consequence = match.group(5)
                if not isinstance(condition, str) or not isinstance(consequence, str):
                    self.logger.log(f"Invalid rule type - condition: {type(condition)}, consequence: {type(consequence)}")
                    continue
                condition = condition.strip()
                consequence = consequence.strip()
                if not condition or not consequence:
                    self.logger.log(f"Empty rule condition or consequence")
                    continue
                rule_map[rule_id or str(len(rule_map)+1)] = {
                    'condition': condition,
                    'consequence': consequence,
                    'is_terminal': False
                }
                if else_consequence:
                    rule_map[f"{rule_id or str(len(rule_map)+1)}_else"] = {
                        'condition': f"NOT ({condition})",
                        'consequence': else_consequence.strip(),
                        'is_terminal': False
                    }
            else:
                fallback = fallback_pattern.match(line)
                if fallback:
                    condition, consequence, else_consequence = fallback.groups()
                    rule_map[str(len(rule_map)+1)] = {
                        'condition': condition.strip(),
                        'consequence': consequence.strip(),
                        'is_terminal': False
                    }
                    rule_map[f"{len(rule_map)+1}_else"] = {
                        'condition': f"NOT ({condition.strip()})",
                        'consequence': else_consequence.strip(),
                        'is_terminal': False
                    }
                else:
                    self.logger.log(f"Skipped non-rule line: {line}")

        # 终端规则识别
        label_names = [label.name.lower() for label in self.meta.labels]
        for rule_id, rule in rule_map.items():
            if rule['consequence'].lower() in label_names:
                rule['is_terminal'] = True
                terminal_rules.append(rule_id)

        # 构建规则链（保持原有逻辑）
        for terminal_rule_id in terminal_rules:
            current_rule_id = terminal_rule_id
            rule_chain = []
            while current_rule_id in rule_map:
                current_rule = rule_map[current_rule_id]
                rule_chain.insert(0, {
                    'condition': current_rule['condition'],
                    'label': current_rule['consequence'] if current_rule['is_terminal'] else None
                })
                # 查找前驱规则
                current_rule_id = None
                for rule_id, rule in rule_map.items():
                    if rule['consequence'] == current_rule_id:
                        current_rule_id = rule_id
                        break
            if rule_chain:
                # 支持 AND 多条件
                condition_pattern = re.compile(
                    r'^\s*([\w\s]+?)\s*(<=|>=|<|>|=)\s*([\d.]+|\".+?\"|\'.+?\'|\w+)\s*$'
                )
                parsed_conditions = []
                valid = True
                for step in rule_chain:
                    for cond in step['condition'].split(' AND '):
                        cond = cond.strip()
                        cond_match = condition_pattern.match(cond)
                        if not cond_match:
                            self.logger.log(f"Invalid condition format: {cond}")
                            valid = False
                            break
                        feature, operator, value = cond_match.groups()
                        feature = ' '.join(feature.split())
                        matched_feature = next((f for f in self.feature_names if f.lower() == feature.lower()), None)
                        if not matched_feature:
                            self.logger.log(f"Unknown feature '{feature}', skipping condition")
                            valid = False
                            break
                        try:
                            value = value.strip('\'\"')
                            numeric_value = float(value) if '.' in value else int(value)
                            value = numeric_value
                        except:
                            pass
                        parsed_conditions.append({
                            'feature': matched_feature,
                            'operator': operator,
                            'value': value,
                            'label': step['label']
                        })
                    if not valid:
                        break
                if valid and parsed_conditions:
                    main_condition = parsed_conditions[0]
                    rule_label = rule_chain[-1]['label'] or main_condition['label']
                    rule_entry = {
                        'condition': main_condition['feature'] + ' ' + main_condition['operator'] + ' ' + str(main_condition['value']),
                        'feature': main_condition['feature'],
                        'operator': main_condition['operator'],
                        'value': main_condition['value'],
                        'label': rule_label,
                        'sub_rules': []
                    }
                    if len(parsed_conditions) > 1:
                        rule_entry['sub_rules'] = [
                            {
                                'condition': c['feature'] + ' ' + c['operator'] + ' ' + str(c['value']),
                                'feature': c['feature'],
                                'operator': c['operator'],
                                'value': c['value'],
                                'label': rule_label
                            } for c in parsed_conditions[1:]
                        ]
                    group_key = f"{rule_entry['feature']}_{rule_entry['operator']}_{rule_entry['value']}"
                    if group_key not in rule_groups:
                        rule_groups[group_key] = {
                            'main_rule': rule_entry,
                            'group_id': group_id
                        }
                        group_id += 1
                        self.logger.log(f"Added new rule group for feature: {rule_entry['feature']}")
                    else:
                        existing = rule_groups[group_key]['main_rule']
                        existing['sub_rules'].extend(rule_entry['sub_rules'])
        for group in rule_groups.values():
            main_rule = group['main_rule']
            if main_rule['sub_rules']:
                seen = set()
                unique_subrules = []
                for r in main_rule['sub_rules']:
                    key = (r['feature'], r['operator'], r['value'])
                    if key not in seen:
                        seen.add(key)
                        unique_subrules.append(r)
                main_rule['sub_rules'] = unique_subrules
            rules.append(main_rule)
        self.logger.log(f"Successfully parsed {len(rules)} valid rules")
        if not rules:
            self.logger.log("LLM规则解析失败，未能提取到任何有效规则，请检查LLM输出格式！")
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