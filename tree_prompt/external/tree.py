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
from ..logger import Logger  

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



class LLMDecisionTree:
    def __init__(self, meta, max_depth, runner=None, rules=None):
        self.meta = meta
        self.max_depth = max_depth
        self.runner = runner
        
        # 存储原始规则文本（用于展示/调试）
        self.raw_rules = rules if rules else []
        
        # 存储结构化解析后的规则（用于实际预测）
        self.parsed_rules = self._parse_rules(self.raw_rules)
        
        # 初始化日志对象
        self.logger = Logger()

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> list[str]:
        """生成决策树规则主流程"""
        try:
            # 数据验证
            if x_train is None or y_train is None:
                self.logger.log("训练数据为空", level="ERROR")
                self.parsed_rules = []
                return []

            # 生成提示词（从dataset.py导入）
            from ..dataset import generate_decision_tree_prompt
            prompt = generate_decision_tree_prompt(
                self.meta, x_train, y_train, self.max_depth
            )
            self.logger.log(f"生成的提示词长度: {len(prompt)}字符")

            # 发送请求并处理响应
            responses = list(self.runner.run([prompt]))
            if not responses:
                self.logger.log("未收到LLM响应", level="WARN")
                return []
                
            # 处理原始规则
            new_raw_rules = self._parse_llm_response(responses[0])
            self.raw_rules.extend(new_raw_rules)  # 保留历史规则
            
            # 解析为结构化规则
            new_parsed_rules = self._parse_rules(new_raw_rules)
            self.parsed_rules.extend(new_parsed_rules)
            
            # 规则验证
            if not self._validate_rules():
                self.logger.log("部分规则未通过最终验证", level="WARN")
                
            self.logger.log(
                f"规则更新完成: 新增原始规则 {len(new_raw_rules)} 条 -> "
                f"有效结构化规则 {len(new_parsed_rules)} 条"
            )
            return new_raw_rules

        except Exception as e:
            self.logger.log(f"训练过程中发生异常: {str(e)}", level="ERROR")
            self.parsed_rules = []
            return []

    def _parse_llm_response(self, response: str) -> list[str]:
        """解析LLM响应为原始规则文本"""
        valid_rules = []
        feature_names = {f.name for f in self.meta.features}

        for line in response.split('\n'):
            line = line.strip()
            if not line.startswith("if ") or " then " not in line:
                continue

            condition = line.split(" then ")[0][3:]
            if not any(fname in condition for fname in feature_names):
                self.logger.log(f"忽略无效规则: 未使用已知特征 - {line}")
                continue

            valid_ops = ['<=', '>=', '<', '>', ' in ', ' not in ']
            if not any(op in condition for op in valid_ops):
                self.logger.log(f"忽略无效规则: 无效操作符 - {line}")
                continue

            valid_rules.append(line)

        return list(dict.fromkeys(valid_rules))

    def _parse_rules(self, rule_texts: list[str]) -> list[dict]:
        """将原始规则文本解析为结构化规则"""
        parsed_rules = []
        feature_names = {f.name for f in self.meta.features}
        
        for rule in rule_texts:
            if not rule.startswith("if ") or " then " not in rule:
                continue
            
            try:
                # 分割条件和预测结果
                condition, prediction = rule[3:].split(" then ", 1)
                
                # 解析条件
                feature, op, value = self._parse_condition(condition.strip())
                if feature not in feature_names:
                    continue
                
                # 解析预测值
                pred_label = self._parse_prediction(prediction.strip())
                
                parsed_rules.append({
                    "feature": feature,
                    "operator": op,
                    "threshold": self._parse_value(op, value),
                    "prediction": pred_label,
                    "raw_rule": rule  # 保留原始文本用于调试
                })
                
            except Exception as e:
                self.logger.log(f"规则解析失败: {rule} - {str(e)}")
                continue
        
        return parsed_rules

    def _parse_condition(self, condition: str) -> tuple:
        """解析条件语句为 (特征名, 操作符, 值)"""
        operator_priority = [' in ', ' not in ', '<=', '>=', '<', '>', '==', '!=']
        for op in operator_priority:
            if op in condition:
                parts = condition.split(op)
                if len(parts) == 2:
                    return parts[0].strip(), op, parts[1].strip()
        return None, None, None

    def _parse_value(self, op: str, value_str: str):
        """根据操作符解析阈值"""
        try:
            # 处理数值型特征
            if op in ['<=', '>=', '<', '>', '==', '!=']:
                return float(value_str)
                
            # 处理类别型特征
            elif op in [' in ', ' not in ']:
                return [v.strip(" '\"") for v in value_str.strip('[]').split(',')]
                
            return value_str.strip(" '\"")
        except:
            return value_str.strip(" '\"")

    def _parse_prediction(self, prediction: str) -> str:
        """解析预测结果，确保符合元数据定义"""
        valid_labels = {label.name for label in self.meta.labels}
        clean_pred = prediction.strip(" '\"")
        return clean_pred if clean_pred in valid_labels else ""

    def predict(self, x_test) -> np.ndarray:
        """应用结构化规则进行预测"""
        if not self.parsed_rules:
            raise ValueError(
                "未找到有效决策规则，请先执行fit()训练或通过构造函数传入规则"
            )
            
        feature_names = [f.name for f in self.meta.features]
        predictions = []
        
        for sample in x_test:
            sample_pred = None
            for rule in self.parsed_rules:
                try:
                    feat_idx = feature_names.index(rule["feature"])
                    sample_value = sample[feat_idx]
                    
                    if self._check_condition(sample_value, rule):
                        sample_pred = rule["prediction"]
                        break
                except (ValueError, IndexError) as e:
                    self.logger.log(f"规则应用异常: {rule['raw_rule']} - {str(e)}")
                    continue
                    
            if sample_pred is None:
                raise ValueError(
                    f"样本未匹配任何规则: {sample}\n"
                    f"可用规则数量: {len(self.parsed_rules)}\n"
                    f"示例规则: {self.parsed_rules[0]['raw_rule'] if self.parsed_rules else '无'}"
                )
                
            predictions.append(sample_pred)
        
        return np.array(predictions)

    def _check_condition(self, sample_value, rule: dict) -> bool:
        """检查样本值是否满足规则条件"""
        op = rule["operator"]
        threshold = rule["threshold"]
        
        try:
            # 数值型比较
            if op in ['<=', '<', '>', '>=', '==', '!=']:
                sample_val = float(sample_value)
                thresh = float(threshold)
                return {
                    '<=': sample_val <= thresh,
                    '<':  sample_val < thresh,
                    '>=': sample_val >= thresh,
                    '>':  sample_val > thresh,
                    '==': sample_val == thresh,
                    '!=': sample_val != thresh
                }.get(op, False)
                
            # 类别型判断
            elif op == ' in ':
                return str(sample_value) in threshold
                
            elif op == ' not in ':
                return str(sample_value) not in threshold
                
            return False
            
        except (TypeError, ValueError):
            # 类型不匹配时的兜底处理
            return str(sample_value) == str(threshold)

    def _validate_rules(self) -> bool:
        """最终规则有效性验证"""
        if not self.parsed_rules:
            return False

        # 检查预测标签有效性
        valid_labels = {label.name for label in self.meta.labels}
        for rule in self.parsed_rules:
            if rule["prediction"] not in valid_labels:
                self.logger.log(f"无效预测标签: {rule['prediction']}")
                return False
                
        return True

    def get_rules(self, raw_format: bool = False):
        """获取规则
        :param raw_format: 是否返回原始文本格式
        """
        return self.raw_rules if raw_format else self.parsed_rules

    def print_debug_info(self):
        """打印调试信息"""
        self.logger.log("=== 决策树调试信息 ===")
        self.logger.log(f"原始规则数量: {len(self.raw_rules)}")
        self.logger.log(f"有效结构化规则: {len(self.parsed_rules)}")
        if self.parsed_rules:
            sample_rule = self.parsed_rules[0]
            self.logger.log(
                "示例规则结构:\n"
                f"- 特征: {sample_rule['feature']}\n"
                f"- 操作符: {sample_rule['operator']}\n"
                f"- 阈值: {sample_rule['threshold']}\n"
                f"- 预测: {sample_rule['prediction']}\n"
                f"- 原始文本: {sample_rule['raw_rule']}"
            )
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
