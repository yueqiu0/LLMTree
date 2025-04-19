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

class DecisionNode:
    def __init__(self, feature=None, op=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # 特征名
        self.op = op            # 操作符（'<=', 'in', 等）
        self.threshold = threshold  # 阈值/类别列表
        self.left = left        # 满足条件的子树
        self.right = right      # 不满足时的下一规则
        self.value = value      # 叶节点的预测值

class LLMDecisionTree:
    def __init__(self, meta, max_depth, runner=None):
        self.meta = meta
        self.max_depth = max_depth
        self.runner = runner
        self.rules = []
        self.decision_tree = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> list[str]:
        """生成决策树规则主流程"""
        try:
            # 数据验证
            if x_train is None or y_train is None:
                Logger.log("训练数据为空")
                return []

            # 生成提示词（从dataset.py导入）
            from ..dataset import generate_decision_tree_prompt
            prompt = generate_decision_tree_prompt(self.meta, x_train, y_train, self.max_depth)
            Logger.log(f"生成的提示词长度: {len(prompt)}字符")

            # 发送请求并处理响应
            responses = list(self.runner.run([prompt]))
            self.rules = self._parse_llm_response(responses[0])
            # 构建决策树
            self.decision_tree = self._parse_llm_response(responses[0])
            
            # 验证规则完备性
            self._validate_training_coverage(x_train)

            if not self._validate_rules():
                Logger.log("部分规则未通过验证")
            
            return self.rules

        except Exception as e:
            
            return []
    def _parse_llm_response(self, response: str) -> DecisionNode:
        root = None
        current_node = None
        
        for line in response.split('\n'):
            line = line.strip()
            if not line.startswith("if ") or " then " not in line:
                continue
            
            condition, prediction = line[3:].split(" then ", 1)
            feature, op, value = self._parse_condition(condition.strip())
            
            if not feature:
                continue
                
            new_node = DecisionNode(
                feature=feature,
                op=op,
                threshold=self._parse_threshold(op, value),
                left=DecisionNode(value=prediction.strip()),
                right=None
            )
            
            if root is None:
                root = current_node = new_node
            else:
                current_node.right = new_node
                current_node = new_node
        
        return root
    
    def _validate_training_coverage(self, x_train):
        """检查训练集是否被完全覆盖"""
        for i, sample in enumerate(x_train):
            try:
                self._predict_single(sample)
            except ValueError as e:
                raise ValueError(
                    f"训练样本#{i} 未被任何规则覆盖:\n"
                    f"样本: {sample}\n"
                    "建议解决方案:\n"
                    "1. 增加 max_depth 参数\n"
                    "2. 检查特征工程是否合理\n"
                    "3. 确保LLM生成的规则完整"
                ) from e
        
    def _predict_single(self, sample) -> str:
        """内部使用的单样本预测方法"""
        node = self.decision_tree
        while node:
            if node.value is not None:
                return node.value
                
            feat_idx = self.meta.features.index(node.feature)
            sample_value = sample[feat_idx]
            
            if self._evaluate_condition(sample_value, node):
                node = node.left
            else:
                node = node.right
        raise ValueError("No matching rule")
    
    def _evaluate_condition(self, sample_value, node: DecisionNode) -> bool:
            """标准化条件判断逻辑"""
            try:
                if node.op in ['<=', '<', '>', '>=']:
                    sample_val = float(sample_value)
                    thresh = float(node.threshold)
                    return {
                        '<=': lambda: sample_val <= thresh,
                        '<':  lambda: sample_val < thresh,
                        '>=': lambda: sample_val >= thresh,
                        '>':  lambda: sample_val > thresh
                    }[node.op]()
                elif node.op == 'in':
                    return str(sample_value) in node.threshold
                return False
            except (ValueError, TypeError):
                raise ValueError(
                    f"类型错误: 特征 {node.feature} 的值 {sample_value} "
                    f"无法与阈值 {node.threshold} 进行比较"
                )
    def predict(self, x_test) -> np.ndarray:
        """强制匹配模式预测"""
        if self.decision_tree is None:
            raise ValueError("请先调用 fit() 方法训练模型")
        
        predictions = []
        feature_names = [f.name for f in self.meta.features]
        
        for sample in x_test:
            node = self.decision_tree
            rule_path = []  # 记录规则路径用于报错
            
            while node:
                rule_path.append(
                    f"{node.feature} {node.op} {node.threshold}"
                )
                
                # 叶节点处理
                if node.value is not None:
                    predictions.append(node.value)
                    break
                    
                # 获取特征值
                try:
                    feat_idx = feature_names.index(node.feature)
                    sample_value = sample[feat_idx]
                except ValueError:
                    raise ValueError(f"特征 '{node.feature}' 不在训练特征集中")
                
                # 条件判断
                if self._evaluate_condition(sample_value, node):
                    node = node.left
                else:
                    node = node.right
            
            # 未匹配任何规则
            if node is None:
                raise ValueError(
                    "样本不匹配任何规则:\n"
                    f"样本特征值: {dict(zip(feature_names, sample))}\n"
                    f"尝试的规则路径: {' -> '.join(rule_path)}"
                )
        
        return np.array(predictions)
    def _parse_threshold(self, op: str, value: str):
        """根据操作符解析阈值"""
        if op in ['<=', '>=', '<', '>']:
            return float(value)
        elif op == 'in':
            return [v.strip(" '\"") for v in value.strip('[]').split(',')]
        return value
    def _parse_llm_response(self, response: str) -> list[str]:
        """解析LLM响应（保留核心解析逻辑）"""
        valid_rules = []
        feature_names = {f.name for f in self.meta.features}

        for line in response.split('\n'):
            line = line.strip()
            if not line.startswith("if ") or " then " not in line:
                continue

            condition = line.split(" then ")[0][3:]
            if not any(fname in condition for fname in feature_names):
                Logger.log(f"忽略无效规则: 未使用已知特征 - {line}")
                continue

            valid_ops = ['<=', '>=', '<', '>', ' in ', ' not in ']
            if not any(op in condition for op in valid_ops):
                Logger.log(f"忽略无效规则: 无效操作符 - {line}")
                continue

            valid_rules.append(line)

        return list(dict.fromkeys(valid_rules))
    def _matches_rule(self, sample, rule):
        """检查样本是否匹配规则"""
        # 实现规则匹配逻辑
        pass
    
    def _validate_rules(self) -> bool:
        """规则验证（保留基础验证逻辑）"""
        if not self.rules:
            return False

        predicted_labels = {r.split(" then ")[1].strip() for r in self.rules}
        required_labels = {l.name for l in self.meta.labels}
        return predicted_labels.issuperset(required_labels)

    def get_rules(self):
        """获取生成的决策规则"""
        return self.rules 

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
