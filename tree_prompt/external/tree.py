from sklearn.preprocessing import OneHotEncoder
import sklearn.tree
import sklearn.ensemble
import numpy as np
import jinja2
import tree_prompt.logger as logger
import re
from xgboost import XGBClassifier
from typing import TYPE_CHECKING
from ..runner import Runner

from .. import dataset
from ..dataset import DatasetMeta


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
    
class LLMDecisionTree(DecisionTree):
    def __init__(self, meta, model_name, temperature, max_depth):
        super().__init__(meta)
        self.model_name = model_name
        self.temperature = temperature
        if not hasattr(meta, 'feature_names'):
            raise AttributeError("meta对象必须实现feature_names方法")
        self.max_depth = max_depth
        self._rules = None
        self.x_train = None  # 存储训练数据
        self.y_train = None  # 存储训练标签
        if not hasattr(meta, 'feature_names'):
            raise AttributeError("meta对象必须实现feature_names方法")
        # 初始化基础模板上下文
        self.base_context = {
        "meta": meta,
        "max_depth": self.max_depth,
        "features": self.meta.feature_names(),
        "labels": self.meta.label_names(),
        "num_features": len(self.meta.feature_names()),
        "num_labels": len(self.meta.label_names()),
       
            
    }
        self.base_context["feature_descriptions"] = self._generate_feature_descriptions()

    def build_tree(self, template: jinja2.Template, runner: Runner, x_train=None, y_train=None, template_params=None) -> bool:
        """构建决策树"""
        try:
            # 存储训练数据
            if x_train is not None:
                self.x_train = x_train
            if y_train is not None:
                self.y_train = y_train

            # 1. 准备模板上下文（关键修改点）
            full_context = {
                **self.base_context,
                "examples": self._generate_sample_data(),
                **(template_params or {})  # 合并外部传入参数
            }

            # 调试输出上下文内容
            logger.log(f"Template Context: {list(full_context.keys())}")
            if "num_features" not in full_context:
                logger.warn("num_features not in template context!")

            # 2. 增强的模板验证
            self._validate_context(
                full_context,
                required_vars=["features", "labels", "num_features"]  # 新增num_features校验
            )

            # 3. 安全渲染模板
            try:
                prompt = template.render(**full_context)
            except jinja2.UndefinedError as e:
                logger.error(f"Missing template variable: {e}")
                return False
                
            logger.log(f"[LLM PROMPT]\n{prompt[:500]}...")  # 限制输出长度

            # 4. 带重试机制的LLM调用
            max_attempts = 3
            for attempt in range(max_attempts):
                responses = list(runner.run([prompt]))
                if responses and responses[0]:
                    llm_response = responses[0]
                    if isinstance(llm_response, list):
                        llm_response = llm_response[0]
                    break
                logger.warn(f"Attempt {attempt+1} failed, retrying...")
            else:
                logger.error("All LLM attempts failed")
                return False

            # 5. 强化的规则解析
            try:
                self._rules = self._parse_response(llm_response)
            except Exception as e:
                logger.error(f"Rule parsing failed: {str(e)}")
                return False
                
            logger.log(f"[GENERATED RULES]\n{self._rules}")
            return True

        except jinja2.TemplateError as e:
            logger.error(f"Template Error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected Error: {str(e)}", exc_info=True)
            return False

    def _generate_sample_data(self):
        """生成示例数据（使用存储的训练数据）"""
        if self.x_train is None or len(self.x_train) == 0:
            return []
        return [dict(zip(self.base_context["features"], sample)) 
                for sample in self.x_train[:3]]
    
    def _generate_feature_descriptions(self):
        """直接通过meta对象获取特征信息"""
        return "\n".join([
            f"{feat.name}: {feat.type}" 
            for feat in self.meta.features  # 不再依赖base_context
        ])

    def _validate_context(self, context, required_vars):
        if "feature_descriptions" not in context:
            context["feature_descriptions"] = "No feature descriptions available"
        for var in required_vars:
            if var not in context:
                raise ValueError(f"Context missing required variable {var}")

    

    def _parse_response(self, response: str) -> str:
        """解析LLM响应生成决策树规则"""
        # 简化解析逻辑，实际需要根据响应格式定制
        lines = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith("if ") or line.startswith("return "):
                lines.append(line)
        return '\n'.join(lines) if lines else None

    def predict(self, x_test, export_rules=False):  # 移除非必要的x_train, y_train参数
        """动态预测（需要先调用build_tree）"""
        if not self._rules:
            raise RuntimeError("Decision tree not built. Call build_tree() first")
        
        # 新增预测逻辑
        predictions = []
        for sample in x_test:
            # 将样本转换为特征字典
            features = dict(zip(self.base_context["features"], sample))
            # 使用规则遍历进行预测
            print("Feature Descriptions:", self.base_context["feature_descriptions"])
            pred_label = self._dynamic_traverse(features, self._rules.split('\n'))
            predictions.append(pred_label)
        
        return predictions, self._rules if export_rules else None

    def _dynamic_traverse(self, features: dict, rules: list) -> int:
        """动态遍历解析生成的决策树规则（改进版）"""
        import re
        
        # 预处理规则：移除空行和注释
        cleaned_rules = []
        for line in rules:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cleaned_rules.append(line)
        
        stack = [ (0, 0) ]  # (当前缩进等级, 规则索引)
        default_label = 0  # 默认返回第一个标签
        
        while stack:
            current_indent, idx = stack.pop()
            
            # 终止条件
            if idx >= len(cleaned_rules):
                continue
                
            # 解析当前行
            line = cleaned_rules[idx]
            
            # 计算实际缩进（按4空格为一级）
            indent_level = (len(line) - len(line.lstrip())) // 4
            
            # 跳过不符合缩进层级的情况
            if indent_level < current_indent:
                continue
                
            # 解析逻辑
            try:
                # Case 1: 条件判断行 (if语句)
                if line.lstrip().startswith("if "):
                    # 使用正则表达式提取特征和阈值
                    pattern = r"if\s+([\w\s]+)\s*([<>]=?)\s*([\d\.]+)\s*:"
                    match = re.match(pattern, line.strip(), re.IGNORECASE)
                    if not match:
                        continue
                    
                    feat_name = match.group(1).strip()
                    operator = match.group(2).strip()
                    threshold = float(match.group(3))
                    
                    # 获取实际特征值
                    feat_value = features.get(feat_name, None)
                    if feat_value is None:
                        # 特征不存在时跳过该条件
                        continue
                    
                    # 执行条件判断
                    condition_met = False
                    if operator == "<=":
                        condition_met = (feat_value <= threshold)
                    elif operator == "<":
                        condition_met = (feat_value < threshold)
                    elif operator == ">=":
                        condition_met = (feat_value >= threshold)
                    elif operator == ">":
                        condition_met = (feat_value > threshold)
                    else:
                        continue  # 无效运算符
                    
                    # 根据判断结果跳转
                    if condition_met:
                        # 进入下一级（缩进+1）
                        stack.append( (indent_level+1, idx+1) )
                    else:
                        # 寻找同级else分支或跳过
                        next_idx = idx + 1
                        while next_idx < len(cleaned_rules):
                            next_line = cleaned_rules[next_idx]
                            next_indent = (len(next_line) - len(next_line.lstrip())) // 4
                            if next_indent == indent_level:
                                if "else:" in next_line.lower():
                                    stack.append( (indent_level+1, next_idx+1) )
                                    break
                                else:
                                    next_idx += 1
                            else:
                                next_idx += 1
                        else:
                            # 没有找到else分支则继续同级
                            stack.append( (indent_level, next_idx) )
                
                # Case 2: 返回标签行 (return语句)
                elif line.lstrip().startswith(("return ", "class: ")):
                    label_pattern = r"(?:return|class:?)\s+([\w\s]+)"
                    match = re.search(label_pattern, line, re.IGNORECASE)
                    if match:
                        pred_label = match.group(1).strip()
                        try:
                            return self.meta.label_names().index(pred_label)
                        except ValueError:
                            # 记录未知标签警告
                            logger.warning(f"Unknown label {pred_label}, using default")
                            return default_label
                    
            except Exception as e:
                logger.error(f"Rule parsing error at line {idx}: {line}\n{str(e)}")
                continue
                
            # 默认情况：继续执行下一条规则
            stack.append( (current_indent, idx+1) )
        
        # 未找到有效返回时返回默认标签
        return default_label
        

    def get_template_context(self):
        """为模板提供动态上下文"""
        return {
            "features": [feat.name for feat in self.meta.features],  # 获取特征名称
            "labels": [label.name for label in self.meta.labels],  # 获取标签名称
            "max_depth": self.max_depth,  # Max depth for the tree
            "num_features": len(self.meta.features),  # 特征数量
            "num_labels": len(self.meta.labels)  # 标签数量
        }

    def _get_feature_ranges(self):
        """获取每个特征的数值范围"""
        ranges = {}
        for i, feat in enumerate(self.meta.features):
            if feat.type == 'numerical':  # 根据特征类型筛选数值型特征
                values = self.x_train[:, i]
                ranges[feat.name] = {  # 使用 feat.name 作为特征名称
                    'min': np.min(values),
                    'max': np.max(values),
                    'mean': np.mean(values)
                }
        return ranges

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
