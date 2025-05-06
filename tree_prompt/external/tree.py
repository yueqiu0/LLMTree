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



# === Step 1: TreeNode类实现 ===
class TreeNode:
    def __init__(self, depth, x, y, parent=None):
        self.depth = depth
        self.x = x
        self.y = y
        self.parent = parent
        self.candidates = []  # n个候选规则
        self.votes = {}       # {rule_str: 得票数}
        self.selected_rule = None  # 最终选定规则
        self.children = []    # 子节点
        self.is_leaf = False
        self.leaf_class = None
        # 新增：路径条件和特征使用追踪
        if parent is not None:
            self.path_conditions = list(parent.path_conditions)
            self.used_features = set(parent.used_features)
        else:
            self.path_conditions = []
            self.used_features = set()

    def add_condition(self, feature, condition):
        """记录当前节点的分裂条件和已用特征"""
        self.path_conditions.append(f"{feature} {condition}")
        self.used_features.add(feature)

    def __repr__(self):
        return f"TreeNode(depth={self.depth}, rule={self.selected_rule}, is_leaf={self.is_leaf}, leaf_class={self.leaf_class})"



# === Step 2: ToTDecisionTree类重构初始化，支持树结构和参数 ===
class ToTDecisionTree(DecisionTree):
    def flat_rules_to_graphviz(self, flat_rules, graph_name="FlatRulesTree"):
        """
        将平面化规则链列表（如 export_flat_rules 输出）转为 dot 格式字符串。
        每条规则为一个节点，节点 label 为完整规则，节点编号顺序连接。
        """
        lines = []
        lines.append(f'digraph {graph_name} {{')
        for idx, rule in enumerate(flat_rules):
            lines.append(f'node{idx} [label="{rule}"]')
        for idx in range(len(flat_rules) - 1):
            lines.append(f'node{idx} -> node{idx+1}')
        lines.append('}')
        return "\n".join(lines)

    def to_graphviz_source(self, features=None, classes=None, graph_name="ToTDecisionTree"):
        """
        生成严格符合如下格式的dot源码：
        node0 [label="ROOT"]
        node1 [label="If xxx ,[node]"]
        node0 -> node1
        node2 [label="If yyy ,THEN label"]
        node1 -> node2
        ...
        """
        lines = []
        node_id_counter = [0]
        node_ids = {}

        def get_node_id(node):
            if node not in node_ids:
                node_ids[node] = f"node{node_id_counter[0]}"
                node_id_counter[0] += 1
            return node_ids[node]

        def dfs(node, parent=None, cond_text=None):
            nid = get_node_id(node)
            if node.depth == 0:
                lines.append(f'{nid} ["ROOT"]')
            elif node.is_leaf:
                if parent and cond_text:
                    label = f'If {cond_text} ,THEN {node.leaf_class}'
                else:
                    label = f'THEN {node.leaf_class}'
                lines.append(f'{nid} ["{label}"]')
            else:
                if cond_text:
                    label = f'If {cond_text} ,[node]'
                else:
                    label = f'If [unknown] ,[node]'
                lines.append(f'{nid} ["{label}"]')
            if parent:
                pid = get_node_id(parent)
                lines.append(f'{pid} -> {nid}')
            if not node.is_leaf and node.selected_rule and isinstance(node.selected_rule, dict):
                conds = node.selected_rule.get("condition", "")
                cond_lines = [l.strip() for l in conds.splitlines() if l.strip()]
                for idx, child in enumerate(node.children):
                    cond_line = cond_lines[idx] if idx < len(cond_lines) else ""
                    dfs(child, node, cond_line)
            elif not node.is_leaf:
                for child in node.children:
                    dfs(child, node, "")

        lines.append(f'digraph {graph_name} {{')
        dfs(self.root)
        lines.append('}')
        return "\n".join(lines)
    def export_flat_rules(self, features=None) -> list[str]:
        """
        导出所有从根到叶的规则，每条规则为 IF cond1 AND cond2 ... THEN label，
        其中 AND 的数量为 max_depth-2（即条件数为 max_depth-1），只输出这些深度的规则。
        修正：去重条件，且只保留最后一个 THEN label。
        """
        rules = []
        max_and = self.max_depth - 2 if hasattr(self, 'max_depth') else 1
        def clean_cond(cond):
            cond = cond.strip()
            if cond.startswith("IF "):
                cond = cond[3:]
            if 'THEN [NODE]' in cond or 'THEN [node]' in cond or 'THEN [Node]' in cond:
                cond = cond.split('THEN')[0].strip()
            if cond.endswith(','):
                cond = cond[:-1].strip()
            return cond
        def dfs(node, path_conds):
            if node is None:
                return
            # 叶子节点直接输出完整路径
            if node.is_leaf:
                if path_conds:
                    rule = " AND ".join(path_conds)
                    rules.append(f"IF {rule} THEN {node.leaf_class}")
                return
            # 非叶子节点，递归所有分支
            if node.selected_rule and isinstance(node.selected_rule, dict):
                conds = node.selected_rule.get("condition", "")
                cond_lines = [l.strip() for l in conds.splitlines() if l.strip()]
                for idx, child in enumerate(node.children):
                    if idx < len(cond_lines):
                        cond = clean_cond(cond_lines[idx])
                        dfs(child, path_conds + [cond])
                    else:
                        dfs(child, path_conds)
            else:
                for child in node.children:
                    dfs(child, path_conds)
        dfs(self.root, [])
        return rules

    def _zero_shot_llm_generate_rule(self, depth, parent_rule, current_path):
        """
        使用LLM和meta信息生成单条决策规则（零样本）。
        调用方式与basic.jinja批量推理部分保持一致。
        """
        prompt = self._build_rule_prompt(
            x=np.zeros((1, len(self.feature_names))),
            y=np.zeros(1),
            depth=depth,
            parent_rule=parent_rule,
            current_path=current_path
        )
        self.logger.log(f"[Zero-Shot LLM RuleGen] prompt:\n{prompt}")
        self.total_prompts += 1
        try:
            responses = next(self.runner.run([prompt]))
            response = responses[0] if isinstance(responses, list) and len(responses) > 0 else responses
            self.logger.log(f"[Zero-Shot LLM RuleGen] 原始回应:\n{response}")
            import json
            if isinstance(response, dict):
                self.logger.log(f"[Zero-Shot LLM RuleGen] 最终返回: {repr(response)}，类型: dict")
                return response
            if isinstance(response, str):
                try:
                    rule = json.loads(response)
                    if isinstance(rule, dict):
                        self.logger.log(f"[Zero-Shot LLM RuleGen] 最终返回: {repr(rule)}，类型: dict(json)")
                        return rule
                except Exception:
                    pass
                import re
                m = re.search(r'condition\s*:\s*(.+?)\s*label\s*:\s*(.+)', response, re.I)
                if m:
                    rule = {"condition": m.group(1).strip(), "label": m.group(2).strip()}
                    self.logger.log(f"[Zero-Shot LLM RuleGen] 最终返回: {repr(rule)}，类型: dict(re)")
                    return rule
                # 新增：如果是多行规则字符串，直接返回字符串
                if response.strip().startswith('IF'):
                    self.logger.log(f"[Zero-Shot LLM RuleGen] 最终返回: {repr(response)}，类型: str(plain)")
                    return {"condition": response.strip(), "label": ""}
        except Exception as e:
            self.logger.log(f"[Zero-Shot LLM RuleGen] 解析异常: {e}")
        self.logger.log(f"[Zero-Shot LLM RuleGen] 最终返回: None")
        return None



    def _zero_shot_llm_vote(self, candidates, depth, parent_rule):
        # Always include rich dataset/feature context
        context_prompt = generate_ToT_tree_prompt(self.meta, np.zeros((1, len(self.feature_names))), np.zeros(1), max_depth=depth).get("prompt", "")
        prompt = f"{context_prompt}\nYou are building a zero-shot decision tree. The following are candidate rules for depth {depth+1}:\n"
        for idx, rule in enumerate(candidates):
            prompt += f"Rule {idx+1}: {rule.get('condition', str(rule))} -> {rule.get('label', '')}\n"
        prompt += "\nPlease select the best rule. Output format: Rule N"
        self.logger.log(f"[Zero-Shot LLM Voting] prompt:\n{prompt}")
        self.total_prompts += 1
        try:
            response = next(self.runner.run(prompt))
            self.logger.log(f"[Zero-Shot LLM投票] 原始回应:\n{response}")
            import re
            m = re.search(r'Rule\s*(\d+)', str(response))
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(candidates):
                    return idx
        except Exception as e:
            self.logger.log(f"[Zero-Shot LLM投票] 解析异常: {e}")
        return 0




    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.root
        if node.is_leaf:
            print(f"{indent}Leaf: class={node.leaf_class}")
        else:
            print(f"{indent}Rule: {node.selected_rule['condition']} -> {node.selected_rule['label']}")
            for child in node.children:
                self.print_tree(child, indent + "  ")
    def __init__(self, meta, max_depth, runner, log_file,
                 candidate_rules_per_node=5, voting_rounds_per_node=3, top_k_rules=2, final_voting_rounds=3):
        super().__init__(meta)
        self.max_depth = max_depth  # 树最大深度
        self.runner = runner        # LLM runner
        self.log_file = log_file    # 日志文件路径
        self.candidate_rules_per_node = candidate_rules_per_node  # 每个节点生成规则数
        self.voting_rounds_per_node = voting_rounds_per_node      # 每个节点投票轮数
        self.top_k_rules = top_k_rules                            # 每个节点保留分支数
        self.final_voting_rounds = final_voting_rounds            # 全局最终投票轮数
        self.root = None            # 最终决策树根节点
        self.candidate_trees = []   # 所有候选树
        self.total_prompts = 0      # 总prompt计数器
        self.feature_names = [f.name for f in meta.features]
        self.rules = []  # 用于build_decision_tree递归规则收集
        log_file_obj = open(self.log_file, 'a', encoding='utf-8', buffering=1)
        self.logger = Logger(log_file_obj)
        self.logger.log(f"=== ToTDecisionTree initialized at {datetime.now().isoformat()} ===")


    def _build_rule_prompt(self, x, y, depth, current_path=None, parent_rule=None, applied_rules=None, num_examples=5):
        """
        构造用于生成规则的prompt（新版，支持自定义格式，集成新版prompt片段和参数传递）
        支持 current_path、applied_rules 等参数灵活传递。
        """
        # 兼容参数
        if current_path is None:
            current_path = "Root"
        if applied_rules is None:
            # 默认用 current_path 作为 applied_rules
            applied_rules = current_path
        # 生成新版prompt（dataset.py新版函数，支持全部上下文参数）
        prompt_dict = generate_ToT_tree_prompt(
            self.meta,
            x,
            y,
            max_depth=depth if depth is not None else self.max_depth,
            num_examples=num_examples,
            current_path=current_path,
            applied_rules=applied_rules
        )
        prompt = prompt_dict.get("prompt", "")
        return prompt
    def get_rules(self):
        """获取当前决策树的所有规则"""
        return self.rules

    def fit(self, x_train=None, y_train=None, with_llm=True):
        """
        只用LLM和meta信息生成决策树（零样本），不依赖训练数据。
        支持多层递归，每一层扫描所有[NODE]节点，current_path为完整规则链。
        修正：循环次数为 max_depth-1，最后一层自动生成叶子节点。
        """
        self.logger.log("=== [Zero-Shot] Start zero-shot decision tree generation (ToT enhanced) ===")
        self.total_prompts = 0
        max_depth = self.max_depth
        self.root = TreeNode(0, x=None, y=None, parent=None)
        # 第一层只有root，current_path为Root
        current_layer = [(self.root, "Root")]
        # 修正：采用while循环，确保所有[NODE]分支都能被递归扩展，直到到达max_depth-1层
        depth = 0
        while current_layer and depth < max_depth - 1:
            self.logger.log(f"[DEBUG][ToT] while-loop begin, depth={depth}, current_layer size={len(current_layer)}, nodes={[repr(n) for n, _ in current_layer]}")
            next_layer = []
            for node, current_path in current_layer:
                self.logger.log(f"[DEBUG][ToT] for-loop, depth={depth}, node={repr(node)}, current_path={current_path}")
                # 终止条件
                if node.depth >= max_depth - 1:
                    self.logger.log(f"[DEBUG][ToT] depth={depth}, node.depth={node.depth} >= max_depth-1, continue as leaf")
                    node.is_leaf = True
                    node.leaf_class = None
                    continue
                # 生成候选规则，数量由 candidate_rules_per_node 控制
                candidates = []
                for _ in range(self.candidate_rules_per_node):
                    rule = self._zero_shot_llm_generate_rule(node.depth, None, current_path)
                    self.logger.log(f"[DEBUG][ToT] depth={depth}, candidate rule raw: {repr(rule)}, type={type(rule)}")
                    if rule is not None:
                        candidates.append(rule)
                if not candidates:
                    self.logger.log(f"[DEBUG][ToT] depth={depth}, no candidates, node set as leaf, continue")
                    node.is_leaf = True
                    node.leaf_class = None
                    continue
                # 投票选出最佳规则
                best_idx = self._zero_shot_llm_vote(candidates, node.depth, None)
                # 打印本轮投票prompt和最佳分支
                self.logger.log(f"[DEBUG][ToT] depth={depth}, 投票prompt如下:\n{getattr(self, 'last_vote_prompt', '')}")
                best_rule = candidates[best_idx]
                self.logger.log(f"[DEBUG][ToT] depth={depth}, 最佳分支: {best_rule}")
                node.selected_rule = best_rule
                node.candidates = candidates
                node.votes = {i: 1 if i == best_idx else 0 for i in range(len(candidates))}
                node.children = []
                # 只扩展最佳分支
                cond = best_rule.get('condition', '')
                self.logger.log(f"[DEBUG][ToT] depth={depth}, cond(raw)={repr(cond)}, type={type(cond)}")
                if not isinstance(cond, str):
                    cond = str(cond)
                    self.logger.log(f"[DEBUG][ToT] depth={depth}, cond 强制转为str: {repr(cond)}")
                if '[NODE]' in cond or '[Node]' in cond or '[node]' in cond:
                    rules_lines = [l.strip() for l in cond.splitlines() if l.strip()]
                    self.logger.log(f"[DEBUG][ToT] depth={depth}, rules_lines: {rules_lines}")
                    for rule_line in rules_lines:
                        self.logger.log(f"[DEBUG][ToT] depth={depth}, rule_line: {repr(rule_line)}")
                        if '[NODE]' in rule_line or '[Node]' in rule_line or '[node]' in rule_line:
                            child_path = (current_path + " AND " + rule_line.replace('THEN [NODE]', '').replace('THEN [Node]', '').replace('THEN [node]', '').strip()).replace('Root AND', 'Root')
                            child = TreeNode(node.depth + 1, x=None, y=None, parent=node)
                            node.children.append(child)
                            next_layer.append((child, child_path))
                            self.logger.log(f"[DEBUG][ToT] 当前depth={depth}, 新child.depth={child.depth}, 发现[NODE]分支: {rule_line}, child_path={child_path}")
                        elif 'THEN' in rule_line:
                            label = rule_line.split('THEN')[-1].strip()
                            child = TreeNode(node.depth + 1, x=None, y=None, parent=node)
                            child.is_leaf = True
                            child.leaf_class = label
                            node.children.append(child)
                            self.logger.log(f"[DEBUG][ToT] depth={depth}, 发现叶子分支: {rule_line}, label={label}")
                else:
                    # 兼容只有一个规则的情况
                    if '[NODE]' in cond or '[Node]' in cond or '[node]' in cond:
                        self.logger.log(f"[DEBUG][ToT] depth={depth}, else-branch, 发现[NODE]分支: {cond}")
                        child_path = (current_path + " AND " + cond.replace('THEN [NODE]', '').replace('THEN [Node]', '').replace('THEN [node]', '').strip()).replace('Root AND', 'Root')
                        child = TreeNode(node.depth + 1, x=None, y=None, parent=node)
                        node.children.append(child)
                        next_layer.append((child, child_path))
                        self.logger.log(f"[DEBUG][ToT] 当前depth={depth}, 新child.depth={child.depth}, 发现[NODE]分支: {cond}, child_path={child_path}")
                    elif 'THEN' in cond:
                        self.logger.log(f"[DEBUG][ToT] depth={depth}, else-branch, 发现叶子分支: {cond}")
                        label = cond.split('THEN')[-1].strip()
                        child = TreeNode(node.depth + 1, x=None, y=None, parent=node)
                        child.is_leaf = True
                        child.leaf_class = label
                        node.children.append(child)
                        self.logger.log(f"[DEBUG][ToT] depth={depth}, 发现叶子分支: {cond}, label={label}")
            self.logger.log(f"[DEBUG][ToT] while-loop end, depth={depth}, next_layer size={len(next_layer)}")
            current_layer = next_layer
            self.logger.log(f"[DEBUG][ToT] 第{depth+1}轮生成树的循环结束，当前层节点数: {len(current_layer)}")
            depth += 1
        # 最后一层：所有 current_layer 节点直接生成叶子
        all_leaf = True
        for node, current_path in current_layer:
            if not node.is_leaf:
                node.is_leaf = True
                node.leaf_class = None
            if not node.is_leaf:
                all_leaf = False
        if not all_leaf:
            self.logger.log("[ERROR][ToT] 最后一层仍有非叶子节点，决策树生成异常，强制退出！")
            import sys
            sys.exit(1)
        # 打印加载后的平面决策树规则
        flat_rules = self.export_flat_rules(features=[f.name for f in self.meta.features] if hasattr(self.meta, 'features') else None)
        print("# === 加载后的决策树平面规则如下 ===")
        for rule in flat_rules:
            print(rule)
        # 打印加载后的平面dot结构源码
        print("\n# === 加载后的平面规则dot源码如下 ===")
        flat_dot = self.flat_rules_to_graphviz(flat_rules, graph_name="FlatRulesTree")
        print(flat_dot)
        # 打印加载后的原始树dot结构源码
        print("\n# === 加载后的决策树结构dot源码如下 ===")
        try:
            dot_source = self.to_graphviz_source(
                features=[f.name for f in self.meta.features] if hasattr(self.meta, 'features') else None,
                classes=[c.name for c in self.meta.labels] if hasattr(self.meta, 'labels') else None,
                graph_name="ToTDecisionTree"
            )
            print(dot_source)
        except Exception as e:
            print(f"[ToT] 决策树dot源码生成异常: {e}")
        # 日志文件也写入两种dot结构
        self.logger.log("# === 加载后的决策树平面规则如下 ===")
        for rule in flat_rules:
            self.logger.log(rule)
        self.logger.log("\n# === 加载后的平面规则dot源码如下 ===")
        self.logger.log(flat_dot)
        self.logger.log("\n# === 加载后的决策树结构dot源码如下 ===")
        try:
            self.logger.log(dot_source)
        except Exception:
            pass
        self.logger.log("=== [Zero-Shot] Decision tree generation finished ===")
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


