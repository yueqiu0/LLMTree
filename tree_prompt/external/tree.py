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
    def predict(self, x_train, y_train, x_test, export_rules=True):
        """
        兼容本地推理流程：用平面化规则对x_test做本地推理。
        """
        if not self.flat_rules or not self.tree_str_lines:
            self.fit(x_train, y_train)
        y_pred = []
        for xi in x_test:
            matched = False
            for rule in self.flat_rules:
                if not rule.startswith("IF "):
                    continue
                try:
                    cond, then = rule.split("THEN", 1)
                except Exception:
                    continue
                cond = cond[3:].strip()  # 去掉 'IF '
                label = then.strip()
                conds = [c.strip() for c in cond.split("AND")]
                satisfied = True
                for c in conds:
                    op_found = False
                    # 支持 = 作为 ==
                    # 优先匹配多字符操作符，避免误分割
                    for op in ["<=", ">=", "==", "!=", "<", ">", "="]:
                        if op in c:
                            feat, val = c.split(op, 1)
                            feat = feat.strip()
                            val = val.strip()
                            idx = next((i for i, f in enumerate(self.feature_names) if f == feat), None)
                            if idx is None:
                                satisfied = False
                                break
                            xval = xi[idx]
                            try:
                                if op == "==" or op == "=":
                                    if str(xval) != val:
                                        satisfied = False
                                elif op == "!=":
                                    if str(xval) == val:
                                        satisfied = False
                                elif op == "<=":
                                    if float(xval) > float(val):
                                        satisfied = False
                                elif op == ">=":
                                    if float(xval) < float(val):
                                        satisfied = False
                                elif op == "<":
                                    if float(xval) >= float(val):
                                        satisfied = False
                                elif op == ">":
                                    if float(xval) <= float(val):
                                        satisfied = False
                            except Exception:
                                satisfied = False
                            op_found = True
                            break
                    if not op_found:
                        satisfied = False
                    if not satisfied:
                        break
                if satisfied:
                    y_pred.append(label)
                    matched = True
                    break
            if not matched:
                y_pred.append(None)
        return y_pred, self.flat_rules
    @staticmethod
    def tree_str_to_flat_rules(tree_str: str) -> list[str]:
        """
        兼容 "IF ..." 和 "Rule: IF ..." 开头，遇到 THEN 截止，忽略 Leaf 节点。
        支持多行AND条件，遇到新IF/Rule: IF时开始新规则，THEN后为结论。
        """
        lines = [l for l in tree_str.splitlines() if l.strip()]
        rules = []
        collecting = False
        cond_parts = []
        for line in lines:
            content = line.strip()
            # 跳过 Leaf 节点
            if content.startswith("Leaf: class="):
                continue
            # 兼容 IF ... 或 Rule: IF ...
            if content.startswith("IF ") or content.startswith("Rule: IF "):
                # 如果正在收集上一条规则且没遇到THEN，丢弃未完成的
                collecting = True
                cond_parts = []
                # 去掉前缀
                if content.startswith("IF "):
                    rule_body = content[3:]
                else:
                    rule_body = content[len("Rule: IF "):]
                # 如果有THEN，直接提取
                if "THEN" in rule_body:
                    before_then, after_then = rule_body.split("THEN", 1)
                    cond = before_then.strip()
                    label = after_then.strip()
                    if cond.endswith(","):
                        cond = cond[:-1].strip()
                    rules.append(f"IF {cond} THEN {label}")
                    collecting = False
                    cond_parts = []
                else:
                    # 没有THEN，先收集条件
                    cond = rule_body.strip()
                    if cond.endswith(","):
                        cond = cond[:-1].strip()
                    cond_parts.append(cond)
            elif collecting:
                # 继续收集AND条件或THEN结论
                if "THEN" in content:
                    before_then, after_then = content.split("THEN", 1)
                    cond = before_then.strip()
                    label = after_then.strip()
                    if cond:
                        if cond.endswith(","):
                            cond = cond[:-1].strip()
                        cond_parts.append(cond)
                    full_cond = " AND ".join([c for c in cond_parts if c])
                    rules.append(f"IF {full_cond} THEN {label}")
                    collecting = False
                    cond_parts = []
                else:
                    # 继续收集AND条件
                    cond = content.strip()
                    if cond.endswith(","):
                        cond = cond[:-1].strip()
                    cond_parts.append(cond)
        if not rules:
            raise RuntimeError(f"[ToTDecisionTree] 平面化规则提取失败！原始print_tree字符串如下：\n{tree_str}")
        return rules
    @staticmethod
    def export_flat_rules(tree_str: str, meta: 'DatasetMeta' = None, logger=None, max_depth=None) -> list[str]:
        """
        解析 print_tree/log 输出的决策树结构字符串，输出所有完整路径规则。
        只保留 AND 数等于 max_depth-2 的规则，避免重复和浅层分支。
        若 logger 不为 None，则将平面化规则打印到日志。
        meta: DatasetMeta对象，用于获取label名称。
        """
        # 先直接保留原始字符串到日志，便于调试
        if logger is not None:
            logger.log("[ToTDecisionTree] print_tree原始字符串如下:")
            logger.log(tree_str)
        rules = ToTDecisionTree.tree_str_to_flat_rules(tree_str)
        # 调试：输出原始解析结果
        if logger is not None:
            logger.log(f"[ToTDecisionTree] tree_str_to_flat_rules原始输出: {rules}")
        filtered_rules = []
        seen = set()
        # 新策略：只保留THEN后面的label是数据集label之一且不是[NODE]的规则
        label_names = None
        if meta is not None and hasattr(meta, 'labels'):
            label_names = set(l.name for l in meta.labels)
        # 兼容静态调用时没有meta的情况
        if label_names is None and rules:
            # 尝试从规则中推断label集合（不推荐，仅兜底）
            label_names = set()
        for rule in rules:
            if 'THEN' in rule:
                after_then = rule.split('THEN', 1)[1].strip()
                # 始终过滤掉THEN为[NODE]/[Node]/[node]的规则
                if after_then in ('[NODE]', '[Node]', '[node]'):
                    continue
                # label_names为空时不过滤label，仅过滤[NODE]，否则要求label在label_names
                if (not label_names or after_then in label_names):
                    if rule not in seen:
                        filtered_rules.append(rule)
                        seen.add(rule)
        # 如果未能获取label_names，则退回原逻辑（但依然过滤[NODE]）
        if not filtered_rules:
            for rule in rules:
                if 'THEN' in rule:
                    after_then = rule.split('THEN', 1)[1].strip()
                    if after_then in ('[NODE]', '[Node]', '[node]'):
                        continue
                if rule not in seen:
                    filtered_rules.append(rule)
                    seen.add(rule)
        if logger is not None:
            logger.log("[ToTDecisionTree] 平面化规则如下:")
            for rule in filtered_rules:
                logger.log(rule)
        return filtered_rules

    def flat_rules_to_graphviz(self, flat_rules, graph_name="FlatRulesTree", log_to_logger=False):
        """
        将平面化规则链列表（如 export_flat_rules 输出）转为 dot 格式字符串。
        每条规则为一个节点，节点 label 为完整规则，节点编号顺序连接。
        若 log_to_logger=True 且 self.logger 存在，则写入日志。
        """
        lines = []
        lines.append(f'digraph {graph_name} {{')
        for idx, rule in enumerate(flat_rules):
            lines.append(f'node{idx} [label=\"{rule}\"]')
        for idx in range(len(flat_rules) - 1):
            lines.append(f'node{idx} -> node{idx+1}')
        lines.append('}')
        dot_str = "\n".join(lines)
        if log_to_logger and hasattr(self, 'logger') and self.logger is not None:
            self.logger.log(f"[ToTDecisionTree] dot决策树如下:\n{dot_str}")
        return dot_str

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
    # 删除无 logger 参数的重复定义，避免重载冲突

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



    def _zero_shot_llm_vote(self, candidates, depth, parent_rule, current_path="Root"):
        """
        对每组候选规则投票5次，统计票数，若有并列最多则在并列者中再次投票，直到唯一最优或达到最大重投次数。
        """
        import re
        max_vote_rounds = 5
        max_tiebreak_rounds = 5
        current_candidates = candidates
        last_llm_condition = None
        while True:
            vote_counts = [0 for _ in range(len(current_candidates))]
            for round_idx in range(max_vote_rounds):
                # 第一轮current_path为Root，后续为上轮大模型原始condition
                prompt_dict = generate_ToT_tree_prompt(
                    self.meta,
                    x_train=self.root.x if hasattr(self, 'root') and self.root is not None else None,
                    y_train=self.root.y if hasattr(self, 'root') and self.root is not None else None,
                    max_depth=depth + 1,
                    num_examples=0,
                    current_path=current_path,
                    applied_rules=None
                )
                intro = prompt_dict.get("intro", "")
                prompt = f"{intro}\nYou are building a zero-shot decision tree. The following are candidate rules for depth {depth+1}:\n"
                for idx, rule in enumerate(current_candidates):
                    prompt += f"Rule {idx+1}: {rule.get('condition', str(rule))} -> {rule.get('label', '')}\n"
                prompt += "\nPlease select the best rule. Output format: Rule N"
                self.logger.log(f"[Zero-Shot LLM Voting][Round {round_idx+1}] prompt:\n{prompt}")
                self.total_prompts += 1
                try:
                    responses = next(self.runner.run([prompt]))
                    response = responses[0] if isinstance(responses, list) and len(responses) > 0 else responses
                    self.logger.log(f"[Zero-Shot LLM投票][Round {round_idx+1}] 原始回应:\n{response}")
                    m = re.search(r'Rule\s*(\d+)', str(response))
                    if m:
                        idx = int(m.group(1)) - 1
                        if 0 <= idx < len(current_candidates):
                            vote_counts[idx] += 1
                except Exception as e:
                    self.logger.log(f"[Zero-Shot LLM投票][Round {round_idx+1}] 解析异常: {e}")
            self.logger.log(f"[Zero-Shot LLM投票] 计票结果: {vote_counts}")
            max_votes = max(vote_counts)
            winners = [i for i, v in enumerate(vote_counts) if v == max_votes]
            if len(winners) == 1:
                winner_idx = winners[0]
                # 记录本轮最佳规则的condition，供下轮current_path用
                best_rule = current_candidates[winner_idx]
                last_llm_condition = best_rule.get('condition', str(best_rule))
                if len(current_candidates) == len(candidates):
                    return winner_idx
                else:
                    for orig_idx, rule in enumerate(candidates):
                        if rule == current_candidates[winner_idx]:
                            return orig_idx
                    return 0
            else:
                self.logger.log(f"[Zero-Shot LLM投票] 出现并列({len(winners)}个)，进入重投轮次")
                current_candidates = [current_candidates[i] for i in winners]
                # 重投时current_path直接用上轮大模型原始condition（去掉Root）
                if last_llm_condition is not None:
                    current_path = last_llm_condition
                if max_tiebreak_rounds <= 0:
                    self.logger.log(f"[Zero-Shot LLM投票] 重投轮次过多，直接返回第一个并列项")
                    for orig_idx, rule in enumerate(candidates):
                        if rule == current_candidates[0]:
                            return orig_idx
                    return 0
                max_tiebreak_rounds -= 1




    def print_tree(self, node=None, indent=""):
        """
        打印决策树结构（print_tree风格字符串）。
        """
        if not self.tree_str_lines:
            print("[ToTDecisionTree] 树结构为空，请先调用 fit 生成树结构！")
            return
        print("\n".join(self.tree_str_lines))
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
        # 新增：保存平面化规则
        self.flat_rules = []
        log_file_obj = open(self.log_file, 'a', encoding='utf-8', buffering=1)
        self.logger = Logger(log_file_obj)
        self.logger.log(f"=== ToTDecisionTree initialized at {datetime.now().isoformat()} ===")

        # 新增：用于保存print_tree风格的树结构字符串
        self.tree_str_lines = []
        self.tree_str = ""


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
        """
        获取当前决策树的所有平面化规则（IF ... AND ...）。
        始终返回 self.flat_rules。
        """
        return self.flat_rules

    def fit(self, x_train=None, y_train=None, with_llm=True):
        """
        只用LLM和meta信息生成决策树（零样本），不依赖训练数据。
        支持多层递归，每一层扫描所有[NODE]节点，current_path为完整规则链。
        修正：循环次数为 max_depth-1，最后一层自动生成叶子节点。
        """
        self.logger.log("=== [Zero-Shot] Start zero-shot decision tree generation (ToT enhanced) ===")
        self.total_prompts = 0
        max_depth = self.max_depth
        self.root = TreeNode(0, x=x_train, y=y_train, parent=None)
        # 第一层只有root，current_path为Root
        current_layer = [(self.root, "Root")]
        # 修正：采用while循环，确保所有[NODE]分支都能被递归扩展，直到到达max_depth-1层
        depth = 0
        while current_layer and depth < max_depth - 1:
            log_depth = depth + 2  # 让日志中的depth从2开始
            self.logger.log(f"[DEBUG][ToT] while-loop begin, depth={log_depth}, current_layer size={len(current_layer)}, nodes={[repr(n) for n, _ in current_layer]}")
            next_layer = []
            for node, current_path in current_layer:
                self.logger.log(f"[DEBUG][ToT] for-loop, depth={log_depth}, node={repr(node)}, current_path={current_path}")
                indent = "  " * node.depth
                # 终止条件
                if node.depth >= max_depth - 1:
                    self.logger.log(f"[DEBUG][ToT] depth={log_depth}, node.depth={node.depth} >= max_depth-1, continue as leaf")
                    node.is_leaf = True
                    node.leaf_class = None
                    continue
                # 生成候选规则，数量由 candidate_rules_per_node 控制
                candidates = []
                for _ in range(self.candidate_rules_per_node):
                    rule = self._zero_shot_llm_generate_rule(node.depth, None, current_path)
                    self.logger.log(f"[DEBUG][ToT] depth={log_depth}, candidate rule raw: {repr(rule)}, type={type(rule)}")
                    if rule is not None:
                        candidates.append(rule)
                if not candidates:
                    self.logger.log(f"[DEBUG][ToT] depth={log_depth}, no candidates, node set as leaf, continue")
                    node.is_leaf = True
                    node.leaf_class = None
                    continue
                # 投票选出最佳规则
                best_idx = self._zero_shot_llm_vote(candidates, node.depth, None)
                # 打印本轮投票prompt和最佳分支
                self.logger.log(f"[DEBUG][ToT] depth={log_depth}, 投票prompt如下:\n{getattr(self, 'last_vote_prompt', '')}")
                best_rule = candidates[best_idx]
                self.logger.log(f"[DEBUG][ToT] depth={log_depth}, 最佳分支: {best_rule}")
                node.selected_rule = best_rule
                node.candidates = candidates
                node.votes = {i: 1 if i == best_idx else 0 for i in range(len(candidates))}
                node.children = []
                # 只扩展最佳分支
                cond = best_rule.get('condition', '')
                self.logger.log(f"[DEBUG][ToT] depth={log_depth}, cond(raw)={repr(cond)}, type={type(cond)}")
                if not isinstance(cond, str):
                    cond = str(cond)
                    self.logger.log(f"[DEBUG][ToT] depth={depth}, cond 强制转为str: {repr(cond)}")
                # 记录Rule节点
                if '[NODE]' in cond or '[Node]' in cond or '[node]' in cond:
                    rules_lines = [l.strip() for l in cond.splitlines() if l.strip()]
                    self.logger.log(f"[DEBUG][ToT] depth={log_depth}, rules_lines: {rules_lines}")
                    for rule_line in rules_lines:
                        self.logger.log(f"[DEBUG][ToT] depth={log_depth}, rule_line: {repr(rule_line)}")
                        if '[NODE]' in rule_line or '[Node]' in rule_line or '[node]' in rule_line:
                            # 记录Rule: ...
                            self.tree_str_lines.append(f"{indent}Rule: {rule_line}")
                            child_path = (current_path + " AND " + rule_line.replace('THEN [NODE]', '').replace('THEN [Node]', '').replace('THEN [node]', '').strip()).replace('Root AND', 'Root')
                            # 修正：只保留第一个 IF 及其后内容
                            if "IF" in child_path:
                                child_path = child_path[child_path.find("IF"):]
                            child = TreeNode(node.depth + 1, x=self.root.x, y=self.root.y, parent=node)
                            node.children.append(child)
                            next_layer.append((child, child_path))
                            self.logger.log(f"[DEBUG][ToT] 当前depth={log_depth}, 新child.depth={child.depth}, 发现[NODE]分支: {rule_line}, child_path={child_path}")
                        elif 'THEN' in rule_line:
                            # 记录Rule: ...
                            self.tree_str_lines.append(f"{indent}Rule: {rule_line}")
                            label = rule_line.split('THEN')[-1].strip()
                            child = TreeNode(node.depth + 1, x=self.root.x, y=self.root.y, parent=node)
                            child.is_leaf = True
                            child.leaf_class = label
                            node.children.append(child)
                            # 记录Leaf: ...
                            leaf_indent = "  " * child.depth
                            self.tree_str_lines.append(f"{leaf_indent}Leaf: class={label}")
                            self.logger.log(f"[DEBUG][ToT] depth={log_depth}, 发现叶子分支: {rule_line}, label={label}")
                else:
                    # 兼容只有一个规则的情况
                    if '[NODE]' in cond or '[Node]' in cond or '[node]' in cond:
                        self.logger.log(f"[DEBUG][ToT] depth={log_depth}, else-branch, 发现[NODE]分支: {cond}")
                        self.tree_str_lines.append(f"{indent}Rule: {cond}")
                        child_path = (current_path + " AND " + cond.replace('THEN [NODE]', '').replace('THEN [Node]', '').replace('THEN [node]', '').strip()).replace('Root AND', 'Root')
                        # 修正：只保留第一个 IF 及其后内容
                        if "IF" in child_path:
                            child_path = child_path[child_path.find("IF"):]
                        child = TreeNode(node.depth + 1, x=self.root.x, y=self.root.y, parent=node)
                        node.children.append(child)
                        next_layer.append((child, child_path))
                        self.logger.log(f"[DEBUG][ToT] 当前depth={log_depth}, 新child.depth={child.depth}, 发现[NODE]分支: {cond}, child_path={child_path}")
                    elif 'THEN' in cond:
                        self.logger.log(f"[DEBUG][ToT] depth={log_depth}, else-branch, 发现叶子分支: {cond}")
                        self.tree_str_lines.append(f"{indent}Rule: {cond}")
                        label = cond.split('THEN')[-1].strip()
                        child = TreeNode(node.depth + 1, x=self.root.x, y=self.root.y, parent=node)
                        child.is_leaf = True
                        child.leaf_class = label
                        node.children.append(child)
                        leaf_indent = "  " * child.depth
                        self.tree_str_lines.append(f"{leaf_indent}Leaf: class={label}")
                        self.logger.log(f"[DEBUG][ToT] depth={log_depth}, 发现叶子分支: {cond}, label={label}")
            self.logger.log(f"[DEBUG][ToT] while-loop end, depth={log_depth}, next_layer size={len(next_layer)}")
            current_layer = next_layer
            self.logger.log(f"[DEBUG][ToT] 第{log_depth-1}轮生成树的循环结束，当前层剩余可分裂点数: {len(current_layer)}")
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
        # 拼接保存print_tree风格的树结构字符串
        self.tree_str = "\n".join(self.tree_str_lines)
        # 自动生成平面化规则并写入日志，只保留最长路径规则
        self.flat_rules = ToTDecisionTree.export_flat_rules(self.tree_str, logger=self.logger, max_depth=self.max_depth)
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





