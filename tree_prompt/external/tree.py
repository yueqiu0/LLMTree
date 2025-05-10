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
from ..dataset import generate_ToT_tree_prompt
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


class ToTDecisionTree:
    def __init__(self, meta, max_depth, runner, log_file, candidate_rules_per_node=3, voting_rounds=3):
        # 初始化基本属性
        self.meta = meta
        self.max_depth = max_depth
        self.runner = runner
        self.log_file = log_file
        self.rules = []  # 存储规则列表
        # 获取特征名称
        self.feature_names = [f.name for f in meta.features]
        # 保存使用的特征到列的映射
        self.feature_name_to_col = {f.name: idx for idx, f in enumerate(meta.features)}
        # Open log file in append mode with line buffering for real-time writing
        log_file_obj = open(self.log_file, 'a', encoding='utf-8', buffering=1)  # 1 means line buffering
        self.logger = Logger(log_file_obj)  # Initialize logger with file object
        # Ensure first log message marks the start
        self.logger.log(f"=== ToTDecision initialized at {datetime.now().isoformat()} ===")
        
        # 用于树构建的内部数据结构
        self.node_queue = []  # 待处理节点队列
        self.token_stats = {  # 统计token使用情况
            "tree_building": {"prompt": 0, "completion": 0, "total": 0},
            "evaluation": [],
            "total_tokens": 0
        }
        
        # 添加规则选择参数
        self.candidate_rules_per_node = candidate_rules_per_node  # 每个节点生成的规则候选数量
        self.voting_rounds = voting_rounds  # 每个节点的投票轮数

    def _vote_for_best_rules(self, rule_candidates, prompt_dict, current_depth, current_path):
        """
        对规则候选进行投票，选择最佳规则组
        
        Args:
            rule_candidates: 规则候选列表
            prompt_dict: 原始提示词字典
            current_depth: 当前深度
            current_path: 当前路径
            
        Returns:
            最佳规则列表
        """
        if not rule_candidates:
            return []
            
        self.logger.log(f"[规则选择] 开始对节点(深度={current_depth}, 路径={current_path or 'Root'})的规则进行投票")
        self.logger.log(f"[规则选择] 规则候选数量: {len(rule_candidates)}")
        
        # 将规则按组组织 - 每次LLM响应生成的是一组规则（可能包含中间节点规则和叶子节点规则）
        # 规则按生成顺序分组
        rule_groups = []
        current_group = []
        
        for rule in rule_candidates:
            # 添加规则到当前组
            current_group.append(rule)
            
            # 每两个规则为一组（一个分割规则和一个叶子规则）
            if len(current_group) == 2:
                rule_groups.append(current_group)
                current_group = []
        
        # 处理剩余的规则
        if current_group:
            rule_groups.append(current_group)
        
        self.logger.log(f"[规则选择] 规则已按组组织，共 {len(rule_groups)} 组")
        for i, group in enumerate(rule_groups):
            self.logger.log(f"[规则选择] 规则组 {i+1}:")
            for j, rule in enumerate(group):
                self.logger.log(f"  规则 {j+1}: {rule}")
        
        # 对规则组进行投票
        best_group_idx = self._vote_for_rule_groups(rule_groups, current_depth, current_path)
        
        if best_group_idx >= 0 and best_group_idx < len(rule_groups):
            selected_rules = rule_groups[best_group_idx]
            self.logger.log(f"[规则选择] 选中规则组 {best_group_idx+1}，包含 {len(selected_rules)} 条规则")
            for idx, rule in enumerate(selected_rules):
                self.logger.log(f"[规则选择] 选中规则 {idx+1}: {rule}")
            return selected_rules
        else:
            self.logger.log(f"[规则选择] 未能选中有效规则组")
            return []
    
    def _vote_for_rule_groups(self, rule_groups, current_depth, current_path):
        """
        对规则组进行投票
        
        Args:
            rule_groups: 规则组列表，每组包含一组相关规则
            current_depth: 当前深度
            current_path: 当前路径
            
        Returns:
            得票最多的规则组索引
        """
        if not rule_groups:
            return -1
            
        self.logger.log(f"[规则投票] 开始对规则组投票, 规则组数: {len(rule_groups)}")
        
        # 如果只有一个规则组，直接返回
        if len(rule_groups) == 1:
            self.logger.log(f"[规则投票] 只有一个规则组，直接选择组 1")
            return 0
        
        # 统计每组规则的投票数
        votes = [0] * len(rule_groups)
        
        # 构建投票提示词
        for round_idx in range(self.voting_rounds):
            self.logger.log(f"[规则投票] 第 {round_idx+1} 轮投票开始")
            
            # 构建投票提示 - 使用英文提示更符合大模型训练分布
            voting_prompt = f"""You are building a decision tree at depth {current_depth}. Select the best rule group from the following candidates.

Current path: {current_path or "Root"}

Rule groups:
"""
            
            for i, group in enumerate(rule_groups):
                voting_prompt += f"Group {i+1}:\n"
                for rule in group:
                    voting_prompt += f"  {rule}\n"
                voting_prompt += "\n"
                
            voting_prompt += f"""
Which group contains the best rules? Please ONLY respond with the group number (e.g., "Group 3").
Do NOT include any explanation or reasoning in your response.
"""
            
            # 发送投票请求
            try:
                for response, token_info in self.runner.run([voting_prompt]):
                    self.logger.log(f"[规则投票] 收到投票响应")
                    
                    # 统计token
                    if isinstance(token_info, dict):
                        self.token_stats["tree_building"]["prompt"] += token_info.get('prompt_tokens', 0)
                        self.token_stats["tree_building"]["completion"] += token_info.get('completion_tokens', 0)
                        self.token_stats["tree_building"]["total"] += token_info.get('total_tokens', 0)
                        self.token_stats["total_tokens"] += token_info.get('total_tokens', 0)
                    
                    # 解析投票结果
                    for resp in response:
                        # 尝试从响应中提取组号
                        import re
                        # 匹配 "Group X" 或 "组 X" 或 "X" 
                        vote_match = re.search(r'(?:Group|组)?\s*(\d+)', resp)
                        if vote_match:
                            group_idx = int(vote_match.group(1)) - 1
                            if 0 <= group_idx < len(rule_groups):
                                votes[group_idx] += 1
                                self.logger.log(f"[规则投票] 投票给规则组 {group_idx+1}")
                            else:
                                self.logger.log(f"[规则投票] 无效规则组索引: {group_idx+1}")
                        else:
                            self.logger.log(f"[规则投票] 无法解析投票结果: {resp[:100]}...")
            except Exception as e:
                self.logger.log(f"[规则投票] 投票过程出错: {str(e)}")
        
        # 找出得票最多的规则组
        max_votes = max(votes) if votes else 0
        best_group_indices = [i for i, v in enumerate(votes) if v == max_votes]
        
        # 如果有多个规则组得票相同，随机选择一个
        if best_group_indices:
            import random
            best_group_idx = random.choice(best_group_indices)
            self.logger.log(f"[规则投票] 投票结束，组 {best_group_idx+1} 获得最高票数: {max_votes}")
            return best_group_idx
        else:
            self.logger.log(f"[规则投票] 投票失败，没有有效投票")
            # 返回第一个规则组作为备选
            if rule_groups:
                self.logger.log(f"[规则投票] 使用默认规则组 1")
                return 0
            return -1
    
    def _generate_rule_candidates(self, prompt, current_depth, current_path):
        """
        生成多个规则候选
        
        Args:
            prompt: 提示词
            current_depth: 当前深度
            current_path: 当前路径
            
        Returns:
            规则候选列表
        """
        self.logger.log(f"[规则生成] 开始为节点(深度={current_depth}, 路径={current_path or 'Root'})生成规则候选")
        rule_candidates = []
        
        # 多次查询LLM
        for i in range(self.candidate_rules_per_node):
            self.logger.log(f"[规则生成] 第 {i+1}/{self.candidate_rules_per_node} 次查询")
            
            try:
                for response, token_info in self.runner.run([prompt]):
                    self.logger.log(f"[规则生成] 收到LLM响应")
                    
                    # 统计token使用
                    if isinstance(token_info, dict):
                        self.token_stats["tree_building"]["prompt"] += token_info.get('prompt_tokens', 0)
                        self.token_stats["tree_building"]["completion"] += token_info.get('completion_tokens', 0)
                        self.token_stats["tree_building"]["total"] += token_info.get('total_tokens', 0)
                        self.token_stats["total_tokens"] += token_info.get('total_tokens', 0)
                    
                    # 解析规则
                    for resp in response:
                        rules = self._extract_rule_texts(resp)
                        self.logger.log(f"[规则生成] 提取到 {len(rules)} 条规则")
                        rule_candidates.extend(rules)
            except Exception as e:
                self.logger.log(f"[规则生成] 生成规则时出错: {str(e)}")
        
        self.logger.log(f"[规则生成] 共生成 {len(rule_candidates)} 条规则候选")
        for i, rule in enumerate(rule_candidates):
            self.logger.log(f"[规则生成] 规则候选 {i+1}: {rule}")
            
        return rule_candidates

    def fit(self, x_train, y_train, with_llm=True):
        """训练决策树模型，生成规则，使用层序遍历构建树"""
        self.logger.log(f"[INFO] 决策树最大深度为{self.max_depth}")
        self.logger.log(f"[DEBUG] fit() called with with_llm={with_llm}, x_train.shape={x_train.shape if hasattr(x_train, 'shape') else 'unknown'}, y_train.shape={y_train.shape if hasattr(y_train, 'shape') else 'unknown'}")

        # 确保有训练数据
        if x_train is None or y_train is None or len(x_train) == 0 or len(y_train) == 0:
            self.logger.log("[ERROR] 训练数据为空，将生成默认规则")
            self._generate_default_rule()
            return

        # 尝试训练模型
        try:
            # 如果不使用LLM，则生成简单的默认规则
            if not with_llm:
                self.logger.log("[INFO] with_llm=False，生成简单默认规则")
                self._generate_simple_rules(x_train, y_train)
                return
            
            # 清空规则列表（以防重复调用fit）
            self.rules = []
                
            # 初始化根节点
            self.node_queue = [{'path': '', 'depth': 1, 'parent_cond': None}]
            
            # 增加调试信息：打印初始队列
            self.logger.log("[树构建] ======== 开始构建决策树 ========")
            self.logger.log(f"[树构建] 初始队列: {self.node_queue}")
            
            # 层序遍历构建树
            queue_iteration = 0  # 跟踪队列迭代次数
            while self.node_queue:
                queue_iteration += 1
                # 取出当前节点
                current_node = self.node_queue.pop(0)
                current_path = current_node['path']
                current_depth = current_node['depth']
                
                self.logger.log(f"\n[树构建] ===== 迭代 {queue_iteration} =====")
                self.logger.log(f"[树构建] 当前处理节点: 深度={current_depth}, 路径={current_path or 'Root'}")
                self.logger.log(f"[树构建] 剩余队列长度: {len(self.node_queue)}")
                
                # 如果已达到最大深度，跳过处理
                if current_depth > self.max_depth:
                    self.logger.log(f"[树构建] 节点已达最大深度 {self.max_depth}，跳过处理")
                    continue
                
                # 调用 generate_ToT_tree_prompt
                prompt_dict = generate_ToT_tree_prompt(
                    meta=self.meta,
                    x_train=x_train,
                    y_train=y_train,
                    max_depth=self.max_depth,
                    depth=current_depth,
                    current_path=current_path or "Root"
                )
                prompt = prompt_dict["prompt"]
                
                # 记录完整prompt到日志
                self.logger.log(f"[BUILD_TREE_PROMPT] 为节点(深度={current_depth}, 路径={current_path or 'Root'})生成提示")
                self.logger.log(f"[BUILD_TREE_PROMPT] 提示长度: {len(prompt)} 字符")
                
                # 修改：生成多个规则候选并投票选择最佳规则
                rule_candidates = self._generate_rule_candidates(prompt, current_depth, current_path)
                selected_rules = self._vote_for_best_rules(rule_candidates, prompt_dict, current_depth, current_path)
                
                if not selected_rules:
                    self.logger.log(f"[ERROR] 无法获取有效规则，尝试使用LLM直接生成规则")
                    # 退回到原始方式：直接调用LLM
                    llm_responses = []
                    try:
                        for response, token_info in self.runner.run([prompt]):
                            self.logger.log(f"[BUILD_TREE_RESPONSE] 收到LLM响应")
                            for idx, r in enumerate(response):
                                self.logger.log(f"[BUILD_TREE_RESPONSE] [{idx}] 长度: {len(r)} 字符")
                                llm_responses.append(r)
                            
                            # 统计token使用
                            if isinstance(token_info, dict):
                                self.token_stats["tree_building"]["prompt"] += token_info.get('prompt_tokens', 0)
                                self.token_stats["tree_building"]["completion"] += token_info.get('completion_tokens', 0)
                                self.token_stats["tree_building"]["total"] += token_info.get('total_tokens', 0)
                                self.token_stats["total_tokens"] += token_info.get('total_tokens', 0)
                    except Exception as e:
                        self.logger.log(f"[ERROR] 调用LLM出错: {str(e)}")
                        continue
                    
                    # 处理LLM响应
                    if not llm_responses:
                        self.logger.log(f"[ERROR] 未收到LLM响应")
                        continue
                    
                    # 解析规则 - 直接解析文本规则
                    selected_rules = self._extract_rule_texts(llm_responses[0])
                    if not selected_rules:
                        self.logger.log(f"[ERROR] 无法提取规则文本")
                        continue
                
                # 处理选定的规则
                self.logger.log(f"[树构建] 处理 {len(selected_rules)} 条选定规则")
                
                # 处理每条文本规则
                for rule_idx, rule_text in enumerate(selected_rules):
                    # 检查是否是中间节点
                    is_node = rule_text.strip().endswith("[NODE]")
                    
                    # 记录规则类型
                    self.logger.log(f"[树构建] 处理规则 {rule_idx+1}/{len(selected_rules)}: {rule_text}")
                    self.logger.log(f"[树构建] 规则类型: {'中间节点' if is_node else '叶子节点'}")
                    
                    # 解析规则文本并创建规则对象
                    rule_obj = self._parse_single_rule(rule_text, current_path)
                    if not rule_obj:
                        self.logger.log(f"[WARNING] 无法解析规则: {rule_text}")
                        continue
                    
                    # 添加到规则列表
                    self.rules.append(rule_obj)
                    
                    # 添加规则日志
                    self.logger.log(f"[树构建] 添加规则: {rule_text}")
                    self.logger.log(f"[树构建] 当前路径: {current_path}, 深度: {current_depth}, 是否中间节点: {is_node}")
                    if isinstance(rule_obj, dict):
                        conditions_str = " AND ".join([f"{c['feature']} {c['operator']} {c['value']}" for c in rule_obj.get('conditions', [])])
                        self.logger.log(f"[树构建] 规则条件: {conditions_str}, 标签: {rule_obj.get('label', '')}")
                        
                        # 打印完整的规则对象结构
                        self.logger.log(f"[树构建] 规则对象结构: {rule_obj}")
                    
                    # 如果是中间节点且未达到最大深度，添加到队列
                    if is_node and current_depth < self.max_depth:
                        # 从规则中提取条件
                        next_path = self._extract_condition_from_rule(rule_obj, current_path)
                        if next_path:
                            # 详细记录新增的节点
                            self.logger.log(f"[树构建] 识别到中间节点 [NODE]，继续扩展树")
                            self.logger.log(f"[树构建] 添加下一层节点: 深度={current_depth+1}, 路径={next_path}")
                            
                            new_node = {
                                'path': next_path,
                                'depth': current_depth + 1,
                                'parent_cond': next_path,
                                'from_rule': rule_text
                            }
                            self.node_queue.append(new_node)
                            
                            # 打印队列信息
                            self.logger.log(f"[树构建] 新增队列节点: {new_node}")
                            self.logger.log(f"[树构建] 当前队列长度: {len(self.node_queue)}")
                    else:
                        if is_node:
                            self.logger.log(f"[树构建] 忽略中间节点 [NODE]，因为已达到最大深度 {self.max_depth}")
                        else:
                            self.logger.log(f"[树构建] 此规则是叶子节点，不再继续扩展")
            
            # 打印当前队列状态
            self.logger.log(f"[树构建] 队列处理完成，最终队列状态: {self.node_queue}")
            
            # 检查规则是否生成
            if not self.rules:
                self.logger.log("[ERROR] No rules were generated during fit.")
                # 如果没有生成规则，则生成默认规则
                self._generate_simple_rules(x_train, y_train)
            else:
                self.logger.log(f"[INFO] 训练完成，生成了 {len(self.rules)} 条规则")
                
                # 打印完整的树结构和规则之间的层次关系
                self.logger.log("[树构建完成] ============== 最终树结构和层次关系 ==============")
                for i, rule in enumerate(self.rules):
                    if isinstance(rule, dict):
                        conditions_str = " AND ".join([f"{c['feature']} {c['operator']} {c['value']}" for c in rule.get('conditions', [])])
                        is_node_str = " [中间节点]" if rule.get('label', '').strip() == '[NODE]' else " [叶子节点]"
                        depth_estimation = len(rule.get('conditions', []))
                        self.logger.log(f"[规则 {i+1}]: IF {conditions_str} THEN {rule.get('label', '')}{is_node_str} (估计深度: {depth_estimation})")
                    else:
                        self.logger.log(f"[规则 {i+1}]: {rule}")
                self.logger.log("[树构建完成] ============== 树结构结束 ==============")
                
                # 打印树的可视化表示
                self.logger.log("[树构建完成] ============== 树的可视化表示 ==============")
                tree_visualization = self._generate_tree_visualization()
                self.logger.log(tree_visualization)
                self.logger.log("[树构建完成] ============== 可视化结束 ==============")
        except Exception as e:
            self.logger.log(f"[ERROR] 训练过程出现异常：{str(e)}")
            import traceback
            self.logger.log(traceback.format_exc())
            # 如果出现任何异常，确保生成默认规则
            self._generate_default_rule()
    
    def _generate_tree_visualization(self):
        """生成树的层次可视化表示"""
        if not self.rules:
            return "树为空"
            
        # 创建节点层次结构
        root = {"label": "Root", "children": []}
        
        # 确定每个规则的层次
        for rule in self.rules:
            if not isinstance(rule, dict) or 'conditions' not in rule:
                continue
                
            conditions = rule.get('conditions', [])
            label = rule.get('label', '')
            
            # 跟踪当前位置
            current_node = root
            
            # 处理每个条件，构建树层次
            for i, cond in enumerate(conditions):
                feature = cond.get('feature', '')
                operator = cond.get('operator', '')
                value = cond.get('value', '')
                
                # 生成条件描述
                cond_str = f"{feature} {operator} {value}"
                
                # 检查是否有现有路径
                found = False
                for child in current_node.get("children", []):
                    if child.get("condition") == cond_str:
                        current_node = child
                        found = True
                        break
                
                # 如果没有找到，创建新节点
                if not found:
                    new_node = {
                        "condition": cond_str,
                        "children": [],
                        "depth": i + 1
                    }
                    current_node.setdefault("children", []).append(new_node)
                    current_node = new_node
            
            # 设置最终节点的标签
            current_node["label"] = label
            current_node["is_node"] = label.strip() == '[NODE]'
        
        # 递归生成可视化
        def visualize_node(node, depth=0, prefix=""):
            result = []
            indent = "  " * depth
            
            if depth == 0:
                result.append(f"{indent}Root")
            else:
                node_type = "中间节点" if node.get("is_node", False) else "叶子节点"
                if "condition" in node and "label" in node:
                    result.append(f"{indent}{prefix}IF {node['condition']} THEN {node['label']} [{node_type}]")
                elif "condition" in node:
                    result.append(f"{indent}{prefix}IF {node['condition']}")
            
            # 处理子节点
            for i, child in enumerate(node.get("children", [])):
                is_last = i == len(node.get("children", [])) - 1
                next_prefix = "└── " if is_last else "├── "
                child_prefix = "    " if is_last else "│   "
                
                result.extend(visualize_node(child, depth + 1, next_prefix))
            
            return result
        
        # 生成可视化
        visualization = "\n".join(visualize_node(root))
        return visualization
    
    def _extract_rule_texts(self, response):
        """从LLM响应中提取规则文本"""
        if not response:
            return []
        
        # 按行分割响应
        lines = response.strip().split('\n')
        rule_texts = []
        
        for line in lines:
            line = line.strip()
            # 过滤空行和非规则行
            if not line or not line.startswith("IF "):
                continue
            rule_texts.append(line)
        
        return rule_texts
    
    def _parse_single_rule(self, rule_text, current_path):
        """解析单条规则文本为规则对象"""
        try:
            # 记录详细的解析过程
            self.logger.log(f"[规则解析] 开始解析规则: {rule_text}")
            self.logger.log(f"[规则解析] 当前路径: {current_path}")
            
            # 正则匹配 IF-THEN 结构
            match = re.match(r'IF\s+(.+?)\s+THEN\s+(.+)', rule_text, re.IGNORECASE)
            if not match:
                self.logger.log(f"[规则解析] [WARNING] 规则格式不匹配: {rule_text}")
                return None
            
            conditions_text, label = match.groups()
            self.logger.log(f"[规则解析] 提取条件文本: {conditions_text}")
            self.logger.log(f"[规则解析] 提取标签: {label}")
            
            # 解析条件
            final_conditions = []
            
            # 如果有当前路径，确保不重复包含
            path_prefix = ""
            if current_path and current_path != "Root":
                path_prefix = current_path + " AND "
                self.logger.log(f"[规则解析] 检测到路径前缀: {path_prefix}")
            
            # 检查条件是否已经包含路径前缀
            full_condition_text = conditions_text
            if path_prefix and not full_condition_text.startswith(path_prefix):
                full_condition_text = path_prefix + full_condition_text
                self.logger.log(f"[规则解析] 合并路径前缀后的完整条件: {full_condition_text}")
            
            # 拆分AND条件
            self.logger.log(f"[规则解析] 拆分条件: {full_condition_text}")
            condition_parts = re.split(r'\s+AND\s+', full_condition_text, flags=re.IGNORECASE)
            self.logger.log(f"[规则解析] 拆分后的条件部分: {condition_parts}")
            
            for condition in condition_parts:
                # 解析特征、操作符和值
                cond_match = re.match(r'^([\w\s_-]+?)\s*(<=|>=|<|>|=|==|!=)\s*([\w\d.\-]+)$', condition.strip())
                if not cond_match:
                    self.logger.log(f"[规则解析] [WARNING] 条件格式不匹配: {condition}")
                    continue
                    
                feature, operator, value = cond_match.groups()
                feature = feature.strip()
                    
                # 尝试将值转换为数字
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
                    
                self.logger.log(f"[规则解析] 解析条件: 特征={feature}, 操作符={operator}, 值={value}")
                
                final_conditions.append({
                    'feature': feature,
                    'operator': operator,
                    'value': value
                })
                
            # 创建规则对象
            rule = {
                'conditions': final_conditions,
                    'label': label.strip()
            }
            
            # 检查是否是NODE中间节点
            is_node = label.strip() == '[NODE]'
            self.logger.log(f"[规则解析] 最终解析规则: {rule}")
            self.logger.log(f"[规则解析] 是否中间节点: {is_node}")
            
            return rule
            
        except Exception as e:
            self.logger.log(f"[ERROR] 解析规则时出错: {str(e)}")
            return None
    
    def _extract_condition_from_rule(self, rule, current_path):
        """从规则对象中提取完整路径条件"""
        self.logger.log(f"[路径提取] 开始从规则提取路径条件")
        self.logger.log(f"[路径提取] 规则: {rule}")
        self.logger.log(f"[路径提取] 当前路径: {current_path}")
        
        if not isinstance(rule, dict) or 'conditions' not in rule:
            self.logger.log(f"[路径提取] 无效规则格式，无法提取路径")
            return None
        
        conditions = []
        # 检查当前路径
        path_conditions = []
        if current_path and current_path != "Root":
            path_parts = re.split(r'\s+AND\s+', current_path, flags=re.IGNORECASE)
            self.logger.log(f"[路径提取] 当前路径拆分: {path_parts}")
            path_conditions = path_parts
        
        # 记录规则中的所有条件
        rule_conditions = []
        for cond in rule['conditions']:
            if all(k in cond for k in ['feature', 'operator', 'value']):
                cond_str = f"{cond['feature']} {cond['operator']} {cond['value']}"
                rule_conditions.append(cond_str)
                
                # 检查此条件是否已在路径中
                if cond_str not in path_conditions:
                    conditions.append(cond_str)
                    self.logger.log(f"[路径提取] 添加新条件: {cond_str}")
                else:
                    self.logger.log(f"[路径提取] 跳过已存在的条件: {cond_str}")
        
        self.logger.log(f"[路径提取] 规则中的所有条件: {rule_conditions}")
        self.logger.log(f"[路径提取] 提取的新条件: {conditions}")
        
        # 合并路径
        if current_path and current_path != "Root" and conditions:
            new_path = current_path + " AND " + " AND ".join(conditions)
            self.logger.log(f"[路径提取] 合并后的完整路径: {new_path}")
            return new_path
        elif conditions:
            new_path = " AND ".join(conditions)
            self.logger.log(f"[路径提取] 无当前路径，新路径: {new_path}")
            return new_path
        else:
            self.logger.log(f"[路径提取] 无法提取条件")
            return None
    
    def _parse_llm_response(self, response):
        """从LLM响应中解析规则（兼容旧版本）"""
        # 使用新方法提取规则
        rule_texts = self._extract_rule_texts(response)
        if not rule_texts:
            return []
        
        # 解析每条规则
        rules = []
        for rule_text in rule_texts:
            rule_obj = self._parse_single_rule(rule_text, "")
            if rule_obj:
                rules.append(rule_obj)
        
        return rules
            
    def get_token_stats(self):
        """返回token使用统计"""
        return self.token_stats

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
                    # 检查rule是否为字典类型
                    if isinstance(rule, dict):
                        rule_label = rule.get('label', '')
                    else:
                        # 如果rule是字符串，直接使用
                        rule_label = str(rule)
                        self.logger.log(f"[WARNING] 预测中遇到字符串规则: {rule_label}")
                    
                    # 如果标签是NODE，跳过这条规则继续匹配
                    if rule_label in ['[NODE]', 'NODE']:
                        continue
                    
                    # 将标签名称转换为标签值
                    label_value = self.meta.get_label_value(rule_label)
                    if label_value is not None:
                        prediction = label_value
                    else:
                        # 如果找不到对应的标签值，记录警告并使用默认标签
                        self.logger.log(f"[WARNING] 找不到标签 '{rule_label}' 对应的值，使用默认标签")
                        if hasattr(self.meta, 'labels') and self.meta.labels:
                            prediction = self.meta.labels[0].value
                        else:
                            prediction = 0
                    break
            if matched_rule_idx is not None:
                rule_hit_count[matched_rule_idx] += 1
                predictions.append(prediction)
            else:
                # 使用默认标签值
                if hasattr(self.meta, 'labels') and self.meta.labels:
                    default_label_value = self.meta.labels[0].value
                else:
                    default_label_value = 0
                self.logger.log(f"[DEBUG] 样本未匹配任何规则，使用默认标签值: {default_label_value}")
                predictions.append(default_label_value)
        # 输出每条规则命中数
        for idx, count in enumerate(rule_hit_count):
            self.logger.log(f"[规则{idx+1}] 命中样本数: {count}")
        return np.array(predictions)

    def _apply_rules(self, sample, rule=None):
        """检查样本是否满足规则
        
        Args:
            sample: 样本数据
            rule: 指定要检查的规则，如果为None则检查所有规则
            
        Returns:
            bool: 如果满足规则则返回True，否则返回False
        """
        rules_to_check = [rule] if rule is not None else self.rules
        
        for r in rules_to_check:
            # 跳过字符串类型的规则
            if not isinstance(r, dict):
                self.logger.log(f"[WARNING] _apply_rules遇到非字典规则: {r}")
                continue
                
            # 检查规则是否有条件
            if 'conditions' not in r or not isinstance(r['conditions'], list):
                continue
                
            # 检查是否满足所有条件
            all_conditions_met = True
            
            for cond in r['conditions']:
                # 检查条件格式是否完整
                if not all(k in cond for k in ['feature', 'operator', 'value']):
                    self.logger.log(f"[WARNING] 不完整的条件: {cond}")
                    all_conditions_met = False
                    break
                    
                feature = cond['feature']
                operator = cond['operator']
                value = cond['value']
                
                # 获取特征值
                if feature in self.feature_name_to_col:
                    feature_idx = self.feature_name_to_col[feature]
                    if feature_idx < len(sample):
                        sample_value = sample[feature_idx]
                    else:
                        self.logger.log(f"[ERROR] 特征索引超出范围: {feature_idx} >= {len(sample)}")
                        all_conditions_met = False
                        break
                else:
                    self.logger.log(f"[ERROR] 未知特征: {feature}")
                    all_conditions_met = False
                    break
                
                # 执行比较操作
                try:
                    # 对数值类型进行比较
                    if isinstance(sample_value, (int, float)) and isinstance(value, (int, float)):
                        if operator == '>=' and not (sample_value >= value):
                            all_conditions_met = False
                            break
                        elif operator == '<=' and not (sample_value <= value):
                            all_conditions_met = False
                            break
                        elif operator == '>' and not (sample_value > value):
                            all_conditions_met = False
                            break
                        elif operator == '<' and not (sample_value < value):
                            all_conditions_met = False
                            break
                        elif (operator == '=' or operator == '==') and not (sample_value == value):
                            all_conditions_met = False
                            break
                        elif operator == '!=' and not (sample_value != value):
                            all_conditions_met = False
                            break
                    # 对字符串类型进行比较
                    else:
                        if (operator == '=' or operator == '==') and not (str(sample_value) == str(value)):
                            all_conditions_met = False
                            break
                        elif operator == '!=' and not (str(sample_value) != str(value)):
                            all_conditions_met = False
                            break
                except Exception as e:
                    self.logger.log(f"[ERROR] 比较操作出错: {str(e)}, feature={feature}, operator={operator}, value={value}, sample_value={sample_value}")
                    all_conditions_met = False
                    break
            
            if all_conditions_met:
                return True
                
        return False

    def export_dict(self) -> dict:
        """导出决策树的规则和元数据为字典格式"""
        return {
            "meta": {
                "features": [feature.name for feature in self.meta.features],
                "labels": [label.name for label in self.meta.labels],
            },
            "max_depth": self.max_depth,
            "rules": self.rules,
        }

    def _generate_default_rule(self):
        """生成最基本的默认规则，确保predict方法能够运行"""
        self.logger.log("[DEBUG] 生成默认规则...")
        
        # 确保每个标签都有对应的规则
        rules = []
        
        # 为每个标签生成一条规则
        if hasattr(self.meta, 'labels') and self.meta.labels:
            for i, label in enumerate(self.meta.labels):
                # 创建一个始终匹配的规则
                condition = {
                    'feature': self.feature_names[0] if self.feature_names else "特征1",
                    'operator': '>=' if i % 2 == 0 else '<',
                    'value': 0.0 if i % 2 == 0 else 0.0
                }
                
                rule = {
                    'conditions': [condition],
                    'label': label.name
                }
                rules.append(rule)
                self.logger.log(f"[DEBUG] 为标签 {label.name} 生成默认规则: {condition}")
        
        # 如果没有标签信息，创建通用规则
        if not rules:
            rule = {
                'conditions': [{
                    'feature': self.feature_names[0] if self.feature_names else "特征1",
                    'operator': '>=',
                    'value': 0.0
                }],
                'label': "default"
            }
            rules.append(rule)
            self.logger.log("[WARNING] 没有标签信息，生成通用默认规则")
        
        self.rules = rules
        self.logger.log(f"[INFO] 成功生成 {len(rules)} 条默认规则")

    def _generate_simple_rules(self, x_train, y_train):
        """生成简单的默认规则，基于特征统计和标签分布"""
        self.logger.log("[DEBUG] 生成简单默认规则...")
        self.logger.log(f"[DEBUG] x_train.shape={x_train.shape if hasattr(x_train, 'shape') else 'unknown'}, y_train.shape={y_train.shape if hasattr(y_train, 'shape') else 'unknown'}")
        
        # 初始化规则列表
        rules = []
        
        try:
            # 获取标签分布
            unique_labels, counts = np.unique(y_train, return_counts=True)
            self.logger.log(f"[DEBUG] 标签分布: {dict(zip(unique_labels, counts))}")
            
            if len(unique_labels) > 0:
                # 为每个唯一标签生成至少一条规则
                for label_value in unique_labels:
                    if len(self.meta.features) > 0:
                        # 选择一个特征
                        feature_idx = 0  # 默认使用第一个特征
                        feature = self.meta.features[feature_idx]
                        feature_name = feature.name
                        feature_values = x_train[:, feature_idx]
                        
                        # 找出此标签对应的样本
                        label_mask = y_train == label_value
                        if np.any(label_mask):
                            label_samples = x_train[label_mask]
                            
                            # 为分类特征和数值特征创建不同的规则
                            if feature.is_categorical and len(label_samples) > 0:
                                # 为分类特征，找出最常见的值
                                unique_values, f_counts = np.unique(label_samples[:, feature_idx], return_counts=True)
                                if len(unique_values) > 0:
                                    best_value = unique_values[np.argmax(f_counts)]
                                    
                                    # 获取标签名称
                                    label_name = self.meta.find_label(label_value).name
                                    self.logger.log(f"[DEBUG] 为标签 {label_name} ({label_value}) 创建基于分类特征 {feature_name} 的规则，值={best_value}")
                                    
                                    # 创建规则
                                    rule = {
                                        'conditions': [{
                                            'feature': feature_name,
                                            'operator': '==',
                                            'value': best_value
                                        }],
                                        'label': label_name
                                    }
                                    rules.append(rule)
                            else:
                                # 为数值特征，使用中位数作为分割点
                                if len(label_samples) > 0:
                                    median_value = np.median(label_samples[:, feature_idx])
                                    
                                    # 获取标签名称
                                    label_name = self.meta.find_label(label_value).name
                                    self.logger.log(f"[DEBUG] 为标签 {label_name} ({label_value}) 创建基于数值特征 {feature_name} 的规则，值={median_value}")
                                    
                                    # 创建规则
                                    rule = {
                                        'conditions': [{
                                            'feature': feature_name,
                                            'operator': '>=',
                                            'value': float(median_value)
                                        }],
                                        'label': label_name
                                    }
                                    rules.append(rule)
                
                # 保存生成的规则
                if rules:
                    self.rules = rules
                    self.logger.log(f"[INFO] 成功生成 {len(rules)} 条简单规则")
                    
                    # 打印生成的规则
                    for idx, rule in enumerate(rules):
                        conditions_str = " AND ".join([f"{c['feature']} {c['operator']} {c['value']}" for c in rule['conditions']])
                        self.logger.log(f"规则 {idx+1}: IF {conditions_str} THEN {rule['label']}")
                else:
                    # 如果没有生成规则，则调用生成默认规则的方法
                    self.logger.log("[WARNING] 未能生成规则，将调用默认规则生成方法")
                    self._generate_default_rule()
            else:
                # 如果没有标签数据，则调用生成默认规则的方法
                self.logger.log("[WARNING] 训练数据中没有标签，将调用默认规则生成方法")
                self._generate_default_rule()
        except Exception as e:
            # 如果出现任何异常，生成默认规则
            self.logger.log(f"[ERROR] 生成简单规则时出现异常：{str(e)}")
            self._generate_default_rule()
    
    def get_rules(self) -> list[str]:
        """返回所有路径规则，格式化为文本形式"""
        if not self.rules:
            self.logger.log("[ERROR] get_rules: self.rules为空！")
            return []
            
        # 支持多条件结构的格式化
        formatted_rules = []
        for idx, rule in enumerate(self.rules, 1):
            conditions = []
            if isinstance(rule, dict) and 'conditions' in rule and isinstance(rule['conditions'], list):
                for cond in rule['conditions']:
                    if all(k in cond for k in ['feature', 'operator', 'value']):
                        cond_str = f"{cond['feature']} {cond['operator']} {cond['value']}"
                        conditions.append(cond_str)
            
            # 检查rule是否为字典类型
            if isinstance(rule, dict):
                then_label = rule.get('label', '')
            else:
                # 如果rule是字符串，直接使用
                then_label = str(rule)
                self.logger.log(f"[WARNING] 规则类型为字符串: {then_label}")
                
            rule_text = f"({idx}) IF {' AND '.join(conditions)} THEN {then_label}"
            if not conditions:
                self.logger.log(f"[WARNING] 规则{idx}条件为空，rule内容: {rule}")
                continue
            formatted_rules.append(rule_text)
            
        if not formatted_rules:
            self.logger.log("[ERROR] get_rules: formatted_rules为空！")
            
        return formatted_rules


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
                        elif (operator == '=' or operator == '=='): match = abs(num_sample - num_value) < 1e-6
                        elif operator == '!=': match = abs(num_sample - num_value) >= 1e-6
                    else:
                        if operator in ['=', '==']: match = str_sample == str_value
                        elif operator == '!=': match = str_sample != str_value
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
                    elif operator in ['=', '==']: match = abs(num_sample - num_value) < 1e-6
                    elif operator == '!=': match = abs(num_sample - num_value) >= 1e-6
                else:
                    if operator in ['=', '==']: match = str_sample == str_value
                    elif operator == '!=': match = str_sample != str_value
                    elif operator in ['>', '<']:
                        match = str_sample == str_value
                if match:
                    return label
        self.logger.log("[ERROR] _apply_rules: 所有规则都未命中，返回默认label！")
        return self.meta.labels[0].value