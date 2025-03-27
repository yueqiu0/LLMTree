import json
import sys
from pathlib import Path
import numpy as np
import yaml
import graphviz
from tree_prompt.model.tree import DecisionTree, Node
import os
import pydot

def load_tree_from_json(json_file):
    """从JSON文件加载决策树模型，支持单树和森林"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    trees = []
    prompts = []
    
    try:
        # 首先尝试加载results下的所有结果
        if 'results' in data:
            all_results = []
            for exp_name, exp_results in data['results'].items():
                if isinstance(exp_results, list):
                    all_results.extend(exp_results)
            
            print(f"发现 {len(all_results)} 个训练结果")
            
            for i, result in enumerate(all_results):
                try:
                    if 'model' in result:
                        model_data = result['model']
                        model_type = model_data.get('type', '未知')
                        print(f"结果 {i+1}, 模型类型: {model_type}")
                        
                        # 检查是否为随机森林模型
                        if 'trees' in model_data.get('model', {}):
                            # 这是随机森林模型
                            forest_data = model_data['model']
                            print(f"检测到随机森林模型, 包含 {len(forest_data['trees'])} 棵树")
                            
                            categories_map = {}
                            if model_data['args'].get('categories'):
                                for k, v in model_data['args']['categories'].items():
                                    categories_map[int(k)] = set(v)
                            
                            for j, tree_data in enumerate(forest_data['trees']):
                                # 创建单棵树
                                tree = DecisionTree(model_data['args'].get('max_depth', 5), categories_map)
                                tree.root_node = load_node_dict(tree_data)
                                trees.append((tree, f"树 {j+1} (结果 {i+1})"))
                                prompts.append(model_data.get('prompt', ''))
                        else:
                            # 这是单棵决策树模型
                            nodes_dict = model_data['model']
                            
                            # 输出节点字典的顶层键，帮助调试
                            print(f"节点字典的顶层键: {list(nodes_dict.keys())}")
                            
                            # 创建决策树
                            categories_map = {}
                            if model_data['args'].get('categories'):
                                for k, v in model_data['args']['categories'].items():
                                    categories_map[int(k)] = set(v)
                            
                            # 加载决策树
                            tree = DecisionTree(model_data['args'].get('max_depth', 5), categories_map)
                            tree.root_node = load_node_dict(nodes_dict)
                            trees.append((tree, f"结果 {i+1}"))
                            prompts.append(model_data.get('prompt', ''))
                except Exception as e:
                    print(f"处理结果 {i} 时出错: {e}")
                    import traceback
                    traceback.print_exc()
        
        # 如果没有找到任何树，尝试直接加载单个模型
        if not trees and 'model' in data:
            model_data = data['model']
            model_type = model_data.get('type', '未知')
            print(f"直接加载模型, 类型: {model_type}")
            
            # 检查是否为随机森林模型
            if 'trees' in model_data.get('model', {}):
                # 这是随机森林模型
                forest_data = model_data['model']
                print(f"检测到随机森林模型, 包含 {len(forest_data['trees'])} 棵树")
                
                categories_map = {}
                if model_data['args'].get('categories'):
                    for k, v in model_data['args']['categories'].items():
                        categories_map[int(k)] = set(v)
                
                for j, tree_data in enumerate(forest_data['trees']):
                    # 创建单棵树
                    tree = DecisionTree(model_data['args'].get('max_depth', 5), categories_map)
                    tree.root_node = load_node_dict(tree_data)
                    trees.append((tree, f"树 {j+1}"))
                    prompts.append(model_data.get('prompt', ''))
            else:
                # 这是单棵决策树模型
                nodes_dict = model_data['model']
                
                # 输出节点字典的顶层键，帮助调试
                print(f"节点字典的顶层键: {list(nodes_dict.keys())}")
                
                # 创建决策树
                categories_map = {}
                if model_data['args'].get('categories'):
                    for k, v in model_data['args']['categories'].items():
                        categories_map[int(k)] = set(v)
                
                # 加载决策树
                tree = DecisionTree(model_data['args'].get('max_depth', 5), categories_map)
                tree.root_node = load_node_dict(nodes_dict)
                trees.append((tree, "单棵树"))
                prompts.append(model_data.get('prompt', ''))
        
        if not trees:
            print("未找到任何决策树模型")
            raise ValueError("JSON文件中未找到任何决策树模型")
        
        print(f"成功加载 {len(trees)} 棵树")
        return trees, data, prompts[0] if prompts else ""
        
    except Exception as e:
        print(f"加载树时出错: {e}")
        import traceback
        traceback.print_exc()
        raise

def load_node_dict(node_dict):
    """递归加载节点字典，增强容错性"""
    if node_dict is None:
        return None
    
    # 输出当前节点字典，帮助调试
    print("节点字典结构:", list(node_dict.keys()))
    # 打印更详细的内容以调试
    print("节点字典详细内容:", node_dict)
    
    # 检查是否是使用数字索引的特殊格式(如血液数据集)
    if all(k.isdigit() for k in node_dict.keys()):
        print("检测到特殊格式(索引键)，尝试构建树...")
        return build_tree_from_indexed_dict(node_dict)
    
    node = Node(None, 1)  # 深度将在后面设置
    
    # 使用get方法代替直接访问，提供默认值
    try:
        # 尝试多种可能的键名
        if 'feature' in node_dict:
            node.split_feature = node_dict['feature']
        elif 'split_feature' in node_dict:
            node.split_feature = node_dict['split_feature']
        else:
            # 如果找不到任何特征键，可能是叶节点
            node.split_feature = None
        
        # 尝试获取分裂值
        if 'split_value' in node_dict:
            node.split_value = node_dict['split_value']
        elif 'value' in node_dict:
            node.split_value = node_dict['value']
        else:
            node.split_value = None
        
        # 尝试获取预测值 - 增加更多可能的键名
        if 'prediction' in node_dict:
            node.leaf_class = node_dict['prediction']
        elif 'class' in node_dict:
            node.leaf_class = node_dict['class']
        elif 'leaf_class' in node_dict:
            node.leaf_class = node_dict['leaf_class']
        elif 'label' in node_dict:
            node.leaf_class = node_dict['label']
        elif 'predict' in node_dict:
            node.leaf_class = node_dict['predict']
        # 兼容树结构中可能出现的其他存储格式
        elif isinstance(node_dict.get('output'), dict) and 'prediction' in node_dict['output']:
            node.leaf_class = node_dict['output']['prediction']
        # 如果有特殊的类别映射，尝试解析
        elif isinstance(node_dict.get('output'), str) and node_dict['output'] in ['是', '1', 'Yes', 'True', 'true']:
            node.leaf_class = 1
        elif isinstance(node_dict.get('output'), str) and node_dict['output'] in ['否', '0', 'No', 'False', 'false']:
            node.leaf_class = 0
        else:
            node.leaf_class = None
        
        print(f"节点处理结果: 特征={node.split_feature}, 值={node.split_value}, 叶类别={node.leaf_class}")
        
        # 检查是否是叶节点的另一种方式：没有子节点且有output字段
        if (not node.leaf_class and 
            'left' not in node_dict and 
            'right' not in node_dict and 
            'left_child' not in node_dict and 
            'right_child' not in node_dict):
            # 这可能是叶节点，尝试推断类别
            if 'output' in node_dict:
                if isinstance(node_dict['output'], (int, float)):
                    node.leaf_class = int(node_dict['output'])
                elif isinstance(node_dict['output'], str):
                    if node_dict['output'].lower() in ['yes', 'true', '1', 'positive', '是']:
                        node.leaf_class = 1
                    else:
                        node.leaf_class = 0
        
        node.is_leaf = node.leaf_class is not None
        node.is_categorical = False  # 默认为数值型
        
        # 尝试获取子节点
        left_child = None
        right_child = None
        
        if 'left' in node_dict:
            left_child = load_node_dict(node_dict['left'])
        elif 'left_child' in node_dict:
            left_child = load_node_dict(node_dict['left_child'])
        
        if 'right' in node_dict:
            right_child = load_node_dict(node_dict['right'])
        elif 'right_child' in node_dict:
            right_child = load_node_dict(node_dict['right_child'])
        
        if left_child:
            left_child.parent = node
            left_child.depth = node.depth + 1
        node.left_child = left_child
        
        if right_child:
            right_child.parent = node
            right_child.depth = node.depth + 1
        node.right_child = right_child
        
        return node
    except Exception as e:
        print(f"处理节点时出错: {e}")
        print("节点字典:", node_dict)
        raise

def build_tree_from_indexed_dict(node_dict):
    """从索引键的字典构建树结构"""
    try:
        # 先创建所有节点
        nodes = {}
        for idx, node_info in node_dict.items():
            idx = int(idx)
            node = Node(None, 1)  # 深度将在后续设置
            
            # 打印详细信息以调试
            print(f"处理索引节点 {idx}, 内容: {node_info}")
            
            # 设置节点属性，使用更安全的get方法
            node.split_feature = node_info.get('feature')
            node.split_value = node_info.get('value')
            
            # 增强叶节点识别能力
            # 尝试获取预测值 - 增加更多可能的键名
            if 'prediction' in node_info:
                node.leaf_class = node_info['prediction']
            elif 'class' in node_info:
                node.leaf_class = node_info['class']
            elif 'leaf_class' in node_info:
                node.leaf_class = node_info['leaf_class']
            elif 'label' in node_info:
                node.leaf_class = node_info['label']
            elif 'predict' in node_info:
                node.leaf_class = node_info['predict']
            # 兼容树结构中可能出现的其他存储格式
            elif isinstance(node_info.get('output'), dict) and 'prediction' in node_info['output']:
                node.leaf_class = node_info['output']['prediction']
            # 如果有特殊的类别映射，尝试解析
            elif isinstance(node_info.get('output'), str) and node_info['output'] in ['是', '1', 'Yes', 'True', 'true']:
                node.leaf_class = 1
            elif isinstance(node_info.get('output'), str) and node_info['output'] in ['否', '0', 'No', 'False', 'false']:
                node.leaf_class = 0
            else:
                # 检查是否是叶节点的特殊情况：没有左右子节点且有output字段
                if 'left' not in node_info and 'right' not in node_info:
                    if 'output' in node_info:
                        if isinstance(node_info['output'], (int, float)):
                            node.leaf_class = int(node_info['output'])
                        elif isinstance(node_info['output'], str):
                            if node_info['output'].lower() in ['yes', 'true', '1', 'positive', '是']:
                                node.leaf_class = 1
                            else:
                                node.leaf_class = 0
            
            node.is_leaf = node.leaf_class is not None
            node.is_categorical = False
            
            print(f"节点 {idx} 处理结果: 特征={node.split_feature}, 值={node.split_value}, 叶类别={node.leaf_class}, 是叶节点={node.is_leaf}")
            
            nodes[idx] = node
        
        # 构建树结构
        root = None
        for idx_str, node_info in node_dict.items():
            idx = int(idx_str)
            if idx not in nodes:
                continue
                
            node = nodes[idx]
            
            # 查找左子节点
            left_idx = node_info.get('left')
            if left_idx is not None and left_idx in nodes:
                node.left_child = nodes[left_idx]
                node.left_child.parent = node
            
            # 查找右子节点
            right_idx = node_info.get('right')
            if right_idx is not None and right_idx in nodes:
                node.right_child = nodes[right_idx]
                node.right_child.parent = node
            
            # 如果这是根节点(没有父节点)
            if not hasattr(node, 'parent') or node.parent is None:
                root = node
        
        # 如果没有找到根节点，使用第一个节点作为根
        if root is None and nodes:
            root = next(iter(nodes.values()))
            print("警告: 未找到明确的根节点，使用第一个节点作为根")
        
        # 设置所有节点的深度
        def set_depths(node, depth=1):
            if node is None:
                return
            node.depth = depth
            set_depths(node.left_child, depth + 1)
            set_depths(node.right_child, depth + 1)
        
        if root:
            set_depths(root)
        
        return root
    except Exception as e:
        print(f"构建索引树时出错: {e}")
        import traceback
        traceback.print_exc()
        # 返回一个简单的单节点树，避免后续处理失败
        node = Node(None, 1)
        node.is_leaf = True
        node.leaf_class = 0
        return node

def load_metadata_from_yml(meta_file_path):
    """从meta.yml文件加载元数据"""
    try:
        with open(meta_file_path, 'r', encoding='utf-8') as f:
            meta_data = yaml.safe_load(f)
            
        # 提取特征和标签信息
        feature_names = [f['name'] for f in meta_data.get('features', [])]
        class_names = [l['name'] for l in meta_data.get('labels', [])]
        
        return feature_names, class_names
    except Exception as e:
        print(f"从元数据文件加载失败: {e}")
        return None, None

def extract_feature_names_from_prompt(prompt):
    """从模型的prompt中提取特征和类别名称"""
    feature_names = []
    class_names = []
    
    # 尝试从prompt中提取特征名称
    if "features" in prompt and "consist" in prompt:
        try:
            features_section = prompt.split("following features:")[1].split("For each data")[0]
            lines = features_section.strip().split("\n")
            for line in lines:
                if ":" in line and "(" in line:
                    feature_name = line.split(":")[0].split(")")[-1].strip()
                    feature_names.append(feature_name)
        except:
            pass
    
    # 尝试从prompt中提取类别名称
    if "result can be" in prompt:
        try:
            classes_section = prompt.split("result can be one of the following:")[1].split("Here's how")[0]
            lines = classes_section.strip().split("\n")
            for line in lines:
                if ":" in line and "(" in line:
                    class_name = line.split(":")[0].split(")")[-1].strip()
                    class_names.append(class_name)
        except:
            pass
    
    # 如果无法提取，返回默认值
    if not feature_names:
        feature_names = [f"特征_{i}" for i in range(10)]
    
    if not class_names:
        class_names = [f"类别_{i}" for i in range(5)]
    
    return feature_names, class_names

def custom_tree_to_graphviz_source(tree, feature_names, class_names, meta_data=None, title=None, node_shape="ellipse", prefix="", output_file=None):
    """自定义方法生成决策树的DOT源代码"""
    lines = []
    node_ids = {}
    next_id = 1
    
    # 添加图形声明不需要在这里，由外部处理
    if not prefix:
        lines.append('digraph Tree {')
        
        # 添加标题
        if title:
            lines.append(f'  label="{title}";')
            lines.append('  labelloc="t";')
        
        lines.append(f'  node [shape={node_shape}, style="filled", color="black", fontname="helvetica"];')
        lines.append('  edge [fontname="helvetica"];')
    
    # 递归生成树节点和边
    def _dfs(node, parent_id=None, left=True):
        if node is None:
            return
        
        # 为当前节点分配ID
        nonlocal next_id
        node_id = f"{prefix}{next_id}"
        node_ids[node] = node_id
        next_id = next_id + 1
        
        # 生成节点标签
        label_parts = []
        
        # 添加决策特征（对于非叶节点）
        if not node.is_leaf:
            feature_idx = node.split_feature
            if feature_idx is not None and 0 <= feature_idx < len(feature_names):
                feature_name = feature_names[feature_idx]
                split_condition = f"{feature_name} <= {node.split_value}" if isinstance(node.split_value, (int, float)) else f"{feature_name} = {node.split_value}"
                label_parts.append(split_condition)
            else:
                label_parts.append(f"特征 {feature_idx} 条件")
        
        # 添加叶节点标签
        if node.is_leaf:
            leaf_val = node.leaf_class
            # 查找正确的类别名称
            label_name = "未知"
            
            # 打印调试信息
            print(f"叶节点: ID={node_id}, 类别={leaf_val}, 类别名=", end="")
            
            # 尝试通过meta_data查找类别名称
            if meta_data and 'labels' in meta_data:
                for l in meta_data.get('labels', []):
                    if int(l.get('value', -1)) == int(leaf_val):
                        label_name = l.get('name', f"未知")
                        break
            
            print(f"{label_name}")
            
            # 格式化标签文本 - 只在叶节点添加类别信息，深度将在后面统一添加
            label_parts.append(f"类别: {label_name}")
        

        
        # 合并所有标签部分
        label = "\\n".join(label_parts)
        
        # 设置节点颜色 - 确保正确处理叶节点
        if node.is_leaf:
            if node.leaf_class == 1:
                fillcolor = "#E58139"  # 正类的颜色
            elif node.leaf_class == 0:
                fillcolor = "#399DE5"  # 负类的颜色
            else:
                fillcolor = "#AAAAAA"  # 未知类别的颜色
            color_saturation = 1.0
        else:
            fillcolor = "#FFFFFF"
            color_saturation = 0.0
        
        # 添加节点定义
        lines.append(f'  {node_id} [label="{label}", fillcolor="{fillcolor}", saturation="{color_saturation}"];')
        
        # 添加边（如果有父节点）
        if parent_id:
            direction = "是" if left else "否"
            lines.append(f'  {parent_id} -> {node_id} [label="{direction}"];')
        
        # 递归处理子节点
        if node.left_child:
            _dfs(node.left_child, node_id, True)
        if node.right_child:
            _dfs(node.right_child, node_id, False)
    
    # 开始递归
    if tree.root_node:
        _dfs(tree.root_node)
    
    # 结束图形定义
    if not prefix:
        lines.append('}')
    
    # 返回源代码而不是尝试渲染
    dot_source = '\n'.join(lines)
    return dot_source

def visualize_multiple_trees(trees_data, feature_names, class_names, output_file, title=None, meta_data=None, node_shape="ellipse"):
    """将多棵树绘制在一个图中"""
    # 开始生成DOT源代码
    lines = []
    lines.append('digraph Forest {')
    
    # 添加标题
    if title:
        safe_title = ''.join(c if c.isalnum() or c.isspace() else '_' for c in title)
        lines.append(f'  label="{safe_title}";')
        lines.append('  labelloc="t";')
    
    lines.append('  rankdir=LR;')  # 左到右布局
    lines.append(f'  node [shape={node_shape}, style="filled", color="black", fontname="helvetica"];')
    lines.append('  edge [fontname="helvetica"];')
    
    # 为每棵树创建子图
    for i, (tree, tree_name) in enumerate(trees_data):
        # 为子图创建一个安全的名称
        safe_tree_name = ''.join(c if c.isalnum() else '_' for c in tree_name)
        lines.append(f'  subgraph cluster_{i} {{')
        lines.append(f'    label="{tree_name}";')
        lines.append('    color=blue;')
        lines.append('    style=rounded;')
        
        # 生成这棵树的DOT代码
        try:
            tree_dot = custom_tree_to_graphviz_source(
                tree, feature_names, class_names, meta_data, None, node_shape, f"tree{i}_"
            )
            
            # 将树的DOT代码添加到子图中
            for line in tree_dot.split('\n'):
                if line and not line.startswith('digraph') and not line.endswith('{') and not line == '}':
                    lines.append(f'    {line}')
        except Exception as e:
            print(f"生成树 {i} 的DOT代码时出错: {e}")
            lines.append(f'    tree{i}_1 [label="无法渲染树 {i}\\n{str(e)}"];')
        
        lines.append('  }')
    
    # 结束图形定义
    lines.append('}')
    
    dot_source = '\n'.join(lines)
    
    # 创建并保存图像
    try:
        dot = graphviz.Source(dot_source)
        dot.render(output_file, format='png', cleanup=True)
        print(f"多树可视化已保存到 {output_file}.png")
    except Exception as e:
        print(f"渲染失败: {e}")
        with open(f"{output_file}.dot", "w") as f:
            f.write(dot_source)
        print(f"DOT源代码已保存到 {output_file}.dot")

class TreeVisualizer:
    def __init__(self, model_meta, output_dir="./output/viz"):
        self.model_meta = model_meta
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def visualize_tree(self, tree, filename):
        """可视化单个决策树并保存"""
        graph = pydot.Dot(graph_type='digraph')
        self._build_tree_graph(tree, tree.root_node, graph)
        graph.write_png(os.path.join(self.output_dir, filename))
        
    def _build_tree_graph(self, tree, node, graph):
        """递归构建树图"""
        if node is None:
            return
            
        # 创建当前节点
        node_id = str(node._id)
        if node.is_leaf:
            # 应用叶节点格式
            self._apply_class_format(tree, node, graph)
        else:
            # 应用内部节点格式
            self._apply_split_format(tree, node, graph)
            
        # 处理子节点
        if node.left_child:
            left_id = str(node.left_child._id)
            if not graph.get_node(left_id):
                self._build_tree_graph(tree, node.left_child, graph)
            graph.add_edge(pydot.Edge(node_id, left_id, label="是"))
            
        if node.right_child:
            right_id = str(node.right_child._id)
            if not graph.get_node(right_id):
                self._build_tree_graph(tree, node.right_child, graph)
            graph.add_edge(pydot.Edge(node_id, right_id, label="否"))
            
    def _apply_split_format(self, tree, node, graph):
        """设置内部节点的样式和标签"""
        if node.split_feature is not None:
            feature_name = self.model_meta.features[node.split_feature].name
            if node.is_categorical:
                label = f"{feature_name} = {node.split_value}"
            else:
                label = f"{feature_name} <= {node.split_value}"
            graph.get_node(str(node._id))[0].set_label(f"{label}\n")
            
    def _apply_class_format(self, tree, node, graph):
        """设置叶节点的样式和标签"""
        if hasattr(node, 'leaf_class') and node.leaf_class is not None:
            # 打印调试信息
            print(f"叶节点标签值: {node.leaf_class}, 可用标签: {[(l.value, l.name) for l in self.model_meta.labels]}")
            
            # 查找标签名称
            label_name = None
            for label in self.model_meta.labels:
                print(f"比较: 节点值={node.leaf_class}, 标签定义值={label.value}, 标签名称={label.name}")
                if int(label.value) == int(node.leaf_class):  # 确保类型一致
                    label_name = label.name
                    print(f"找到匹配标签: {label_name}")
                    break
            
            if label_name is None:
                label_name = f"类别: {node.leaf_class}"
            else:
                label_name = f"类别: {label_name}"
            
            # 设置叶节点标签和样式
            graph.get_node(str(node._id))[0].set_label(f"{label_name}\n")
            
            # 根据类别设置不同颜色
            if node.leaf_class == 3:  # good类(值为3)设为灰色
                fillcolor = "#AAAAAA"
            else:  # unacceptable类(值为0)设为蓝色
                fillcolor = "#6699CC"
            
            graph.get_node(str(node._id))[0].set_fillcolor(fillcolor)
            graph.get_node(str(node._id))[0].set_style('filled')
            graph.get_node(str(node._id))[0].set_shape('ellipse')

def print_tree_info(tree):
    """输出树的基本信息"""
    # 计算树深度
    def get_depth(node):
        if node is None:
            return 0
        if node.is_leaf:
            return node.depth
        return max(get_depth(node.left_child), get_depth(node.right_child))
    
    # 计算节点数量
    def count_nodes(node):
        if node is None:
            return 0
        return 1 + count_nodes(node.left_child) + count_nodes(node.right_child)
    
    # 计算叶子节点数量
    def count_leaves(node):
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return count_leaves(node.left_child) + count_leaves(node.right_child)
    
    # 收集分裂特征
    def collect_features(node, features=None):
        if features is None:
            features = set()
        if node is None:
            return features
        if not node.is_leaf and node.split_feature is not None:  # 确保split_feature不是None
            features.add(node.split_feature)
        collect_features(node.left_child, features)
        collect_features(node.right_child, features)
        return features
    
    depth = get_depth(tree.root_node)
    nodes = count_nodes(tree.root_node)
    leaves = count_leaves(tree.root_node)
    features = collect_features(tree.root_node)
    
    # 过滤掉None值，然后再排序
    valid_features = [f for f in features if f is not None]
    
    print(f"  最大深度: {depth}")
    print(f"  总节点数: {nodes}")
    print(f"  叶节点数: {leaves}")
    print(f"  使用的特征: {sorted(valid_features) if valid_features else '无有效特征'}")

def main():
    if len(sys.argv) < 2:
        print("用法: python visualize_trees.py <json文件> [meta文件路径] [--merge] [--ellipse]")
        return
        
    json_files = []
    meta_file = None
    dataset_type = None
    merge_trees = False
    node_shape = "ellipse"  # 默认使用椭圆形
    
    # 处理命令行参数
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.endswith('.json'):
            json_files.append(arg)
        elif arg.endswith('.yml') or arg.endswith('.yaml'):
            meta_file = arg
        elif arg in ['car', 'blood', 'diabetes']:
            dataset_type = arg
        elif arg == '--merge':
            merge_trees = True
        elif arg == '--ellipse':
            node_shape = "ellipse"
        elif arg == '--box':
            node_shape = "box"
        i += 1
    
    if not json_files:
        print("错误: 请提供至少一个JSON文件")
        return
    
    # 根据JSON文件名或显式指定的数据集类型推断元数据文件
    if not meta_file:
        if dataset_type:
            meta_file = f"dataset/{dataset_type}/meta.yml"
        else:
            # 从JSON文件名推断数据集类型
            for known_type in ['car', 'blood', 'diabetes']:
                if any(known_type in json_file for json_file in json_files):
                    meta_file = f"dataset/{known_type}/meta.yml"
                    break
            
            # 如果仍然没有找到，使用默认值
            if not meta_file:
                meta_file = "dataset/car/meta.yml"
    
    print(f"使用元数据文件: {meta_file}")
    print(f"节点形状: {node_shape}")
    
    # 加载特征和类别名称
    if meta_file and os.path.exists(meta_file):
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta_data = yaml.safe_load(f)
            
        feature_names = [f['name'] for f in meta_data.get('features', [])]
        class_names = [l['name'] for l in meta_data.get('labels', [])]
        
        # 创建简单的类来存储元数据信息
        class SimpleFeature:
            def __init__(self, name, desc="", is_categorical=False, categories=None):
                self.name = name
                self.desc = desc
                self.is_categorical = is_categorical
                self.categories = categories or []
                
        class SimpleLabel:
            def __init__(self, name, value, desc=""):
                self.name = name
                self.value = value
                self.desc = desc
        
        # 创建简化版元数据对象
        class SimpleMetaData:
            def __init__(self, name):
                self.name = name
                self.features = []
                self.labels = []
        
        # 手动构建元数据对象
        model_meta = SimpleMetaData(meta_data.get('name', '未知'))
        
        # 添加特征
        for f in meta_data.get('features', []):
            is_cat = 'categories' in f
            cats = list(f['categories'].keys()) if is_cat else []
            feature = SimpleFeature(f['name'], f.get('desc', ''), is_cat, cats)
            model_meta.features.append(feature)
        
        # 添加标签
        for l in meta_data.get('labels', []):
            # 确保value是整数类型
            label_value = int(l.get('value', 0))
            label = SimpleLabel(l['name'], label_value, l.get('desc', ''))
            print(f"加载标签: 名称={l['name']}, 值={label_value}")
            model_meta.labels.append(label)
    else:
        feature_names = []
        class_names = []
        model_meta = None
    
    # 创建输出目录
    output_dir = Path("output/visualize")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if merge_trees and len(json_files) == 1:
        # 只处理一个文件，但要将其所有树合并显示
        json_file = json_files[0]
        print(f"\n处理文件: {json_file}")
        
        try:
            # 加载树（可能是多棵）
            trees_data, data, prompt = load_tree_from_json(json_file)
            
            if len(trees_data) <= 1:
                print("只有一棵树，无需合并显示")
                merge_trees = False
            else:
                # 如果无法从元数据加载，则尝试从提示中提取
                if not feature_names or not class_names:
                    extracted_features, extracted_classes = extract_feature_names_from_prompt(prompt)
                    
                    if not feature_names and extracted_features:
                        feature_names = extracted_features
                    
                    if not class_names and extracted_classes:
                        class_names = extracted_classes
                
                # 确保特征名称和类别名称足够
                max_feature_idx = 0
                for tree, _ in trees_data:
                    used_features = collect_used_features(tree.root_node)
                    if used_features:
                        max_feature_idx = max(max_feature_idx, max(used_features))
                
                # 如果特征名称不足，补充默认名称
                while len(feature_names) <= max_feature_idx:
                    feature_names.append(f"特征_{len(feature_names)}")
                
                # 补充类别名称
                if not class_names or len(class_names) < 2:
                    class_names = ["类别0", "类别1"]
                
                # 设置合并图的输出文件路径
                output_file = output_dir / f"{Path(json_file).stem}_merged"
                
                # 可视化合并的树
                print(f"\n生成合并的树图 ({len(trees_data)} 棵树)")
                visualize_multiple_trees(
                    trees_data, 
                    feature_names, 
                    class_names, 
                    str(output_file),
                    title=f"合并森林 _ {Path(json_file).stem}",
                    meta_data=meta_data,  # 传递原始YAML数据
                    node_shape=node_shape
                )
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 如果不合并或者有多个文件，分别处理每个文件
    if not merge_trees or len(json_files) > 1:
        for json_file in json_files:
            print(f"\n处理文件: {json_file}")
            
            try:
                # 加载树（可能是多棵）
                trees_data, data, prompt = load_tree_from_json(json_file)
                
                # 如果无法从元数据加载，则尝试从提示中提取
                if not feature_names or not class_names:
                    extracted_features, extracted_classes = extract_feature_names_from_prompt(prompt)
                    
                    if not feature_names and extracted_features:
                        feature_names = extracted_features
                    
                    if not class_names and extracted_classes:
                        class_names = extracted_classes
                
                # 为每棵树创建可视化
                for i, (tree, tree_name) in enumerate(trees_data):
                    print(f"\n处理树 {i+1}/{len(trees_data)}: {tree_name}")
                    
                    # 设置输出文件路径
                    if len(trees_data) == 1:
                        output_file = output_dir / Path(json_file).stem
                    else:
                        output_file = output_dir / f"{Path(json_file).stem}_tree{i+1}"
                    
                    # 确保特征名称和类别名称数量与树中使用的一致
                    used_features = collect_used_features(tree.root_node)
                    max_feature_idx = max(used_features) if used_features else 0
                    
                    # 如果特征名称不足，补充默认名称
                    while len(feature_names) <= max_feature_idx:
                        feature_names.append(f"特征_{len(feature_names)}")
                    
                    # 补充类别名称
                    if not class_names or len(class_names) < 2:
                        class_names = ["类别0", "类别1"]
                        
                    print(f"使用特征名称: {feature_names}")
                    print(f"使用类别名称: {class_names}")
                    
                    # 使用正确的model_meta对象初始化TreeVisualizer
                    visualizer = TreeVisualizer(model_meta)
                    visualizer.visualize_tree(tree, str(output_file))
                    
                    # 输出树信息
                    print("\n决策树信息:")
                    print_tree_info(tree)
                
            except Exception as e:
                print(f"处理文件 {json_file} 时出错: {e}")
                import traceback
                traceback.print_exc()

# 辅助函数，收集树中使用的所有特征
def collect_used_features(node, features=None):
    if features is None:
        features = set()
    
    if node is None:
        return features
    
    if (not getattr(node, 'is_leaf', False) and 
        getattr(node, 'split_feature', None) is not None):  # 确保split_feature不是None
        features.add(node.split_feature)
    
    collect_used_features(getattr(node, 'left_child', None), features)
    collect_used_features(getattr(node, 'right_child', None), features)
    
    return features

if __name__ == "__main__":
    main() 