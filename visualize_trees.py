import json
import sys
from pathlib import Path
import numpy as np
import yaml
import graphviz
from tree_prompt.model.tree import DecisionTree, Node
import os
import pydot

class IdGenerator:
    def __init__(self):
        self.next_id = 0
        
    def get_id(self):
        id_val = self.next_id
        self.next_id += 1
        return id_val

# 全局ID生成器
id_generator = IdGenerator()

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
                            
                            # 提取特征排序信息
                            feature_map = extract_feature_shuffle_map(data)
                            
                            # 加载决策树
                            tree = DecisionTree(model_data['args'].get('max_depth', 5), categories_map)
                            tree.root_node = load_node_dict(nodes_dict, feature_map)
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
                
                # 提取特征排序信息
                feature_map = extract_feature_shuffle_map(data)
                
                # 加载决策树
                tree = DecisionTree(model_data['args'].get('max_depth', 5), categories_map)
                tree.root_node = load_node_dict(nodes_dict, feature_map)
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

def load_node_dict(node_dict, feature_map=None):
    """递归加载节点字典，支持特征映射"""
    if node_dict is None:
        return None
    
    # 输出当前节点字典，帮助调试
    print("节点字典结构:", list(node_dict.keys()))
    # 打印更详细的内容以调试
    print("节点字典详细内容:", node_dict)
    
    # 检查是否是使用数字索引的特殊格式(如血液数据集)
    if all(k.isdigit() for k in node_dict.keys()):
        print("检测到特殊格式(索引键)，尝试构建树...")
        return build_tree_from_indexed_dict(node_dict, feature_map)
    
    node = Node(None, 1)  # 深度将在后面设置
    node._id = id_generator.get_id()  # 添加ID属性
    
    # 使用get方法代替直接访问，提供默认值
    try:
        # 尝试多种可能的键名
        if 'feature' in node_dict:
            feature_idx = node_dict['feature']
            if feature_map and feature_idx is not None:
                # 使用反向映射，将训练特征索引映射回原始特征索引
                feature_key = str(feature_idx)
                if feature_key in feature_map:
                    node.split_feature = feature_map[feature_key]
                    print(f"特征映射: 训练特征{feature_idx} -> 原始特征{node.split_feature}")
                else:
                    node.split_feature = feature_idx
            else:
                node.split_feature = feature_idx
            
        elif 'split_feature' in node_dict:
            original_feature = node_dict['split_feature']
            if feature_map and original_feature is not None and str(original_feature) in feature_map:
                node.split_feature = feature_map[str(original_feature)]
            else:
                node.split_feature = original_feature
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
            left_child = load_node_dict(node_dict['left'], feature_map)
        elif 'left_child' in node_dict:
            left_child = load_node_dict(node_dict['left_child'], feature_map)
        
        if 'right' in node_dict:
            right_child = load_node_dict(node_dict['right'], feature_map)
        elif 'right_child' in node_dict:
            right_child = load_node_dict(node_dict['right_child'], feature_map)
        
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

def build_tree_from_indexed_dict(node_dict, feature_map=None):
    """从索引键的字典构建树结构，支持特征映射"""
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
            
            # 在处理特征时应用映射
            if feature_map and node.split_feature is not None and str(node.split_feature) in feature_map:
                node.split_feature = feature_map[str(node.split_feature)]
            
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

def custom_tree_to_graphviz_source(tree, feature_names, class_names, meta_data=None, feature_map=None, title=None, node_shape="ellipse", prefix="", output_file=None):
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
        
        lines.append(f'  node [shape={node_shape}, style="filled", color="black"')
        lines.append('  edge ;')
    
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
        
        # 处理非叶节点的特征信息
        if not node.is_leaf and node.split_feature is not None:
            feature_idx = node.split_feature
            # 应用特征名称
            if isinstance(feature_idx, (int, float)) and 0 <= feature_idx < len(feature_names):
                feature_name = feature_names[feature_idx]
            else:
                feature_name = f"特征{feature_idx}"
            
            # 获取特征类型信息
            is_categorical = determine_feature_type(feature_idx, feature_name, node.split_value, meta_data, node)
            
            # 根据特征类型选择正确的显示格式
            if is_categorical:
                split_condition = f"{feature_name} = {node.split_value}"
            else:
                split_condition = f"{feature_name} <= {node.split_value}"
                
            label_parts.append(split_condition)
        
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
        
        # 递归处理子节点 - 恢复正确的递归逻辑
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

def visualize_multiple_trees(trees_data, output_file, feature_names, class_names, meta_data=None, node_shape="ellipse"):
    """改进的多树合并可视化，每棵树垂直排列但整体横向排布"""
    # 确保输出文件有.png扩展名
    if not output_file.endswith('.png'):
        output_file = output_file + '.png'
    
    # 创建类别值到名称的映射字典
    class_map = {}
    if meta_data and 'labels' in meta_data:
        for label_info in meta_data['labels']:
            if 'value' in label_info and 'name' in label_info:
                # 确保使用正确的类型进行键映射
                try:
                    # 尝试整数映射
                    class_map[int(label_info['value'])] = label_info['name']
                except (ValueError, TypeError):
                    # 如果失败，尝试原始类型映射
                    class_map[label_info['value']] = label_info['name']
                print(f"添加标签映射: {label_info['value']} -> {label_info['name']}")

    # 如果没有从元数据中获取到映射，则使用class_names列表创建映射
    if not class_map and class_names:
        for i, name in enumerate(class_names):
            class_map[i] = name

    print(f"创建类别映射: {class_map}")
        
    # 创建一个包含独立子图的GraphViz图 - 整个图从左到右方向
    G = pydot.Dot(graph_type="digraph", 
                 compound=True, 
                 rankdir="LR",  # 整体从左到右
                 nodesep=0.7,   # 增加节点间距
                 ranksep=1.0,   # 增加层间距
                 label="合并森林 _ " + os.path.basename(output_file).split('.')[0])
    
    # 为每棵树创建一个独立的子图
    for i, (tree, name) in enumerate(trees_data):
        tree_id = f"tree{i}"
        
        # 创建子图，每棵树内部是从上到下排列
        subgraph = pydot.Cluster(tree_id, 
                                label=f"树 {i+1}: {name}", 
                                color="blue", 
                                style="rounded",
                                fontsize="16",  # 增加字体大小
                                rankdir="TB")  # 子图内从上到下
        
        # 重置ID生成器，确保每棵树使用独立的ID空间
        global id_generator
        id_generator = IdGenerator()
        
        # 添加树的所有节点到子图，传递类别映射
        add_tree_to_subgraph(tree, tree.root_node, subgraph, feature_names, class_map, 
                            meta_data, tree_id + "_", node_shape)
        
        G.add_subgraph(subgraph)
    
    # 保存合并后的图
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 设置更大的图像尺寸，确保所有树能够正确显示
    G.set_size('"20,15"')  # 增加为20英寸宽，15英寸高
    G.set_dpi("300")       # 设置更高的DPI（每英寸点数）
    
    # 设置全局字体大小
    G.set_fontsize("14")   # 增加全局字体大小
    
    # 添加图例（如果需要）
    G.set_margin("1,1")    # 设置页边距
    
    # 输出高质量PNG图像
    G.write_png(output_file)
    print(f"多树可视化已保存到 {output_file}")

class TreeVisualizer:
    def __init__(self, model_meta, output_dir="./"):
        self.model_meta = model_meta
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def visualize_tree(self, tree, output_file, title=None, node_shape="ellipse"):
        """将单棵决策树可视化为PNG图像"""
        # 创建输出目录
        output_dir = Path(os.path.dirname(output_file))
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建指向输出文件的绝对路径
        if not output_file.endswith('.png'):
            output_file = output_file + '.png'
        
        graph = pydot.Dot(graph_type='digraph')
        graph.set_rankdir('TB')  # 从上到下布局
        graph.set_node_defaults(shape=node_shape, style='filled', color='black')
        graph.set_edge_defaults(fontsize="12")
        
        if title:
            graph.set_label(title)
        
        # 递归构建树图
        if tree.root_node:
            self._build_tree_graph(tree, tree.root_node, graph)
        
        # 设置更高质量的输出
        graph.set_size('"12,10"')  # 12英寸宽，10英寸高
        graph.set_dpi("300")       # 设置更高的DPI（每英寸点数）
        graph.set_margin("0.5,0.5")  # 设置页边距
        
        # 保存为PNG图像
        graph.write_png(output_file)
        print(f"决策树可视化已保存到 {output_file}")
        
        return output_file
        
    def _build_tree_graph(self, tree, node, graph):
        """递归构建树图"""
        if node is None:
            return
            
        # 创建当前节点
        node_id = str(node._id)
        
        # 先将节点添加到图中，然后再设置属性
        graph.add_node(pydot.Node(node_id))
        
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
        node_id = str(node._id)
        nodes = graph.get_node(node_id)
        
        if not nodes:
            print(f"警告: 找不到节点 {node_id}")
            return
        
        graph_node = nodes[0]
        
        if node.split_feature is not None:
            try:
                # 直接使用节点的特征索引 - 这已经是映射后的原始特征索引
                feature_idx = node.split_feature
                
                # 获取特征名称 - 直接使用原始特征索引
                feature_name = f"特征{feature_idx}"
                if hasattr(self.model_meta, 'features') and isinstance(self.model_meta.features, list):
                    if 0 <= feature_idx < len(self.model_meta.features):
                        feature = self.model_meta.features[feature_idx]
                        if hasattr(feature, 'name'):
                            feature_name = feature.name
                        elif isinstance(feature, dict) and 'name' in feature:
                            feature_name = feature['name']
                
                # 判断特征是否为分类型
                is_categorical = False
                if hasattr(self.model_meta, 'features') and isinstance(self.model_meta.features, list):
                    if 0 <= feature_idx < len(self.model_meta.features):
                        feature = self.model_meta.features[feature_idx]
                        # 根据特征对象类型判断是否为分类特征
                        if hasattr(feature, 'is_categorical') and feature.is_categorical:
                            is_categorical = True
                        elif hasattr(feature, 'type') and feature.type in ['categorical', 'category']:
                            is_categorical = True
                        elif isinstance(feature, dict):
                            if feature.get('type') in ['categorical', 'category']:
                                is_categorical = True
                
                # 从值类型判断
                if isinstance(node.split_value, str) and not node.split_value.replace('.', '', 1).isdigit():
                    is_categorical = True
                
                # 根据特征类型设置操作符
                if is_categorical:
                    label = f"{feature_name} = {node.split_value}"
                else:
                    label = f"{feature_name} <= {node.split_value}"
                
                # 设置节点标签和样式
                graph_node.set_label(label)
                
                # 设置内部节点的样式 - 与多树显示一致
                graph_node.set_fillcolor("white")
                graph_node.set_fontcolor("black")
                graph_node.set_style("filled")
                graph_node.set_fontsize("14")
                graph_node.set_width("1.5")
                graph_node.set_height("0.8")
                
            except Exception as e:
                print(f"设置内部节点失败: {e}")
                import traceback
                traceback.print_exc()
                graph_node.set_label(f"特征{node.split_feature}")
                graph_node.set_fillcolor("white")
                graph_node.set_fontcolor("black")
                graph_node.set_style("filled")
                graph_node.set_fontsize("14")
    
    def _apply_class_format(self, tree, node, graph):
        """设置叶节点的样式和标签"""
        node_id = str(node._id)
        nodes = graph.get_node(node_id)
        if not nodes:
            return
        
        graph_node = nodes[0]
        
        try:
            # 获取叶节点的类别
            label_value = node.leaf_class
            print(f"叶节点标签值: {label_value}", end="")
            
            # 获取可用的标签
            labels = []
            if self.model_meta and hasattr(self.model_meta, 'labels'):
                labels = [(label.value, label.name) for label in self.model_meta.labels]
                print(f", 可用标签: {labels}")
            
            # 查找匹配的标签
            label_name = None
            for value, name in labels:
                print(f"比较: 节点值={label_value}, 标签定义值={value}, 标签名称={name}")
                if value == label_value:
                    label_name = name
                    print(f"找到匹配标签: {label_name}")
                    break
            
            if label_name is None:
                label_name = f"类别{label_value}"
            
            # 设置叶节点的标签和样式 - 与多棵树显示一致
            fillcolor = "#add8e6" if label_value == 0 else "#d3d3d3"  # 蓝色/灰色
            label = f"类别: {label_name}"
            
            # 更新：添加更多样式设置以与多树匹配
            graph_node.set_label(label)
            graph_node.set_style("filled")
            graph_node.set_fillcolor(fillcolor)
            graph_node.set_fontsize("14")  # 设置字体大小
            graph_node.set_width("1.2")    # 设置节点宽度
            graph_node.set_height("0.8")   # 设置节点高度
        except Exception as e:
            print(f"设置叶节点样式时出错: {e}")
            graph_node.set_label("叶节点")

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
                    str(output_file),
                    feature_names, 
                    class_names, 
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

def extract_feature_shuffle_map(json_data):
    """从JSON中提取特征随机打乱的映射关系"""
    shuffle_map = None
    
    # 先尝试从顶层args中找特征映射
    if 'args' in json_data and 'feature_shuffle_map' in json_data['args']:
        shuffle_map = json_data['args']['feature_shuffle_map']
        print(f"在顶层args中找到特征映射")
    
    # 再尝试从model.args中找
    elif 'model' in json_data and 'args' in json_data['model']:
        args = json_data['model']['args']
        if 'feature_shuffle_map' in args:
            shuffle_map = args['feature_shuffle_map']
            print(f"在model.args中找到特征映射")
    
    # 最后尝试在results中找
    elif 'results' in json_data:
        print(f"在results中查找特征映射")
        for size, results in json_data['results'].items():
            if isinstance(results, list) and results:
                for result in results:
                    # 直接在result.args中查找
                    if 'args' in result and 'feature_shuffle_map' in result['args']:
                        shuffle_map = result['args']['feature_shuffle_map']
                        print(f"在results[{size}].args中找到特征映射")
                        break
                    # 在result.model.args中查找
                    elif 'model' in result and 'args' in result['model']:
                        args = result['model']['args']
                        if 'feature_shuffle_map' in args:
                            shuffle_map = args['feature_shuffle_map']
                            print(f"在results[{size}].model.args中找到特征映射")
                            break
                if shuffle_map:
                    break
    
    # 如果找到了特征映射信息，直接返回训练到原始的映射关系
    if shuffle_map:
        print(f"找到特征随机打乱映射: {shuffle_map}")
        # 直接返回训练->原始的映射，不需要创建反向映射
        return shuffle_map
    
    print("未找到特征映射信息，将使用默认映射")
    return None

def parse_attributes(attr_str):
    """将GraphViz属性字符串解析为字典"""
    if not attr_str or not attr_str.strip():
        return {}
    
    # 去除开头的'['和结尾的']'
    attr_str = attr_str.strip()
    if attr_str.startswith('['):
        attr_str = attr_str[1:]
    if attr_str.endswith('];'):
        attr_str = attr_str[:-2]
    elif attr_str.endswith(']'):
        attr_str = attr_str[:-1]
    
    attrs = {}
    # 简单解析，不处理嵌套引号的复杂情况
    parts = []
    in_quotes = False
    current_part = ""
    
    # 分割属性，处理引号内的逗号
    for char in attr_str:
        if char == '"':
            in_quotes = not in_quotes
            current_part += char
        elif char == ',' and not in_quotes:
            parts.append(current_part.strip())
            current_part = ""
        else:
            current_part += char
    
    if current_part.strip():
        parts.append(current_part.strip())
    
    # 解析键值对
    for part in parts:
        if '=' in part:
            key, value = part.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # 去除值两侧的引号
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
                
            attrs[key] = value
    
    return attrs

def add_tree_to_subgraph(tree, node, subgraph, feature_names, class_map, meta_data, prefix="", node_shape="ellipse"):
    """递归添加树节点到子图"""
    if node is None:
        return
    
    # 为节点创建唯一ID
    node_id = prefix + str(node._id)
    
    # 创建节点和设置样式
    if node.is_leaf:
        # 叶节点
        try:
            # 确定叶节点类别
            label_value = node.leaf_class
            print(f"叶节点: ID={node_id}, 类别={label_value}", end="")
            
            # 使用类别映射来查找类别名称
            label_name = class_map.get(label_value, f"未知({label_value})")
            print(f", 类别名={label_name}")
            
            # 设置叶节点样式 - 根据类别决定颜色
            fillcolor = "#add8e6" if label_value == 0 else "#d3d3d3"  # 蓝色/灰色
            label = f"类别: {label_name}"
            subgraph.add_node(pydot.Node(node_id, label=label, 
                                    shape=node_shape, style="filled", 
                                    fillcolor=fillcolor, fontsize="14",
                                    width="1.2", height="0.8"))
        except Exception as e:
            print(f"设置叶节点失败: {e}")
            subgraph.add_node(pydot.Node(node_id, label=f"叶节点({label_value})", 
                                    shape=node_shape, style="filled", fontsize="14"))
    else:
        # 内部节点
        try:
            # 直接使用节点的特征索引 - 这已经是映射后的原始特征索引
            feature_idx = node.split_feature
            
            # 获取特征名称 - 直接使用原始特征索引访问feature_names
            feature_name = f"特征{feature_idx}"
            if feature_names and 0 <= feature_idx < len(feature_names):
                feature_name = feature_names[feature_idx]
            
            # 改进的特征类型判断逻辑
            is_categorical = False
            
            # 1. 首先检查节点自身的is_categorical属性
            if hasattr(node, 'is_categorical') and node.is_categorical:
                is_categorical = True
                print(f"节点 {node_id}: 从节点属性判断为分类特征")
            
            # 2. 从特征值类型判断 - 如果是非数字字符串，则为分类特征
            elif isinstance(node.split_value, str):
                try:
                    float(node.split_value)  # 尝试转换为数字
                except ValueError:
                    is_categorical = True
                    print(f"节点 {node_id}: 从特征值类型判断为分类特征")
            
            # 3. 从元数据中获取特征信息
            elif meta_data and 'features' in meta_data:
                for feature in meta_data['features']:
                    # 通过特征名称匹配
                    if 'name' in feature and feature['name'] == feature_name:
                        if feature.get('type') in ['categorical', 'category', 'string', 'enum']:
                            is_categorical = True
                            print(f"节点 {node_id}: 从元数据名称判断为分类特征")
                            break
                    
                    # 通过特征索引匹配
                    elif 'index' in feature and feature['index'] == feature_idx:
                        if feature.get('type') in ['categorical', 'category', 'string', 'enum']:
                            is_categorical = True
                            print(f"节点 {node_id}: 从元数据索引判断为分类特征")
                            break
            
            # 4. 特定特征名称启发式判断
            if not is_categorical:
                categorical_keywords = ['sex', 'gender', 'type', 'category', 'class', 'color', 'status']
                for keyword in categorical_keywords:
                    if keyword.lower() in feature_name.lower():
                        is_categorical = True
                        print(f"节点 {node_id}: 从特征名称启发式判断为分类特征")
                        break
            
            # 根据特征类型设置操作符
            if is_categorical:
                label = f"{feature_name} = {node.split_value}"
            else:
                label = f"{feature_name} <= {node.split_value}"
                
            # 设置节点样式
            subgraph.add_node(pydot.Node(node_id, label=label, 
                                    shape=node_shape, style="filled", 
                                    fillcolor="white", fontsize="14",
                                    fontname="helvetica",  # 确保字体一致
                                    width="1.5", height="0.8"))
            
            # 输出特征类型判断结果用于调试
            print(f"节点 {node_id}: 特征={feature_name}, 是否分类={is_categorical}, 操作符={'=' if is_categorical else '<='}")
            
        except Exception as e:
            print(f"设置内部节点失败: {e}")
            import traceback
            traceback.print_exc()
            subgraph.add_node(pydot.Node(node_id, label="内部节点", 
                                  shape=node_shape, style="filled", fontsize="14",
                                  fontname="helvetica"))
    
    # 递归处理子节点
    if node.left_child:
        left_id = prefix + str(node.left_child._id)
        add_tree_to_subgraph(tree, node.left_child, subgraph, feature_names, class_map, meta_data, prefix, node_shape)
        subgraph.add_edge(pydot.Edge(node_id, left_id, label="是", fontsize="12", fontname="helvetica"))
        
    if node.right_child:
        right_id = prefix + str(node.right_child._id)
        add_tree_to_subgraph(tree, node.right_child, subgraph, feature_names, class_map, meta_data, prefix, node_shape)
        subgraph.add_edge(pydot.Edge(node_id, right_id, label="否", fontsize="12", fontname="helvetica"))

# 添加通用的特征类型判断函数
def determine_feature_type(feature_idx, feature_name, split_value, meta_data=None, node=None):
    """通用特征类型判断函数，返回特征是否为分类型"""
    is_categorical = False
    
    # 1. 从元数据判断 - 通过索引或名称匹配
    if meta_data and 'features' in meta_data:
        for i, feature in enumerate(meta_data['features']):
            # 索引匹配 (两种可能的索引标识方式)
            idx_match = ('index' in feature and feature['index'] == feature_idx) or (i == feature_idx)
            # 名称匹配
            name_match = ('name' in feature and feature['name'] == feature_name)
            
            if idx_match or name_match:
                # 检查特征类型
                if feature.get('type') in ['categorical', 'category', 'string', 'enum']:
                    return True
                # 检查是否定义了分类值列表
                if 'categories' in feature or 'values' in feature:
                    return True
                break
    
    # 2. 从值类型判断 - 非数字字符串通常是分类特征
    if isinstance(split_value, str):
        # 尝试转换为数字
        try:
            float(split_value)
            # 成功转换，可能是数值型 (保持当前状态)
        except ValueError:
            # 不能转换为数字，确定是分类型
            return True
    
    # 3. 从节点属性判断 (如果提供了节点)
    if node and hasattr(node, 'is_categorical'):
        return node.is_categorical
    
    return is_categorical

def determine_feature_name(feature_idx, meta_data, feature_names=None):
    """根据特征索引获取特征的实际名称"""
    # 从meta_data中获取特征映射
    feature_map = {}
    if meta_data and 'args' in meta_data and 'feature_shuffle_map' in meta_data['args']:
        # 获取训练特征->原始特征映射
        feature_map = meta_data['args']['feature_shuffle_map']
    
    # 尝试应用特征映射获取原始特征索引
    original_idx = feature_idx
    if feature_map and str(feature_idx) in feature_map:
        original_idx = feature_map[str(feature_idx)]
        print(f"特征映射: 训练特征{feature_idx} -> 原始特征{original_idx}")
    
    # 使用原始特征索引获取特征名称
    feature_name = f"特征{feature_idx}"
    if feature_names and original_idx < len(feature_names):
        feature_name = feature_names[original_idx]
    
    return feature_name, original_idx

if __name__ == "__main__":
    main() 