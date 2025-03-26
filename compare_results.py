import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib
import os
import platform
from matplotlib import font_manager
import warnings

def setup_chinese_font():
    """设置中文字体"""
    system = platform.system()
    
    # 检测系统类型并尝试对应的常见中文字体
    if system == 'Windows':
        font_list = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']
    elif system == 'Darwin':  # macOS
        font_list = ['PingFang SC', 'Heiti SC', 'STHeiti', 'STSong', 'STFangsong']
    else:  # Linux 或其他系统
        font_list = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'AR PL UMing CN', 'AR PL KaitiM GB', 'Noto Sans CJK SC']
    
    # 尝试查找系统上可用的中文字体
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    
    # 打印可用字体，便于调试
    print("系统可用字体:")
    for af in sorted(set(available_fonts)):
        if any(c > '\u4e00' and c < '\u9fff' for c in af):  # 含有中文字符的字体名
            print(f" - {af}")
    
    # 尝试已知字体列表
    font_found = False
    for font_name in font_list:
        if font_name in available_fonts:
            try:
                plt.rcParams['font.family'] = font_name
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                
                # 测试字体
                fig = plt.figure(figsize=(1, 1))
                plt.text(0.5, 0.5, '测试中文显示')
                plt.close(fig)
                
                print(f"成功使用中文字体: {font_name}")
                font_found = True
                break
            except Exception as e:
                print(f"尝试使用字体 {font_name} 失败: {e}")
                continue
    
    # 如果已知列表中没有找到，尝试所有可能包含中文的字体
    if not font_found:
        for font_name in available_fonts:
            # 尝试可能的中文字体
            if any(c > '\u4e00' and c < '\u9fff' for c in font_name) or 'CJK' in font_name:
                try:
                    plt.rcParams['font.family'] = font_name
                    plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                    plt.rcParams['axes.unicode_minus'] = False
                    
                    # 测试字体
                    fig = plt.figure(figsize=(1, 1))
                    plt.text(0.5, 0.5, '测试中文显示')
                    plt.close(fig)
                    
                    print(f"成功使用中文字体: {font_name}")
                    font_found = True
                    break
                except:
                    continue
    


def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_metrics_from_baseline(data):
    """从结果数据中提取性能指标 - 基线版本"""
    train_sizes = sorted([int(k) for k in data['results'].keys()])
    metrics = {
        'train_size': [],
        'run': [],
        'auc': [],
        'accuracy': []
    }
    
    for size in train_sizes:
        for i, result in enumerate(data['results'][str(size)]):
            metrics['train_size'].append(size)
            metrics['run'].append(i)
            
            # 尝试提取AUC和准确率
            if 'auc' in result:
                metrics['auc'].append(result['auc'])
            elif 'llm_tree' in result:  # 适应不同的字段名称
                metrics['auc'].append(result['llm_tree'])
            else:
                metrics['auc'].append(None)
                
            if 'accuracy' in result:
                metrics['accuracy'].append(result['accuracy'])
            else:
                metrics['accuracy'].append(None)
    
    return pd.DataFrame(metrics)

def extract_metrics_from_test(data):
    """从结果数据中提取性能指标 - 测试版本"""
    train_sizes = sorted([int(k) for k in data['results'].keys()])
    metrics = {
        'train_size': [],
        'run': [],
        'llm_tree': [],  # Combined model
        'tree': []       # Tree only
    }
    
    for size in train_sizes:
        for i, result in enumerate(data['results'][str(size)]):
            metrics['train_size'].append(size)
            metrics['run'].append(i)
            metrics['llm_tree'].append(result.get('llm_tree', None))
            metrics['tree'].append(result.get('tree', None))
    
    return pd.DataFrame(metrics)

def calculate_statistics(df, group_col='train_size'):
    """计算每个训练集大小的平均指标"""
    # 确定需要聚合的列
    agg_cols = {}
    for col in df.columns:
        if col not in [group_col, 'run'] and not pd.isna(df[col]).all():
            agg_cols[col] = ['mean', 'std']
    
    return df.groupby(group_col).agg(agg_cols).reset_index()

def compare_results(no_tree_df, with_tree_df):
    """比较有无树规则的结果"""
    no_tree_stats = calculate_statistics(no_tree_df)
    with_tree_stats = calculate_statistics(with_tree_df)
    
    # 合并结果用于比较
    comparison = pd.DataFrame({
        'train_size': no_tree_stats['train_size'],
        'no_tree_auc': no_tree_stats['auc']['mean'],
        'no_tree_auc_std': no_tree_stats['auc']['std'],
        'with_tree_auc': with_tree_stats['auc']['mean'],
        'with_tree_auc_std': with_tree_stats['auc']['std'],
    })
    
    # 添加准确率列，如果存在
    if 'accuracy' in no_tree_stats.columns and 'accuracy' in with_tree_stats.columns:
        comparison['no_tree_acc'] = no_tree_stats['accuracy']['mean']
        comparison['no_tree_acc_std'] = no_tree_stats['accuracy']['std']
        comparison['with_tree_acc'] = with_tree_stats['accuracy']['mean']
        comparison['with_tree_acc_std'] = with_tree_stats['accuracy']['std']
        comparison['acc_improvement'] = comparison['with_tree_acc'] - comparison['no_tree_acc']
    
    # 计算改进
    comparison['auc_improvement'] = comparison['with_tree_auc'] - comparison['no_tree_auc'] 
    
    return comparison

def create_visualizations(comparison, test_stats, model_info, output_path):
    """创建可视化图表"""
    # 创建输出目录
    output_dir = Path("output/visualize")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确保output_path是完整路径
    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    # 更新输出路径为output/visualize目录
    output_path = output_dir / output_path.stem
    
    plt.figure(figsize=(16, 10))
    
    # 使用中文标题和标签
    labels = {
        'title1': '不同样本量下的AUC比较',
        'title3': '树规则带来的AUC改进',
        'xlabel': '训练样本数',
        'ylabel1': 'AUC',
        'ylabel3': 'AUC改进',
        'label1': '无树规则',
        'label2': '有树规则',
    }
    
    # 检查是否有准确率数据
    has_accuracy = 'no_tree_acc' in comparison.columns
    
    if has_accuracy:
        labels.update({
            'title2': '不同样本量下的准确率比较',
            'title4': '树规则带来的准确率改进',
            'ylabel2': '准确率',
            'ylabel4': '准确率改进',
        })
        num_plots = 4
    else:
        num_plots = 2
    
    # AUC比较图
    plt.subplot(2, num_plots//2, 1)
    plt.errorbar(comparison['train_size'], comparison['no_tree_auc'], 
                 yerr=comparison['no_tree_auc_std'], fmt='o-', label=labels['label1'])
    plt.errorbar(comparison['train_size'], comparison['with_tree_auc'], 
                 yerr=comparison['with_tree_auc_std'], fmt='s-', label=labels['label2'])
    
    # 如果有测试数据，添加到图表
    if test_stats is not None:
        # 确保得到的是标量而不是Series
        llm_tree_mean = float(test_stats['llm_tree']['mean'].iloc[0]) if isinstance(test_stats['llm_tree']['mean'], pd.Series) else float(test_stats['llm_tree']['mean'])
        tree_mean = float(test_stats['tree']['mean'].iloc[0]) if isinstance(test_stats['tree']['mean'], pd.Series) else float(test_stats['tree']['mean'])
        
        train_size = int(test_stats['train_size'].iloc[0])
        
        plt.scatter([train_size], [llm_tree_mean], 
                   marker='*', s=150, color='red', label='测试集LLM+树', zorder=5)
        plt.scatter([train_size], [tree_mean], 
                   marker='*', s=150, color='green', label='测试集树', zorder=5)
    
    plt.xlabel(labels['xlabel'], fontsize=12)
    plt.ylabel(labels['ylabel1'], fontsize=12)
    plt.title(labels['title1'], fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(comparison['train_size'])
    
    # 准确率比较图
    if has_accuracy:
        plt.subplot(2, 2, 2)
        plt.errorbar(comparison['train_size'], comparison['no_tree_acc'], 
                    yerr=comparison['no_tree_acc_std'], fmt='o-', label=labels['label1'])
        plt.errorbar(comparison['train_size'], comparison['with_tree_acc'], 
                    yerr=comparison['with_tree_acc_std'], fmt='s-', label=labels['label2'])
        plt.xlabel(labels['xlabel'], fontsize=12)
        plt.ylabel(labels['ylabel2'], fontsize=12)
        plt.title(labels['title2'], fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(comparison['train_size'])
    
    # AUC改进图
    plt.subplot(2, num_plots//2, num_plots//2 + 1)
    bars = plt.bar(comparison['train_size'].astype(str), comparison['auc_improvement'], color='skyblue')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    plt.xlabel(labels['xlabel'], fontsize=12)
    plt.ylabel(labels['ylabel3'], fontsize=12)
    plt.title(labels['title3'], fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 给每个柱子添加数值标签
    for bar in bars:
        height = bar.get_height()
        text_y_pos = height + 0.01 if height > 0 else height - 0.025
        plt.text(bar.get_x() + bar.get_width()/2., text_y_pos,
                f'{height:.3f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # 准确率改进图
    if has_accuracy:
        plt.subplot(2, 2, 4)
        bars = plt.bar(comparison['train_size'].astype(str), comparison['acc_improvement'], color='lightgreen')
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        plt.xlabel(labels['xlabel'], fontsize=12)
        plt.ylabel(labels['ylabel4'], fontsize=12)
        plt.title(labels['title4'], fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 给每个柱子添加数值标签
        for bar in bars:
            height = bar.get_height()
            text_y_pos = height + 0.005 if height > 0 else height - 0.015
            plt.text(bar.get_x() + bar.get_width()/2., text_y_pos,
                    f'{height:.3f}',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    plt.tight_layout(pad=3.0)

    suptitle = "树规则辅助LLM在糖尿病预测任务中的性能评估"
    # model_subtitle = ""
    # if model_info:
    #     model_subtitle = f"模型: {model_info['model']}"
    #     if 'max_depth' in model_info:
    #         model_subtitle += f" | 决策树深度: {model_info['max_depth']}"
    #     if 'lambda' in model_info:
    #         model_subtitle += f" | λ={model_info['lambda']:.1f}"
    #     if 'train_size' in model_info:
    #         model_subtitle += f" | 样本数: {model_info['train_size']}"

    plt.suptitle(suptitle, fontsize=16, y=1.02)
    # if model_info:
    #     plt.figtext(0.5, 0.96, model_subtitle, ha='center', fontsize=14)
    
    # 保存为多种格式，使用output/visualize目录
    plt.savefig(f"{output_path}_visualizations.png", dpi=300, bbox_inches='tight')

    
    # 单独创建平均改进图
    plt.figure(figsize=(10, 6))
    x = np.arange(len(comparison['train_size']))
    width = 0.35
    
    bar1 = plt.bar(x - width/2 if has_accuracy else x, comparison['auc_improvement'], 
                   width, label='AUC改进', color='skyblue')
    
    if has_accuracy:
        bar2 = plt.bar(x + width/2, comparison['acc_improvement'], 
                      width, label='准确率改进', color='lightgreen')
    
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    plt.xlabel('训练样本数', fontsize=12)
    plt.ylabel('性能改进', fontsize=12)
    title = '树规则带来的性能改进'
    if model_info and 'model' in model_info:
        title += f"\n模型: {model_info['model']}"
        if 'lambda' in model_info:
            title += f" | λ={model_info['lambda']:.1f}"
    
    plt.title(title, fontsize=14)
    plt.xticks(x, comparison['train_size'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # 给每个柱子添加数值标签
    for bar in bar1:
        height = bar.get_height()
        text_y_pos = height + 0.005 if height > 0 else height - 0.015
        plt.text(bar.get_x() + bar.get_width()/2., text_y_pos,
                f'{height:.3f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    if has_accuracy:
        for bar in bar2:
            height = bar.get_height()
            text_y_pos = height + 0.005 if height > 0 else height - 0.015
            plt.text(bar.get_x() + bar.get_width()/2., text_y_pos,
                    f'{height:.3f}',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}_improvements.png", dpi=300, bbox_inches='tight')
    # try:
    #     plt.savefig(f"{output_path.stem}_improvements.pdf", bbox_inches='tight')
    # except:
    #     pass
    
    # 如果有测试数据，创建一个单独的详细测试结果图
    if test_stats is not None:
        # 确保得到的是标量而不是Series
        llm_tree_mean = float(test_stats['llm_tree']['mean'].iloc[0]) if isinstance(test_stats['llm_tree']['mean'], pd.Series) else float(test_stats['llm_tree']['mean'])
        tree_mean = float(test_stats['tree']['mean'].iloc[0]) if isinstance(test_stats['tree']['mean'], pd.Series) else float(test_stats['tree']['mean'])
        
        plt.figure(figsize=(12, 6))
        
        # 左图：AUC对比
        plt.subplot(1, 2, 1)
        plt.bar([0, 1], [llm_tree_mean, tree_mean], color=['royalblue', 'forestgreen'])
        plt.xticks([0, 1], ['LLM+树规则', '仅树规则'])
        plt.ylabel('AUC', fontsize=12)
        plt.title(f'AUC对比 (样本量={int(test_stats["train_size"].iloc[0])})', fontsize=14)
        
        # 添加数值标签
        for i, v in enumerate([llm_tree_mean, tree_mean]):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
        
        # 右图：树规则详情
        plt.subplot(1, 2, 2)
        rules_text = model_info.get('rules', '未找到规则')
        lines = rules_text.split('\n')
        for i, row in enumerate(lines):
            if row.strip():
                plt.text(0, 1-i*0.10, row, fontsize=10, ha='left', va='top')
        plt.axis('off')
        plt.title('训练生成的树规则', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f"{output_path}_test_details.png", dpi=300, bbox_inches='tight')
    
def extract_tree_rules(test_data):
    """从测试数据中提取树规则"""
    train_size = list(test_data['results'].keys())[0]
    result = test_data['results'][train_size][0]  # 取第一个结果
    
    if 'model' in result and 'prompt' in result['model']:
        prompt = result['model']['prompt']
        
        # 从prompt中提取规则部分
        rules_start = prompt.find("The rules are as follows:")
        rules_end = prompt.find("For data points that are not covered by any rule")
        
        if rules_start != -1 and rules_end != -1:
            rules_text = prompt[rules_start:rules_end].strip()
            return rules_text
    
    return "未找到规则"

def extract_model_info(test_data):
    """提取模型信息和关键参数"""
    model_info = {}
    
    if 'args' in test_data:
        args = test_data['args']
        
        # 提取模型名称
        if 'runner_args' in args and 'model_name' in args['runner_args']:
            model_name = args['runner_args']['model_name']
            # 简化模型名称
            if '/' in model_name:
                model_name = model_name.split('/')[-1]
            model_info['model'] = model_name
        
        # 提取训练参数
        if 'strategy_args' in args:
            if 'max_depth' in args['strategy_args']:
                model_info['max_depth'] = args['strategy_args']['max_depth']
            if 'lambda_' in args['strategy_args']:
                model_info['lambda'] = args['strategy_args']['lambda_']
        
        # 提取训练样本数
        if 'train_sizes' in args:
            model_info['train_size'] = args['train_sizes'][0]
    
    # 提取树规则
    model_info['rules'] = extract_tree_rules(test_data)
    
    return model_info
    
def main(no_tree_file, with_tree_file, test_file, output_file):
    """主函数"""
    # 创建输出目录
    output_dir = Path("output/visualize")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 将输出文件路径转换为output/visualize目录下
    if isinstance(output_file, str):
        output_file = Path(output_file)
    
    output_file = output_dir / output_file.name
    
    # 设置中文字体
    setup_chinese_font()
    
    print(f"正在加载数据文件...")
    print(f"无树规则文件: {no_tree_file}")
    print(f"有树规则文件: {with_tree_file}")
    if test_file:
        print(f"测试文件: {test_file}")
    
    # 加载JSON文件
    no_tree_data = load_json(no_tree_file)
    with_tree_data = load_json(with_tree_file)
    
    # 加载测试文件（如果有）
    test_data = None
    if test_file:
        test_data = load_json(test_file)
    
    # 提取模型信息
    model_info = None
    if test_data:
        model_info = extract_model_info(test_data)
        print(f"提取的模型信息: {model_info}")
    
    # 提取数据
    print("提取基线数据...")
    no_tree_df = extract_metrics_from_baseline(no_tree_data)
    print(f"无树规则数据形状: {no_tree_df.shape}")
    print(no_tree_df.head())
    
    print("提取有树规则数据...")
    with_tree_df = extract_metrics_from_baseline(with_tree_data)
    print(f"有树规则数据形状: {with_tree_df.shape}")
    print(with_tree_df.head())
    
    # 比较结果
    print("比较性能差异...")
    comparison = compare_results(no_tree_df, with_tree_df)
    print("比较结果:")
    print(comparison)
    
    # 提取测试统计数据
    test_stats = None
    if test_data:
        print("提取测试数据...")
        test_df = extract_metrics_from_test(test_data)
        test_stats = calculate_statistics(test_df)
        print("测试统计:")
        print(test_stats)
    
    
    # 生成可视化
    print("生成可视化图表...")
    create_visualizations(comparison, test_stats, model_info, output_file)
    
    # 打印关键发现
    print("\n======= 关键发现 =======")
    print(f"平均AUC改进: {comparison['auc_improvement'].mean():.4f}")
    if 'acc_improvement' in comparison.columns:
        print(f"平均准确率改进: {comparison['acc_improvement'].mean():.4f}")
    
    print(f"AUC改进最大的训练集大小: {comparison.loc[comparison['auc_improvement'].idxmax(), 'train_size']} 样本 (+{comparison['auc_improvement'].max():.4f})")
    if 'acc_improvement' in comparison.columns:
        print(f"准确率改进最大的训练集大小: {comparison.loc[comparison['acc_improvement'].idxmax(), 'train_size']} 样本 (+{comparison['acc_improvement'].max():.4f})")
    
    # 创建报告并保存到output/visualize目录
    report_path = output_dir / f"{Path(output_file).stem}_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 树规则对糖尿病数据集预测性能的评估分析\n\n")
        
        if model_info:
            f.write(f"## 模型信息\n\n")
            f.write(f"- 模型: {model_info.get('model', '未知')}\n")
            if 'max_depth' in model_info:
                f.write(f"- 决策树最大深度: {model_info.get('max_depth', '未知')}\n")
            if 'lambda' in model_info:
                f.write(f"- 学习率 λ: {model_info.get('lambda', '未知')}\n")
            if 'train_size' in model_info:
                f.write(f"- 训练样本数: {model_info.get('train_size', '未知')}\n")
            f.write(f"\n### 生成的树规则\n\n```\n{model_info.get('rules', '未找到规则')}\n```\n\n")
        
        f.write("## 性能指标对比\n\n")
        f.write("### AUC值对比（平均值）\n")
        f.write("| 训练集大小 | 无树规则 | 有树规则 | 树规则提升 |\n")
        f.write("|-----------|---------|---------|----------|\n")
        for _, row in comparison.iterrows():
            f.write(f"| {int(row['train_size']):11d} | {row['no_tree_auc']:.4f} | {row['with_tree_auc']:.4f} | {row['auc_improvement']:.4f} |\n")
        
        if 'no_tree_acc' in comparison.columns:
            f.write("\n### 准确率对比（平均值）\n")
            f.write("| 训练集大小 | 无树规则 | 有树规则 | 树规则提升 |\n")
            f.write("|-----------|---------|---------|----------|\n")
            for _, row in comparison.iterrows():
                f.write(f"| {int(row['train_size']):11d} | {row['no_tree_acc']:.4f} | {row['with_tree_acc']:.4f} | {row['acc_improvement']:.4f} |\n")
        
        if test_stats is not None:
            f.write("\n### 测试结果\n")
            f.write("| 训练集大小 | LLM+树规则 | 仅树规则 | 差异 |\n")
            f.write("|-----------|----------|---------|------|\n")
            for i, row in test_stats.iterrows():
                llm_tree_mean = float(row['llm_tree']['mean']) if isinstance(row['llm_tree']['mean'], pd.Series) else row['llm_tree']['mean']
                tree_mean = float(row['tree']['mean']) if isinstance(row['tree']['mean'], pd.Series) else row['tree']['mean']
                diff = llm_tree_mean - tree_mean
                f.write(f"| {int(row['train_size']):11d} | {llm_tree_mean:.4f} | {tree_mean:.4f} | {diff:.4f} |\n")
        
        f.write("\n## 主要发现\n\n")
        f.write(f"1. **整体性能提升**：使用树规则辅助后，在大多数训练集大小下，模型性能都得到了提升。平均AUC提升{comparison['auc_improvement'].mean():.4f}")
        if 'acc_improvement' in comparison.columns:
            f.write(f"，平均准确率提升{comparison['acc_improvement'].mean():.4f}")
        f.write("。\n\n")
        
        best_size_auc = comparison.loc[comparison['auc_improvement'].idxmax(), 'train_size']
        f.write(f"2. **样本量影响**：AUC改进最显著的是{int(best_size_auc)}个样本配置(+{comparison['auc_improvement'].max():.4f})")
        
        if 'acc_improvement' in comparison.columns:
            best_size_acc = comparison.loc[comparison['acc_improvement'].idxmax(), 'train_size']
            f.write(f"，准确率改进最显著的是{int(best_size_acc)}个样本配置(+{comparison['acc_improvement'].max():.4f})")
        f.write("。\n\n")
        
        neg_improvements = comparison[comparison['auc_improvement'] < 0]
        if not neg_improvements.empty:
            f.write(f"3. **负面影响**：在{', '.join([str(int(size)) for size in neg_improvements['train_size']])}个样本配置下，树规则反而略微降低了AUC。\n\n")
        

    
    print(f"详细分析报告已保存到: {report_path}")
    print(f"可视化图表已保存到: {output_dir / f'{Path(output_file).stem}_visualizations.png'}")
    print(f"性能改进图已保存到: {output_dir / f'{Path(output_file).stem}_improvements.png'}")
    if test_stats is not None:
        print(f"测试详情图已保存到: {output_dir / f'{Path(output_file).stem}_test_details.png'}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='比较两个评估JSON文件的性能差异')
    parser.add_argument('no_tree_file', type=str, help='不使用树规则的评估结果JSON文件')
    parser.add_argument('with_tree_file', type=str, help='使用树规则的评估结果JSON文件')
    parser.add_argument('output_file', type=str, help='输出比较结果的文件名(将保存在output/visualize目录)')
    parser.add_argument('--test_file', type=str, help='测试结果JSON文件(可选)', default=None)
    
    args = parser.parse_args()
    
    main(args.no_tree_file, args.with_tree_file, args.test_file, args.output_file)