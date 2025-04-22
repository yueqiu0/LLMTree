import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def setup_chinese_font():
    """简化版中文字体设置"""
    plt.rcParams['font.family'] = 'Microsoft YaHei'  # 直接指定常见字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def load_simple_metrics(file_path):
    """加载简化指标"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 提取所有结果中的指标
    metrics = {'auc': [], 'accuracy': []}
    for size_data in data['results'].values():
        for run in size_data:
            metrics['auc'].append(run['auc'])
            metrics['accuracy'].append(run['accuracy'])
    
    # 计算平均值
    return {
        'auc_mean': np.mean(metrics['auc']),
        'acc_mean': np.mean(metrics['accuracy']),
        'auc_std': np.std(metrics['auc']),
        'acc_std': np.std(metrics['accuracy'])
    }

def create_simple_visualization(metrics1, metrics2, labels, output_path):
    """创建简化对比图"""
    plt.figure(figsize=(12, 6))
    
    # AUC对比
    plt.subplot(1, 2, 1)
    plt.bar([0, 1], 
            [metrics1['auc_mean'], metrics2['auc_mean']],
            yerr=[metrics1['auc_std'], metrics2['auc_std']],
            color=['skyblue', 'lightgreen'])
    plt.xticks([0, 1], labels)
    plt.ylabel('AUC')
    plt.title('AUC对比')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 准确率对比
    plt.subplot(1, 2, 2)
    plt.bar([0, 1], 
            [metrics1['acc_mean'], metrics2['acc_mean']],
            yerr=[metrics1['acc_std'], metrics2['acc_std']],
            color=['orange', 'pink'])
    plt.xticks([0, 1], labels)
    plt.ylabel('准确率')
    plt.title('准确率对比')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存至: {output_path}")

if __name__ == "__main__":
    # 参数设置
    file1 = "results/baseline.json"  # 替换为实际文件路径
    file2 = "results/with_tree.json"
    output = "output/comparison.png"
    
    # 设置字体
    setup_chinese_font()
    
    # 加载数据
    metrics_baseline = load_simple_metrics(file1)
    metrics_tree = load_simple_metrics(file2)
    
    # 生成可视化
    create_simple_visualization(
        metrics_baseline, 
        metrics_tree,
        labels=['基准模型', '树增强模型'],
        output_path=output
    )