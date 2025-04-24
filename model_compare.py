import json
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import re

def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_baseline_llm_metrics(json_data, train_size):
    if train_size not in json_data['results']:
        return []
    
    auc_values = []
    for experiment in json_data['results'][train_size]:
        if 'auc' in experiment:
            auc_values.append(experiment['auc'])
    return auc_values

def extract_baseline_dt_metrics(json_data, train_size, metric_type='llm_dt'):
    if train_size not in json_data['results']:
        return []
    
    auc_values = []
    for experiment in json_data['results'][train_size]:
        if metric_type == 'llm_dt' and 'auc' in experiment:
            auc_values.append(experiment['auc'])
        elif metric_type == 'dt' and 'tree_auc' in experiment:
            auc_values.append(experiment['tree_auc'])
    return auc_values

def extract_ot_metrics(json_data, train_size, metric_type='llm_ot'):
    if train_size not in json_data['results']:
        return []
    
    auc_values = []
    for experiment in json_data['results'][train_size]:
        if metric_type == 'llm_ot' and 'llm_tree' in experiment:
            auc_values.append(experiment['llm_tree'])
        elif metric_type == 'ot' and 'tree' in experiment:
            auc_values.append(experiment['tree'])
    return auc_values

def calculate_average_auc(json_data, train_sizes=["2", "4", "8", "16", "32"], file_type='baseline_llm', metric_type='llm'):
    averages = {}
    for size in train_sizes:
        auc_values = []
        
        if file_type == 'baseline_llm':
            auc_values = extract_baseline_llm_metrics(json_data, size)
        elif file_type == 'baseline_dt':
            if metric_type == 'llm_dt':
                auc_values = extract_baseline_dt_metrics(json_data, size, 'llm_dt')
            elif metric_type == 'dt':
                auc_values = extract_baseline_dt_metrics(json_data, size, 'dt')
        elif file_type == 'ot':
            if metric_type == 'llm_ot':
                auc_values = extract_ot_metrics(json_data, size, 'llm_ot')
            elif metric_type == 'ot':
                auc_values = extract_ot_metrics(json_data, size, 'ot')
        
        if auc_values:
            averages[size] = np.mean(auc_values)
        else:
            averages[size] = None
    return averages

def create_llm_visualization(data, output_path=None):
    plt.figure(figsize=(22, 16))
    plt.style.use('ggplot')
    
    datasets = ["abalone", "blood", "car", "diabetes"]
    train_sizes = ["2", "4", "8", "16", "32"]
    
    models = [
        "llm", 
        "llm_dt", 
        "llm_ot_rank_wo_sup", 
        "llm_ot_rank_w_sup_known", 
        "llm_ot_rank_w_sup_unknown", 
        "llm_ot_meta_rule_wo_sup"
    ]
    
    model_labels = [
        "LLM", 
        "LLM+DT", 
        "LLM+OT (Rank w/o Sup)", 
        "LLM+OT (Rank w/ Sup Known)", 
        "LLM+OT (Rank w/ Sup Unknown)", 
        "LLM+OT (Meta Rule w/o Sup)"
    ]
    
    colors = [
        "#3498db",  # Blue
        "#2ecc71",  # Green
        "#e74c3c",  # Red
        "#9b59b6",  # Purple
        "#f1c40f",  # Yellow
        "#e67e22"   # Orange
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    axes = axes.flatten()
    
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        
        group_width = 0.8
        bar_width = group_width / len(models)
        group_positions = np.arange(len(train_sizes))
        
        for j, (model, label, color) in enumerate(zip(models, model_labels, colors)):
            bar_positions = group_positions - group_width/2 + (j+0.5)*bar_width
            
            auc_values = []
            for size in train_sizes:
                if model in data and dataset in data[model] and size in data[model][dataset]:
                    value = data[model][dataset][size]
                    if value is None:
                        error_msg = f"ERROR: None value found for dataset={dataset}, size={size}, model={model}"
                        print(error_msg)
                        value = 0
                    auc_values.append(value)
                else:
                    print(f"Missing data for dataset={dataset}, size={size}, model={model}")
                    auc_values.append(0)
            
            print(f"Dataset: {dataset}, Model: {model}, AUC values: {auc_values}")
            
            bars = ax.bar(bar_positions, auc_values, width=bar_width, 
                         color=color, label=label)
            
            for bar, value in zip(bars, auc_values):
                if value > 0:  
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', 
                           fontsize=8, rotation=45)
        
        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('Average AUC', fontsize=12)
        ax.set_title(f'Dataset: {dataset.capitalize()}', fontsize=14)
        
        ax.set_xticks(group_positions)
        ax.set_xticklabels(train_sizes)
        
        ax.set_ylim(0.5, 1.0)
        
        if i == 0:  
            ax.legend(title="Model Type", loc='upper center', bbox_to_anchor=(0.5, -0.15),
                     fancybox=True, shadow=True, ncol=3)
        
        ax.grid(True, linestyle='--', alpha=0.7)
    
    fig.suptitle('AUC Comparison Across Different LLM Prediction Models\n' +
                'Model: Qwen/Qwen2.5-72B-Instruct-Turbo (Alpha=0.8, Threshold=0.7, Beta=0.10, Delta=1)', 
                fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.15)  
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {output_path}")
    
    plt.close()  

def create_tree_visualization(data, output_path=None):
    plt.figure(figsize=(22, 16))
    plt.style.use('ggplot')
    
    datasets = ["abalone", "blood", "car", "diabetes"]
    train_sizes = ["2", "4", "8", "16", "32"]
    
    models = [
        "dt", 
        "ot_rank_wo_sup", 
        "ot_rank_w_sup_known", 
        "ot_rank_w_sup_unknown", 
        "ot_meta_rule_wo_sup"
    ]
    
    model_labels = [
        "DT", 
        "OT (Rank w/o Sup)", 
        "OT (Rank w/ Sup Known)", 
        "OT (Rank w/ Sup Unknown)", 
        "OT (Meta Rule w/o Sup)"
    ]
    
    colors = [
        "#3498db",  # Blue
        "#2ecc71",  # Green
        "#e74c3c",  # Red
        "#9b59b6",  # Purple
        "#f1c40f"   # Yellow
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    axes = axes.flatten()
    
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        
        group_width = 0.8
        bar_width = group_width / len(models)
        group_positions = np.arange(len(train_sizes))
        
        for j, (model, label, color) in enumerate(zip(models, model_labels, colors)):
            bar_positions = group_positions - group_width/2 + (j+0.5)*bar_width
            
            auc_values = []
            for size in train_sizes:
                if model in data and dataset in data[model] and size in data[model][dataset]:
                    value = data[model][dataset][size]
                    if value is None:
                        error_msg = f"ERROR: None value found for dataset={dataset}, size={size}, model={model}"
                        print(error_msg)
                        value = 0
                    auc_values.append(value)
                else:
                    print(f"Missing data for dataset={dataset}, size={size}, model={model}")
                    auc_values.append(0)
            
            print(f"Dataset: {dataset}, Model: {model}, AUC values: {auc_values}")
            
            bars = ax.bar(bar_positions, auc_values, width=bar_width, 
                         color=color, label=label)
            

            for bar, value in zip(bars, auc_values):
                if value > 0:  
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', 
                           fontsize=8, rotation=45)
        
        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('Average AUC', fontsize=12)
        ax.set_title(f'Dataset: {dataset.capitalize()}', fontsize=14)
        
        ax.set_xticks(group_positions)
        ax.set_xticklabels(train_sizes)
        
        ax.set_ylim(0.5, 1.0)
        
        if i == 0:  
            ax.legend(title="Model Type", loc='upper center', bbox_to_anchor=(0.5, -0.15),
                     fancybox=True, shadow=True, ncol=3)
        
        ax.grid(True, linestyle='--', alpha=0.7)
    
    fig.suptitle('AUC Comparison Across Different Tree Prediction Models\n' +
                'Model: Qwen/Qwen2.5-72B-Instruct-Turbo (Alpha=0.8, Threshold=0.7, Beta=0.10, Delta=1)', 
                fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.15) 
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {output_path}")
    
    plt.close() 

def main():
    parser = argparse.ArgumentParser(description='Compare AUC values across different models')
    parser.add_argument('--input_dir', type=str, default='./output',
                        help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default='./output/visualize',
                        help='Directory for output charts')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    file_patterns = {
        "llm": "evaluate/baseline_llm_{dataset}.json",
        "llm_dt": "evaluate/baseline_dt_{dataset}.json",
        "dt": "evaluate/baseline_dt_{dataset}.json",
        "llm_ot_rank_wo_sup": "train/llm_rank_wo_sup_test_alpha_0dot8_{dataset}.json",
        "ot_rank_wo_sup": "train/llm_rank_wo_sup_test_alpha_0dot8_{dataset}.json",
        "llm_ot_rank_w_sup_known": "train/llm_rank_w_sup_known_test_threshold_0dot70_{dataset}.json",
        "ot_rank_w_sup_known": "train/llm_rank_w_sup_known_test_threshold_0dot70_{dataset}.json",
        "llm_ot_rank_w_sup_unknown": "train/llm_rank_w_sup_unknown_test_threshold_0dot70_beta_0dot10_{dataset}.json",
        "ot_rank_w_sup_unknown": "train/llm_rank_w_sup_unknown_test_threshold_0dot70_beta_0dot10_{dataset}.json",
        "llm_ot_meta_rule_wo_sup": "train/pre_meta_rule_wo_sup_test_delta_1_{dataset}.json",
        "ot_meta_rule_wo_sup": "train/pre_meta_rule_wo_sup_test_delta_1_{dataset}.json"
    }
    
    datasets = ["abalone", "blood", "car", "diabetes"]
    train_sizes = ["2", "4", "8", "16", "32"]
    
    llm_data = {}
    tree_data = {}

    for model, file_pattern in file_patterns.items():
        llm_data[model] = {}
        tree_data[model] = {}
        
        for dataset in datasets:
            file_path = os.path.join(args.input_dir, file_pattern.format(dataset=dataset))
            
            if os.path.exists(file_path):
                try:
                    json_data = load_json_file(file_path)
                    
                    if model in ["llm"]:
                        file_type = 'baseline_llm'
                        averages = calculate_average_auc(json_data, train_sizes, file_type, 'llm')
                        llm_data[model][dataset] = averages
                    
                    elif model in ["llm_dt", "dt"]:
                        file_type = 'baseline_dt'
                        metric_type = 'llm_dt' if model == "llm_dt" else 'dt'
                        averages = calculate_average_auc(json_data, train_sizes, file_type, metric_type)
                        
                        if model == "llm_dt":
                            llm_data[model][dataset] = averages
                        else:
                            tree_data[model][dataset] = averages
                    
                    else:
                        file_type = 'ot'
                        if model.startswith("llm_"):
                            metric_type = 'llm_ot'
                            averages = calculate_average_auc(json_data, train_sizes, file_type, metric_type)
                            llm_data[model][dataset] = averages
                        else:
                            metric_type = 'ot'
                            averages = calculate_average_auc(json_data, train_sizes, file_type, metric_type)
                            tree_data[model][dataset] = averages
                    
                    print(f"Processed {file_path} for {model}:{dataset}")
                
                except Exception as e:
                    print(f"Error processing {file_path} for {model}:{dataset}: {e}")
            else:
                print(f"File not found: {file_path}")
    
    llm_output_path = os.path.join(args.output_dir, 'llm_model_comparison.png')
    tree_output_path = os.path.join(args.output_dir, 'tree_model_comparison.png')
    
    create_llm_visualization(llm_data, llm_output_path)
    create_tree_visualization(tree_data, tree_output_path)

if __name__ == "__main__":
    main()
