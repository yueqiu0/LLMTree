import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def extract_metadata_and_accuracy(json_data):
    # 提取模型名
    model = json_data["args"]["runner_args"]["model_name"]
    # 提取数据集名
    dataset_path = json_data["args"]["dataset_args"]["data_file"]
    dataset = Path(dataset_path).parts[-2]

    # 提取所有 accuracy
    all_acc = []
    for result_list in json_data["results"].values():
        for run in result_list:
            if "accuracy" in run:
                acc = run["accuracy"]
                all_acc.append(acc)

    mean_acc = np.mean(all_acc) if all_acc else 0.0
    std_acc = np.std(all_acc) if all_acc else 0.0

    return dataset, model, mean_acc, std_acc

def format_entry(mean, std):
    return f"{mean:.2f}±{std:.2f}"

def main(folder_path):
    table_data = defaultdict(dict)  # {dataset: {model_meta_flag: acc_str}}

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            dataset, model, mean_acc, std_acc = extract_metadata_and_accuracy(json_data)
            meta_flag = "w/metadata" if "_w_meta_" in filename else "w/o metadata"

            model_clean = model.split("/")[-1]  # e.g., "DeepSeek-V3"
            key = f"{model_clean} ({meta_flag})"

            table_data[dataset][key] = format_entry(mean_acc, std_acc)

    # 获取所有模型列名（按顺序）
    all_keys = sorted({k for d in table_data.values() for k in d})
    datasets = sorted(table_data.keys())

    # 输出表格
    print("Table 1: The test accuracy of LLMs with or without the metadata about the dataset.")
    print("We run five trials and report the mean and standard deviation.\n")
    header = ["Datasets"] + all_keys
    print("\t".join(header))
    for dataset in datasets:
        row = [dataset] + [table_data[dataset].get(col, "") for col in all_keys]
        print("\t".join(row))

if __name__ == "__main__":
    folder = "./output/evaluate"  # 替换为你的文件夹路径，如：r"C:\Users\chenx\output_json"
    main(folder)
