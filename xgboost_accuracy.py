import os
import json
import numpy as np

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get("results", {})
    dataset_name = os.path.basename(filepath).replace("xgboost_baseline_", "").replace(".json", "")
    
    print(f"数据集: {dataset_name}")
    print(f"{'Train Size':>10} | {'Accuracy Mean ± Std':>20}")
    print("-" * 35)
    
    for train_size_str, result_list in sorted(results.items(), key=lambda x: int(x[0])):
        accuracies = [r.get("tree_accuracy", None) for r in result_list if r.get("tree_accuracy") is not None]
        if accuracies:
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            print(f"{train_size_str:>10} | {mean_acc:.3f} ± {std_acc:.3f}")
        else:
            print(f"{train_size_str:>10} | {'No accuracy data'}")
    print("\n")

def main():
    eval_dir = "./output/evaluate"
    json_files = [f for f in os.listdir(eval_dir) if f.endswith(".json") and f.startswith("xgboost_baseline_")]

    for filename in json_files:
        filepath = os.path.join(eval_dir, filename)
        process_file(filepath)

if __name__ == "__main__":
    main()
