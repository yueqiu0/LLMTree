import os
import sys
import yaml

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_yaml(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(content, f, allow_unicode=True, sort_keys=False)

def write_if_not_exists(path, content):
    if not os.path.exists(path):
        ensure_dir(os.path.dirname(path))
        save_yaml(path, content)
        print(f"✓ Created: {path}")
    else:
        print(f"- Skipped (exists): {path}")

def generate_configs(dataset_root):
    dataset_root = os.path.abspath(dataset_root)
    dataset_names = [
        name for name in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, name))
    ]

    current_dir = os.path.abspath(os.path.dirname(__file__))
    base_dir = os.path.join(current_dir, "config", "base", "dataset")  # ✅ 更新位置
    eval_dir = os.path.join(current_dir, "config", "evaluate")
    train_dir = os.path.join(current_dir, "config", "train")

    for name in dataset_names:
        dataset_file = f"{name}.yml"

        # 1. base yaml （✅ 修正路径）
        base_path = os.path.join(base_dir, dataset_file)
        base_content = {
            "config": {
                "dataset_args": {
                    "data_file": f"dataset/{name}/data.csv",
                    "meta_file": f"dataset/{name}/meta.yml",
                    "format": "csv",
                    "shuffle_column": True
                }
            }
        }
        write_if_not_exists(base_path, base_content)

        # 2. evaluate/chatgpt-xxx.yml
        chatgpt_eval_path = os.path.join(eval_dir, f"chatgpt-{name}.yml")
        chatgpt_eval_content = {
            "base_configs": [
                f"../base/dataset/{name}.yml",
                "../base/runner/chatgpt.yml",
                "../base/common.yml",
                "./tree/simple.yml",
                "./common.yml"
            ],
            "config": {
                "exp_name": f"chatgpt-{name}-baseline"
            }
        }
        write_if_not_exists(chatgpt_eval_path, chatgpt_eval_content)

        # 3. evaluate/random_forest-xxx.yml
        rf_eval_path = os.path.join(eval_dir, f"random_forest-{name}.yml")
        rf_eval_content = {
            "base_configs": [
                f"../base/dataset/{name}.yml",
                "../base/common.yml",
                "./tree/random_forest.yml",
                "./common.yml"
            ],
            "config": {
                "exp_name": f"{name}-random_forest",
                "tree_only": True
            }
        }
        write_if_not_exists(rf_eval_path, rf_eval_content)

        # 4. evaluate/xgboost-xxx.yml
        xgb_eval_path = os.path.join(eval_dir, f"xgboost-{name}.yml")
        xgb_eval_content = {
            "base_configs": [
                f"../base/dataset/{name}.yml",
                "../base/common.yml",
                "./tree/xgboost.yml",
                "./common.yml"
            ],
            "config": {
                "exp_name": f"{name}-xgboost",
                "tree_only": True
            }
        }
        write_if_not_exists(xgb_eval_path, xgb_eval_content)

        # 5. train/chatgpt-xxx-known.yml
        known_train_path = os.path.join(train_dir, f"chatgpt-{name}-known.yml")
        known_train_content = {
            "base_configs": [
                f"../base/dataset/{name}.yml",
                "../base/runner/chatgpt.yml",
                "../base/common.yml",
                "./strategy/known_class.yml",
                "./common.yml"
            ],
            "config": {
                "exp_name": f"chatgpt-{name}"
            }
        }
        write_if_not_exists(known_train_path, known_train_content)

        # 6. train/chatgpt-xxx-multiple.yml
        multiple_train_path = os.path.join(train_dir, f"chatgpt-{name}-multiple.yml")
        multiple_train_content = {
            "base_configs": [
                f"../base/dataset/{name}.yml",
                "../base/runner/chatgpt.yml",
                "../base/common.yml",
                "./strategy/feature_bagging.yml",
                "./common.yml"
            ],
            "config": {
                "exp_name": f"chatgpt-{name}"
            }
        }
        write_if_not_exists(multiple_train_path, multiple_train_content)

        # 7. train/chatgpt-xxx.yml
        unknown_train_path = os.path.join(train_dir, f"chatgpt-{name}.yml")
        unknown_train_content = {
            "base_configs": [
                f"../base/dataset/{name}.yml",
                "../base/runner/chatgpt.yml",
                "../base/common.yml",
                "./strategy/unknown_class.yml",
                "./common.yml"
            ],
            "config": {
                "exp_name": f"chatgpt-{name}"
            }
        }
        write_if_not_exists(unknown_train_path, unknown_train_content)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_yaml_configs.py <dataset_folder>")
        sys.exit(1)

    dataset_root = sys.argv[1]
    generate_configs(dataset_root)
