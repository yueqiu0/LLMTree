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
    base_dir = os.path.join(current_dir, "config", "base", "dataset")
    eval_dir = os.path.join(current_dir, "config", "evaluate")
    train_dir = os.path.join(current_dir, "config", "train")

    for name in dataset_names:
        dataset_file = f"{name}.yml"

        # 1️⃣ base/dataset/*.yml
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

        # 2️⃣ evaluate/chatgpt-xxx.yml
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

        # 3️⃣ evaluate/random_forest-xxx.yml
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

        # 4️⃣ evaluate/xgboost-xxx.yml
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

        # 5️⃣ evaluate/CoT-xxx.yml
        cot_eval_path = os.path.join(eval_dir, f"CoT-{name}.yml")
        cot_eval_content = {
            "base_configs": [
                f"../base/dataset/{name}.yml",
                "../base/runner/chatgpt.yml",
                "./CoT_llm_tree.yml",
                "tree/CoT.yml",
                "./common.yml"
            ],
            "config": {
                "exp_name": f"CoT-{name}-baseline",
                "template": "template/basic.jinja",
                "tree_only": False
            }
        }
        write_if_not_exists(cot_eval_path, cot_eval_content)

        # 6️⃣ evaluate/ToT-xxx.yml
        tot_eval_path = os.path.join(eval_dir, f"ToT-{name}.yml")
        tot_eval_content = {
            "base_configs": [
                f"../base/dataset/{name}.yml",
                "../base/runner/chatgpt.yml",
                "./ToT_llm_tree.yml",
                "tree/ToT.yml",
                "./common.yml"
            ],
            "config": {
                "exp_name": f"ToT-{name}-baseline",
                "template": "template/basic.jinja",
                "tree_only": False
            }
        }
        write_if_not_exists(tot_eval_path, tot_eval_content)

        # 7️⃣ evaluate/Iot-xxx.yml
        iot_eval_path = os.path.join(eval_dir, f"Iot-{name}.yml")
        iot_eval_content = {
            "base_configs": [
                f"../base/dataset/{name}.yml",
                "../base/runner/chatgpt.yml",
                "./direct_llm_tree.yml",
                "tree/llm_gen_directly.yml",
                "./common.yml"
            ],
            "config": {
                "exp_name": f"llm_gen_tree-{name}-baseline",
                "template": "template/basic.jinja",
                "tree_only": False
            }
        }
        write_if_not_exists(iot_eval_path, iot_eval_content)

        # 8️⃣ train/chatgpt-xxx-known.yml
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

        # 9️⃣ train/chatgpt-xxx-multiple.yml
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

        # 🔟 train/chatgpt-xxx.yml
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
