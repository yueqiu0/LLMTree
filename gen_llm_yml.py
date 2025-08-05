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

def generate_llm_gen_directly_configs(dataset_root):
    dataset_root = os.path.abspath(dataset_root)
    dataset_names = [
        name for name in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, name))
    ]

    current_dir = os.path.abspath(os.path.dirname(__file__))
    eval_dir = os.path.join(current_dir, "config", "evaluate")

    for name in dataset_names:
        filename = f"llm_gen_directly-{name}.yml"
        output_path = os.path.join(eval_dir, filename)

        content = {
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

        write_if_not_exists(output_path, content)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python gen_llm_gen_directly_yml.py <dataset_folder>")
        sys.exit(1)

    dataset_root = sys.argv[1]
    generate_llm_gen_directly_configs(dataset_root)
