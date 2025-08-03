import os
import logging
import re
from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from openai import OpenAI
from tqdm import tqdm
import sys
import io

# -------- 配置 --------
API_KEY =
BASE_URL = "https://api.ppinfra.com/v3/openai"
MODEL_ID = "deepseek/deepseek-v3-0324"

DATA_PATH = "dataset/diabetes/data.csv"
LOG_DIR = "./output/log"
LOG_FILE = os.path.join(LOG_DIR, "run.log")
MAX_CART_TREES = 5
MAX_DEPTH = 3
LLM_CALLS_PER_TYPE = 100

# -------- 修复编码 --------
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# -------- 创建日志 --------
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# -------- 读取数据 --------
def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    return X, y

# -------- CART训练 --------
def train_cart_trees(X, y, n_trees=MAX_CART_TREES, max_depth=MAX_DEPTH):
    trees = []
    for i in range(n_trees):
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42 + i)
        tree.fit(X, y)
        trees.append(tree)
    return trees

# -------- CART树转规则 --------
def tree_to_rules(tree: DecisionTreeClassifier, feature_names):
    tree_ = tree.tree_
    paths = []

    def recurse(node, conditions):
        if tree_.feature[node] != _tree.TREE_UNDEFINED and len(conditions) < 2:
            feature = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], conditions + [f"{feature} < {threshold:.5f}"])
            recurse(tree_.children_right[node], conditions + [f"{feature} >= {threshold:.5f}"])
        else:
            values = tree_.value[node][0]
            class_id = values.argmax()
            rule = f"IF {' AND '.join(conditions) if conditions else 'True'} THEN {'yes' if class_id == 1 else 'no'}"
            paths.append(rule)

    recurse(0, [])
    return paths

# -------- 提取规则签名 --------
def extract_rule_signature(rule):
    cond_part, then_part = rule.split("THEN")
    conds = cond_part.strip()[3:].strip()
    label = then_part.strip().lower()
    pattern = r"([a-zA-Z_ ]+?)\s*(>=|<)"
    items = re.findall(pattern, conds)
    items = [(feat.strip(), op) for feat, op in items]
    return tuple(items), label

# -------- 对比规则准确率 --------
def compare_rules_accuracy(cart_rules, llm_rules):
    def extract_full_signature(rule):
        cond_part, then_part = rule.split("THEN")
        conds = cond_part.strip()[3:].strip()
        label = then_part.strip().lower()
        # 提取特征 + 符号
        pattern = r"([a-zA-Z_ ]+?)\s*(>=|<)"
        items = re.findall(pattern, conds)
        items = [(feat.strip(), op) for feat, op in items]
        return tuple(items), label

    cart_sigs = [extract_full_signature(r) for r in cart_rules]
    correct = 0
    wrong = 0

    for rule in llm_rules:
        sig = extract_full_signature(rule)
        if sig in cart_sigs:
            correct += 1
        else:
            wrong += 1

    return correct, wrong



# -------- 解析文本树为规则 --------

import re

import re

def parse_tree_text_to_rules(tree_text: str):
    lines = tree_text.strip().split('\n')
    logger.info(f"开始解析树文本，共{len(lines)}行")
    rules = []
    stack = []  # 当前路径条件，存成三元组 (feature, op, threshold)

    for idx, line in enumerate(lines):
        # 去除行开头所有 "|", 空格，再去除 "---" 和空格
        content = line
        # 先去掉开头的所有管道符和空格
        content = re.sub(r"^(\|\s*)+", "", content)
        # 再去掉 "---" 和它前后的空格
        content = content.lstrip('-').strip()

        if not content:
            continue

        is_leaf = content.lower().startswith("class")

        # 根据前面去除的管道和缩进，计算层级：统计line中 '|   ' 的次数作为层级
        level = line.count('|   ')

        if is_leaf:
            label_raw = content.split(":")[-1].strip()
            label = label_raw.lower()
            if label in ("1", "class 1", "yes"):
                label = "yes"
            elif label in ("0", "class 0", "no"):
                label = "no"
            else:
                label = "yes" if "1" in label else "no"

            # 拼接条件字符串
            cond_strs = []
            for feat, op, thr in stack[:level]:
                cond_strs.append(f"{feat} {op} {thr}")
            rule = "IF " + " AND ".join(cond_strs) + f" THEN {label}"
            rules.append(rule)
        else:
            # 解析条件，提取 特征、符号、阈值
            # 格式示例："Glucose >= 127.50"
            m = re.match(r"([a-zA-Z_ ]+?)\s*(>=|<)\s*([\d\.]+)", content)
            if m:
                feat = m.group(1).strip()
                op = m.group(2)
                thr = float(m.group(3))
                if len(stack) > level:
                    stack[level] = (feat, op, thr)
                    stack = stack[:level+1]
                else:
                    stack.append((feat, op, thr))
            else:
                logger.warning(f"无法解析条件行: {content}")

    logger.info(f"解析得到规则数: {len(rules)}")
    return rules



def extract_features_from_rule(rule):
    cond_part = rule.split("THEN")[0].strip()
    cond_part = cond_part[3:].strip()  # 去掉开头 IF
    # 提取特征名，确保只匹配 '>=' 和 '<'
    feats = re.findall(r"([a-zA-Z_ ]+?)\s*(?:>=|<)", cond_part)
    feats = [f.strip() for f in feats]
    return tuple(feats)




# -------- 解析文本规则列表 --------
def parse_rules_text(rules_text):
    lines = [line.strip() for line in rules_text.strip().splitlines() if line.strip()]
    return lines

# -------- 调用 LLM --------
def call_llm_generate_tree(client, prompt):
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        stream=False
    )
    return response.choices[0].message.content

def call_llm_generate_rule(client, prompt):
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return response.choices[0].message.content

# -------- 主流程 --------
def main():
    logger.info("开始加载数据")
    X, y = load_data(DATA_PATH)
    feature_names = list(X.columns)
    logger.info(f"特征名: {feature_names}")

    logger.info(f"训练 {MAX_CART_TREES} 棵 CART 树，最大深度 {MAX_DEPTH}")
    cart_trees = train_cart_trees(X, y)

    cart_all_rules = []
    for i, tree in enumerate(cart_trees):
        rules = tree_to_rules(tree, feature_names)
        logger.info(f"Cart Tree {i} 规则:")
        for r in rules:
            logger.info("  " + r)
        cart_all_rules.extend(rules)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    prompt_tree = f"""Dataset information:
- Name: diabetes
- Task: diagnostically predict whether or not a patient has diabetes.
- Label meaning: diagnosis

Features:
- Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age

Labels:
- no (0): the patient does not have diabetes
- yes (1): the patient has diabetes

---

Task:
Generate a decision tree in textual format for binary classification.

Strict requirements:
- Each decision path must contain at most **two conditions**.
- Each path must involve at most **two distinct features**.
- Use only comparison operators: `>=` and `<`.
- Use indentation with '|---' for tree structure.
- Leaf nodes must be labeled as `class: 0` or `class: 1` only.
- Output only the tree without any extra explanation.
Example output format:

|--- BloodPressure >= 75.00
|   |--- Pregnancies < 4.00
|   |   |--- class: 1
|   |--- Pregnancies >= 4.00
|   |   |--- class: 0
|--- BloodPressure < 75.00
|   |--- Insulin >= 100.00
|   |   |--- class: 1
|   |--- Insulin < 100.00
|   |   |--- class: 0

"""


    prompt_rule = f"""Dataset information:
- Name: diabetes
- Task: diagnostically predict whether or not a patient has diabetes.
- Label meaning: diagnosis

Features:
- Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age

Labels:
- no (0): the patient does not have diabetes
- yes (1): the patient has diabetes

---

Task:
Generate a complete just one decision rules for classification.

Strict requirements:
- Each rule must contain at most two conditions joined by `AND`.
- Each condition must use only `>=` and `<` operators.
- Each rule must use at most **two distinct features**, and feature order matters.
- Format:

IF <condition1> [AND <condition2>] THEN <no|yes>

Example:

IF Glucose >= 127.50 AND BMI >= 30.00 THEN class: 1
IF Glucose >= 127.50 AND BMI < 30.00 THEN class: 0
Output only the rules."""


    logger.info("开始调用大模型生成树并解析规则")
    correct_tree_rules = 0
    wrong_tree_rules = 0
    for i in tqdm(range(LLM_CALLS_PER_TYPE), desc="生成树并拆规则"):
        try:
            tree_text = call_llm_generate_tree(client, prompt_tree)
            logger.info(f"[大模型生成树 {i}] 内容:\n{tree_text}")

            logger.info(f"大模型生成树原始文本:\n{tree_text}")
            rules = parse_tree_text_to_rules(tree_text)
            if not rules:
                logger.warning("规则解析为空，跳过")
                continue
            c, w = compare_rules_accuracy(cart_all_rules, rules)
            logger.info(f"[Tree {i}] 正确规则数: {c}, 错误规则数: {w}")
            correct_tree_rules += c
            wrong_tree_rules += w
        except Exception as e:
            logger.error(f"Tree {i} 调用异常: {e}")

    total_tree = correct_tree_rules + wrong_tree_rules
    acc_tree = correct_tree_rules / total_tree if total_tree > 0 else 0
    logger.info(f"大模型生成树拆规则整体准确率: {acc_tree:.4f}")

    logger.info("开始调用大模型生成规则")
    correct_rule_rules = 0
    wrong_rule_rules = 0
    for i in tqdm(range(LLM_CALLS_PER_TYPE), desc="生成规则"):
        try:
            logging.info(f"大模型生成树原始文本:\n{tree_text}")
            rules_text = call_llm_generate_rule(client, prompt_rule)
            logger.info(f"[大模型生成规则 {i}] 内容:\n{rules_text}")
            rules = parse_rules_text(rules_text)
            if not rules:
                logger.warning("规则列表为空，跳过")
                continue
            c, w = compare_rules_accuracy(cart_all_rules, rules)
            logger.info(f"[Rule {i}] 正确规则数: {c}, 错误规则数: {w}")
            correct_rule_rules += c
            wrong_rule_rules += w
        except Exception as e:
            logger.error(f"Rule {i} 调用异常: {e}")

    total_rule = correct_rule_rules + wrong_rule_rules
    acc_rule = correct_rule_rules / total_rule if total_rule > 0 else 0
    logger.info(f"大模型直接生成规则整体准确率: {acc_rule:.4f}")

# -------- 入口 --------
if __name__ == "__main__":
    main()
