# Tree-as-a-Prompt

The code of paper [From Prompts to Trees: Effective LLM-Guided TreeGeneration for Few-Shot Tabular Classification]().

## Setup

### Install Dependencies

**With conda**

Create environment from `environment.yml`:

```
conda env create -f environment.yml
```

or install the dependencies manually:

```
conda create -n tree-prompt python=3.11
conda activate tree-prompt
conda install numpy scikit-learn graphviz jinja2 pyyaml tqdm openai
```

**With pip**

Python version >= 3.10 is required.

```
pip install numpy scikit-learn graphviz jinja2 pyyaml tqdm openai
```

### OpenAI API Access

For ChatGPT experiments, you need to have an OpenAI API key. Please note that the training cost may be high (for example, up to $10+ to train a tree using 16 samples).

### Deploy Vicuna/Llama2 (Optional)

We recommend using [TogetherAI]( https://api.together.xyz/v1) to deploy an OpenAI API service of  Qwen/Qwen2.5-72B-Instruct-Turbo. Please refer to the documentation for deployment details.

The model used in our experiments is [ Qwen/Qwen2.5-72B-Instruct-Turbo]
## Run Experiments

We provide a set of experiment configurations to reproduce our results. Since `temperature` is set to 0, the result is expected to be mostly deterministic, but it can still vary across runs, so we can't guarantee the reproducibility.

### Steps

**Before running experiments, set `OPENAI_API_KEY` environment variable.** The key can also be set via command-line arguments or configuration files, but environment variable is the most convenient way.

```bash
export OPENAI_API_KEY=sk-xxx
```

**Run `train.py` to train and evaluate our model.** You need to specify the provided configuration file (in `config/train`) and training sizes.

```
python train.py --config config/train/<config-name>.yml --train-sizes <size> 
```

**Run `evaluate.py` to evaluate baselines.** You need to specify the provided configuration file (in `config/evaluate`) .

```
python3 evaluate.py --config config/evaluate/<config-name>.yml --use-tree-rules <0|1>
```

For more details about the experiment settings, please refer to the [Experiment Settings](#experiment-settings) section.

### Reproduction of Table 1

These experiments are to evaluate the performance of our approach with GPT-3.5 on different datasets.

- To train our model (LLMT),change to the `feature/pre_meta_rule_sup_known` branch and use `chatgpt-<dataset>.yml` in `config/train`.
- To evaluate baselines (DT,IO-Tree,CoT,ToT), change to the right branches I mentioned below and use `chatgpt-<dataset>.yml` in `config/evaluate`.
- To evaluate XGB,change to the `main` branch and use `xgboost-<dataset>.yml` in `config/evaluate`.

**Examples:**

```bash
# LLMT
git checkout feature/pre_meta_rule_sup_known&&python3 train.py --config config/train/chatgpt-diabetes.yml 
# DT/LLM+DT
git checkout main&&python3 evaluate.py --config config/evaluate/chatgpt-diabetes.yml 
# XGB
git checkout main&&python3 evaluate.py --config config/evaluate/xgboost-diabetes.yml
#RF
git checkout main&&python3 evaluate.py  --config config/evaluate/random_forest_diabetes.yml
# IO-Tree
git checkout tree_gen_by_llm_directly&&python evaluate.py  --config config/evaluate/llm_gen_directly-diabetes.yml --tree-type LLM --with-llm 0 
#CoT
git checkout CoT_Tree&&python evaluate.py  --config config/evaluate/CoT-diabetes.yml --tree-type CoT --with-llm 0 
#ToT
git checkout  fix/ToT_tree&& python evaluate.py  --config config/evaluate/ToT-diabetes.yml -tree-type ToT --with-llm 0 

```
### Reproduction of Table 2

These experiments are to evaluate our approach's performance with multiple trees using feature bagging. The training size is fixed to 8.

- To train our model (OT/LLM+OT), use `chatgpt-<dataset>-multiple.yml` in `config/train`.
- To evaluate RF, use `random_forest-<dataset>.yml` in `config/evaluate`.
- To evaluate XGB, use `xgboost-<dataset>.yml` in `config/evaluate`.
- To evaluate CoT, use `CoT-<dataset>.yml` in `config/evaluate`.
- To evaluate ToT, use `ToT-<dataset>.yml` in `config/evaluate`.
- To evaluate IO-Tree, use `llm_gen_directly-<dataset>.yml` in `config/evaluate`.

**Examples:**

```bash
# OT/LLM+OT
python3 train.py --config config/train/chatgpt-diabetes-multiple.yml --train-sizes 8
# RF
python3 evaluate.py --config config/evaluate/random_forest-diabetes.yml --train-sizes 8
```



**Examples:**

```bash
#LLMT
git checkout feature/pre_meta_rule_sup_known&&python3 train.py --config config/train/chatgpt-diabetes.yml 

```

## Experiment Settings

You can use either configuration files, command-line arguments or both to configure experiment settings. Command-line arguments have higher priority than configuration files and can be used to override the settings.

For example:

```
python train.py --config <config-file> --train-sizes 4 8
```

This command reads the configurations from file `<config-file>` and overrides `train_sizes` to `[4, 8]`.

The complete list of arguments are as follows. The corresponding command-line argument keys are listed in the comments.

### Train

These are the settings for training our model.

```yaml
config:
  # training strategy (unknown_class, known_class, feature_bagging) [--strategy]
  strategy: unknown_class
  # training strategy arguments
  strategy_args:
    # number of trees in forest (only for feature_bagging) [--num-trees]
    num_trees: 3
    # max depth of tree [--max-depth]
    max_depth: 3

  # runner type (openai_api) [--runner]
  runner: openai_api
  # runner arguments
  runner_args:
    # openai api key (for openai access only) [--openai-api-key]
    openai_api_key: sk-xxx
    # openai api base url (for non-openai APIs) [--openai-api-base]
    openai_api_base: https://localhost:7800/v1
    # model name [--model-name]
    model_name:  Qwen/Qwen2.5-72B-Instruct-Turbo
    # maximum number of parallel request [--parallel-batch-size]
    parallel_batch_size: 6
    # time in seconds between two requests [--request-interval]
    request_interval: 0.2
    # request timeout in seconds [--timeout]
    timeout: 30

  # dataset arguments
  dataset_args: # see sections below
    # dataset data file [--dataset-data-file]
    data_file: dataset/diabetes/data.csv
    # dataset meta file [--dataset-meta-file]
    meta_file: dataset/diabetes/meta.yml
    # dataset format (csv, libsvm) [--dataset-format]
    format: csv
    # whether to shuffle feature order [--shuffle-column]
    shuffle_column: true

  # results output directory [--output-dir]
  output_dir: output/chatgpt
  # random seed [--random-seed]
  random_seed: 0
  # train sizes [--train-sizes]
  train_sizes: [2, 4, 8, 16]
  # number of samples per query during training [--train-batch]
  train_batch: 8
  # number of trials (different random seeds) per train size [--num-tests-per-set]
  num_tests_per_set: 5
  # test size [--test-size]
  test_size: 100
  # number of test samples per query [--test-batch]
  test_batch: 8
  # prompt template for generationg rules (jinja2 file) [--template]
  template: template/train/meta_rule.jinja
  #prompt template for supervising tree generation (jinja2 file) [--template]
  supervise_template: template/train/supervision.jinja
  # table-to-text serializer type (tabular, text, list) [--serializer]
  serializer: tabular
```

### Evaluate

Most of evaluation settings are the same as training settings, except for the following:

**Not applicable for evaluation**
- `strategy`
- `strategy_args`
- `train_batch`


**Only applicable for evaluation**

```yaml
config:
  # whether to use decision rules from the tree [--use-tree-rules]
  use_tree_rules: false
  # whether to evaluate tree model itself only [--tree-only]
  tree_only: false
  # tree model type (simple, xgboost, random_forest) [--tree-type]
  tree_type: random_forest
  # tree model arguments
  tree_args:
    # max depth of tree [--max-depth]
    max_depth: 3
    # number of trees (only for xgboost and random_forest) [--num-trees]
    num_trees: 3
  with-llm: 0
  #whether to use LLM to reason with the tree generated
```

### Base Configuration Files

To consistently share some settings across different tasks (e.g. dataset, runner, credentials), you can configure them in a seperate file and "import" them using `base_configs` in the main config, with their relative paths. We provide a set of base configurations we use in our experiments at `config/base/**`,, `config/train/common.yml`, `config/evaluate/tree/*` and `config/evaluate/common.yml`.

```yaml
base_configs:
- ../base/dataset/diabetes.yml
- ../base/runner/chatgpt.yml
- ../base/common.yml
- ./strategy/unknown_class.yml
- ./common.yml

config:
  exp_name: chatgpt-diabetes
```
