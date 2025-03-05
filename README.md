# Tree-as-a-Prompt

The code of paper [Tree-as-a-Prompt: Boosting Black-Box Large Language Models on Few-Shot Classification of Tabular Data]().

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

We recommend using [FastChat](https://github.com/lm-sys/FastChat) to deploy an OpenAI API service of Vicuna/Llama2. Please refer to the documentation for deployment details.

The model used in our experiments is [`lmsys/vicuna-7b-v1.5`](https://huggingface.co/lmsys/vicuna-7b-v1.5).

## Run Experiments

We provide a set of experiment configurations to reproduce our results. Since `temperature` is set to 0, the result is expected to be mostly deterministic, but it can still vary across runs, so we can't guarantee the reproducibility.

### Steps

**Before running experiments, set `OPENAI_API_KEY` environment variable.** The key can also be set via command-line arguments or configuration files, but environment variable is the most convenient way.

```bash
export OPENAI_API_KEY=sk-xxx
```

**Run `train.py` to train and evaluate our model.** You need to specify the provided configuration file (in `config/train`), training sizes and loss parameters. The loss parameters for each experiment are listed below.

```
python train.py --config config/train/<config-name>.yml --train-sizes <size> --loss-lambda <lambda> --loss-mu <mu>
```

As hyperparameters are tuned for each size, you need to run the experiments for each size separately.

**Run `evaluate.py` to evaluate baselines.** You need to specify the provided configuration file (in `config/evaluate`) and whether to incorporate decision rules from the tree (`--use-tree-rules`).

```
python3 evaluate.py --config config/evaluate/<config-name>.yml --use-tree-rules <0|1>
```

For more details about the experiment settings, please refer to the [Experiment Settings](#experiment-settings) section.

### Reproduction of Table 1

These experiments are to evaluate the performance of our approach with GPT-3.5 on different datasets.

- To train our model (OT/LLM+OT), use `chatgpt-<dataset>.yml` in `config/train`.
- To evaluate baselines (LLM/DT/LLM+DT), use `chatgpt-<dataset>.yml` in `config/evaluate`.
- To evaluate XGB, use `xgboost-<dataset>.yml` in `config/evaluate`.

The hyperparameters used in these experiments are listed in the following table. The unknown class is not included when $\mu = \infty$, please either set `--loss-mu` to a large value ($\ge 1000$) or use `chatgpt-<dataset>-known.yml` as the configuration file for these cases.

**diabetes**

| Size | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| $\lambda$ | 2 | 0.1 | 4 | 4 | 4 | 4 |
| $\mu$ | 0 | 1 | 0.6 | 0.2 | $\infty$ | $\infty$ |

**car**

| Size | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| $\lambda$ | 0.1 | 1 | 1 | 0.5 | 8 | 8 |
| $\mu$ | 0.6 | 0 | 0.2 | 0.2 | $\infty$ | $\infty$ |

**blood**

| Size | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| $\lambda$ | 0.1 | 0.5 | 4 | 0.1 | 0.1 | 1 |
| $\mu$ | 1 | 2 | 2 | 0.2 | 0.2 | $\infty$ |

**abalone**

| Size | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| $\lambda$ | 0.1 | 0.1 | 0.1 | 1 | 1 | 1 |
| $\mu$ | 0 | 0 | 2 | 0.2 | 0.2 | $\infty$ |

**Examples:**

```bash
# OT/LLM+OT
python3 train.py --config config/train/chatgpt-diabetes.yml --train-sizes 8 --loss-lambda 4 --loss-mu 0.2
# LLM
python3 evaluate.py --config config/evaluate/chatgpt-diabetes.yml --use-tree-rules 0
# DT/LLM+DT
python3 evaluate.py --config config/evaluate/chatgpt-diabetes.yml --use-tree-rules 1
# XGB
python3 evaluate.py --config config/evaluate/xgboost-diabetes.yml
```

### Reproduction of Table 2

These experiments are to evaluate our approach's performance with multiple trees using feature bagging. The training size is fixed to 8.


- To train our model (OT/LLM+OT), use `chatgpt-<dataset>-multiple.yml` in `config/train`.
- To evaluate RF, use `random_forest-<dataset>.yml` in `config/evaluate`.

The hyperparameters used in these experiments are listed in the following table.

| Dataset | $\lambda$ | $\mu$ |
| ---------- | --- | --- |
| diabetes | 1 | 0.2 |
| car | 0.1 | 0.6 |
| blood | 0.1 | 0.6 |
| abalone | 1 | 0.2 |

**Examples:**

```bash
# OT/LLM+OT
python3 train.py --config config/train/chatgpt-diabetes-multiple.yml --train-sizes 8 --loss-lambda 1 --loss-mu 0.2
# RF
python3 evaluate.py --config config/evaluate/random_forest-diabetes.yml --train-sizes 8
```

### Reproduction of Table 3

These experiments are to study the effect of unknown option and the regularization term. The training size is fixed to 8.

- To evaluate our model without the unknown class,
  - use `chatgpt-<dataset>-known.yml` in `config/evaluate`, or
  - use `chatgpt-<dataset>.yml` in `config/evaluate`, and set `--loss-mu` to a large value.
- To evaluate our model without the regularization term, use `chatgpt-<dataset>.yml` in `config/evaluate` and set `--loss-lambda` to 0.

**Examples:**

```bash
# LLM+OT (no unknown class)
python3 train.py --config config/train/chatgpt-diabetes-known.yml --train-sizes 8 --loss-lambda 4
# LLM+OT (no regularization)
python3 train.py --config config/train/chatgpt-diabetes.yml --train-sizes 8 --loss-lambda 0 --loss-mu 0
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
  # experiment name [--exp-name]
  exp_name: chatgpt-diabetes

  # training strategy (unknown_class, known_class, feature_bagging) [--strategy]
  strategy: unknown_class
  # training strategy arguments
  strategy_args:
    # number of trees in forest (only for feature_bagging) [--num-trees]
    num_trees: 3
    # max depth of tree [--max-depth]
    max_depth: 3
    # lambda for regularization [--loss-lambda]
    lambda: 1
    # mu for unknown class penalty (not applicable for known_class) [--loss-mu]
    mu: 0.2
    # number of bins of histogram [--hist-nbins]
    hist_nbins: 10

  # runner type (openai_api) [--runner]
  runner: openai_api
  # runner arguments
  runner_args:
    # openai api key (for openai access only) [--openai-api-key]
    openai_api_key: sk-xxx
    # openai api base url (for non-openai APIs) [--openai-api-base]
    openai_api_base: https://localhost:7800/v1
    # model name [--model-name]
    model_name: gpt-3.5-turbo-0613
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
  # prompt template (jinja2 file) [--template]
  template: template/train/basic.jinja
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
```

### Base Configuration Files

To consistently share some settings across different tasks (e.g. dataset, runner, credentials), you can configure them in a seperate file and "import" them using `base_configs` in the main config, with their relative paths. We provide a set of base configurations we use in our experiments at `config/base/**`, `config/train/strategy/*`, `config/train/common.yml`, `config/evaluate/tree/*` and `config/evaluate/common.yml`.

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
