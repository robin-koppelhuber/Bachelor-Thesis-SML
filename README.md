[![Python Version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Frobin-koppelhuber%2FBachelor-Thesis-SML%2Fmaster%2Fpyproject.toml&query=%24.project%5B%27requires-python%27%5D&label=python&color=blue&logo=python&logoColor=white)](https://github.com/robin-koppelhuber/Bachelor-Thesis-SML/blob/master/pyproject.toml) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) ![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)

## About

- This benchmark is part of my bachelor thesis at the ETH Statistical Machine Learning Lab ([ETH SML](https://sml.inf.ethz.ch/))
- We quantify the statistical learning theory bias of the function class of models parameterized by the convex set of weights of fine-tuned expert models that share the same pre-training model, in the multi-task (MTL) and multi-objective (MOL) setting
- This investigation is motivated by the fact that most model merging methods, especially data-free and zero shot methods, constrain themselves to the aforementioned convex set. Even though this is somewhat theoretically principled with investigations of linear mode connectivity and other phenomena as well as most importantly performance considerations, the results of this benchmark could prompt the development of novel methods. For more information see (*link final thesis*)
- To this end, we benchmark multiple methods constrained to the convex set as well as more unconstrained methods like retraining from the pre-training model with chebyshev scalarization (...) or EPO search
---
- Work in progress!


## Install

Install `uv` ([docs](https://docs.astral.sh/uv/getting-started/installation/)), then sync for your accelerator:

```bash
uv sync --extra cpu          # CPU only
uv sync --extra cuda         # NVIDIA GPU
uv sync --extra cuda --dev   # + dev tools (linting, tests)
```

Copy `.env.example` to `.env` and fill in `WANDB_API_KEY` and `HF_TOKEN`.

### Download datasets and models

```bash
# Download all datasets and models used by any benchmark
uv run python scripts/setup_data.py --all-benchmarks
uv run python scripts/setup_models.py --all-benchmarks

# Or limit to the currently selected benchmark (default: poc)
uv run python scripts/setup_data.py
uv run python scripts/setup_models.py
```

---

## Running

```bash
# Default config (glue_2_label benchmark, ties method)
uv run python main.py

# Select benchmark and method
uv run python main.py benchmark=glue-2-label method=chebyshev device=cuda

# Multi-run sweep (Hydra -m)
uv run python main.py -m benchmark=poc method=averaging,ties,chebyshev device=cuda

# Force retrain (ignore cached models)
uv run python main.py benchmark=glue-2-label method=epo benchmark.force_retrain=true
```

### Benchmarks

| Config | Tasks |
|---|---|
| `poc` | ag_news, imdb, mnli, mrpc |
| `glue-2-label` | cola, mrpc, qnli, sst2 |
| `recovery-cola` / `recovery-mrpc` / `recovery-qnli` / `recovery-sst2` | Single-task recovery |

### Methods

| Config | Description |
|---|---|
| `averaging` | Weighted task-vector averaging |
| `ties` | TIES merging (magnitude-based conflict resolution) |
| `chebyshev` | Fine-tuning with Chebyshev scalarization |
| `epo` | Exact Pareto Optimization search |
| `self_position` | Self-positioning method |

---

## Cluster (ETH Euler)

The `configs/cluster/euler.yaml` override redirects all paths to `$SCRATCH`. Assumes the repo is at `$HOME/Bachelor-Thesis-SML` (symlink or clone).

**One-time setup** (login node):
```bash
bash scripts/cluster/setup_euler.sh
```

**Submit a job:**
```bash
sbatch --export=BENCHMARK=glue-2-label,METHOD=chebyshev \
  scripts/cluster/run_benchmark.slurm

# With additional Hydra overrides:
sbatch --export=BENCHMARK=glue-2-label,METHOD=chebyshev,\
EXTRA_ARGS="seed=123 wandb.group=sweep1" \
  scripts/cluster/run_benchmark.slurm
```

**Pull results to local machine:**
```bash
bash scripts/cluster/extract_results.sh [your_nethz]
```

Results are also synced to W&B automatically during the run.

---

## Config reference

The framework uses [Hydra](https://hydra.cc/) for configuration. All configs are in [configs/](configs/).

### Key command-line overrides

| Option | Values | Description |
|---|---|---|
| `benchmark` | `poc`, `glue-2-label`, `recovery-*` | Benchmark |
| `method` | see Methods table above | Merging method |
| `device` | `auto`, `cpu`, `cuda`, `xpu` | Compute device |
| `seed` | integer (default `42`) | Random seed |
| `cluster` | `euler` | Redirect paths to `$SCRATCH` |
| `benchmark.mode` | `train_eval`, `train_only`, `eval_only` | Execution mode |
| `benchmark.cache_enabled` | `true`, `false` | Use cached models/evals |
| `benchmark.force_retrain` | `true`, `false` | Ignore cached models |
| `benchmark.evaluation.batch_size` | integer (default `32`) | Eval batch size |
| `benchmark.evaluation.num_samples` | integer or `null` | Limit eval samples |
| `wandb.mode` | `online`, `offline`, `disabled` | W&B tracking mode |
| `wandb.group` | string | Group runs in W&B |
| `wandb.tags` | list e.g. `[tag1,tag2]` | W&B tags |
| `logging.level` | `DEBUG`, `INFO`, `WARNING` | Log verbosity |

### Custom configs

```yaml
# configs/method/my_method.yaml
# @package _global_
method:
  name: my_method
  class_path: src.methods.my_method.MyMethod
  params:
    param1: value1
```

```yaml
# configs/benchmark/my_benchmark.yaml
# @package _global_
benchmark:
  name: my_benchmark
  tasks: [task1, task2]
  preference_vectors:
    - [0.5, 0.5]
```

---

## Troubleshooting

**W&B disabled:**
```bash
uv run python main.py wandb.mode=disabled
```

**Module not found:**
```bash
uv pip install -e .
```

---

## Benchmark design

- Base model: [roberta-base](https://huggingface.co/FacebookAI/roberta-base)
- Fine-tuned checkpoints: [textattack/roberta-base-*](https://huggingface.co/textattack) for all tasks
- Task vectors with different classification head sizes are zero-padded before merging; MNLI label order is remapped
- All models share the same pre-training point (required for linear mode connectivity)
---
- todo: expand on benchmark explanation, design decisions etc.

### (tmp) old poc benchmark explanation
- We will first implement a proof of concept of the methodology before moving to bigger & harder to train models, more datasets and different architectures
- As a base model we use [roberta-base](https://huggingface.co/FacebookAI/roberta-base), an improved version of [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
- For supervised fine tuning we will use the textattack roberta-base checkpoints for the following datastes
  1. [AG News](https://huggingface.co/datasets/wangrongsheng/ag_news) datasets - classifying news articles with topic; [HF textattact/roberta-base-ag-news](https://huggingface.co/textattack/roberta-base-ag-news)
  2. [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) dataset - classifying movie reviews into positive and negative; [HF textattac/roberta-base-imdb](https://huggingface.co/textattack/roberta-base-imdb#textattack-model-card)
  3. [MNLI](https://huggingface.co/datasets/SetFit/mnli) dataset - classify if how a premise fits a hypothesis; [HF textattack/roberta-base-MNLI](https://huggingface.co/textattack/roberta-base-MNLI)
  4. [MRPC dataset](https://huggingface.co/datasets/SetFit/mrpc) - decide whether a text is a Paraphrase of another; [HF textattack/roberta-base-MRPC](https://huggingface.co/textattack/roberta-base-MRPC)
- For more potential roberta-base fine tuned models from text-attack see [here](https://huggingface.co/textattack/models?search=roberta-base)
- As an alternative pre-training point the [ibm-research/ColD-Fusion-...-seed0](https://huggingface.co/ibm-research/models?search=seed0&sort=created&p=1) models could be used; merged model from 35 fine tuned models on different tasks from bert-base-uncased ([paper](https://arxiv.org/abs/2212.01378))
- For additional unlabeled data required by some merging methods we could use [wikitext-103](https://huggingface.co/datasets/Salesforce/wikitext)

## Resources

- [Fusion Bench](https://github.com/tanganke/fusion_bench/tree/main) — reference implementations of task arithmetic methods
- [Mergekit](https://github.com/arcee-ai/mergekit) — application-focused model merging toolkit
- [MergeBench](https://github.com/uiuctml/MergeBench) — benchmark suite
