![Python Version 3.13.9](img.shields.io)
## About
- This project is part of my bachelor thesis at the ETH Statistical Machine Learning Lab ([ETH SML](https://sml.inf.ethz.ch/))
- The main goal of this project is to investigate some hypothesis about the performance of different model classes in the preference aware model merging regime
- These insights should eventually inform the development of a novel method
---
- Work in progress!


## Install

- install `uv` ([docs](https://docs.astral.sh/uv/getting-started/installation/))
- depending on your platform / accelerator (`cpu`, `cuda`, `xpu`) run

```bash
# --dev is optional
uv sync --extra cpu --dev
```

### Running the Benchmark

Basic usage:

```bash
# Run with default config
uv run python main.py

# Override specific settings, see config reference below
uv run python main.py benchmark=poc -m method=averaging,ties device=cuda
```

#### Evaluation-Only Mode (Training Methods)

For training-based methods (Chebyshev, EPO), you can re-evaluate saved models without retraining:

```bash
# Enable evaluation-only mode
uv run python -m src.benchmarks.poc.run method=chebyshev benchmark.training.evaluate_only=true

# Or evaluate a specific saved model
uv run python -m src.benchmarks.poc.evaluate_saved model_path=checkpoints/training/model.safetensors
```

Models are cached in `checkpoints/training/` with filenames like `chebyshev_a1b2c3d4e5f6.safetensors`

#### Sync W&B Artifacts

Download all models and artifacts from a W&B run:

```bash
uv run python scripts/sync_wandb_artifacts.py --entity YOUR_ENTITY --project YOUR_PROJECT --run-id RUN_ID
```

### Download Datasets and Models

Before running benchmarks, download the required data:

```bash
# Download all POC datasets
uv run python scripts/setup_data.py

# Download all POC models
uv run python scripts/setup_models.py
```

## Benchmark setup

### Proof on concept

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

---

- Methods to compare (given a preference vector)
  - Preference aware task arithmetic method with conflict avoidance, e.g. TIES merging
  - Fine-tuning from roberta-base with chebishev-scalarization and e.g. Adam optimizer
  - Fine-tuning from roberta-base with preference aware MGDA method, e.g. Exact Pareto Optimization search (EPO search) ([paper](https://arxiv.org/abs/2108.00597))
- Additionally preference aware AdaMerge could be considered
- With wikitext as unlabeled data a pseudo-labeling approach is not feasible for most fine-tuned tasks

#### Assumptions made for this evaluation

- Given multiple tasks or objectives and a preference vector, we want to find a model that finds the best trade-off for this preference vector (formalize!)
- Theoretically, we have the following model classes
  1. Convex combination of single objective / task optimal models; unpredictably altered by different 'interference avoidance' techniques from 'task arithmetic' methods
  2. Class of models obtainable by fine tuning from shared pre-training point towards to a single pre trained vector via linear scalarization or chebyshev scalarization
  3. Class of models obtainable via preference aware by MGDAs
- For class 1,
  - we assume that all models were fine-tuned from a shared pre-trained model to avoid the permutation invariance problem
  - For now we also assume similar hyperparameter & regularization to avoid e.g. magnitude conflicts
  - We assume that we have access to full fine-tuned task vectors, not just LoRAs, to avoid the 'different subspaces' problem when merging different LoRAs (Anisotropic vs. Isotropic)
  - **Task vectors with different classification head sizes** (due to different num_labels) are zero-padded to match the maximum size before merging, preserving all fine-tuned weights while allowing merging across tasks.
- These assumptions are necessary to have linear mode connectivity
- If we would drop the "optimize for given preference" requirement we would have access to a way bigger set of methods

## Config

The framework uses Hydra for configuration management. All configs are in [configs/](configs/):

- [configs/config.yaml](configs/config.yaml) - Main configuration
- [configs/benchmark/](configs/benchmark/) - Benchmark configs
- [configs/method/](configs/method/) - Merging method configs
- [configs/model/](configs/model/) - Model configs
- [configs/dataset/](configs/dataset/) - Dataset configs

#### Creating Custom Configs

**Custom Method:**

```yaml
# configs/method/my_method.yaml
# @package _global_
method:
  name: my_method
  class_path: src.methods.my_method.MyMethod
  params:
    param1: value1
    param2: value2
```

**Custom Benchmark:**

```yaml
# configs/benchmark/my_benchmark.yaml
# @package _global_
benchmark:
  name: my_benchmark
  tasks:
    - task1
    - task2
  preference_vectors:
    - [0.5, 0.5]
```

### Command Line Options

| Category          | Option                             | Values                              | Description                |
| ----------------- | ---------------------------------- | ----------------------------------- | -------------------------- |
| **Core**          | `benchmark`                        | `poc`                               | Benchmark configuration    |
|                   | `method`                           | `ties`, `chebyshev`, `epo`          | Merging method             |
|                   | `model`                            | `roberta_base`                      | Base model                 |
|                   | `device`                           | `auto`, `cpu`, `cuda`, `xpu`        | Compute device             |
|                   | `seed`                             | Integer (default: `42`)             | Random seed                |
| **W&B**           | `wandb.mode`                       | `online`, `offline`, `disabled`     | W&B tracking mode          |
|                   | `wandb.project`                    | String                              | Project name               |
|                   | `wandb.entity`                     | String                              | W&B entity/team            |
|                   | `wandb.group`                      | String                              | Run grouping               |
|                   | `wandb.tags`                       | List                                | Tags (e.g., `[tag1,tag2]`) |
|                   | `wandb.notes`                      | String                              | Run notes                  |
| **Logging**       | `logging.level`                    | `DEBUG`, `INFO`, `WARNING`, `ERROR` | Log level                  |
|                   | `logging.console_format`           | `simple`, `detailed`, `json`        | Console output format      |
|                   | `logging.log_to_file`              | `true`, `false`                     | Save logs to file          |
|                   | `logging.log_to_wandb`             | `true`, `false`                     | Log to W&B                 |
| **Paths**         | `paths.data_dir`                   | Path                                | Dataset directory          |
|                   | `paths.checkpoint_dir`             | Path                                | Checkpoint directory       |
|                   | `paths.output_dir`                 | Path                                | Output directory           |
|                   | `paths.log_dir`                    | Path                                | Log directory              |
| **Evaluation**    | `benchmark.evaluation.batch_size`  | Integer (default: `32`)             | Evaluation batch size      |
|                   | `benchmark.evaluation.num_samples` | Integer or `null`                   | Limit evaluation samples   |
|                   | `benchmark.save_merged_models`     | `true`, `false`                     | Save merged models         |
|                   | `benchmark.save_task_vectors`      | `true`, `false`                     | Save task vectors          |
| **Method Params** | `method.params.k`                  | Float (default: `0.2`)              | TIES top-k fraction        |
|                   | `method.params.lambda_merge`       | Float (default: `1.0`)              | TIES merge coefficient     |

### Quick Examples

```bash
# Change method and device
uv run python main.py method=averaging device=cuda

# W&B configuration
uv run python main.py wandb.mode=offline wandb.tags=[experiment1,roberta]

# Debug mode with detailed logging
uv run python main.py logging.level=DEBUG logging.console_format=detailed

# Custom paths
uv run python main.py paths.data_dir=/custom/path/data
```

## Troubleshooting

### W&B Not Available

W&B is optional. The framework works without it:

```bash
uv run python main.py wandb.mode=disabled
```

Or install it:

```bash
uv sync --group cluster
```

### Missing Dependencies

Reinstall all dependencies:

```bash
uv sync
```

For CUDA support:

```bash
uv sync --extra cuda
```

If you get 'module src not found errors'

```bash
uv pip install -e .
```

## Resources

- Previous work / repos on Model merging benchmarks
  - [Fusion bench](https://github.com/tanganke/fusion_bench/tree/main): implements many "task arithmetic" methods and has some overarching framework for managing models, datasets and configs
  - [Mergekit](https://github.com/arcee-ai/mergekit): most popular open-source package (6.5k stars); more application focused, does not seem useful except their "Raw PyTorch Model Merging" (mergekit-pytorch) module
  - [Merge bench](https://github.com/uiuctml/MergeBench):

## Todos

- Package project into container, e.g. docker -> apptainer
- Implement EPO search

#### Packages to add

- Joblib for caching (currently own implementation, getting out of hand)
- Peft for LoRAs


## Deprecated

### Methods to implement

- AdaMerge ([github](https://github.com/EnnengYang/AdaMerging), [paper](https://arxiv.org/abs/2310.02575))
