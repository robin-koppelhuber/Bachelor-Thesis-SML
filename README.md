## About

## Install
- install `uv` ([docs](https://docs.astral.sh/uv/getting-started/installation/))
- depending on your platform / accelerator (`cpu`, `cuda`, `xpu`) run

```bash
# --dev is optional
uv sync --extra cpu --dev
```

## Todos
#### Packages to add
- Peft for LoRAs
- Huggingface Hub
- Hydra for configurations
- dotenv

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
- These assumptions are necessary to have linear mode connectivity
- If we would drop the "optimize for given preference" requirement we would have access to a way bigger set of methods

## Resources
- Previous work / repos on Model merging benchmarks
    - [Fusion bench](https://github.com/tanganke/fusion_bench/tree/main): implements many "task arithmetic" methods and has some overarching framework for managing models, datasets and configs
    - [Mergekit](https://github.com/arcee-ai/mergekit): most popular open-source package (6.5k stars); more application focused, does not seem useful except their "Raw PyTorch Model Merging" (mergekit-pytorch) module
    - [Merge bench](https://github.com/uiuctml/MergeBench): 

## Deprecated
### Methods to implement
- AdaMerge ([github](https://github.com/EnnengYang/AdaMerging), [paper](https://arxiv.org/abs/2310.02575))
