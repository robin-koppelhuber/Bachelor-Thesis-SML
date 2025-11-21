## About

## Install
- install `uv` ([docs](https://docs.astral.sh/uv/getting-started/installation/))
- depending on your plattform / accelerator (`cpu`, `cuda`, `xpu`) run

```bash
# --dev is optional
uv sync --extra cpu --dev
```

## Todos
### Methods to implement
- AdaMerge ([github](https://github.com/EnnengYang/AdaMerging), [paper](https://arxiv.org/abs/2310.02575))
---
- Previous work / repos on Model merging benchmarks
    - [Fusion bench](https://github.com/tanganke/fusion_bench/tree/main): implements many "task arithmetic" methods and has some overarching framework for managing models, datasets and configs
    - [Mergekit](https://github.com/arcee-ai/mergekit): most popular open-source package (6.5k stars); more application focused, does not seem usefull except their "Raw PyTorch Model Merging" (mergekit-pytorch) module
    - [Merge bench](https://github.com/uiuctml/MergeBench): 

### Packages to add
- Peft for LoRAs
- Huggingface Hub
- Hydra for configurations
- dotenv

