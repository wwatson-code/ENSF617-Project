# AI-Generated Image Detection Experiments

This repository contains our ENSF 617 project work on binary detection of real vs AI-generated images using the Defactify image dataset. So far, we have built baseline image classifiers, patch-based models, and residual/autocorrelation pipelines, and we have investigated issues around prompt structure, class imbalance, and evaluation bias.

## Repository Layout

- `prototype-notebooks/`
  - experiment notebooks for the main model variants
- `tools/`
  - `data_prep.py` for downloading and organizing the dataset
  - `build_balanced_splits.py` for creating balanced prompt-level subsets when needed
- `research-material/`
  - reference paper(s) and background material
- `requirements.txt`
  - Python dependencies
- `.env.example`
  - example environment-variable file for local setup

Generated artifacts such as downloaded data, W&B logs, checkpoints, and notebook caches are ignored through `.gitignore`.

## Setup

Run commands from the repository root.

```bash
pip install -r requirements.txt
```

If you want W&B logging, export your API key before running the notebooks:

```bash
export WANDB_API_KEY=your_key_here
```

The notebooks now read `WANDB_API_KEY` from the environment instead of storing it directly in the notebook source.

## Current Experiments

- `genai_classifier_images.ipynb`
  - baseline raw-image classifier using ResNet transfer learning
- `genai_classifier_vgg16_random_patch.ipynb`
  - VGG-16 on random `224x224` image patches
- `genai_classifier_resnet18_residual_autocorr.ipynb`
  - ResNet-18 on noise residual autocorrelation maps
- `genai_classifier_resnet18_residual_autocorr_prompt_sampler.ipynb`
  - residual/autocorrelation model with prompt-aware train sampling
- `genai_classifier_resnet18_residual_autocorr_prompt_train_flat_eval.ipynb`
  - prompt-aware training with flat manifest-driven validation/test evaluation

## Notes

- The notebooks assume data lives under `output/` relative to the repo root.
- Validation and test evaluation should be treated carefully because prompt structure and class balance differ across splits.
- Accuracy alone can be misleading; balanced accuracy and per-class recall are important for this project.
