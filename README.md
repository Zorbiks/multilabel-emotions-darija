# Moroccan and Algerian Arabic Emotion Detection

## Overview

This project explores multi-label emotion detection in Moroccan Darija and Algerian Arabic. The goal is to classify short Arabic text snippets according to the emotions they express, including categories such as joy, sadness, anger, fear, surprise, and disgust.

The work is implemented in the notebook at [main.ipynb](main.ipynb) and is structured as a research comparison rather than a single-model demo. It evaluates multiple modeling strategies and reports their performance under the same experimental setup.

## Project objective

The main task is to take a short Arabic sentence and predict all emotions present in it, not just a single label. This makes the problem a multi-label classification task, where each input can be associated with multiple emotion classes at the same time.

## Repository structure

- [main.ipynb](main.ipynb): main experimental notebook containing the full pipeline
- [predictions_archive/](predictions_archive): saved model predictions for different experiments and seeds
- [results_archive/](results_archive): final result tables and aggregated outputs
- [tests_archive/](tests_archive): validation scripts and deliverable checks
- [README.md](README.md): project overview and workflow summary

## Research workflow

### 1. Setup and data loading

- install the required dependencies
- load the Arabic emotion datasets
- define the train/validation/test splitting strategy

### 2. Data audit and cleaning

- inspect text quality and noise
- detect duplicate or problematic examples
- analyze script mixing, emoji usage, repeated letters, and similar artifacts
- remove noisy duplicates from the training set without leaking information into validation/test sets

### 3. Baselines and shortcut methods

- TF-IDF + classical machine learning classifiers
- simple heuristic and shortcut-style models

### 4. Modeling experiments

- zero-shot and NLI-inspired approaches
- custom neural network baselines
- frozen pretrained language models
- fine-tuned pretrained models, including multitask variants

### 5. Evaluation and reporting

- compute macro-F1 scores for each model and each language/dialect
- save prediction outputs and result tables
- compare the best-performing approaches at the end of the study

## Expected outputs

The pipeline generates:

- model predictions for validation and test examples
- per-model performance metrics
- comparison tables across experiments
- final benchmark summaries identifying the strongest approaches

## Notes

This project is designed to be reproducible and comparative. It emphasizes systematic evaluation across many model families and uses validation metrics to compare effectiveness fairly across dialectal Arabic data.

## How to use it

Open [main.ipynb](main.ipynb) in Kaggle or VS Code and run the cells in order to reproduce the pipeline, including dataset processing, model training, evaluation, and output generation.
