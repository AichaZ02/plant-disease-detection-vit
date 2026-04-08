# Plant Disease Detection using Vision Foundation Models

A comparative study of CNN-based transfer learning and Vision Transformer fine-tuning 
for plant disease classification, with Grad-CAM interpretability analysis.

## Results

| Model | Validation Accuracy | Epochs | Strategy |
|-------|-------------------|--------|----------|
| ResNet50 | 96.2% | 5 | Transfer learning (final layer only) |
| ViT-B/16 | 99.7% | 3 | Full fine-tuning |

## Dataset

[PlantVillage](https://huggingface.co/datasets/Hemg/new-plant-diseases-dataset) — 70,295 images 
across 38 plant disease classes covering 14 crop species.

## Methods

**ResNet50 (Day 2)**
- Pretrained on ImageNet, final classification layer replaced and fine-tuned
- Only ~1M parameters trained out of 25M total
- Achieved 96.2% validation accuracy in 5 epochs

**ViT-B/16 (Day 3)**
- Pretrained on ImageNet-21k (14M images), fully fine-tuned on plant data
- Global self-attention allows the model to capture disease patterns across the entire leaf
- Achieved 99.7% validation accuracy in 3 epochs

**Grad-CAM Interpretability (Day 4)**
- Applied Grad-CAM to visualize which leaf regions the ViT model focuses on
- Results show the model correctly attends to diseased areas rather than background
- Interpretability is critical for real-world agricultural deployment where trust matters

## Grad-CAM Visualizations

Grad-CAM visualizations are available in the notebook output directly.
Open `plant-disease-detection.ipynb` to see the heatmaps inline.

## Key Observations

- ViT outperforms ResNet50 by ~3.8% despite being trained on the same data
- ViT converges faster (3 epochs vs 5) but requires more time per epoch (~30 min vs ~5 min)
- Grad-CAM confirms the model attends to disease-relevant leaf regions, not background
- PlantVillage is a controlled dataset — real field conditions would be significantly harder

## Technical Stack

- Python, PyTorch, HuggingFace Transformers
- Google Colab / Kaggle (T4 GPU)
- pytorch-grad-cam for interpretability

## Context

This project was developed as part of a self-directed research preparation for a PhD 
application in Foundation Models for plant phenotyping (Inria / AgriScienceFM project).
