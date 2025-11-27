# Automatic Waste Container Classification

**Course:** Audiovisual Data Processing (Processamento de Dados Audiovisuais)
**Institution:** Universidade da Beira Interior
**Author:** Bernardo Vieira Henriques
**Date:** 2025/2026

## Project Overview

This project addresses the problem of automatic image classification for urban waste containers using Deep Learning techniques. The goal is to develop a computer vision system capable of recognizing seven distinct types of waste containers to support applications in smart city management and autonomous waste collection.

The solution implements and compares two distinct approaches:
1.  **Custom CNN:** A Convolutional Neural Network designed and trained from scratch to serve as a baseline.
2.  **Transfer Learning:** Fine-tuning state-of-the-art architectures (ResNet18, DenseNet121, MobileNetV2, and EfficientNet-B0) to assess the trade-offs between accuracy and computational efficiency.

## Dataset

The system classifies images into the following seven categories based on the provided dataset:
* **Indiferenciado** (General Waste)
* **Papel** (Paper - Blue)
* **Plástico** (Plastic - Yellow)
* **Vidro** (Glass - Green)
* **Orgânico** (Organic - Brown)
* **Pilhas** (Batteries - Red)
* **Óleo** (Oil - Orange)

The dataset is split into Training, Validation, and Test sets using a **Stratified Shuffle Split** strategy to maintain the class distribution balance across all subsets, which is critical given the class imbalance in the source data.

## Key Features

* **Data Preprocessing:** Implementation of image resizing, normalization, and on-the-fly Data Augmentation (rotation, flips, color jitter) to improve model robustness against lighting and angle variations.
* **Handling Class Imbalance:** Integration of **Focal Loss** to mitigate the impact of imbalanced classes during the training phase.
* **Architecture Flexibility:** A modular "Model Factory" design pattern allows for easy switching between different backbone architectures (ResNet, DenseNet, MobileNet, EfficientNet).
* **Explainability:** Integration of **Grad-CAM** (Gradient-weighted Class Activation Mapping) to visualize the specific regions of interest that influence the model's decision, ensuring the model focuses on the container rather than the background.
* **Metrics:** Comprehensive evaluation using Accuracy, Loss, Precision, Recall, and F1-Score.

## Project Structure

```text
project_root/
│
├── dataset_waste_container/      # Source images organized by class folders
├── models/                       # Directory where trained models (.pth) are saved
├── results/                      # Generated plots, indices, and Grad-CAM visualizations
│
├── scripts/                      # Source code
│   └── problem analysis.py       # Problem analysis and approach planning
│   ├── cnn.py                    # Definition of the custom CNN architecture
│   ├── data_transformation.py    # Data transforms and augmentation pipelines
│   ├── models_factory.py         # Factory to load and adapt pre-trained models
│   ├── train_cnn.py              # Training loop for the custom CNN
│   ├── train_finetune.py         # Training loop for Transfer Learning models
│   ├── test.py                   # Evaluation script for the custom CNN
│   └── test_finetune.py          # Evaluation script for Transfer Learning models
│
├── README.md                     # Project documentation
└── requirements.txt              # List of Python dependencies required