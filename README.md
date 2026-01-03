# COVID-19 ResNet Project

## Overview
This project explores the application of deep learning, specifically the ResNet architecture, to the classification of COVID-19 from medical imaging data. The main goal is to leverage convolutional neural networks (CNNs) to distinguish between COVID-19 positive and negative cases using image data, providing insights into the effectiveness of transfer learning and modern neural network architectures in medical diagnostics.

## Notebook Summary
The notebook `HW5P1T1.ipynb` contains the following key sections:
- **Data Loading and Preprocessing:** Utilizes a custom data loader to import and preprocess the dataset, including normalization and augmentation techniques to improve model generalization.
- **Model Architecture:** Implements a ResNet-based model, either from scratch or using pretrained weights, to perform image classification.
- **Training and Evaluation:** Trains the model on the dataset, monitors performance metrics such as accuracy and loss, and evaluates the model on a validation/test set.
- **Results and Analysis:** Presents the results, including confusion matrices, ROC curves, and a discussion of the model's strengths and limitations.

## Analysis
The project demonstrates that transfer learning with ResNet can achieve strong performance on COVID-19 image classification tasks, even with limited data. Data augmentation and careful preprocessing are crucial for improving model robustness. The results highlight the potential of deep learning in assisting medical professionals with rapid and accurate diagnosis, though further validation on larger and more diverse datasets is recommended for real-world deployment.

## Files
- `HW5P1T1.ipynb`: Main notebook containing code, experiments, and analysis.
- `data_loader.py`: Script for loading and preprocessing the dataset.

## Requirements
- Python 3.x
- PyTorch or TensorFlow (depending on implementation)
- NumPy, Matplotlib, and other standard data science libraries

## Usage
1. Ensure all dependencies are installed.
2. Unzip the dataset.
3. Run the notebook `HW5P1T1.ipynb` to reproduce the experiments and results.

## Acknowledgements
This project is for educational purposes and is based on publicly available COVID-19 imaging datasets. Please cite original data sources if using this work for research or publication.