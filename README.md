# Exploring Self-Supervised Learning with SimCLR

This repository hosts a comprehensive exploration into self-supervised learning using the SimCLR framework, focusing on the contrastive learning approach. The project is divided into three main parts: dataset implementation for contrastive model training, the SimCLR loss function and training step, and applying transfer learning strategies for model evaluation on a downstream classification task.

## Overview

The project explores the application of SimCLR for self-supervised learning on the PneumoniaMNIST dataset, which includes downsampled chest X-ray images. It covers:

- **Part A:** Implementation of a custom dataset class for contrastive learning, including a data augmentation pipeline for generating positive pairs.
- **Part B:** Implementation of the SimCLR loss function and training steps within a PyTorch Lightning framework.
- **Part C:** Application of linear probing and fine-tuning transfer learning strategies to evaluate the performance of pre-trained models on a downstream task of pneumonia detection.

## Installation

Ensure you have Python 3.x installed. Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include PyTorch, PyTorch Lightning, torchvision, torchmetrics, medmnist, matplotlib, and numpy.

## Usage

To run the notebook:

1. **Setup:** Ensure all dependencies are installed. If using Google Colab, uncomment the setup cell to install PyTorch Lightning and MedMNIST.
2. **Part A:** Run the cells under "Part A" to implement and visualize the dataset with augmented positive pairs.
3. **Part B:** Follow the instructions under "Part B" to implement the SimCLR loss function and incorporate it into the training step.
4. **Part C:** Execute the steps in "Part C" for linear probing and fine-tuning to evaluate the model's performance on the pneumonia detection task.

## Contributing

Contributions to enhance the project, add new features, or improve documentation are welcome. Please fork the repository, make your changes, and submit a pull request.

## Evaluation

An evaluation is included, discussing the results of applying linear probing and fine-tuning on two pre-trained models. It provides insights into the effectiveness of self-supervised learning models pre-trained with different strategies when applied to medical image classification tasks.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
