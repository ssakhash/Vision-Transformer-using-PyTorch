# Vision-Transformer-from-Scratch

This repository contains a simple implementation of a Vision Transformer (ViT), a type of model that applies the transformer architecture (primarily used for natural language processing) to computer vision tasks. Here, we use the CIFAR-100 dataset for the demonstration.

## Overview

The key idea behind Vision Transformers is to treat an image just like a sequence of tokens in a sentence. The image is broken down into fixed-size patches, which are then flattened into a sequence of 1D vectors. This sequence is then passed through a transformer encoder. The transformer's attention mechanism allows the model to weigh the contribution of each patch based on its context, or its relationship with other patches, in determining the final output.

## Requirements

The code is implemented in Python, and uses the following libraries:
- PyTorch
- torchvision

You can install the requirements via `pip`:

```sh
pip install torch torchvision
```

## Usage
Clone this repository:
```sh
git clone https://github.com/yourusername/vision-transformer.git](https://github.com/ssakhash/Vision-Transformer-from-Scratch.git
```
Run the Python script:
```sh
python ViT.py
```
This will start the training process. The model will go through 10 epochs on the CIFAR-100 dataset and print the training loss after each epoch.

## Model Configuration
You can customize the model configuration by changing the Config class in main.py. Here are the configuration parameters:
- 'img_size': The size of the input images (32 for CIFAR-100).
- 'patch_size': The size of the patches that the image is divided into.
- 'num_classes': The number of classes in the dataset (100 for CIFAR-100).
- 'dim': The dimensionality of the patch embeddings.
- 'depth': The number of transformer blocks.
- 'heads': The number of attention heads in the multi-head attention mechanism.
- 'mlp_dim': The dimensionality of the feed-forward neural network inside the transformer.
- 'channels': The number of channels in the input images (3 for RGB images).
- 'dropout': The dropout rate.

## Contributing
Contributions to this repository are welcome. You can contribute by opening an issue or by creating a pull request.
