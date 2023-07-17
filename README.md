# Vision-Transformer-using-PyTorch

This repository contains an updated implementation of a Vision Transformer (ViT), a model that applies the transformer architecture (which was originally designed for natural language processing) to computer vision tasks. We demonstrate the use of the model with the CIFAR-100 dataset.

## Overview

Vision Transformers' main concept is to handle a picture in the same way as a string of tokens in a language. A series of 1D vectors are created by flattening the image's fixed-size patches into a series of 1D vectors. A transformer encoder is then used to feed this sequence through. The model may weigh each patch's contribution depending on its context, or its relationship to other patches, to determine the output owing to the transformer's attention mechanism.

## Requirements

The code is implemented in Python, and uses the following libraries:
- PyTorch
- torchvision
- tqdm

You can install the requirements via `pip`:

```sh
pip install torch torchvision tqdm
```

## Usage
Clone this repository:
```sh
git clone https://github.com/ssakhash/Vision-Transformer-from-Scratch.git
```
Run the Python script:
```sh
python ViT.py
```
This command initiates the training process. The model goes through 20 epochs on the CIFAR-100 dataset with data augmentation and L2 regularization. It also includes learning rate scheduling and uses the tqdm library to visualize the progress of data loading and model training. The training loss after each epoch gets displayed on the console.

## Model Configuration
You can customize the model configuration by changing the Config class in main.py. Here are the configuration parameters:
- 'img_size': The size of the input images (32 for CIFAR-100).
- 'patch_size': The size of the patches the image is divided into.
- 'num_classes': The number of classes in the dataset (100 for CIFAR-100).
- 'dim': The dimensionality of the patch embeddings.
- 'depth': The number of transformer blocks.
- 'heads': The number of attention heads in the multi-head attention mechanism.
- 'mlp_dim': The dimensionality of the feed-forward neural network inside the transformer.
- 'channels': The number of channels in the input images (3 for RGB images).
- 'dropout': The dropout rate.
- 'weight_decay': The coefficient for L2 regularization in the Adam optimizer.

## Contributing
Contributions to this repository are welcome. You can contribute by opening an issue or by creating a pull request.
