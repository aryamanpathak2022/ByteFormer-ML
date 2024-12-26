# Bytes Are All You Need: Transformers Operating Directly On File Bytes

This repository contains the implementation of the research paper ["Bytes Are All You Need"](https://arxiv.org/pdf/2306.00238), adapted for the CIFAR-10 dataset. Our work involves implementing both Vision Transformer (ViT) and ByteFormer architectures to evaluate their performance on image classification tasks.

---

## Overview

The objective of this implementation is to explore the application of transformers on file bytes rather than traditional image RGB formats. We:

1. Implemented the Vision Transformer (ViT) to classify CIFAR-10 images in their original RGB format.
2. Developed the ByteFormer by converting CIFAR-10 RGB images into byte representations and leveraging transformer architecture for byte-level embeddings and processing.

---

## CIFAR-10 Dataset

CIFAR-10 is a collection of 60,000 32x32 RGB images distributed across 10 classes. It serves as a benchmark dataset for image classification tasks.

### Preprocessing

- **ViT**: Images are used in their standard RGB format (32x32x3).
- **ByteFormer**: Images are converted to bytes using TIFF encoding, resulting in a flattened vector of 3072 bytes per image.

---

## Vision Transformer (ViT) Implementation

We implemented the ViT architecture to process RGB images. The architecture consists of:

1. **Patch Embedding**: The 32x32 RGB image is divided into patches of size 4x4.
2. **Positional Encoding**: Positional information is added to the patch embeddings.
3. **Transformer Encoder**: Consisting of:
   - Multi-head self-attention layers.
   - Feed-forward neural network (FFN).
   - Layer normalization and dropout for regularization.

### Model Parameters

- Input shape: `(32, 32, 3)`
- Patch size: `4`
- Embedding dimension: `64`
- Number of heads: `4`
- Feed-forward dimension: `128`
- Number of transformer layers: `8`
- Number of classes: `10`

### Code Snippet

```python
vit_model = create_vit_model(
    input_shape=(32, 32, 3),
    patch_size=4,
    embed_dim=64,
    num_heads=4,
    ff_dim=128,
    num_layers=8,
    num_classes=10
)
```

---

## ByteFormer Implementation

ByteFormer processes the byte representation of images. The key steps are:

1. **Byte Conversion**: RGB images are converted to a byte array of size `3072` using TIFF encoding.
2. **Byte Embedding**: Each byte (0-255) is mapped to a 32-dimensional embedding.
3. **1D Convolution**: A convolutional layer with filters and ReLU activation is applied for initial feature extraction.
4. **Positional Encoding**: Added to the byte embeddings.
5. **Transformer Encoder**: Similar to ViT, consisting of multi-head attention, FFN, layer normalization, and dropout.

### Model Parameters

- Input shape: `(3072,)`
- Byte vocabulary size: `256`
- Byte embedding dimension: `128`
- Convolutional filters: `32`
- Patch size: `16`
- Transformer embedding dimension: `32`
- Number of heads: `4`
- Feed-forward dimension: `256`
- Number of transformer layers: `4`
- Number of classes: `10`

### Code Snippet

```python
model = create_vit_byte_model(
    input_shape=(3072, ),
    byte_vocab_size=256,
    byte_embed_dim=128,
    conv_filters=32,
    patch_size=16,
    embed_dim=32,
    num_heads=4,
    ff_dim=256,
    num_layers=4,
    num_classes=10
)
```

---

## Architecture



### Explanation of the Image

![VIT](https://github.com/user-attachments/assets/3d52d2a3-5a7d-4b6d-8ef5-1a972c66b95f)

![ByteFormer](https://github.com/user-attachments/assets/98413f2d-794a-46a1-b8fd-1e6ea1cdd9bd)



The diagram illustrates the workflow for both ViT and ByteFormer architectures:

- **ViT Path**: RGB image is patch-embedded, passed through positional encoding, and processed via transformer encoders.
- **ByteFormer Path**: Image bytes are byte-embedded, passed through a 1D convolution layer, followed by positional encoding and transformer encoders.

---

---

## References

- Original Paper: [Bytes Are All You Need](https://arxiv.org/pdf/2306.00238)
- CIFAR-10 Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)



## Contributors

1. Aryaman Pathak

2. Aditya Priyadarshi

---

## License

This project is licensed under the [MIT License](./LICENSE).

