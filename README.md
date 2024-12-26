# Bytes Are All You Need: CIFAR-10 Implementation

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

## Vision Transformer Model Summary

| Layer (type)                    | Output Shape           | Param #       |
|---------------------------------|------------------------|---------------|
| input_layer (InputLayer)        | (None, 32, 32, 3)      | 0             |
| patch_embedding (PatchEmbedding)| (None, None, 64)       | 3,136         |
| positional_encoding (PositionalEncoding)| (None, 64, 64) | 0             |
| transformer_encoder_block (TransformerEncoderBlock) | (None, 64, 64) | 83,200        |
| transformer_encoder_block_1 (TransformerEncoderBlock)| (None, 64, 64) | 83,200        |
| transformer_encoder_block_2 (TransformerEncoderBlock)| (None, 64, 64) | 83,200        |
| transformer_encoder_block_3 (TransformerEncoderBlock)| (None, 64, 64) | 83,200        |
| transformer_encoder_block_4 (TransformerEncoderBlock)| (None, 64, 64) | 83,200        |
| transformer_encoder_block_5 (TransformerEncoderBlock)| (None, 64, 64) | 83,200        |
| transformer_encoder_block_6 (TransformerEncoderBlock)| (None, 64, 64) | 83,200        |
| transformer_encoder_block_7 (TransformerEncoderBlock)| (None, 64, 64) | 83,200        |
| layer_normalization_16 (LayerNormalization) | (None, 64, 64) | 128           |
| flatten (Flatten)               | (None, 4096)           | 0             |
| dense_17 (Dense)                | (None, 128)            | 524,416       |
| dropout_24 (Dropout)            | (None, 128)            | 0             |
| dense_18 (Dense)                | (None, 10)             | 1,290         |

**Total params:** 1,194,570 (4.56 MB)  
**Trainable params:** 1,194,570 (4.56 MB)  
**Non-trainable params:** 0 (0.00 B)

---

## ByteFormer Model Summary


| Layer (type)                        | Output Shape              | Param #      |
|-------------------------------------|---------------------------|--------------|
| input_layer_9 (InputLayer)          | (None, 3072)              | 0            |
| embedding (Embedding)               | (None, 3072, 128)         | 32,768       |
| conv1d (Conv1D)                     | (None, 1536, 32)          | 12,320       |
| positional_encoding_1               | (None, 1536, 32)          | 0            |
| (PositionalEncoding)                |                           |              |
| transformer_encoder_block_8         | (None, 1536, 32)          | 33,600       |
| (TransformerEncoderBlock)           |                           |              |
| transformer_encoder_block_9         | (None, 1536, 32)          | 33,600       |
| (TransformerEncoderBlock)           |                           |              |
| transformer_encoder_block_10        | (None, 1536, 32)          | 33,600       |
| (TransformerEncoderBlock)           |                           |              |
| transformer_encoder_block_11        | (None, 1536, 32)          | 33,600       |
| (TransformerEncoderBlock)           |                           |              |
| layer_normalization_25              | (None, 1536, 32)          | 64           |
| (LayerNormalization)                |                           |              |
| flatten_1 (Flatten)                 | (None, 49152)             | 0            |
| dense_28 (Dense)                    | (None, 256)               | 12,583,168   |
| dropout_37 (Dropout)                | (None, 256)               | 0            |
| dense_29 (Dense)                    | (None, 10)                | 2,570        |

**Total params:** 12,765,290 (48.70 MB)  
**Trainable params:** 12,765,290 (48.70 MB)  
**Non-trainable params:** 0 (0.00 B)






## Architecture



### Explanation of the Image

![VIT](https://github.com/user-attachments/assets/d2350a36-e512-45b7-a185-9945a49cd01d)

![ByteFormer](https://github.com/user-attachments/assets/07d2207c-44ab-4272-9c6d-2284e434f51a)





The diagram illustrates the workflow for both ViT and ByteFormer architectures:

- **ViT Path**: RGB image is patch-embedded, passed through positional encoding, and processed via transformer encoders.
- **ByteFormer Path**: Image bytes are byte-embedded, passed through a 1D convolution layer, followed by positional encoding and transformer encoders.



---

## Future Implications

This approach of directly training models on bytes offers several advantages:

- Generality: The same architecture can be applied to any file type, making it highly adaptable for diverse datasets.

- Simplified Pipeline: Eliminates the need for modality-specific preprocessing, such as RGB processing for images or tokenization for text.

- Efficiency: Direct byte processing can reduce the complexity of converting data into intermediate formats.

By leveraging byte-level data, we can develop models that are more universally applicable and capable of handling a wide range of input types without significant architectural changes.

---
## References

- Original Paper: [Bytes Are All You Need](https://arxiv.org/pdf/2306.00238)
- CIFAR-10 Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)




## Contributors

1. Aryaman Pathak @aryamanpathak2022

2. Aditya Priyadarshi @ap5967ap

---


