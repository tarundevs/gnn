# Multimodal GNN for Node Classification

A PyTorch implementation of a Multimodal Graph Neural Network (GNN) for node classification tasks, specifically designed for knowledge graphs containing multiple types of nodes (text, images, IRIs). The model uses Graph Attention Networks (GAT) with embedding caching for efficient processing of multimodal data.

## Features

- **Multimodal Processing**: Handles various node types including:
  - Text nodes (using DistilBERT embeddings)
  - Image nodes (using CLIP embeddings)
  - IRI nodes
  - Blank nodes

- **Advanced Architecture**:
  - Multiple Graph Attention layers
  - Residual connections
  - Batch and layer normalization
  - Configurable number of attention heads
  - Dropout for regularization

- **Efficient Data Processing**:
  - Embedding caching system for fast retraining
  - Batch processing for large graphs
  - Progress bars for training monitoring

- **Training Optimizations**:
  - Learning rate scheduling
  - Gradient clipping
  - Early stopping
  - Best model checkpointing

## Requirements

```txt
torch
torch-geometric
transformers
tqdm
kgbench
fire
scikit-learn
numpy
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| dataset | 'dmg777k' | Dataset name |
| hidden_dim | 256 | Hidden layer dimension |
| output_dim | 128 | Output dimension |
| num_layers | 3 | Number of GAT layers |
| num_heads | 8 | Number of attention heads |
| dropout | 0.3 | Dropout rate |
| lr | 0.0005 | Learning rate |
| weight_decay | 1e-4 | Weight decay for regularization |
| epochs | 200 | Maximum number of epochs |
| patience | 20 | Early stopping patience |

## Model Architecture

The model consists of several key components:

1. **Multimodal Encoder**:
   - CLIP for image encoding
   - DistilBERT for text encoding
   - Custom embedding for IRIs

2. **Graph Neural Network**:
   - Multiple GAT layers with residual connections
   - Layer normalization after each GAT layer
   - Batch normalization for input features
   - Dense output layer with dropout

## Data Processing

The system uses a caching mechanism for embeddings:

1. First run generates and caches embeddings
2. Subsequent runs use cached embeddings
3. Automatic cache management based on parameters

Cache location: `cached_embeddings/`

## Results

Training progress is displayed, showing:
- Current epoch
- Loss value
- Training accuracy
- Validation accuracy

Final model checkpoints are saved in the `checkpoints/` directory.
