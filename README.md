# Shakespeare Text Generation Model

## Overview
This project implements a Bigram Language Model using PyTorch to generate text in the style of Shakespeare. The model uses a transformer-based architecture with self-attention mechanisms to learn and generate text from a given input corpus.

## Features
- Transformer-based neural network architecture
- Multi-head self-attention mechanism
- Position and token embeddings
- TensorBoard logging for tracking training progress
- Text generation capability

## Prerequisites
- Python 3.8+
- PyTorch
- TensorBoard
- CUDA (optional, for GPU acceleration)

## Model Architecture
- Token Embedding Layer
- Position Embedding Layer
- Multiple Transformer Blocks
  - Multi-Head Self-Attention
  - Feed-Forward Neural Network
- Layer Normalization
- Language Model Head

### Hyperparameters
- Batch Size: 64
- Block Size: 256
- Embedding Dimension: 64
- Number of Heads: 4
- Number of Layers: 4
- Learning Rate: 3e-4

## Installation
1. Clone the repository
2. Install required dependencies:
```bash
pip install torch tensorboard
```

## Preparation
1. Prepare your input text file named `input.txt`
2. The script will automatically:
   - Tokenize the input text
   - Create train/validation splits
   - Generate a vocabulary

## Training
Run the script to train the model:
```bash
python shakespeare_model.py
```

The training process will:
- Train for 5000 iterations
- Evaluate loss every 500 iterations
- Log metrics to TensorBoard
- Save model checkpoints

## Text Generation
After training, the model will:
- Generate 2000 tokens printed to console
- Save 10,000 generated tokens to `more.txt`

## TensorBoard Visualization
View training metrics:
```bash
tensorboard --logdir=runs
```

## Device Support
Automatically detects and uses:
- CUDA (NVIDIA GPU)
- MPS (Apple Silicon)
- CPU (Fallback)

## Output
- Console will display:
  - Device being used
  - Model parameter count
  - Training loss at intervals
- Generated text in console and `more.txt`

## License
[Insert your license here]

## Acknowledgements
Inspired by Andrej Karpathy's neural network tutorials and the GPT architecture.
