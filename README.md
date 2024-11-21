# Shakespeare Text Generation Model

## Overview
A PyTorch-based Bigram Language Model that generates text in the style of Shakespeare, utilizing a transformer-based architecture with advanced natural language processing techniques.

## Features
### Model
- Transformer-based neural network architecture
- Multi-head self-attention mechanism
- Position and token embeddings
- TensorBoard logging for tracking training progress
- Text generation capability

### API Endpoints

Start API
```bash
python model_api.py
```
API Endpoints

#### /generate

- Method: GET

- Response:

  ```json
  {
    "generated_text": "string",
  }
  ```

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

## Application Startup

### Frontend
1. Navigate to frontend directory
```bash
cd frontend
```

2. Install dependencies
```bash
npm install
```

3. Start development server
```bash
npm run dev
```

### Backend
1. Start Python API
```bash
python model_api.py
```

### Access
- Frontend: `http://localhost:3000`
- API Endpoints: `http://localhost:8000`

### Development Mode
- Frontend runs on port 3000
- Backend API runs on port 8000
- Ensure both are running simultaneously

## Model Preparation
1. Prepare your input text file named `input.txt`
2. The script will automatically:
   - Tokenize the input text
   - Create train/validation splits
   - Generate a vocabulary

## Training
Run the script to train the model:
```bash
python bigram.py
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

## Potential Improvements
### Advanced Tokenization
I'll revise the "Potential Improvements" section to describe Karpathy's tokenization tutorial more specifically in the context of improving the Shakespeare text generation app:

## Potential Improvements
### Advanced Tokenization Inspired by Karpathy's Approach
Karpathy's tutorial introduces a more sophisticated tokenization approach that could significantly enhance our Shakespeare text generation model:

1. **Byte Pair Encoding (BPE)**: Implement a dynamic tokenization method that:
   - Learns token representations from the corpus
   - Handles out-of-vocabulary words more effectively
   - Reduces vocabulary size while maintaining semantic meaning
   - Captures subword and character-level nuances in Shakespeare's text

2. **Tokenization Benefits**:
   - Improved handling of rare words and character combinations
   - More efficient representation of Shakespeare's complex language
   - Potential reduction in model complexity
   - Better generalization across different text styles

3. **Implementation Considerations**:
   - Modify current character-level tokenization
   - Implement iterative token merging algorithm
   - Create custom vocabulary based on Shakespeare corpus
   - Potentially reduce model's computational complexity

The BPE approach could provide more nuanced text generation, capturing the linguistic intricacies of Shakespearean language more precisely than the current character-level tokenization.

## Acknowledgements
Inspired by Andrej Karpathy's neural network tutorials and the GPT architecture.