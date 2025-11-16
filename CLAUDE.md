# Backprop from Scratch

In this working directory, I'm building backpropagation from the ground up for learning purposes. The goal is to understand the low-level concepts of backpropagation and neural network training through hands-on implementation.

## Project Goals

1. **Configurable Model Architecture** - A flexible neural network that stores weights, activations, and gradients
2. **Web-based Visualization** - Flask server with real-time heatmap visualization of all network matrices
3. **Low-level Backpropagation** - Manual implementation of forward/backward passes with minimal library reliance

## Current Status

### ✅ Completed Components

#### 1. Model ([model.py](model.py))
- Fully implemented neural network with configurable architecture
- GELU activation function with derivative
- Xavier weight initialization
- Forward pass with activation storage
- Backward pass computing gradients via chain rule
- Parameter update via gradient descent
- Supports arbitrary number of hidden layers

**Key Features:**
- Input dimension: Configurable
- Hidden layers: Configurable count and dimension
- Output dimension: Configurable
- Stores activations, gradients, and weights for visualization

#### 2. Visualizer ([visualizer.py](visualizer.py), [templates/index.html](templates/index.html))
- Flask web server running on port 7000
- Two-column layout: Forward Pass (left, green) | Backward Pass (right, orange/red)
- Real-time heatmap visualization of:
  - Weights and biases
  - Activations (pre and post)
  - Gradients
- Global color normalization across all matrices
- Color scale legend with min/p25/p50/p75/max markers
- "Run Single" button to trigger forward/backward pass with random data

**API Endpoints:**
- `/api/state` - Get current model state (weights, activations, gradients)
- `/api/run_single` - Run single forward/backward pass

#### 3. Dataset ([dataset.py](dataset.py))
- Abstract `Dataset` base class for extensibility
- `SequenceDataset` implementation for sequence learning tasks
- Features:
  - Configurable vocabulary, sequence length, and sample count
  - Two encoding types: one-hot and embedding
  - Two task types: next-token prediction and sequence classification
  - **Master sequences**: Pattern-based learning with subsequence extraction
  - Full dataset generation upfront (stored in memory)
  - `get_batch()` method for random sampling
  - `inspect_samples()` method for human-readable data inspection

**Master Sequence Architecture:**
Instead of random sequences, the dataset can use "master sequences" - longer fixed sequences from which training samples are extracted as subsequences. This creates learnable patterns.

Example:
```python
master_sequences = [
    ['a', 'b', 'c', 'c', 'a', 'a', 'b', 'b'],  # M1
    ['a', 'd', 'd', 'a', 'c', 'c', 'a', 'd']   # M2
]
# Training samples extracted:
# 'adda' → 'c'  (from M2: addaccad)
# 'caab' → 'b'  (from M1: abccaabb)
```

**Dataset Configuration Options:**

Three ways to configure sequence generation:

1. **Explicit Master Sequences**:
```python
'master_sequences': [
    ['a', 'b', 'c', 'c', 'a', 'a', 'b', 'b'],
    ['a', 'd', 'd', 'a', 'c', 'c', 'a', 'd']
]
```

2. **Auto-generated Master Sequences** (current):
```python
'num_master_sequences': 3,
'master_sequence_length': (8, 12)  # Random length in range [8, 12]
# Or use fixed length: 'master_sequence_length': 10
```

3. **Fully Random** (no master sequences):
```python
# Omit master sequence parameters
# Generates completely random sequences
```

**Current Configuration:**
- Vocabulary: ['a', 'b', 'c', 'd']
- Sequence length: 5
- Number of samples: 100
- Task: Next-token prediction
- Master sequences: 3 sequences with random lengths between 8 and 12

#### 4. Main Script ([main.py](main.py))
- Orchestrates dataset, model, trainer, and visualizer
- Automatically matches model dimensions to dataset
- Test forward/backward pass before starting server
- Clean startup output with progress indicators

### 🚧 In Progress

#### Trainer ([trainer.py](trainer.py))
- Skeleton structure created with comprehensive method signatures
- Not yet implemented (all methods are `pass` statements)
- Planned features:
  - Main training loop (train, train_epoch, train_batch)
  - Loss functions (MSE and cross-entropy)
  - Loss gradient computation
  - Evaluation on validation data
  - Utilities (batching, shuffling)
  - Callbacks and progress logging

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    main.py                          │
│  Orchestrates all components and starts server      │
└─────────────────────────────────────────────────────┘
          │
          ├──> dataset.py (SequenceDataset)
          │    - Generates training data
          │    - Provides batching interface
          │
          ├──> model.py (Model)
          │    - Forward pass
          │    - Backward pass
          │    - Parameter updates
          │
          ├──> trainer.py (Trainer) [NOT IMPLEMENTED]
          │    - Training loops
          │    - Loss computation
          │
          └──> visualizer.py (Flask Server)
               - Web UI for visualization
               - Real-time heatmaps
```

## File Structure

```
backprop/
├── model.py           # Neural network implementation
├── dataset.py         # Dataset generation and batching
├── trainer.py         # Training loop (skeleton)
├── visualizer.py      # Flask server
├── templates/
│   └── index.html     # Web visualization UI
├── main.py            # Main orchestration script
└── CLAUDE.md          # This file
```

## How to Run

First, activate the virtual environment:
```bash
source ~/venvs/basic/bin/activate
```

Then run the main script:
```bash
cd backprop
python main.py
```

Then visit http://127.0.0.1:7000 in your browser to see the visualization.

## Next Steps

1. **Implement Trainer methods** - Complete the training loop, loss functions, and evaluation
2. **Training integration** - Connect trainer with dataset and model for end-to-end training
3. **Test on actual learning** - Train the network on the sequence prediction task and observe convergence
4. **Add more visualizations** - Loss curves, accuracy metrics, prediction outputs

## Technical Details

**Model Architecture:**
- Input layer: Xavier initialization, GELU activation
- Hidden layers: Xavier initialization, GELU activation, configurable count
- Output layer: Xavier initialization, no activation (raw logits)

**Visualization:**
- Global min/max normalization ensures consistent color scale
- Matrices are compact (max 10px cells) and centered
- Biases displayed as horizontal rows below weight matrices
- Separate color scales for forward (blue→green) and backward (yellow→red) passes

**Dataset:**
- Task: Next-token prediction from sequence patterns
- 100 training samples extracted from 3 randomly generated master sequences
- Master sequence lengths: Random between 8 and 12 tokens
- Training samples are 5-token subsequences randomly extracted from master sequences
- One-hot encoding: 4 tokens × 4 positions = 16-dimensional input
- Output: 4-dimensional one-hot vector for target token
- This creates **learnable patterns** instead of random noise
- Example master sequences: `'dccccbca'`, `'bbdaacddab'`, `'dabbabaadabc'`
