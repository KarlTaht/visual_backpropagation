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
- Training controls: Run Single Step, Run Batch, Run Full Epoch
- Visualization mode toggle: "Visualize Only" (no weight updates) or "Train" (update weights)
- Dynamic hyperparameter controls:
  - Learning rate adjustment
  - Batch size configuration
  - Loss function selection (MSE or Cross-Entropy)
- Multi-run loss chart with Plotly (compare training runs side-by-side)
- Epoch times displayed in milliseconds

**API Endpoints:**
- `/api/state` - Get current model state (weights, activations, gradients)
- `/api/run_single` - Run single forward/backward pass (with optional weight update)
- `/api/train_batch` - Run a full batch of training
- `/api/trainer_config` - Get/update trainer configuration
- `/api/update_batch_size` - Update batch size
- `/api/update_loss_type` - Switch between MSE and Cross-Entropy
- `/api/runs` - List all training runs
- `/api/runs/create` - Create new training run
- `/api/runs/{id}/activate` - Switch to a different run
- `/api/runs/{id}/reset` - Reset run to initial state
- `/api/runs/{id}/rename` - Rename a run
- `/api/runs/{id}` (DELETE) - Delete a run
- `/api/runs/losses` - Get loss history for all runs (for multi-run chart)

#### 3. Training Run Management ([training_run.py](training_run.py))
- `TrainingRun` class encapsulates:
  - Run metadata (ID, name, creation time)
  - Training configuration (learning rate, batch size, loss type)
  - Weight snapshots (initial and current state)
  - Training history (losses, epoch times, sample counts)
  - Serialization to/from dictionary for persistence
- `TrainingRunManager` class manages:
  - Multiple concurrent training runs
  - Active run switching with automatic weight loading
  - Run lifecycle (create, activate, reset, delete, rename)
  - Cross-run loss comparison for visualization
- Enables:
  - Experimenting with different hyperparameters
  - Comparing loss curves across configurations
  - Resetting runs to initial state
  - Preserving training history

#### 4. Dataset ([dataset.py](dataset.py))
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

#### 5. Gradient Visualizer ([gradient_visualizer.py](gradient_visualizer.py))
- Main entry point for the gradient visualization tool
- Orchestrates dataset, model, trainer, and visualizer
- Automatically matches model dimensions to dataset
- Test forward/backward pass before starting server
- Clean startup output with progress indicators

#### 6. Trainer ([trainer.py](trainer.py))
- Training loop with configurable hyperparameters
- Loss functions: MSE and Cross-Entropy
- Loss gradient computation
- Training statistics tracking
- Uses PyTorch DataLoader for batching (gradient computation remains manual NumPy)

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              gradient_visualizer.py                 │
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
          ├──> trainer.py (Trainer)
          │    - Training loops
          │    - Loss computation
          │
          ├──> training_run.py (TrainingRunManager)
          │    - Manages multiple training runs
          │    - Weight snapshots and history
          │
          └──> visualizer.py (Flask Server)
               - Web UI for visualization
               - Real-time heatmaps
               - Training run management
               - Multi-run comparison charts
```

## File Structure

```
backprop/
├── model.py               # Neural network implementation
├── dataset.py             # Dataset generation and batching
├── trainer.py             # Training loop with loss functions
├── training_run.py        # Training run management system
├── visualizer.py          # Flask server with run management
├── gradient_visualizer.py # Main entry point (gradient visualization)
├── templates/
│   └── index.html         # Web visualization UI
├── static/
│   ├── css/
│   │   └── main.css       # Extracted styles
│   └── js/
│       ├── api.js         # API communication
│       ├── charts.js      # Plotly chart rendering
│       ├── heatmaps.js    # Matrix visualization
│       ├── controls.js    # Training controls
│       ├── runs.js        # Run management
│       ├── distributions.js # Distribution analysis
│       └── main.js        # Entry point
├── tests/                 # Test suite
└── CLAUDE.md              # This file
```

## How to Run

First, activate the virtual environment:
```bash
source ~/venvs/basic/bin/activate
```

Then run the gradient visualizer:
```bash
cd backprop
python gradient_visualizer.py
```

Then visit http://127.0.0.1:7000 in your browser to see the visualization.

## Next Steps

1. **Persist training runs** - Save/load runs to disk for session persistence
2. **Add accuracy metrics** - Track prediction accuracy alongside loss
3. **Gradient analysis** - Add gradient magnitude/flow visualizations
4. **Model checkpointing** - Save best model weights during training
5. **Early stopping** - Implement automatic training termination based on loss plateau

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
