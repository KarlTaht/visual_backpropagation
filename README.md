# Backpropagation from Scratch

A neural network implementation with manual backpropagation and real-time visualization, built for learning purposes. All gradient computation uses raw NumPy.

## Quick Start

```bash
# Activate virtual environment
source ~/venvs/basic/bin/activate

# Run the visualizer
python main.py

# Visit http://127.0.0.1:7000
```

## Features

- **Manual Backpropagation** - Forward/backward passes implemented from scratch in NumPy
- **Real-time Visualization** - Heatmaps of weights, activations, and gradients
- **Training Run Management** - Create, compare, and switch between experiments
- **Dynamic Hyperparameters** - Adjust learning rate, batch size, and loss function on-the-fly
- **Multi-run Comparison** - Plotly charts comparing loss curves across runs

## Architecture

**Data Flow:**
```
main.py orchestrates:
  SequenceDataset → Model → Trainer → TrainingRunManager → Flask Visualizer
```

**Components:**

- **Model** (`model.py`) - Neural network with manual forward/backward passes, GELU activation, Xavier initialization. Stores weights, activations, and gradients for visualization.

- **Dataset** (`dataset.py`) - Generates sequence data using "master sequences" - fixed patterns from which training samples are extracted as subsequences, creating learnable patterns.

- **Trainer** (`trainer.py`) - Training loop with MSE and cross-entropy loss. Uses PyTorch DataLoader for batching (gradient computation is manual NumPy).

- **Training Run Manager** (`training_run.py`) - Manages multiple training experiments with weight snapshots, configuration tracking, and loss history.

- **Visualizer** (`visualizer.py`) - Flask server on port 7000 with real-time heatmaps, training controls, and multi-run comparison charts.

## Web UI Features

- **Visualization Modes**: Toggle between "Visualize Only" (no weight updates) and "Train" (update weights)
- **Training Controls**: Run Single Step, Run Batch, Run Full Epoch
- **Hyperparameter Controls**: Adjust learning rate, batch size, loss function (MSE/Cross-Entropy)
- **Run Management**: Create new runs, switch between runs, reset to initial state, rename, delete
- **Loss Charts**: Interactive Plotly charts comparing loss curves across different training runs

## Configuration

Edit `main.py` to modify dataset, model, and training parameters:

```python
# Dataset
'vocabulary': ['a', 'b', 'c', 'd']
'sequence_length': 5
'task_type': 'next_token'  # or 'sequence_classification'
'num_master_sequences': 3
'master_sequence_length': (8, 12)

# Model
'hidden_dimension': 32
'hidden_layers': 2

# Training
'learning_rate': 0.001
'batch_size': 32
'loss_type': 'mse'  # or 'cross_entropy'
```

## Master Sequence Concept

Training samples are subsequences of fixed "master sequences":
```
Master: 'abccaabb'
Sample: 'caab' → 'b' (extracted from master)
```

This creates learnable patterns instead of random noise.

## Testing

```bash
pytest tests/ -v                              # All tests
pytest tests/test_master_sequences.py -v     # Single file
pytest tests/ -k "test_explicit"             # Pattern match
```

## Dependencies

- NumPy (neural network math)
- Flask (visualization server)
- PyTorch (DataLoader only)
- Plotly.js (loss charts, loaded via CDN)
- pytest (testing)
