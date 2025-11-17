#!/usr/bin/env python3

from model import Model
from dataset import SequenceDataset
from trainer import Trainer
from visualizer import initialize_server
import numpy as np

def get_dataset_config():
    return {
        'vocabulary': ['a', 'b', 'c', 'd'],
        'sequence_length': 5,
        'num_samples': 100,
        'encoding_type': 'one_hot',
        'task_type': 'next_token',
        # Auto-generate master sequences for pattern-based learning
        'num_master_sequences': 3,
        'master_sequence_length': (8, 12)  # Random length between 8 and 12
    }

def get_model_config(dataset):
    return {
        'input_dimension': dataset.get_input_dim(),
        'hidden_dimension': 32,
        'hidden_layers': 2,
        'output_dimension': dataset.get_output_dim()
    }


def get_trainer_config():
    return {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10,
        'loss_type': 'mse'
    }


def main():
    print("=" * 60)
    print("Starting the Backpropagation Learning Project")
    print("=" * 60)

    # === Dataset Configuration ===
    print("\n[1/4] Setting up dataset...")
    
    # Generate dataset
    dataset = SequenceDataset(get_dataset_config())
    dataset_info = dataset.get_info()
    print(f"  ✓ Dataset generated")
    print(f"    - Type: {dataset_info['task_type']}")
    print(f"    - Vocabulary: {dataset_info['vocabulary']}")
    print(f"    - Sequence length: {dataset_info['sequence_length']}")
    print(f"    - Total samples: {dataset_info['num_samples']}")
    print(f"    - Encoding: {dataset_info['encoding_type']}")
    print(f"    - Dataset shape: X={dataset_info['dataset_shape']['X']}, y={dataset_info['dataset_shape']['y']}")

    # Display master sequence info if available
    if dataset_info['master_sequences'] is not None:
        ms_info = dataset_info['master_sequences']
        print(f"    - Master sequences: {ms_info['num_sequences']}")
        for i, seq in enumerate(ms_info['sequences']):
            seq_str = ''.join(seq)
            print(f"      M{i+1}: '{seq_str}' (length {len(seq)})")

    # === Display Sample Data ===
    dataset.inspect_samples(num_samples=3)

    # === Model Configuration ===
    print("\n[2/4] Initializing model...")
    model_config = get_model_config(dataset)
    model = Model(model_config)

    print(f"  ✓ Model initialized")
    print(f"    - Input dimension: {model_config['input_dimension']}")
    print(f"    - Hidden dimension: {model_config['hidden_dimension']}")
    print(f"    - Hidden layers: {model_config['hidden_layers']}")
    print(f"    - Output dimension: {model_config['output_dimension']}")

    # === Trainer Configuration ===
    print("\n[3/4] Setting up trainer...")
    trainer_config = get_trainer_config()
    trainer = Trainer(model, dataset, trainer_config)

    print(f"  ✓ Trainer configured")
    print(f"    - Learning rate: {trainer_config['learning_rate']}")
    print(f"    - Batch size: {trainer_config['batch_size']}")
    print(f"    - Epochs: {trainer_config['epochs']}")
    print(f"    - Loss type: {trainer_config['loss_type']}")
    print(f"    - Using PyTorch DataLoader with automatic shuffling")


    # === Start Visualization Server ===
    print("\n[4/4] Starting visualization server...")
    initialize_server(model, trainer, dataset, host='127.0.0.1', port=7000, debug=True)


if __name__ == "__main__":
    main()