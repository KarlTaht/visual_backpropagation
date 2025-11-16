from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Global model, dataset, and trainer references (will be set when server is initialized)
model = None
dataset = None
trainer = None
current_example = None  # Stores the latest training example with predictions
training_stats = {
    'current_epoch': 0,
    'total_epochs': 0,
    'losses': [],
    'is_training': False,
    'samples_per_epoch': 0,
    'epoch_times': [],  # Time taken for each epoch in seconds
    'total_samples_trained': 0
}

@app.route('/')
def index():
    """Main page for the visualizer."""
    return render_template('index.html', title='Toy Network Visualizer')

@app.route('/api/state')
def get_state():
    """API endpoint to get current model state."""
    if model is None:
        return jsonify({'error': 'Model not initialized'}), 400

    state = model.get_state()
    # Convert numpy arrays to lists for JSON serialization
    serializable_state = _make_serializable(state)
    return jsonify(serializable_state)

@app.route('/api/dataset')
def get_dataset():
    """API endpoint to get dataset information."""
    if dataset is None:
        return jsonify({'error': 'Dataset not initialized'}), 400

    dataset_info = dataset.get_info()
    # Convert to serializable format
    serializable_info = _make_serializable(dataset_info)
    return jsonify(serializable_info)

@app.route('/api/current_example')
def get_current_example():
    """API endpoint to get the current training example with predictions."""
    if current_example is None:
        return jsonify({'error': 'No example available yet. Run a forward pass first.'}), 400

    # Convert to serializable format
    serializable_example = _make_serializable(current_example)
    return jsonify(serializable_example)

@app.route('/api/run_single', methods=['POST'])
def run_single():
    """Run a single training step using the trainer."""
    global current_example, training_stats

    if trainer is None:
        return jsonify({'error': 'Trainer not initialized'}), 400

    import numpy as np

    try:
        # Get a single batch from dataset
        X, y = dataset.get_batch(batch_size=1)

        # Forward pass
        output = model.forward(X)

        # Use trainer to compute loss and gradients (using trainer's loss function)
        loss = trainer.compute_loss(output, y)
        loss_gradient = trainer.compute_loss_gradient(output, y)

        # Backward pass
        model.backward(loss_gradient)

        # Update parameters
        model.update_parameters(trainer.learning_rate)

        # Store current example information
        current_example = {
            'input': X,
            'target': y,
            'prediction': output,
            'loss': float(loss),
            'has_dataset': dataset is not None,
            'loss_type': trainer.loss_type
        }

        # Add to training stats
        training_stats['losses'].append(float(loss))

        return jsonify({
            'success': True,
            'message': f'Single training step completed (loss: {trainer.loss_type})',
            'loss': float(loss),
            'loss_type': trainer.loss_type,
            'output_shape': list(output.shape)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _make_serializable(obj):
    """Recursively convert numpy arrays to lists for JSON serialization."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    else:
        return obj

@app.route('/api/training_stats')
def get_training_stats():
    """API endpoint to get current training statistics."""
    global training_stats
    return jsonify(training_stats)


@app.route('/api/train_epoch', methods=['POST'])
def train_epoch():
    """Run one full epoch of training."""
    global training_stats

    if trainer is None:
        return jsonify({'error': 'Trainer not initialized'}), 400

    import time

    try:
        training_stats['is_training'] = True

        # Track timing
        start_time = time.time()

        # Train for one epoch
        epoch_loss = trainer.train_epoch()

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Update stats
        training_stats['current_epoch'] += 1
        training_stats['losses'].append(float(epoch_loss))
        training_stats['epoch_times'].append(elapsed_time)

        # Calculate samples trained in this epoch
        samples_in_epoch = len(dataset)
        training_stats['total_samples_trained'] += samples_in_epoch
        training_stats['samples_per_epoch'] = samples_in_epoch

        training_stats['is_training'] = False

        return jsonify({
            'success': True,
            'message': f'Epoch {training_stats["current_epoch"]} completed',
            'epoch': training_stats['current_epoch'],
            'loss': float(epoch_loss),
            'elapsed_time': elapsed_time,
            'samples': samples_in_epoch
        })
    except Exception as e:
        training_stats['is_training'] = False
        return jsonify({'error': str(e)}), 500


@app.route('/api/reset_training', methods=['POST'])
def reset_training():
    """Reset training statistics."""
    global training_stats

    training_stats['current_epoch'] = 0
    training_stats['losses'] = []
    training_stats['is_training'] = False
    training_stats['epoch_times'] = []
    training_stats['total_samples_trained'] = 0

    return jsonify({
        'success': True,
        'message': 'Training statistics reset'
    })


@app.route('/api/reset_model', methods=['POST'])
def reset_model():
    """Reset model weights and training statistics."""
    global training_stats

    if model is None:
        return jsonify({'error': 'Model not initialized'}), 400

    try:
        # Reinitialize model weights
        model.initialize_parameters()

        # Reset training statistics
        training_stats['current_epoch'] = 0
        training_stats['losses'] = []
        training_stats['is_training'] = False
        training_stats['epoch_times'] = []
        training_stats['total_samples_trained'] = 0

        return jsonify({
            'success': True,
            'message': 'Model and training statistics reset'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train_n_epochs', methods=['POST'])
def train_n_epochs():
    """Run N epochs of training."""
    global training_stats
    from flask import request
    import time

    if trainer is None:
        return jsonify({'error': 'Trainer not initialized'}), 400

    data = request.get_json()
    n_epochs = data.get('n_epochs', 1)

    try:
        training_stats['is_training'] = True

        results = []
        for i in range(n_epochs):
            start_time = time.time()
            epoch_loss = trainer.train_epoch()
            elapsed_time = time.time() - start_time

            training_stats['current_epoch'] += 1
            training_stats['losses'].append(float(epoch_loss))
            training_stats['epoch_times'].append(elapsed_time)

            samples_in_epoch = len(dataset)
            training_stats['total_samples_trained'] += samples_in_epoch

            results.append({
                'epoch': training_stats['current_epoch'],
                'loss': float(epoch_loss),
                'elapsed_time': elapsed_time
            })

        training_stats['is_training'] = False

        return jsonify({
            'success': True,
            'message': f'{n_epochs} epoch(s) completed',
            'results': results
        })
    except Exception as e:
        training_stats['is_training'] = False
        return jsonify({'error': str(e)}), 500


@app.route('/api/trainer_config')
def get_trainer_config():
    """Get current trainer configuration."""
    if trainer is None:
        return jsonify({'error': 'Trainer not initialized'}), 400

    return jsonify({
        'learning_rate': trainer.learning_rate,
        'batch_size': trainer.batch_size,
        'epochs': trainer.epochs,
        'loss_type': trainer.loss_type
    })


@app.route('/api/update_learning_rate', methods=['POST'])
def update_learning_rate():
    """Update the trainer's learning rate."""
    from flask import request

    if trainer is None:
        return jsonify({'error': 'Trainer not initialized'}), 400

    data = request.get_json()
    new_lr = data.get('learning_rate')

    if new_lr is None or new_lr <= 0:
        return jsonify({'error': 'Invalid learning rate'}), 400

    trainer.learning_rate = float(new_lr)

    return jsonify({
        'success': True,
        'message': f'Learning rate updated to {new_lr}',
        'learning_rate': trainer.learning_rate
    })


@app.route('/api/test_master_sequences')
def test_master_sequences():
    """Test model on master sequences."""
    if model is None or dataset is None:
        return jsonify({'error': 'Model or dataset not initialized'}), 400

    import numpy as np

    # Get master sequences from dataset
    dataset_info = dataset.get_info()
    master_seq_info = dataset_info.get('master_sequences')

    if master_seq_info is None:
        return jsonify({'error': 'No master sequences available'}), 400

    master_sequences = master_seq_info['sequences']
    sequence_length = dataset_info['sequence_length']

    results = []
    total_correct = 0
    total_predictions = 0

    for seq_idx, master_seq in enumerate(master_sequences):
        # Generate all possible windows from this master sequence
        seq_results = []

        for i in range(len(master_seq) - sequence_length):
            # Extract input subsequence and target
            input_tokens = master_seq[i:i+sequence_length-1]
            target_token = master_seq[i+sequence_length-1]

            # Encode input
            input_encoded = dataset.encode_sequence(input_tokens)
            input_batch = np.array([input_encoded])

            # Run forward pass
            output = model.forward(input_batch)

            # Get predicted token
            predicted_idx = np.argmax(output[0])
            predicted_token = dataset.vocabulary[predicted_idx]

            # Get probability distribution
            probs = output[0].tolist()

            is_correct = (predicted_token == target_token)
            if is_correct:
                total_correct += 1
            total_predictions += 1

            seq_results.append({
                'input': ''.join(input_tokens),
                'target': target_token,
                'predicted': predicted_token,
                'correct': is_correct,
                'probabilities': probs,
                'confidence': float(np.max(output[0]))
            })

        results.append({
            'master_sequence': ''.join(master_seq),
            'predictions': seq_results
        })

    accuracy = total_correct / total_predictions if total_predictions > 0 else 0

    return jsonify({
        'success': True,
        'accuracy': accuracy,
        'total_correct': total_correct,
        'total_predictions': total_predictions,
        'master_sequences': results
    })


def initialize_server(model_instance, trainer_instance, dataset_instance=None, host='127.0.0.1', port=7000, debug=True):
    """
    Initialize and run the Flask visualization server.

    Args:
        model_instance: The Model object to visualize
        trainer_instance: The Trainer object
        dataset_instance: The Dataset object (optional)
        host: Host address for the server
        port: Port number for the server
        debug: Enable debug mode
    """
    global model, dataset, trainer, training_stats
    model = model_instance
    dataset = dataset_instance
    trainer = trainer_instance

    # Initialize training stats with trainer config
    training_stats['total_epochs'] = trainer.epochs
    training_stats['samples_per_epoch'] = len(dataset) if dataset else 0

    print(f"\n🚀 Starting Toy Network Visualizer at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)

