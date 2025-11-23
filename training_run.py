import time
import numpy as np
from copy import deepcopy


class TrainingRun:
    """
    Encapsulates the state and history of a single training run.
    Stores configuration, weights snapshots, and training metrics.
    """

    def __init__(self, run_id, name=None, config=None, initial_weights=None):
        """
        Initialize a new training run.

        Args:
            run_id: Unique identifier for this run
            name: Human-readable name for the run (optional)
            config: Dictionary containing training configuration
            initial_weights: Snapshot of model weights at run start
        """
        self.run_id = run_id
        self.name = name or f"Run {run_id}"
        self.created_at = time.time()

        # Training configuration
        self.config = config or {
            'learning_rate': 0.001,
            'batch_size': 32,
            'loss_type': 'mse',
            'hidden_layers': 2,
            'hidden_dimension': 32
        }

        # Weight snapshots
        self.initial_weights = initial_weights
        self.current_weights = deepcopy(initial_weights) if initial_weights else None

        # Training history
        self.losses = []
        self.epoch_times = []
        self.current_epoch = 0
        self.total_samples_trained = 0

        # Gradient norm tracking (per layer)
        self.gradient_norms = {
            'Input Layer': [],
            'Output Layer': []
        }
        # Hidden layers will be added dynamically based on model architecture

        # Run state
        self.is_active = False
        self.is_training = False

    def snapshot_weights(self, model):
        """
        Take a snapshot of the model's current weights.

        Args:
            model: The Model instance to snapshot
        """
        weights = {
            'input_weights': model.input_layer_weights.copy(),
            'input_biases': model.input_layer_biases.copy(),
            'hidden_weights': model.hidden_layers_weights.copy(),
            'hidden_biases': model.hidden_layer_biases.copy(),
            'output_weights': model.output_layer_weights.copy(),
            'output_biases': model.output_layer_biases.copy()
        }
        return weights

    def save_initial_weights(self, model):
        """Save the model's current weights as the initial state for this run."""
        self.initial_weights = self.snapshot_weights(model)
        self.current_weights = deepcopy(self.initial_weights)

    def update_current_weights(self, model):
        """Update the current weights snapshot from the model."""
        self.current_weights = self.snapshot_weights(model)

    def load_weights_into_model(self, model, use_initial=False):
        """
        Load this run's weights into the model.

        Args:
            model: The Model instance to load weights into
            use_initial: If True, load initial weights; otherwise load current weights
        """
        weights = self.initial_weights if use_initial else self.current_weights

        if weights is None:
            raise ValueError("No weights available to load")

        model.input_layer_weights = weights['input_weights'].copy()
        model.input_layer_biases = weights['input_biases'].copy()
        model.hidden_layers_weights = weights['hidden_weights'].copy()
        model.hidden_layer_biases = weights['hidden_biases'].copy()
        model.output_layer_weights = weights['output_weights'].copy()
        model.output_layer_biases = weights['output_biases'].copy()

    def record_epoch(self, loss, elapsed_time, samples_in_epoch, gradient_norms=None):
        """
        Record the completion of an epoch.

        Args:
            loss: Average loss for the epoch
            elapsed_time: Time taken for the epoch in seconds
            samples_in_epoch: Number of samples trained in the epoch
            gradient_norms: Optional dict of gradient norms per layer
        """
        self.losses.append(float(loss))
        self.epoch_times.append(elapsed_time)
        self.current_epoch += 1
        self.total_samples_trained += samples_in_epoch

        if gradient_norms:
            self._record_gradient_norms(gradient_norms)

    def record_batch(self, loss, batch_size, gradient_norms=None):
        """
        Record the completion of a single batch.

        Args:
            loss: Loss for the batch
            batch_size: Number of samples in the batch
            gradient_norms: Optional dict of gradient norms per layer
        """
        self.losses.append(float(loss))
        self.total_samples_trained += batch_size

        if gradient_norms:
            self._record_gradient_norms(gradient_norms)

    def record_single_step(self, loss, gradient_norms=None):
        """
        Record the completion of a single training step.

        Args:
            loss: Loss for the single sample
            gradient_norms: Optional dict of gradient norms per layer
        """
        self.losses.append(float(loss))
        self.total_samples_trained += 1

        if gradient_norms:
            self._record_gradient_norms(gradient_norms)

    def _record_gradient_norms(self, gradient_norms):
        """
        Record gradient norms for each layer.

        Args:
            gradient_norms: Dict mapping layer names to L2 norms
        """
        for layer_name, norm in gradient_norms.items():
            # Initialize layer if not seen before (for hidden layers)
            if layer_name not in self.gradient_norms:
                self.gradient_norms[layer_name] = []
            self.gradient_norms[layer_name].append(float(norm))

    def reset_history(self):
        """Reset training history while keeping configuration and initial weights."""
        self.losses = []
        self.epoch_times = []
        self.current_epoch = 0
        self.total_samples_trained = 0

        # Reset gradient norms
        self.gradient_norms = {
            'Input Layer': [],
            'Output Layer': []
        }

        # Reset current weights to initial state
        if self.initial_weights:
            self.current_weights = deepcopy(self.initial_weights)

    def get_stats(self):
        """
        Get current training statistics.

        Returns:
            Dictionary containing training statistics
        """
        return {
            'run_id': self.run_id,
            'name': self.name,
            'current_epoch': self.current_epoch,
            'losses': self.losses,
            'epoch_times': self.epoch_times,
            'total_samples_trained': self.total_samples_trained,
            'is_active': self.is_active,
            'is_training': self.is_training,
            'config': self.config,
            'created_at': self.created_at
        }

    def get_summary(self):
        """
        Get a brief summary of the run for display.

        Returns:
            Dictionary containing run summary
        """
        current_loss = self.losses[-1] if self.losses else None
        best_loss = min(self.losses) if self.losses else None
        avg_epoch_time = (
            sum(self.epoch_times) / len(self.epoch_times)
            if self.epoch_times else None
        )

        return {
            'run_id': self.run_id,
            'name': self.name,
            'current_epoch': self.current_epoch,
            'current_loss': current_loss,
            'best_loss': best_loss,
            'total_samples_trained': self.total_samples_trained,
            'avg_epoch_time': avg_epoch_time,
            'config': self.config,
            'is_active': self.is_active
        }

    def to_dict(self):
        """
        Serialize the entire run to a dictionary.

        Returns:
            Dictionary containing all run data
        """
        def convert_numpy(obj):
            """Recursively convert numpy arrays to lists."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        return {
            'run_id': self.run_id,
            'name': self.name,
            'created_at': self.created_at,
            'config': self.config,
            'initial_weights': convert_numpy(self.initial_weights),
            'current_weights': convert_numpy(self.current_weights),
            'losses': self.losses,
            'gradient_norms': self.gradient_norms,
            'epoch_times': self.epoch_times,
            'current_epoch': self.current_epoch,
            'total_samples_trained': self.total_samples_trained,
            'is_active': self.is_active,
            'is_training': self.is_training
        }

    @classmethod
    def from_dict(cls, data):
        """
        Create a TrainingRun from a dictionary.

        Args:
            data: Dictionary containing run data

        Returns:
            TrainingRun instance
        """
        def convert_to_numpy(obj):
            """Recursively convert lists back to numpy arrays where appropriate."""
            if isinstance(obj, dict):
                # Check if this looks like a weights dict
                if any(key.endswith('weights') or key.endswith('biases') for key in obj.keys()):
                    return {k: np.array(v) if isinstance(v, list) else convert_to_numpy(v)
                            for k, v in obj.items()}
                else:
                    return {k: convert_to_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return obj  # Keep lists as lists (for losses, etc.)
            else:
                return obj

        run = cls(
            run_id=data['run_id'],
            name=data.get('name'),
            config=data.get('config'),
            initial_weights=convert_to_numpy(data.get('initial_weights'))
        )

        run.created_at = data.get('created_at', time.time())
        run.current_weights = convert_to_numpy(data.get('current_weights'))
        run.losses = data.get('losses', [])
        run.gradient_norms = data.get('gradient_norms', {'Input Layer': [], 'Output Layer': []})
        run.epoch_times = data.get('epoch_times', [])
        run.current_epoch = data.get('current_epoch', 0)
        run.total_samples_trained = data.get('total_samples_trained', 0)
        run.is_active = data.get('is_active', False)
        run.is_training = data.get('is_training', False)

        return run


class TrainingRunManager:
    """
    Manages multiple training runs and tracks the active run.
    """

    def __init__(self):
        """Initialize the run manager."""
        self.runs = {}  # run_id -> TrainingRun
        self.active_run_id = None
        self.next_run_id = 1

    def create_run(self, model, trainer, name=None):
        """
        Create a new training run.

        Args:
            model: The Model instance (for weight snapshot)
            trainer: The Trainer instance (for config)
            name: Optional name for the run

        Returns:
            The created TrainingRun
        """
        run_id = f"run_{self.next_run_id:03d}"
        self.next_run_id += 1

        # Create config from trainer
        config = {
            'learning_rate': trainer.learning_rate,
            'batch_size': trainer.batch_size,
            'loss_type': trainer.loss_type,
            'hidden_layers': model.num_hidden_layers,
            'hidden_dimension': model.hidden_dimension
        }

        # Create the run
        run = TrainingRun(run_id, name=name, config=config)
        run.save_initial_weights(model)

        self.runs[run_id] = run

        return run

    def get_run(self, run_id):
        """Get a run by ID."""
        return self.runs.get(run_id)

    def get_active_run(self):
        """Get the currently active run."""
        if self.active_run_id:
            return self.runs.get(self.active_run_id)
        return None

    def set_active_run(self, run_id, model, trainer):
        """
        Set a run as active and load its state.

        Args:
            run_id: ID of the run to activate
            model: Model instance to load weights into
            trainer: Trainer instance to configure
        """
        if run_id not in self.runs:
            raise ValueError(f"Run {run_id} not found")

        # Deactivate current run
        if self.active_run_id and self.active_run_id in self.runs:
            self.runs[self.active_run_id].is_active = False

        # Activate new run
        run = self.runs[run_id]
        run.is_active = True
        self.active_run_id = run_id

        # Load weights into model
        run.load_weights_into_model(model, use_initial=False)

        # Configure trainer to match run config
        trainer.learning_rate = run.config['learning_rate']
        trainer.batch_size = run.config['batch_size']
        trainer.loss_type = run.config['loss_type']

        return run

    def delete_run(self, run_id):
        """Delete a run."""
        if run_id in self.runs:
            # Can't delete active run
            if run_id == self.active_run_id:
                self.active_run_id = None
            del self.runs[run_id]
            return True
        return False

    def list_runs(self):
        """Get summaries of all runs."""
        return [run.get_summary() for run in self.runs.values()]

    def get_all_losses(self):
        """Get losses from all runs for multi-series plotting."""
        return {
            run_id: {
                'name': run.name,
                'losses': run.losses,
                'config': run.config
            }
            for run_id, run in self.runs.items()
        }

    def get_all_gradient_norms(self):
        """Get gradient norms from all runs for multi-series plotting."""
        return {
            run_id: {
                'name': run.name,
                'gradient_norms': run.gradient_norms,
                'config': run.config
            }
            for run_id, run in self.runs.items()
        }
