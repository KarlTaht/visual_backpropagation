import numpy as np
from torch.utils.data import DataLoader

class Trainer:
    """
    Trainer class for training neural network models.
    Handles the training loop, loss calculation, and optimization.
    """

    def __init__(self, model, dataset, config):
        """
        Initialize the trainer.

        Args:
            model: The Model instance to train
            dataset: the Dataset used to train the model
            config: Training configuration dict with keys like:
                - learning_rate: float
                - batch_size: int
                - epochs: int
                - loss_type: str ('mse', 'cross_entropy', etc.)
        """
        self.model = model
        self.dataset = dataset
        self.learning_rate = config.get('learning_rate', 0.01)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 10)
        self.loss_type = config.get('loss_type', 'mse')

        # Create PyTorch DataLoader for elegant batching and shuffling
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Automatically shuffle each epoch
            drop_last=False  # Keep last batch even if smaller
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }


    # === Main Training Loop ===
    def train(self):
        """
        Main training loop.

        Returns:
            Training history dictionary
        """
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)

        for epoch in range(self.epochs):
            # Train for one epoch
            train_loss = self.train_epoch()

            # Record history
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)

            # Print progress
            self.print_progress(epoch, train_loss)

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

        return self.history


    def train_epoch(self):
        """
        Train for one epoch using PyTorch DataLoader.

        Returns:
            tuple: (average_loss, final_gradient_norms) where final_gradient_norms is from the last batch
        """
        epoch_loss = 0.0
        num_batches = 0
        final_gradient_norms = None

        # DataLoader automatically handles shuffling and batching
        for X_batch, y_batch in self.train_loader:
            # Convert from PyTorch tensors to numpy arrays
            X_batch = X_batch.numpy()
            y_batch = y_batch.numpy()

            # Train on this batch
            batch_loss, gradient_norms = self.train_batch(X_batch, y_batch)

            epoch_loss += batch_loss
            num_batches += 1
            final_gradient_norms = gradient_norms  # Keep last batch's gradient norms

        # Return average loss and gradient norms from final batch
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss, final_gradient_norms


    def train_batch(self, X_batch, y_batch):
        """
        Train on a single batch.

        Args:
            X_batch: Batch input data (batch_size, input_dim)
            y_batch: Batch target data (batch_size, output_dim)

        Returns:
            tuple: (loss, gradient_norms) where gradient_norms is a dict of L2 norms per layer
        """
        # Forward pass
        predictions = self.model.forward(X_batch)

        # Compute loss (for logging)
        loss = self.compute_loss(predictions, y_batch)

        # Compute loss gradient
        loss_gradient = self.compute_loss_gradient(predictions, y_batch)

        # Backward pass
        self.model.backward(loss_gradient)

        # Compute gradient norms before updating
        gradient_norms = self.model.compute_gradient_norms()

        # Update parameters
        self.model.update_parameters(self.learning_rate)

        return loss, gradient_norms


    # === Loss Functions ===
    def compute_loss(self, predictions, targets):
        """
        Compute loss between predictions and targets.

        Args:
            predictions: Model predictions (batch_size, output_dim)
            targets: Target values (batch_size, output_dim)

        Returns:
            Scalar loss value
        """
        if self.loss_type == 'mse':
            return self.mse_loss(predictions, targets)
        elif self.loss_type == 'cross_entropy':
            return self.cross_entropy_loss(predictions, targets)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


    def compute_loss_gradient(self, predictions, targets):
        """
        Compute gradient of loss with respect to predictions.

        Args:
            predictions: Model predictions (batch_size, output_dim)
            targets: Target values (batch_size, output_dim)

        Returns:
            Gradient of loss w.r.t. predictions (batch_size, output_dim)
        """
        if self.loss_type == 'mse':
            return self.mse_loss_gradient(predictions, targets)
        elif self.loss_type == 'cross_entropy':
            return self.cross_entropy_loss_gradient(predictions, targets)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


    def mse_loss(self, predictions, targets):
        """
        Mean Squared Error loss.

        Args:
            predictions: Model predictions (batch_size, output_dim)
            targets: Target values (batch_size, output_dim)

        Returns:
            MSE loss value (scalar)
        """
        return np.mean((predictions - targets) ** 2)


    def mse_loss_gradient(self, predictions, targets):
        """
        Gradient of MSE loss.

        Args:
            predictions: Model predictions (batch_size, output_dim)
            targets: Target values (batch_size, output_dim)

        Returns:
            Gradient of MSE loss (batch_size, output_dim)
        """
        # d/dx[(pred - target)^2] = 2(pred - target)
        # Average over batch: divide by batch_size
        batch_size = predictions.shape[0]
        return 2 * (predictions - targets) / batch_size


    def cross_entropy_loss(self, predictions, targets):
        """
        Cross-entropy loss (for classification).

        Args:
            predictions: Model predictions (logits, batch_size, num_classes)
            targets: Target values (one-hot encoded, batch_size, num_classes)

        Returns:
            Cross-entropy loss value (scalar)
        """
        # Apply softmax to get probabilities
        exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

        # Cross-entropy: -sum(target * log(prob))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1 - epsilon)

        # Compute cross-entropy
        ce_loss = -np.sum(targets * np.log(probs)) / predictions.shape[0]
        return ce_loss


    def cross_entropy_loss_gradient(self, predictions, targets):
        """
        Gradient of cross-entropy loss with softmax.

        Args:
            predictions: Model predictions (logits, batch_size, num_classes)
            targets: Target values (one-hot encoded, batch_size, num_classes)

        Returns:
            Gradient of cross-entropy loss (batch_size, num_classes)
        """
        # Apply softmax to get probabilities
        exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

        # Gradient of softmax + cross-entropy: prob - target
        batch_size = predictions.shape[0]
        return (probs - targets) / batch_size


    # === Evaluation ===
    def evaluate(self, X, y):
        """
        Evaluate model on given data.

        Args:
            X: Input data
            y: Target data

        Returns:
            Average loss on the dataset
        """
        pass


    # === Callbacks & Logging ===
    def on_epoch_end(self, epoch, train_loss, val_loss=None):
        """
        Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            train_loss: Training loss for this epoch
            val_loss: Optional validation loss
        """
        pass


    def print_progress(self, epoch, train_loss, val_loss=None):
        """
        Print training progress.

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Optional validation loss
        """
        progress_str = f"Epoch [{epoch + 1}/{self.epochs}] - Train Loss: {train_loss:.6f}"

        if val_loss is not None:
            progress_str += f" - Val Loss: {val_loss:.6f}"

        print(progress_str)
