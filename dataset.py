import numpy as np
from abc import ABC, abstractmethod


class Dataset(ABC):
    """
    Base class for datasets.
    Provides interface for generating training/validation data.
    """

    def __init__(self, config):
        """
        Initialize dataset.

        Args:
            config: Configuration dictionary
        """
        self.config = config


    @abstractmethod
    def generate(self):
        """
        Generate the full dataset.

        Returns:
            Tuple of (X, y) where X is all inputs and y is all targets
        """
        pass


    @abstractmethod
    def get_batch(self, batch_size):
        """
        Get a batch of data from the dataset.

        Args:
            batch_size: Number of samples in the batch

        Returns:
            Tuple of (X_batch, y_batch) sampled from the dataset
        """
        pass


    @abstractmethod
    def get_info(self):
        """
        Get dataset information.

        Returns:
            Dictionary with dataset metadata
        """
        pass


class SequenceDataset(Dataset):
    """
    Dataset for sequence learning tasks.
    Generates sequences from a vocabulary with configurable properties.
    """

    def __init__(self, config):
        """
        Initialize sequence dataset.

        Args:
            config: Configuration dict with keys:
                - vocabulary: list of tokens (e.g., ['a', 'b', 'c'])
                - sequence_length: int, length of input sequences
                - num_samples: int, total number of samples in the dataset
                - encoding_type: str, 'one_hot' or 'embedding'
                - task_type: str, 'next_token' or 'sequence_classification'
                - master_sequences: list of sequences (optional), e.g., [['a','b','c'], ['d','e','f']]
                - num_master_sequences: int (optional), number of master sequences to generate
                - master_sequence_length: int or tuple (optional), length or (min, max) range for master sequences
        """
        super().__init__(config)

        self.vocabulary = config['vocabulary']
        self.vocab_size = len(self.vocabulary)
        self.sequence_length = config['sequence_length']
        self.num_samples = config.get('num_samples', 1000)
        self.encoding_type = config.get('encoding_type', 'one_hot')
        self.task_type = config.get('task_type', 'next_token')

        # Create token-to-index mapping
        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocabulary)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

        # Set up master sequences
        if 'master_sequences' in config:
            # Use provided master sequences
            self.master_sequences = config['master_sequences']
        elif 'num_master_sequences' in config:
            # Generate random master sequences
            num_sequences = config['num_master_sequences']
            length_spec = config.get('master_sequence_length', (8, 16))

            # Support both single length and (min, max) range
            if isinstance(length_spec, tuple):
                min_length, max_length = length_spec
            else:
                min_length = max_length = length_spec

            self.master_sequences = self._generate_master_sequences(
                num_sequences,
                min_length,
                max_length
            )
        else:
            # No master sequences - use fully random generation
            self.master_sequences = None

        # Generate the full dataset
        self.X, self.y = self.generate()


    # === PyTorch DataLoader Compatibility ===
    def __len__(self):
        """
        Return the number of samples in the dataset.
        Required for PyTorch DataLoader compatibility.
        """
        return self.num_samples


    def __getitem__(self, idx):
        """
        Get a single sample by index.
        Required for PyTorch DataLoader compatibility.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (X, y) for the sample at index idx
        """
        return self.X[idx], self.y[idx]


    def _generate_master_sequences(self, num_sequences, min_length, max_length):
        """
        Generate random master sequences from vocabulary.

        Args:
            num_sequences: Number of master sequences to generate
            min_length: Minimum length of each master sequence
            max_length: Maximum length of each master sequence

        Returns:
            List of master sequences, where each sequence is a list of tokens
        """
        sequences = []
        for _ in range(num_sequences):
            # Randomly choose length in range [min_length, max_length]
            length = np.random.randint(min_length, max_length + 1)

            sequence = [
                self.vocabulary[np.random.randint(0, self.vocab_size)]
                for _ in range(length)
            ]
            sequences.append(sequence)
        return sequences


    def _extract_subsequence_from_master(self):
        """
        Extract a random subsequence from a random master sequence.

        Returns:
            List of tokens forming a subsequence of length self.sequence_length
        """
        # Randomly select a master sequence
        master_idx = np.random.randint(0, len(self.master_sequences))
        master_seq = self.master_sequences[master_idx]

        # Check if master sequence is long enough
        if len(master_seq) < self.sequence_length:
            raise ValueError(
                f"Master sequence {master_idx} has length {len(master_seq)}, "
                f"but sequence_length is {self.sequence_length}. "
                f"Master sequences must be at least as long as sequence_length."
            )

        # Randomly select a starting position
        max_start = len(master_seq) - self.sequence_length
        start_idx = np.random.randint(0, max_start + 1)

        # Extract subsequence
        subsequence = master_seq[start_idx : start_idx + self.sequence_length]

        return subsequence


    # === Vocabulary & Encoding ===
    def get_vocabulary(self):
        """
        Get the vocabulary.

        Returns:
            List of tokens in vocabulary
        """
        return self.vocabulary


    def get_vocab_size(self):
        """
        Get vocabulary size.

        Returns:
            Integer vocabulary size
        """
        return self.vocab_size


    def encode_token(self, token):
        """
        Encode a single token.

        Args:
            token: Token to encode

        Returns:
            Encoded representation (one-hot vector or embedding index)
        """
        idx = self.token_to_idx[token]

        if self.encoding_type == 'one_hot':
            encoding = np.zeros(self.vocab_size)
            encoding[idx] = 1
            return encoding
        elif self.encoding_type == 'embedding':
            return idx
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")


    def encode_sequence(self, sequence):
        """
        Encode a sequence of tokens.

        Args:
            sequence: List of tokens

        Returns:
            Encoded sequence (batch_size=1, seq_len * vocab_size for one_hot)
        """
        if self.encoding_type == 'one_hot':
            # Concatenate all one-hot vectors into a single vector
            encoded = np.concatenate([self.encode_token(token) for token in sequence])
            return encoded
        elif self.encoding_type == 'embedding':
            # Return sequence of indices
            return np.array([self.encode_token(token) for token in sequence])
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")


    def decode_token(self, encoding):
        """
        Decode an encoded token back to its original form.

        Args:
            encoding: One-hot vector or index

        Returns:
            Token string
        """
        if self.encoding_type == 'one_hot':
            idx = np.argmax(encoding)
        else:
            idx = int(encoding)

        return self.idx_to_token[idx]


    # === Dataset Generation ===
    def generate(self):
        """
        Generate the full dataset.

        Returns:
            Tuple of (X, y) where:
                - X: All input sequences (num_samples, input_dim)
                - y: All target sequences or tokens (num_samples, output_dim)
        """
        X_all = []
        y_all = []

        for _ in range(self.num_samples):
            # Generate sequence (either from master sequences or random)
            if self.master_sequences is not None:
                # Extract subsequence from a random master sequence
                sequence = self._extract_subsequence_from_master()
            else:
                # Generate fully random sequence
                sequence = [
                    self.vocabulary[np.random.randint(0, self.vocab_size)]
                    for _ in range(self.sequence_length)
                ]

            # Create input/target pairs based on task type
            if self.task_type == 'next_token':
                # Predict next token given sequence
                X = self.encode_sequence(sequence[:-1])  # All but last token
                y = self.encode_token(sequence[-1])      # Last token as target
            elif self.task_type == 'sequence_classification':
                # Classify entire sequence
                X = self.encode_sequence(sequence)
                # For now, use a simple classification: sum of token indices % num_classes
                # This can be customized for specific tasks
                label_idx = sum([self.token_to_idx[token] for token in sequence]) % 2
                y = np.zeros(2)  # Binary classification
                y[label_idx] = 1
            else:
                raise ValueError(f"Unknown task type: {self.task_type}")

            X_all.append(X)
            y_all.append(y)

        return np.array(X_all), np.array(y_all)


    def get_batch(self, batch_size):
        """
        Get a batch of data by randomly sampling from the dataset.

        Args:
            batch_size: Number of samples in the batch

        Returns:
            Tuple of (X_batch, y_batch) where:
                - X_batch: Batch input sequences (batch_size, input_dim)
                - y_batch: Batch target sequences or tokens (batch_size, output_dim)
        """
        # Randomly sample indices
        indices = np.random.randint(0, self.num_samples, size=batch_size)

        return self.X[indices], self.y[indices]


    def inspect_samples(self, num_samples=10):
        """
        Print human-readable samples from the dataset.

        Args:
            num_samples: Number of samples to display (default: 10)
        """
        print(f"\n    Sample training examples (first {num_samples}):")
        print("    " + "-" * 50)

        for i in range(min(num_samples, self.num_samples)):
            # Get encoded samples
            X_sample = self.X[i]
            y_sample = self.y[i]

            # Decode based on encoding type
            if self.encoding_type == 'one_hot':
                # For one-hot encoding, decode each chunk
                input_tokens = []

                # Input is (sequence_length - 1) tokens concatenated for next_token task
                if self.task_type == 'next_token':
                    for j in range(self.sequence_length - 1):
                        token_encoding = X_sample[j * self.vocab_size : (j + 1) * self.vocab_size]
                        token = self.decode_token(token_encoding)
                        input_tokens.append(token)
                else:  # sequence_classification
                    for j in range(self.sequence_length):
                        token_encoding = X_sample[j * self.vocab_size : (j + 1) * self.vocab_size]
                        token = self.decode_token(token_encoding)
                        input_tokens.append(token)

                input_seq = ''.join(input_tokens)

                # Decode target
                if self.task_type == 'next_token':
                    target_token = self.decode_token(y_sample)
                    print(f"    [{i:2d}] Input: '{input_seq}' -> Target: '{target_token}'")
                else:  # sequence_classification
                    label = np.argmax(y_sample)
                    print(f"    [{i:2d}] Input: '{input_seq}' -> Label: {label}")

            elif self.encoding_type == 'embedding':
                # For embedding encoding, indices are directly stored
                if self.task_type == 'next_token':
                    input_tokens = [self.idx_to_token[idx] for idx in X_sample]
                    target_token = self.idx_to_token[int(y_sample)]
                    input_seq = ''.join(input_tokens)
                    print(f"    [{i:2d}] Input: '{input_seq}' -> Target: '{target_token}'")
                else:  # sequence_classification
                    input_tokens = [self.idx_to_token[idx] for idx in X_sample]
                    label = np.argmax(y_sample)
                    input_seq = ''.join(input_tokens)
                    print(f"    [{i:2d}] Input: '{input_seq}' -> Label: {label}")


    # === Dataset Info ===
    def get_info(self):
        """
        Get dataset information.

        Returns:
            Dictionary with dataset metadata
        """
        info = {
            'dataset_type': 'SequenceDataset',
            'vocabulary': self.vocabulary,
            'vocab_size': self.vocab_size,
            'sequence_length': self.sequence_length,
            'num_samples': self.num_samples,
            'encoding_type': self.encoding_type,
            'task_type': self.task_type,
            'input_dim': self.get_input_dim(),
            'output_dim': self.get_output_dim(),
            'dataset_shape': {
                'X': self.X.shape,
                'y': self.y.shape
            }
        }

        # Add master sequence information if available
        if self.master_sequences is not None:
            info['master_sequences'] = {
                'num_sequences': len(self.master_sequences),
                'sequence_lengths': [len(seq) for seq in self.master_sequences],
                'sequences': self.master_sequences
            }
        else:
            info['master_sequences'] = None

        return info


    def get_input_dim(self):
        """Calculate input dimension based on encoding and task."""
        if self.encoding_type == 'one_hot':
            if self.task_type == 'next_token':
                return self.vocab_size * (self.sequence_length - 1)
            else:
                return self.vocab_size * self.sequence_length
        else:
            if self.task_type == 'next_token':
                return self.sequence_length - 1
            else:
                return self.sequence_length


    def get_output_dim(self):
        """Calculate output dimension based on task."""
        if self.task_type == 'next_token':
            if self.encoding_type == 'one_hot':
                return self.vocab_size
            else:
                return 1  # Single index
        elif self.task_type == 'sequence_classification':
            return 2  # Binary classification for now
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
