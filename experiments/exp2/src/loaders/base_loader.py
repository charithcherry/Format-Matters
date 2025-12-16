"""
Base Loader Interface for Experiment 2
Defines unified interface for all format loaders
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import numpy as np
import logging
from pathlib import Path


class BaseLoader(ABC):
    """
    Abstract base class for data loaders
    All format-specific loaders must implement this interface
    """

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 1024,
        shuffle: bool = True,
        random_state: int = 42,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize loader

        Args:
            data_dir: Directory containing format-specific data files
            batch_size: Batch size for iteration
            shuffle: Whether to shuffle data
            random_state: Random seed for reproducibility
            logger: Logger instance
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.logger = logger or logging.getLogger(__name__)

        self.rng = np.random.RandomState(random_state)

    @abstractmethod
    def load_train(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load training data

        Returns:
            X_train, y_train as numpy arrays
        """
        pass

    @abstractmethod
    def load_val(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load validation data

        Returns:
            X_val, y_val as numpy arrays
        """
        pass

    @abstractmethod
    def get_batch_iterator(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: Optional[int] = None
    ):
        """
        Create batch iterator for data

        Args:
            X: Feature array
            y: Target array
            batch_size: Batch size (uses self.batch_size if None)

        Yields:
            (X_batch, y_batch) tuples
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the loaded data

        Returns:
            Dictionary with metadata (num_samples, num_features, etc.)
        """
        pass

    def _shuffle_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Shuffle data arrays

        Args:
            X: Feature array
            y: Target array

        Returns:
            Shuffled X, y
        """
        if self.shuffle:
            indices = self.rng.permutation(len(X))
            return X[indices], y[indices]
        return X, y

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"data_dir={self.data_dir}, "
                f"batch_size={self.batch_size}, "
                f"shuffle={self.shuffle})")
