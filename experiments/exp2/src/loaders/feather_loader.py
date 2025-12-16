"""
Feather Loader for Experiment 2
"""

import pandas as pd
import pyarrow.feather as feather
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import logging
import sys

sys.path.append(str(Path(__file__).parent.parent))
from loaders.base_loader import BaseLoader


class FeatherLoader(BaseLoader):
    """Loader for Feather format"""

    def __init__(self, data_dir: Path, target_col: str = 'target', batch_size: int = 1024,
                 shuffle: bool = True, random_state: int = 42, use_columns: Optional[list] = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(data_dir, batch_size, shuffle, random_state, logger)
        self.target_col = target_col
        self.use_columns = use_columns
        self.train_path = self.data_dir / 'train.feather'
        self.val_path = self.data_dir / 'val.feather'

        if not self.train_path.exists():
            raise FileNotFoundError(f"Train file not found: {self.train_path}")
        if not self.val_path.exists():
            raise FileNotFoundError(f"Val file not found: {self.val_path}")

    def load_train(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training data"""
        df = pd.read_feather(self.train_path, columns=self.use_columns)
        self.logger.info(f"Loaded {len(df)} train samples, {len(df.columns)} columns")

        # Extract target
        if self.target_col in df.columns:
            y = df[self.target_col].values
            X_df = df.drop(columns=[self.target_col])
        else:
            y = df.iloc[:, -1].values
            X_df = df.iloc[:, :-1]

        # Data is already clean (numeric only, no NaN) from builder
        X = X_df.values.astype(np.float32)
        self.logger.info(f"Train data shape: X={X.shape}, y={y.shape}")
        return self._shuffle_data(X, y)

    def load_val(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load validation data"""
        df = pd.read_feather(self.val_path, columns=self.use_columns)
        self.logger.info(f"Loaded {len(df)} val samples, {len(df.columns)} columns")

        # Extract target
        if self.target_col in df.columns:
            y = df[self.target_col].values
            X_df = df.drop(columns=[self.target_col])
        else:
            y = df.iloc[:, -1].values
            X_df = df.iloc[:, :-1]

        # Data is already clean (numeric only, no NaN) from builder
        X = X_df.values.astype(np.float32)
        self.logger.info(f"Val data shape: X={X.shape}, y={y.shape}")
        return X, y

    def get_batch_iterator(self, X: np.ndarray, y: np.ndarray, batch_size: Optional[int] = None):
        batch_size = batch_size or self.batch_size
        num_samples = len(X)
        num_batches = (num_samples + batch_size - 1) // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            yield X[start_idx:end_idx], y[start_idx:end_idx]

    def get_metadata(self) -> Dict[str, Any]:
        train_table = feather.read_table(self.train_path)
        val_table = feather.read_table(self.val_path)
        return {
            'format': 'feather',
            'train_samples': train_table.num_rows,
            'val_samples': val_table.num_rows,
            'total_samples': train_table.num_rows + val_table.num_rows,
            'num_features': train_table.num_columns - 1,
            'num_columns': train_table.num_columns,
            'target_col': self.target_col,
            'train_path': str(self.train_path),
            'val_path': str(self.val_path)
        }
