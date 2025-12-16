"""
LMDB Loader for Experiment 2
"""

import lmdb
import pickle
import msgpack
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import logging
import sys

sys.path.append(str(Path(__file__).parent.parent))
from loaders.base_loader import BaseLoader


class LMDBLoader(BaseLoader):
    """Loader for LMDB format"""

    def __init__(self, data_dir: Path, target_col: str = 'target', compression: str = 'pickle',
                 batch_size: int = 1024, shuffle: bool = True, random_state: int = 42,
                 logger: Optional[logging.Logger] = None):
        super().__init__(data_dir, batch_size, shuffle, random_state, logger)
        self.target_col = target_col
        self.compression = compression
        self.train_path = self.data_dir / 'train.lmdb'
        self.val_path = self.data_dir / 'val.lmdb'

        if not self.train_path.exists():
            raise FileNotFoundError(f"Train LMDB not found: {self.train_path}")
        if not self.val_path.exists():
            raise FileNotFoundError(f"Val LMDB not found: {self.val_path}")

    def load_train(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training data"""
        X, y = self._load_lmdb(self.train_path)
        self.logger.info(f"Train data shape: X={X.shape}, y={y.shape}")
        return self._shuffle_data(X, y)

    def load_val(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load validation data"""
        X, y = self._load_lmdb(self.val_path)
        self.logger.info(f"Val data shape: X={X.shape}, y={y.shape}")
        return X, y

    def _load_lmdb(self, lmdb_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from LMDB"""
        env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=True, meminit=False)
        samples = []

        try:
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    if self.compression == 'msgpack':
                        row_dict = msgpack.unpackb(value, raw=False)
                    else:
                        row_dict = pickle.loads(value)
                    samples.append(row_dict)
        finally:
            env.close()

        self.logger.info(f"Loaded {len(samples)} samples from LMDB")

        # Convert to DataFrame first
        df = pd.DataFrame(samples)

        # Extract target
        if self.target_col in df.columns:
            y = df[self.target_col].values
            X_df = df.drop(columns=[self.target_col])
        else:
            y = df.iloc[:, -1].values
            X_df = df.iloc[:, :-1]

        # Data is already clean from builder
        X = X_df.values.astype(np.float32)
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
        env_train = lmdb.open(str(self.train_path), readonly=True, lock=False)
        env_val = lmdb.open(str(self.val_path), readonly=True, lock=False)

        try:
            with env_train.begin() as txn:
                train_samples = txn.stat()['entries']
            with env_val.begin() as txn:
                val_samples = txn.stat()['entries']
        finally:
            env_train.close()
            env_val.close()

        return {
            'format': 'lmdb',
            'compression': self.compression,
            'train_samples': train_samples,
            'val_samples': val_samples,
            'total_samples': train_samples + val_samples,
            'train_path': str(self.train_path),
            'val_path': str(self.val_path)
        }
