"""
Tabular Data Preprocessing Utilities for Experiment 2
Ensures identical preprocessing across all formats
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path


class TabularPreprocessor:
    """
    Unified preprocessing for tabular data
    Ensures identical transformations across all formats
    """

    def __init__(
        self,
        val_split: float = 0.2,
        random_state: int = 42,
        sample_fraction: Optional[float] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize preprocessor

        Args:
            val_split: Fraction of data for validation
            random_state: Random seed for reproducibility
            sample_fraction: If set, sample this fraction of data (for testing)
            logger: Logger instance
        """
        self.val_split = val_split
        self.random_state = random_state
        self.sample_fraction = sample_fraction
        self.logger = logger or logging.getLogger(__name__)

        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.numerical_cols: List[str] = []
        self.target_col: Optional[str] = None

    def load_and_split_data(
        self,
        data_path: Path,
        target_col: str,
        nrows: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data and split into train/validation

        Args:
            data_path: Path to CSV file
            target_col: Name of target column
            nrows: Number of rows to load (for testing)

        Returns:
            train_df, val_df
        """
        self.logger.info(f"Loading data from {data_path}")
        self.logger.info(f"Reading {'all' if nrows is None else nrows} rows...")

        # Load data
        df = pd.read_csv(data_path, nrows=nrows)
        self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        self.logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Sample if requested
        if self.sample_fraction and self.sample_fraction < 1.0:
            original_size = len(df)
            df = df.sample(frac=self.sample_fraction, random_state=self.random_state)
            self.logger.info(f"Sampled {len(df)} rows ({self.sample_fraction*100:.1f}%) from {original_size}")

        # Check target column
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        self.target_col = target_col

        # Split into train/validation
        train_df, val_df = train_test_split(
            df,
            test_size=self.val_split,
            random_state=self.random_state,
            shuffle=True
        )

        self.logger.info(f"Split: {len(train_df)} train, {len(val_df)} validation")

        return train_df, val_df

    def identify_column_types(self, df: pd.DataFrame, target_col: str):
        """
        Identify categorical and numerical columns

        Args:
            df: DataFrame to analyze
            target_col: Target column name
        """
        self.logger.info("Identifying column types...")

        # All columns except target are features
        self.feature_cols = [col for col in df.columns if col != target_col]

        # Identify categorical columns (object dtype or few unique values)
        self.categorical_cols = []
        self.numerical_cols = []

        for col in self.feature_cols:
            dtype = df[col].dtype

            if dtype == 'object' or dtype.name == 'category':
                self.categorical_cols.append(col)
            elif dtype in ['int64', 'int32', 'float64', 'float32']:
                # If integer with few unique values, treat as categorical
                n_unique = df[col].nunique()
                if dtype in ['int64', 'int32'] and n_unique < 20:
                    self.categorical_cols.append(col)
                else:
                    self.numerical_cols.append(col)

        self.logger.info(f"Identified {len(self.categorical_cols)} categorical, "
                        f"{len(self.numerical_cols)} numerical columns")

        if self.logger.level <= logging.DEBUG:
            self.logger.debug(f"Categorical columns: {self.categorical_cols[:10]}...")
            self.logger.debug(f"Numerical columns: {self.numerical_cols[:10]}...")

    def preprocess_features(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        fit: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features: encode categoricals, scale numericals

        Args:
            train_df: Training dataframe
            val_df: Validation dataframe (optional)
            fit: If True, fit transformers on train data

        Returns:
            X_train, X_val (or just X_train if val_df is None)
        """
        self.logger.info("Preprocessing features...")

        # Identify column types if not done
        if not self.feature_cols:
            self.identify_column_types(train_df, self.target_col)

        # Handle missing values
        self.logger.info("Handling missing values...")
        train_df = train_df.copy()
        if val_df is not None:
            val_df = val_df.copy()

        # Fill numerical missing with median
        for col in self.numerical_cols:
            if train_df[col].isna().any():
                median_val = train_df[col].median()
                train_df[col].fillna(median_val, inplace=True)
                if val_df is not None:
                    val_df[col].fillna(median_val, inplace=True)

        # Fill categorical missing with mode
        for col in self.categorical_cols:
            if train_df[col].isna().any():
                mode_val = train_df[col].mode()[0] if not train_df[col].mode().empty else 'MISSING'
                train_df[col].fillna(mode_val, inplace=True)
                if val_df is not None:
                    val_df[col].fillna(mode_val, inplace=True)

        # Encode categorical features
        if self.categorical_cols:
            self.logger.info(f"Encoding {len(self.categorical_cols)} categorical columns...")

            for col in self.categorical_cols:
                if fit or col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    train_df[col] = self.label_encoders[col].fit_transform(train_df[col].astype(str))
                else:
                    train_df[col] = self._safe_transform(self.label_encoders[col], train_df[col].astype(str))

                if val_df is not None:
                    val_df[col] = self._safe_transform(self.label_encoders[col], val_df[col].astype(str))

        # Extract feature arrays
        X_train = train_df[self.feature_cols].values.astype(np.float32)
        X_val = val_df[self.feature_cols].values.astype(np.float32) if val_df is not None else None

        # Scale numerical features
        if self.numerical_cols:
            self.logger.info("Scaling numerical features...")

            if fit:
                self.scaler = StandardScaler()
                X_train = self.scaler.fit_transform(X_train)
            else:
                if self.scaler is None:
                    raise ValueError("Scaler not fitted. Call with fit=True first.")
                X_train = self.scaler.transform(X_train)

            if X_val is not None:
                X_val = self.scaler.transform(X_val)

        self.logger.info(f"Preprocessed features: {X_train.shape}")

        if X_val is not None:
            return X_train, X_val
        else:
            return X_train, None

    def preprocess_target(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract and preprocess target variable

        Args:
            train_df: Training dataframe
            val_df: Validation dataframe (optional)

        Returns:
            y_train, y_val (or just y_train if val_df is None)
        """
        self.logger.info("Preprocessing target...")

        y_train = train_df[self.target_col].values
        y_val = val_df[self.target_col].values if val_df is not None else None

        self.logger.info(f"Target shape: {y_train.shape}")
        self.logger.info(f"Target unique values: {np.unique(y_train)}")

        return y_train, y_val

    def _safe_transform(self, encoder: LabelEncoder, values: pd.Series) -> np.ndarray:
        """
        Safely transform values, handling unseen categories

        Args:
            encoder: Fitted LabelEncoder
            values: Values to transform

        Returns:
            Transformed values
        """
        # Handle unseen categories by assigning them to a default class
        known_classes = set(encoder.classes_)
        values_array = values.values

        # Replace unseen values with the first known class
        default_class = encoder.classes_[0]
        mask = ~pd.Series(values_array).isin(known_classes)

        if mask.any():
            self.logger.warning(f"Found {mask.sum()} unseen categories, replacing with '{default_class}'")
            values_array = values_array.copy()
            values_array[mask] = default_class

        return encoder.transform(values_array)

    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """Get statistics about preprocessing"""
        return {
            'num_features': len(self.feature_cols),
            'num_categorical': len(self.categorical_cols),
            'num_numerical': len(self.numerical_cols),
            'target_col': self.target_col,
            'scaler_fitted': self.scaler is not None,
            'num_label_encoders': len(self.label_encoders)
        }


if __name__ == '__main__':
    # Test preprocessor with sample data
    print("Testing TabularPreprocessor...")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    df = pd.DataFrame({
        'num_feature_1': np.random.randn(n_samples),
        'num_feature_2': np.random.randn(n_samples),
        'cat_feature_1': np.random.choice(['A', 'B', 'C'], n_samples),
        'cat_feature_2': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })

    # Save sample data
    sample_path = Path('../../exp2_output/sample_data.csv')
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(sample_path, index=False)

    # Test preprocessor
    preprocessor = TabularPreprocessor(
        val_split=0.2,
        random_state=42,
        logger=logger
    )

    train_df, val_df = preprocessor.load_and_split_data(
        data_path=sample_path,
        target_col='target'
    )

    X_train, X_val = preprocessor.preprocess_features(train_df, val_df, fit=True)
    y_train, y_val = preprocessor.preprocess_target(train_df, val_df)

    print(f"\nPreprocessed data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_val: {y_val.shape}")

    stats = preprocessor.get_preprocessing_stats()
    print(f"\nPreprocessing stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nPreprocessor test completed successfully!")
