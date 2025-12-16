"""
ML Training Script for Experiment 2
Trains Random Forest classifier on different formats
"""

import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging
import sys

sys.path.append(str(Path(__file__).parent))

from utils.logger import setup_logger
from utils.metrics import ResourceMonitor, MetricsCollector
from loaders import CSVLoader, LMDBLoader, ParquetLoader, FeatherLoader


def train_and_evaluate(
    format_name: str,
    data_dir: Path,
    output_dir: Path,
    target_col: str = 'target',
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Train and evaluate model on a specific format

    Args:
        format_name: Format name ('csv', 'lmdb', 'parquet', 'feather')
        data_dir: Directory containing format data
        output_dir: Directory for outputs
        target_col: Target column name
        n_estimators: Number of trees
        max_depth: Max tree depth
        random_state: Random seed
        logger: Logger instance

    Returns:
        Dictionary with results
    """
    if logger is None:
        logger = setup_logger(f'train_{format_name}')

    logger.info("=" * 80)
    logger.info(f"Training on {format_name.upper()} format")
    logger.info("=" * 80)

    print(f"\n{'='*60}")
    print(f"Training on {format_name.upper()} format")
    print(f"{'='*60}")

    collector = MetricsCollector(logger=logger)
    monitor = ResourceMonitor(interval=0.5, logger=logger)

    collector.add_metric('format', format_name)
    collector.add_metric('data_dir', str(data_dir))

    try:
        # Initialize loader
        logger.info("Initializing data loader...")
        collector.start_timer('init_loader')

        if format_name == 'csv':
            loader = CSVLoader(data_dir, target_col=target_col, logger=logger)
        elif format_name == 'lmdb':
            loader = LMDBLoader(data_dir, target_col=target_col, logger=logger)
        elif format_name == 'parquet':
            loader = ParquetLoader(data_dir, target_col=target_col, logger=logger)
        elif format_name == 'feather':
            loader = FeatherLoader(data_dir, target_col=target_col, logger=logger)
        else:
            raise ValueError(f"Unknown format: {format_name}")

        collector.stop_timer('init_loader')

        # Get metadata
        metadata = loader.get_metadata()
        collector.add_metrics(metadata)
        logger.info(f"Loaded metadata: {metadata['train_samples']} train, {metadata['val_samples']} val")

        # Load training data
        logger.info("Loading training data...")
        print("[Train] Loading training data...")

        monitor.start()
        collector.start_timer('load_train')
        X_train, y_train = loader.load_train()
        load_train_time = collector.stop_timer('load_train')
        monitor.stop()

        load_resources = monitor.get_summary()
        collector.add_metrics({f'load_train_{k}': v for k, v in load_resources.items()})

        logger.info(f"Loaded train data: {X_train.shape} in {load_train_time:.2f}s")
        print(f"[Train] Loaded {X_train.shape[0]:,} samples, {X_train.shape[1]} features in {load_train_time:.2f}s")

        # Load validation data
        logger.info("Loading validation data...")
        print("[Train] Loading validation data...")

        monitor = ResourceMonitor(interval=0.5, logger=logger)
        monitor.start()
        collector.start_timer('load_val')
        X_val, y_val = loader.load_val()
        load_val_time = collector.stop_timer('load_val')
        monitor.stop()

        load_resources = monitor.get_summary()
        collector.add_metrics({f'load_val_{k}': v for k, v in load_resources.items()})

        logger.info(f"Loaded val data: {X_val.shape} in {load_val_time:.2f}s")
        print(f"[Train] Loaded {X_val.shape[0]:,} samples in {load_val_time:.2f}s")

        # Train model
        logger.info(f"Training Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")
        print(f"[Train] Training Random Forest (trees={n_estimators}, depth={max_depth})...")

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )

        monitor = ResourceMonitor(interval=0.5, logger=logger)
        monitor.start()
        collector.start_timer('train_model')

        model.fit(X_train, y_train)

        train_time = collector.stop_timer('train_model')
        monitor.stop()

        train_resources = monitor.get_summary()
        collector.add_metrics({f'train_{k}': v for k, v in train_resources.items()})

        logger.info(f"Training completed in {train_time:.2f}s")
        print(f"[Train] Training completed in {train_time:.2f}s")

        # Evaluate on training set
        logger.info("Evaluating on training set...")
        collector.start_timer('eval_train')
        y_train_pred = model.predict(X_train)
        eval_train_time = collector.stop_timer('eval_train')

        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')

        logger.info(f"Train accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"[Train] Train accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")

        collector.add_metric('train_accuracy', train_acc)
        collector.add_metric('train_f1', train_f1)

        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        collector.start_timer('eval_val')
        y_val_pred = model.predict(X_val)
        eval_val_time = collector.stop_timer('eval_val')

        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')

        logger.info(f"Val accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"[Train] Val accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")

        collector.add_metric('val_accuracy', val_acc)
        collector.add_metric('val_f1', val_f1)

        # Calculate end-to-end latency
        total_time = load_train_time + load_val_time + train_time
        collector.add_metric('end_to_end_time_s', total_time)

        logger.info(f"End-to-end time: {total_time:.2f}s")
        print(f"[Train] End-to-end time: {total_time:.2f}s")

        # Save results
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = collector.get_all_metrics()
        results_path = output_dir / f'{format_name}_results.json'

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved results to {results_path}")
        print(f"[Train] Saved results to {results_path}")

        logger.info("=" * 80)
        print(f"{'='*60}\n")

        return results

    except Exception as e:
        logger.error(f"Error training on {format_name}: {e}", exc_info=True)
        print(f"[Train] ERROR: {e}")
        raise


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train ML model on format')
    parser.add_argument('--format', type=str, required=True,
                       choices=['csv', 'lmdb', 'parquet', 'feather'])
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--target-col', type=str, default='target')
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    logger = setup_logger(
        f'train_{args.format}',
        log_dir=Path('../exp2_logs'),
        debug_mode=args.debug
    )

    results = train_and_evaluate(
        format_name=args.format,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        target_col=args.target_col,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        logger=logger
    )

    print("\nResults Summary:")
    for key, value in results.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")
