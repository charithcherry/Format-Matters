"""
Feather Format Builder for Experiment 2
Builds columnar Feather format (Apache Arrow IPC)
"""

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Optional, Dict, Any, Literal
import logging
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from utils.metrics import MetricsCollector


def build_feather(
    input_path: Path,
    output_dir: Path,
    labels_path: Optional[Path] = None,
    nrows: Optional[int] = None,
    compression: Literal['uncompressed', 'lz4', 'zstd'] = 'lz4',
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Build Feather format (columnar Arrow IPC)

    Args:
        input_path: Path to input CSV file
        output_dir: Directory for output Feather files
        nrows: Number of rows to process (None = all)
        compression: Compression codec ('uncompressed', 'lz4', 'zstd')
        logger: Logger instance

    Returns:
        Dictionary with build statistics
    """
    if logger is None:
        logger = setup_logger('build_feather', log_dir=Path('../../exp2_logs'))

    logger.info("=" * 80)
    logger.info("Feather Format Builder (Columnar Arrow IPC)")
    logger.info("=" * 80)

    print("\n[Feather Builder] Starting Feather format build...")
    print(f"[Feather Builder] Input: {input_path}")
    print(f"[Feather Builder] Output: {output_dir}")
    print(f"[Feather Builder] Rows to process: {nrows if nrows else 'ALL'}")
    print(f"[Feather Builder] Compression: {compression}")

    # Initialize metrics
    collector = MetricsCollector(logger=logger)
    collector.start_timer('total_build_time')

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_train = output_dir / 'train.feather'
    output_val = output_dir / 'val.feather'

    try:
        # Read input data
        logger.info(f"Reading input CSV: {input_path}")
        print(f"[Feather Builder] Reading input CSV...")

        collector.start_timer('read_time')
        df = pd.read_csv(input_path, nrows=nrows)
        read_time = collector.stop_timer('read_time')

        logger.info(f"Read {len(df)} rows, {len(df.columns)} columns in {read_time:.2f}s")
        print(f"[Feather Builder] Read {len(df):,} rows, {len(df.columns)} columns")

        # Merge with labels if provided
        if labels_path:
            logger.info(f"Merging with labels from: {labels_path}")
            print(f"[Feather Builder] Merging with labels...")
            labels_df = pd.read_csv(labels_path)
            df = df.merge(labels_df, on='customer_ID', how='inner')
            logger.info(f"Merged data: {len(df)} rows after join")
            print(f"[Feather Builder] Merged: {len(df):,} rows (with labels)")

        # Get memory usage
        memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        collector.add_metric('input_memory_mb', memory_mb)
        logger.info(f"Memory usage: {memory_mb:.2f} MB")

        # Data cleaning and preprocessing
        logger.info("Cleaning and preprocessing data...")
        print(f"[Feather Builder] Cleaning data...")

        initial_rows = len(df)
        df = df.drop_duplicates()
        rows_removed = initial_rows - len(df)

        if rows_removed > 0:
            logger.info(f"Removed {rows_removed} duplicate rows")
            print(f"[Feather Builder] Removed {rows_removed:,} duplicate rows")

        collector.add_metric('duplicates_removed', rows_removed)
        # Keep only numeric columns (drop categoricals to avoid encoding issues)
        initial_cols = len(df.columns)
        numeric_cols = df.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns.tolist()
        
        # Always keep target column
        if 'target' in df.columns and 'target' not in numeric_cols:
            numeric_cols.append('target')
        
        df = df[numeric_cols]
        cols_dropped = initial_cols - len(df.columns)
        
        logger.info(f"Kept {len(df.columns)} numeric columns, dropped {cols_dropped} categorical columns")
        
        # Fill NaN values with median (for numeric columns)
        nan_counts = df.isna().sum().sum()
        if nan_counts > 0:
            logger.info(f"Filling {nan_counts} NaN values with column medians")
            df = df.fillna(df.median())
        
        collector.add_metric('cols_dropped_categorical', cols_dropped)
        collector.add_metric('nans_filled', nan_counts)


        # Split into train/val with stratification
        logger.info("Splitting into train/validation (80/20, stratified, shuffled)...")
        print(f"[Feather Builder] Splitting into train/validation (stratified)...")

        # Use stratified split to maintain class balance
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify=df['target'] if 'target' in df.columns else None
        )

        logger.info(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")
        print(f"[Feather Builder] Train: {len(train_df):,} rows, Val: {len(val_df):,} rows")

        # Write Feather files
        logger.info("Writing Feather files...")
        print(f"[Feather Builder] Writing Feather files...")

        collector.start_timer('write_time')

        # Convert to PyArrow table
        train_table = pa.Table.from_pandas(train_df)
        val_table = pa.Table.from_pandas(val_df)

        # Write Feather V2 (Arrow IPC) with compression
        feather.write_feather(
            train_table,
            output_train,
            compression=compression,
            version=2  # Use Feather V2 (Arrow IPC format)
        )
        logger.info(f"Wrote train Feather: {output_train}")

        feather.write_feather(
            val_table,
            output_val,
            compression=compression,
            version=2
        )
        logger.info(f"Wrote validation Feather: {output_val}")

        write_time = collector.stop_timer('write_time')
        print(f"[Feather Builder] Wrote files in {write_time:.2f}s")

        # Get file sizes
        train_size_mb = output_train.stat().st_size / (1024 ** 2)
        val_size_mb = output_val.stat().st_size / (1024 ** 2)
        total_size_mb = train_size_mb + val_size_mb

        collector.add_metric('train_size_mb', train_size_mb)
        collector.add_metric('val_size_mb', val_size_mb)
        collector.add_metric('total_size_mb', total_size_mb)
        collector.add_metric('num_output_files', 2)

        logger.info(f"Train file size: {train_size_mb:.2f} MB")
        logger.info(f"Val file size: {val_size_mb:.2f} MB")
        logger.info(f"Total size: {total_size_mb:.2f} MB")

        print(f"[Feather Builder] Train file: {train_size_mb:.2f} MB")
        print(f"[Feather Builder] Val file: {val_size_mb:.2f} MB")
        print(f"[Feather Builder] Total size: {total_size_mb:.2f} MB")

        # Calculate compression ratio
        input_size_mb = input_path.stat().st_size / (1024 ** 2)
        compression_ratio = total_size_mb / input_size_mb if input_size_mb > 0 else 1.0

        collector.add_metric('input_size_mb', input_size_mb)
        collector.add_metric('compression_ratio', compression_ratio)

        logger.info(f"Compression ratio vs input: {compression_ratio:.4f}x")
        print(f"[Feather Builder] Compression ratio: {compression_ratio:.4f}x")

        # Calculate average bytes per row
        avg_bytes_per_row = (total_size_mb * 1024 ** 2) / len(df)
        collector.add_metric('avg_bytes_per_row', avg_bytes_per_row)

        logger.info(f"Average bytes per row: {avg_bytes_per_row:.2f}")
        print(f"[Feather Builder] Avg bytes/row: {avg_bytes_per_row:.2f}")

        # Get Arrow schema info
        logger.info("Arrow schema info:")
        print(f"[Feather Builder] Schema:")
        for i, field in enumerate(train_table.schema):
            logger.info(f"  Column {i}: {field.name} ({field.type})")
            if i < 5:  # Print first 5 columns
                print(f"  {field.name}: {field.type}")

        if len(train_table.schema) > 5:
            print(f"  ... and {len(train_table.schema) - 5} more columns")

        collector.add_metric('num_columns', len(train_table.schema))

        # Total build time
        total_time = collector.stop_timer('total_build_time')

        logger.info(f"Total build time: {total_time:.2f}s")
        print(f"[Feather Builder] Total build time: {total_time:.2f}s")

        # Summary statistics
        stats = {
            'format': 'feather',
            'compression': compression,
            'version': 2,
            'input_rows': initial_rows,
            'output_rows': len(df),
            'duplicates_removed': rows_removed,
            'train_rows': len(train_df),
            'val_rows': len(val_df),
            'num_columns': len(df.columns),
            'train_size_mb': train_size_mb,
            'val_size_mb': val_size_mb,
            'total_size_mb': total_size_mb,
            'input_size_mb': input_size_mb,
            'compression_ratio': compression_ratio,
            'avg_bytes_per_row': avg_bytes_per_row,
            'num_output_files': 2,
            'read_time_s': read_time,
            'write_time_s': write_time,
            'total_build_time_s': total_time,
            'output_paths': {
                'train': str(output_train),
                'val': str(output_val)
            }
        }

        # Save metrics
        metrics_path = output_dir / 'build_metrics.json'
        collector.add_metrics(stats)
        collector.save_metrics(metrics_path)

        logger.info("=" * 80)
        logger.info("Feather Build Complete!")
        logger.info("=" * 80)

        print(f"\n[Feather Builder] + Build complete!")
        print(f"[Feather Builder] Metrics saved to: {metrics_path}")

        return stats

    except Exception as e:
        logger.error(f"Error building Feather format: {e}", exc_info=True)
        print(f"\n[Feather Builder] X Error: {e}")
        raise


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build Feather format')
    parser.add_argument('--input', type=str, required=True, help='Input CSV path')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--nrows', type=int, default=None, help='Number of rows to process')
    parser.add_argument('--compression', type=str, default='lz4',
                       choices=['uncompressed', 'lz4', 'zstd'],
                       help='Compression codec')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    logger = setup_logger(
        'build_feather',
        log_dir=Path('../../exp2_logs'),
        debug_mode=args.debug
    )

    stats = build_feather(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        nrows=args.nrows,
        compression=args.compression,
        logger=logger
    )

    print("\nBuild Statistics:")
    for key, value in stats.items():
        if key != 'output_paths':
            print(f"  {key}: {value}")
