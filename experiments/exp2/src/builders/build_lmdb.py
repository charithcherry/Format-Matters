"""
LMDB Format Builder for Experiment 2
Builds row-oriented LMDB database with each row as a pickled dictionary
"""

import pandas as pd
import lmdb
import pickle
import msgpack
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


def build_lmdb(
    input_path: Path,
    output_dir: Path,
    labels_path: Optional[Path] = None,
    nrows: Optional[int] = None,
    compression: Literal['none', 'pickle', 'msgpack'] = 'pickle',
    map_size: Optional[int] = None,  # If None, auto-calculate based on data size
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Build LMDB format (row-based key-value storage)

    Args:
        input_path: Path to input CSV file
        output_dir: Directory for output LMDB databases
        nrows: Number of rows to process (None = all)
        compression: Serialization method ('none', 'pickle', 'msgpack')
        map_size: Maximum LMDB map size in bytes
        logger: Logger instance

    Returns:
        Dictionary with build statistics
    """
    if logger is None:
        logger = setup_logger('build_lmdb', log_dir=Path('../../exp2_logs'))

    logger.info("=" * 80)
    logger.info("LMDB Format Builder (Row-Oriented)")
    logger.info("=" * 80)

    print("\n[LMDB Builder] Starting LMDB format build...")
    print(f"[LMDB Builder] Input: {input_path}")
    print(f"[LMDB Builder] Output: {output_dir}")
    print(f"[LMDB Builder] Rows to process: {nrows if nrows else 'ALL'}")
    print(f"[LMDB Builder] Serialization: {compression}")

    # Initialize metrics
    collector = MetricsCollector(logger=logger)
    collector.start_timer('total_build_time')

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_train = output_dir / 'train.lmdb'
    output_val = output_dir / 'val.lmdb'

    try:
        # Read input data
        logger.info(f"Reading input CSV: {input_path}")
        print(f"[LMDB Builder] Reading input CSV...")

        collector.start_timer('read_time')
        df = pd.read_csv(input_path, nrows=nrows)
        read_time = collector.stop_timer('read_time')

        logger.info(f"Read {len(df)} rows, {len(df.columns)} columns in {read_time:.2f}s")
        print(f"[LMDB Builder] Read {len(df):,} rows, {len(df.columns)} columns")

        # Merge with labels if provided
        if labels_path:
            logger.info(f"Merging with labels from: {labels_path}")
            print(f"[LMDB Builder] Merging with labels...")
            labels_df = pd.read_csv(labels_path)
            df = df.merge(labels_df, on='customer_ID', how='inner')
            logger.info(f"Merged data: {len(df)} rows after join")
            print(f"[LMDB Builder] Merged: {len(df):,} rows (with labels)")

        # Data cleaning and preprocessing
        logger.info("Cleaning and preprocessing data...")
        print(f"[LMDB Builder] Cleaning data...")

        initial_rows = len(df)
        df = df.drop_duplicates()
        rows_removed = initial_rows - len(df)

        if rows_removed > 0:
            logger.info(f"Removed {rows_removed} duplicate rows")
            print(f"[LMDB Builder] Removed {rows_removed:,} duplicate rows")

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
        print(f"[LMDB Builder] Splitting into train/validation (stratified)...")

        # Use stratified split to maintain class balance
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify=df['target'] if 'target' in df.columns else None
        )

        logger.info(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")
        print(f"[LMDB Builder] Train: {len(train_df):,} rows, Val: {len(val_df):,} rows")

        # Auto-calculate map_size if not provided
        if map_size is None:
            # Estimate: ~3 KB per row for pickled data, then double for safety
            estimated_bytes_per_row = 3000
            estimated_total_bytes = len(df) * estimated_bytes_per_row
            map_size = int(estimated_total_bytes * 2)  # 2x safety margin

            # Ensure minimum 1 GB
            map_size = max(map_size, 1 * 1024 ** 3)

            logger.info(f"Auto-calculated map_size: {map_size / 1024**3:.2f} GB (for {len(df)} rows)")
            print(f"[LMDB Builder] Auto map_size: {map_size / 1024**3:.2f} GB")

        # Build LMDB databases
        collector.start_timer('write_time')

        print(f"[LMDB Builder] Building train database...")
        train_bytes = _build_lmdb_db(
            df=train_df,
            output_path=output_train,
            compression=compression,
            map_size=map_size,
            logger=logger
        )

        print(f"[LMDB Builder] Building validation database...")
        val_bytes = _build_lmdb_db(
            df=val_df,
            output_path=output_val,
            compression=compression,
            map_size=map_size,
            logger=logger
        )

        write_time = collector.stop_timer('write_time')
        print(f"[LMDB Builder] Wrote databases in {write_time:.2f}s")

        # Use actual data bytes (not pre-allocated map_size) for fair comparison
        train_size_mb = train_bytes / (1024 ** 2)
        val_size_mb = val_bytes / (1024 ** 2)
        total_size_mb = train_size_mb + val_size_mb

        # Also track on-disk size (includes map_size overhead)
        train_disk_mb = _get_dir_size(output_train) / (1024 ** 2)
        val_disk_mb = _get_dir_size(output_val) / (1024 ** 2)
        total_disk_mb = train_disk_mb + val_disk_mb

        collector.add_metric('train_size_mb', train_size_mb)
        collector.add_metric('val_size_mb', val_size_mb)
        collector.add_metric('total_size_mb', total_size_mb)
        collector.add_metric('train_data_bytes', train_bytes)
        collector.add_metric('val_data_bytes', val_bytes)
        collector.add_metric('num_output_files', 2)

        logger.info(f"Train data: {train_size_mb:.2f} MB (disk: {train_disk_mb:.2f} MB with map_size)")
        logger.info(f"Val data: {val_size_mb:.2f} MB (disk: {val_disk_mb:.2f} MB with map_size)")
        logger.info(f"Total data size: {total_size_mb:.2f} MB (actual bytes)")
        logger.info(f"Total disk size: {total_disk_mb:.2f} MB (includes {map_size / 1024**3:.1f} GB map_size overhead)")

        print(f"[LMDB Builder] Train data: {train_size_mb:.2f} MB ({train_bytes:,} bytes)")
        print(f"[LMDB Builder] Val data: {val_size_mb:.2f} MB ({val_bytes:,} bytes)")
        print(f"[LMDB Builder] Total data size: {total_size_mb:.2f} MB (for comparison)")
        print(f"[LMDB Builder] Total disk size: {total_disk_mb:.2f} MB (includes map_size overhead)")

        # Calculate compression ratio
        input_size_mb = input_path.stat().st_size / (1024 ** 2)
        compression_ratio = total_size_mb / input_size_mb if input_size_mb > 0 else 1.0

        collector.add_metric('input_size_mb', input_size_mb)
        collector.add_metric('compression_ratio', compression_ratio)

        logger.info(f"Compression ratio vs input: {compression_ratio:.4f}x")
        print(f"[LMDB Builder] Compression ratio: {compression_ratio:.4f}x")

        # Storage efficiency
        avg_bytes_per_row = (train_bytes + val_bytes) / len(df)
        collector.add_metric('avg_bytes_per_row', avg_bytes_per_row)

        logger.info(f"Average bytes per row: {avg_bytes_per_row:.2f}")
        print(f"[LMDB Builder] Avg bytes/row: {avg_bytes_per_row:.2f}")

        # Total build time
        total_time = collector.stop_timer('total_build_time')

        logger.info(f"Total build time: {total_time:.2f}s")
        print(f"[LMDB Builder] Total build time: {total_time:.2f}s")

        # Summary statistics
        stats = {
            'format': 'lmdb',
            'compression': compression,
            'input_rows': initial_rows,
            'output_rows': len(df),
            'duplicates_removed': rows_removed,
            'train_rows': len(train_df),
            'val_rows': len(val_df),
            'num_columns': len(df.columns),
            'train_size_mb': train_size_mb,
            'val_size_mb': val_size_mb,
            'total_size_mb': total_size_mb,
            'train_disk_mb': train_disk_mb,
            'val_disk_mb': val_disk_mb,
            'total_disk_mb': total_disk_mb,
            'input_size_mb': input_size_mb,
            'compression_ratio': compression_ratio,
            'avg_bytes_per_row': avg_bytes_per_row,
            'num_output_files': 2,
            'read_time_s': read_time,
            'write_time_s': write_time,
            'total_build_time_s': total_time,
            'map_size_gb': map_size / (1024 ** 3),
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
        logger.info("LMDB Build Complete!")
        logger.info("=" * 80)

        print(f"\n[LMDB Builder] + Build complete!")
        print(f"[LMDB Builder] Metrics saved to: {metrics_path}")

        return stats

    except Exception as e:
        logger.error(f"Error building LMDB format: {e}", exc_info=True)
        print(f"\n[LMDB Builder] X Error: {e}")
        raise


def _build_lmdb_db(
    df: pd.DataFrame,
    output_path: Path,
    compression: str,
    map_size: int,
    logger: logging.Logger
) -> int:
    """
    Build a single LMDB database from DataFrame

    Args:
        df: Input DataFrame
        output_path: Output LMDB path
        compression: Serialization method
        map_size: LMDB map size
        logger: Logger instance

    Returns:
        Total bytes written
    """
    logger.info(f"Creating LMDB database: {output_path}")

    # Create LMDB environment
    env = lmdb.open(
        str(output_path),
        map_size=map_size,
        writemap=True,
        map_async=True,
        sync=False
    )

    total_bytes = 0

    try:
        with env.begin(write=True) as txn:
            for idx, row in enumerate(df.itertuples(index=False)):
                # Convert row to dictionary
                row_dict = row._asdict()

                # Serialize based on compression method
                if compression == 'pickle':
                    value_bytes = pickle.dumps(row_dict, protocol=pickle.HIGHEST_PROTOCOL)
                elif compression == 'msgpack':
                    # Convert to serializable types
                    row_dict_serializable = {
                        k: (int(v) if isinstance(v, (int, float)) else str(v))
                        for k, v in row_dict.items()
                    }
                    value_bytes = msgpack.packb(row_dict_serializable)
                else:  # 'none' - still need to serialize, use pickle
                    value_bytes = pickle.dumps(row_dict, protocol=pickle.HIGHEST_PROTOCOL)

                # Store with integer key
                key = str(idx).encode('ascii')
                txn.put(key, value_bytes)

                total_bytes += len(value_bytes)

                if (idx + 1) % 10000 == 0:
                    logger.debug(f"Processed {idx + 1} rows...")

    finally:
        env.close()

    logger.info(f"Wrote {len(df)} rows to LMDB ({total_bytes:,} bytes)")

    return total_bytes


def _get_dir_size(path: Path) -> int:
    """Get total size of directory in bytes"""
    total_size = 0

    if path.is_file():
        return path.stat().st_size

    if path.is_dir():
        for item in path.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size

    return total_size


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build LMDB format')
    parser.add_argument('--input', type=str, required=True, help='Input CSV path')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--nrows', type=int, default=None, help='Number of rows to process')
    parser.add_argument('--compression', type=str, default='pickle',
                       choices=['none', 'pickle', 'msgpack'], help='Serialization method')
    parser.add_argument('--map-size', type=int, default=50, help='LMDB map size in GB')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    logger = setup_logger(
        'build_lmdb',
        log_dir=Path('../../exp2_logs'),
        debug_mode=args.debug
    )

    stats = build_lmdb(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        nrows=args.nrows,
        compression=args.compression,
        map_size=args.map_size * 1024 ** 3,
        logger=logger
    )

    print("\nBuild Statistics:")
    for key, value in stats.items():
        if key != 'output_paths':
            print(f"  {key}: {value}")
