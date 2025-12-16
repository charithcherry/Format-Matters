"""
Main Runner Script for Experiment 2
Orchestrates the entire experimental pipeline:
1. Build all formats
2. Train models on each format
3. Generate plots and comparisons
"""

import sys
import time
import json
from pathlib import Path
import argparse
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / 'exp2_src'))

from utils.logger import setup_logger
from builders import build_csv, build_lmdb, build_parquet, build_feather
from train import train_and_evaluate


def run_experiment(
    input_data: Path,
    labels_data: Path,
    output_base: Path,
    nrows: int = None,
    skip_build: bool = False,
    formats: list = None,
    debug: bool = False
):
    """
    Run complete Experiment 2 pipeline

    Args:
        input_data: Path to input train_data.csv
        labels_data: Path to train_labels.csv
        output_base: Base output directory
        nrows: Number of rows to process (None = all)
        skip_build: Skip format building if already done
        formats: List of formats to run (None = all)
        debug: Enable debug logging
    """
    # Setup main logger
    logger = setup_logger(
        'exp2_main',
        log_dir=Path('exp2_logs'),
        debug_mode=debug
    )

    logger.info("=" * 80)
    logger.info("EXPERIMENT 2: Tabular Data Format Comparison")
    logger.info("=" * 80)

    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Tabular Data Format Comparison")
    print("=" * 80)
    print(f"Input data: {input_data}")
    print(f"Output base: {output_base}")
    print(f"Rows to process: {nrows if nrows else 'ALL'}")
    print(f"Debug mode: {debug}")
    print("=" * 80 + "\n")

    # Define formats to test (LMDB first to catch capacity issues early)
    all_formats = ['lmdb', 'csv', 'parquet', 'feather']
    formats_to_run = formats if formats else all_formats

    logger.info(f"Formats to test: {formats_to_run}")
    print(f"Formats to test: {', '.join(formats_to_run)}\n")

    # Create output directories
    formats_dir = output_base / 'formats'
    runs_dir = output_base / 'runs'
    formats_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Results storage
    build_stats = {}
    train_results = {}

    # STEP 1: Build all formats
    if not skip_build:
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Building Formats")
        logger.info("="*80)
        print("\n" + "="*80)
        print("STEP 1: Building Formats")
        print("="*80 + "\n")

        format_builders = {
            'csv': (build_csv, {}),
            'lmdb': (build_lmdb, {'compression': 'pickle'}),  # Uses builder default (5 GB)
            'parquet': (build_parquet, {'compression': 'snappy'}),
            'feather': (build_feather, {'compression': 'lz4'})
        }

        for fmt in formats_to_run:
            if fmt not in format_builders:
                logger.warning(f"Unknown format: {fmt}, skipping")
                continue

            builder_func, kwargs = format_builders[fmt]
            output_dir = formats_dir / fmt

            logger.info(f"\nBuilding {fmt.upper()} format...")
            print(f"\n{'-'*60}")
            print(f"Building {fmt.upper()} format")
            print(f"{'-'*60}")

            try:
                start_time = time.time()

                stats = builder_func(
                    input_path=input_data,
                    output_dir=output_dir,
                    labels_path=labels_data,
                    nrows=nrows,
                    logger=logger,
                    **kwargs
                )

                build_time = time.time() - start_time
                build_stats[fmt] = stats

                logger.info(f"+ {fmt.upper()} build complete in {build_time:.2f}s")
                logger.info(f"  Size: {stats['total_size_mb']:.2f} MB")
                logger.info(f"  Compression ratio: {stats['compression_ratio']:.4f}x")

                print(f"\n+ {fmt.upper()} build complete")
                print(f"  Time: {build_time:.2f}s")
                print(f"  Size: {stats['total_size_mb']:.2f} MB")
                print(f"  Ratio: {stats['compression_ratio']:.4f}x")

            except Exception as e:
                logger.error(f"X Failed to build {fmt}: {e}", exc_info=True)
                print(f"\nX Failed to build {fmt}: {e}")
                continue

        # Save build stats
        build_stats_path = runs_dir / 'build_stats.json'
        with open(build_stats_path, 'w') as f:
            json.dump(build_stats, f, indent=2)

        logger.info(f"\nSaved build statistics to {build_stats_path}")
        print(f"\nSaved build statistics to {build_stats_path}")

    else:
        logger.info("Skipping format building (--skip-build enabled)")
        print("Skipping format building (--skip-build enabled)\n")

    # STEP 2: Train models on each format
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Training Models")
    logger.info("="*80)
    print("\n" + "="*80)
    print("STEP 2: Training Models")
    print("="*80 + "\n")

    for fmt in formats_to_run:
        data_dir = formats_dir / fmt

        if not data_dir.exists():
            logger.warning(f"Data directory not found for {fmt}: {data_dir}")
            print(f"! Data directory not found for {fmt}, skipping")
            continue

        logger.info(f"\nTraining on {fmt.upper()} format...")
        print(f"\n{'-'*60}")
        print(f"Training on {fmt.upper()} format")
        print(f"{'-'*60}")

        try:
            start_time = time.time()

            results = train_and_evaluate(
                format_name=fmt,
                data_dir=data_dir,
                output_dir=runs_dir,
                target_col='target',
                n_estimators=100,
                max_depth=10,
                random_state=42,
                logger=logger
            )

            train_time = time.time() - start_time
            train_results[fmt] = results

            logger.info(f"+ {fmt.upper()} training complete in {train_time:.2f}s")
            logger.info(f"  Val accuracy: {results.get('val_accuracy', 0):.4f}")

            print(f"\n+ {fmt.upper()} training complete")
            print(f"  Time: {train_time:.2f}s")
            print(f"  Val accuracy: {results.get('val_accuracy', 0):.4f}")

        except Exception as e:
            logger.error(f"X Failed to train on {fmt}: {e}", exc_info=True)
            print(f"\nX Failed to train on {fmt}: {e}")
            continue

    # Save training results
    train_results_path = runs_dir / 'train_results.json'
    with open(train_results_path, 'w') as f:
        json.dump(train_results, f, indent=2)

    logger.info(f"\nSaved training results to {train_results_path}")
    print(f"\nSaved training results to {train_results_path}")

    # STEP 3: Generate comparison summary
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Results Summary")
    logger.info("="*80)
    print("\n" + "="*80)
    print("STEP 3: Results Summary")
    print("="*80 + "\n")

    # Create comparison table
    print("Format Comparison:")
    print(f"{'-'*80}")
    print(f"{'Format':<12} {'Size (MB)':<12} {'Load Time':<12} {'Train Time':<12} {'Val Acc':<10}")
    print(f"{'-'*80}")

    for fmt in formats_to_run:
        if fmt not in train_results:
            continue

        results = train_results[fmt]

        # Get size from build_stats (not train_results)
        size_mb = build_stats.get(fmt, {}).get('total_size_mb', 0) if build_stats else 0
        load_time = results.get('load_train_total', 0) + results.get('load_val_total', 0)
        train_time = results.get('train_model_total', 0)
        val_acc = results.get('val_accuracy', 0)

        print(f"{fmt.upper():<12} {size_mb:<12.2f} {load_time:<12.2f} {train_time:<12.2f} {val_acc:<10.4f}")

    print(f"{'-'*80}\n")

    logger.info("=" * 80)
    logger.info("EXPERIMENT 2 COMPLETE!")
    logger.info("=" * 80)
    print("=" * 80)
    print("EXPERIMENT 2 COMPLETE!")
    print("=" * 80 + "\n")

    return build_stats, train_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run Experiment 2: Tabular Data Format Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on sample (10K rows) with all formats
  python run_exp2.py --input exp2_data/train_data.csv --nrows 10000

  # Run on full data (takes longer)
  python run_exp2.py --input exp2_data/train_data.csv

  # Skip building if already done
  python run_exp2.py --input exp2_data/train_data.csv --skip-build

  # Run only specific formats
  python run_exp2.py --input exp2_data/train_data.csv --formats csv parquet

  # Enable debug logging
  python run_exp2.py --input exp2_data/train_data.csv --nrows 10000 --debug
        """
    )

    parser.add_argument('--input', type=str, required=True,
                       help='Path to input train_data.csv')
    parser.add_argument('--labels', type=str, required=True,
                       help='Path to train_labels.csv')
    parser.add_argument('--output', type=str,
                       default='exp2_outputs',
                       help='Output base directory')
    parser.add_argument('--nrows', type=int, default=None,
                       help='Number of rows to process (None = all, use 10000 for testing)')
    parser.add_argument('--skip-build', action='store_true',
                       help='Skip format building (use existing formats)')
    parser.add_argument('--formats', nargs='+',
                       choices=['csv', 'lmdb', 'parquet', 'feather'],
                       help='Specific formats to run (default: all)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')

    args = parser.parse_args()

    # Convert to Path objects
    input_path = Path(args.input)
    labels_path = Path(args.labels)
    output_path = Path(args.output)

    # Validate input
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    if not labels_path.exists():
        print(f"ERROR: Labels file not found: {labels_path}")
        sys.exit(1)

    # Run experiment
    try:
        build_stats, train_results = run_experiment(
            input_data=input_path,
            labels_data=labels_path,
            output_base=output_path,
            nrows=args.nrows,
            skip_build=args.skip_build,
            formats=args.formats,
            debug=args.debug
        )

        print("\n+ Experiment completed successfully!")
        print(f"\nResults saved to: {output_path}/runs/")

    except KeyboardInterrupt:
        print("\n\n! Experiment interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\nX Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
