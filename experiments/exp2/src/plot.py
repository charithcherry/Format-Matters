"""
Plotting Script for Experiment 2
Generates comparison plots for format analysis
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def load_results(runs_dir: Path):
    """Load experimental results"""
    build_stats_path = runs_dir / 'build_stats.json'
    train_results_path = runs_dir / 'train_results.json'

    with open(build_stats_path) as f:
        build_stats = json.load(f)

    with open(train_results_path) as f:
        train_results = json.load(f)

    return build_stats, train_results


def plot_comparisons(build_stats, train_results, output_dir: Path):
    """Generate all comparison plots"""
    output_dir.mkdir(parents=True, exist_ok=True)

    formats = list(train_results.keys())

    # Plot 1: File sizes (use build_stats, not train_results)
    plt.figure(figsize=(10, 6))
    sizes = [build_stats[f].get('total_size_mb', 0) for f in formats]
    plt.bar(formats, sizes, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    plt.ylabel('File Size (MB)')
    plt.title('Storage Efficiency Comparison')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'file_sizes.png', dpi=300)
    plt.close()

    # Plot 2: Load times
    plt.figure(figsize=(10, 6))
    load_times = [train_results[f].get('load_train_total', 0) + train_results[f].get('load_val_total', 0) for f in formats]
    plt.bar(formats, load_times, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    plt.ylabel('Load Time (s)')
    plt.title('Data Loading Performance')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'load_times.png', dpi=300)
    plt.close()

    # Plot 3: Training times
    plt.figure(figsize=(10, 6))
    train_times = [train_results[f].get('train_model_total', 0) for f in formats]
    plt.bar(formats, train_times, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    plt.ylabel('Training Time (s)')
    plt.title('Model Training Performance')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'train_times.png', dpi=300)
    plt.close()

    # Plot 4: Accuracy comparison
    plt.figure(figsize=(10, 6))
    val_accs = [train_results[f].get('val_accuracy', 0) * 100 for f in formats]
    plt.bar(formats, val_accs, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Model Performance Comparison')
    plt.ylim([min(val_accs) - 5, max(val_accs) + 5])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy.png', dpi=300)
    plt.close()

    # Plot 5: End-to-end comparison
    plt.figure(figsize=(12, 6))
    metrics = ['Load Time', 'Train Time', 'Val Accuracy']
    x = np.arange(len(formats))
    width = 0.25

    load_norm = np.array([train_results[f].get('load_train_total', 0) + train_results[f].get('load_val_total', 0) for f in formats])
    train_norm = np.array([train_results[f].get('train_model_total', 0) for f in formats])
    acc_norm = np.array([train_results[f].get('val_accuracy', 0) * 100 for f in formats])

    # Normalize to [0, 100] for comparison
    load_norm = (load_norm / load_norm.max()) * 100
    train_norm = (train_norm / train_norm.max()) * 100

    plt.bar(x - width, load_norm, width, label='Load Time', color='#3498db')
    plt.bar(x, train_norm, width, label='Train Time', color='#e74c3c')
    plt.bar(x + width, acc_norm, width, label='Val Accuracy', color='#2ecc71')

    plt.ylabel('Normalized Score')
    plt.title('Overall Performance Comparison')
    plt.xticks(x, formats)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_comparison.png', dpi=300)
    plt.close()

    print(f"\n✓ Generated plots in {output_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate plots for Experiment 2')
    parser.add_argument('--runs-dir', type=str, default='../exp2_output/runs')
    parser.add_argument('--output-dir', type=str, default='../exp2_output/plots')

    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)

    build_stats, train_results = load_results(runs_dir)
    plot_comparisons(build_stats, train_results, output_dir)
