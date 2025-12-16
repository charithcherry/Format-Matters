"""
Generate all plots for the Format Matters paper
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

# Set style for consistent, publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Define consistent color scheme as per paper specifications
COLORS = {
    'CSV': '#808080',      # Gray
    'LMDB': '#DC143C',     # Red
    'WebDataset': '#FFA500',  # Orange
    'TFRecord': '#800080',    # Purple
    'Parquet': '#1f77b4',     # Blue
    'Feather': '#2ca02c',     # Green
}

# Base paths
BASE_DIR = Path(r'C:\Users\arjya\Fall 2025\Systems for ML\Project 1\SML')
EXP1_DIR = BASE_DIR / 'format-matters' / 'exp1_runs' / '20251127-145553' / 'train_baselines'
EXP2_DIR = BASE_DIR / 'format-matters' / 'exp2_outputs' / 'runs'
OUTPUT_DIR = BASE_DIR / 'plots'

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

def load_exp1_data():
    """Load Experiment 1 data from summary CSV"""
    summary_path = EXP1_DIR / 'summary.csv'
    df = pd.read_csv(summary_path)
    # Filter to epoch 3 only
    df = df[df['epoch'] == 3]
    return df

def load_exp2_data():
    """Load Experiment 2 data from JSON files"""
    train_results_path = EXP2_DIR / 'train_results.json'
    build_stats_path = EXP2_DIR / 'build_stats.json'

    with open(train_results_path, 'r') as f:
        train_results = json.load(f)

    with open(build_stats_path, 'r') as f:
        build_stats = json.load(f)

    return train_results, build_stats

def plot1_image_throughput():
    """
    PLOT 1: Image Format Throughput Comparison (Exp 1)
    Bar chart showing samples/second for each format
    """
    print("Generating Plot 1: Image Format Throughput Comparison...")

    df = load_exp1_data()

    # Extract throughput data
    formats = ['csv', 'lmdb', 'tfrecord', 'webdataset']
    format_labels = ['CSV', 'LMDB', 'TFRecord', 'WebDataset']
    throughputs = []

    for fmt in formats:
        fmt_data = df[df['format'] == fmt]
        if len(fmt_data) > 0:
            throughputs.append(fmt_data['train_samples_per_sec'].values[0])
        else:
            throughputs.append(0)

    # Calculate variance
    variance = (max(throughputs) - min(throughputs)) / np.mean(throughputs) * 100

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(format_labels, throughputs,
                   color=[COLORS[label] for label in format_labels],
                   edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Throughput (samples/sec)', fontweight='bold')
    ax.set_xlabel('Data Format', fontweight='bold')
    ax.set_title('Experiment 1: Image Format Throughput Comparison\n' +
                 f'(Variance: {variance:.1f}%)', fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(throughputs) * 1.15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plot1_image_throughput.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: plot1_image_throughput.png")

def plot2_image_storage():
    """
    PLOT 2: Image Format Storage Efficiency (Exp 1)
    Log-scale bar chart showing storage sizes
    """
    print("Generating Plot 2: Image Format Storage Efficiency...")

    # Storage data from Table 1 in the paper
    formats = ['CSV', 'WebDataset', 'TFRecord', 'LMDB']
    storage_mb = [35, 4430, 4060, 19070]

    # Create plot with log scale
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(formats, storage_mb,
                   color=[COLORS[f] for f in formats],
                   edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f} MB' if height < 1000 else f'{height/1000:.1f} GB',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_yscale('log')
    ax.set_ylabel('Storage Size (MB, log scale)', fontweight='bold')
    ax.set_xlabel('Data Format', fontweight='bold')

    # Calculate ratio
    ratio = storage_mb[3] / storage_mb[0]  # LMDB / CSV
    ax.set_title(f'Experiment 1: Image Format Storage Efficiency\n' +
                 f'(LMDB is {ratio:.0f}× larger than CSV)', fontweight='bold', pad=20)

    ax.grid(axis='y', alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plot2_image_storage.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: plot2_image_storage.png")

def plot3_tabular_storage():
    """
    PLOT 3: Tabular Format Storage Comparison (Exp 2)
    Grouped bar chart: Data size vs Disk size
    """
    print("Generating Plot 3: Tabular Format Storage Comparison...")

    _, build_stats = load_exp2_data()

    formats = ['CSV', 'Parquet', 'Feather', 'LMDB']
    data_sizes = []
    disk_sizes = []

    for fmt in formats:
        fmt_key = fmt.lower()
        data_sizes.append(build_stats[fmt_key]['total_size_mb'])
        disk_sizes.append(build_stats[fmt_key].get('total_disk_mb',
                                                     build_stats[fmt_key]['total_size_mb']))

    # Create grouped bar chart
    x = np.arange(len(formats))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, data_sizes, width, label='Data Size',
                    color=[COLORS[f] for f in formats], alpha=0.8,
                    edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, disk_sizes, width, label='Disk Size',
                    color=[COLORS[f] for f in formats], alpha=0.5,
                    edgecolor='black', linewidth=1.2, hatch='///')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height/1000:.2f}' if height >= 1000 else f'{height:.0f}',
                    ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Storage Size (MB)', fontweight='bold')
    ax.set_xlabel('Data Format', fontweight='bold')
    ax.set_title('Experiment 2: Tabular Format Storage Comparison\n' +
                 '(LMDB has significant disk overhead)', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(formats)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plot3_tabular_storage.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: plot3_tabular_storage.png")

def plot4_tabular_loading():
    """
    PLOT 4: Tabular Format Loading Performance (Exp 2)
    Bar chart showing total load time
    """
    print("Generating Plot 4: Tabular Format Loading Performance...")

    train_results, _ = load_exp2_data()

    formats = ['Feather', 'Parquet', 'CSV', 'LMDB']
    load_times = []

    for fmt in formats:
        fmt_key = fmt.lower()
        train_load = train_results[fmt_key]['load_train_mean']
        val_load = train_results[fmt_key]['load_val_mean']
        load_times.append(train_load + val_load)

    # Calculate speedups relative to CSV
    csv_time = load_times[2]
    speedups = [csv_time / t for t in load_times]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(formats, load_times,
                   color=[COLORS[f] for f in formats],
                   edgecolor='black', linewidth=1.2)

    # Add value labels with speedup
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if speedups[i] > 1:
            label = f'{height:.2f}s\n({speedups[i]:.1f}× faster)'
        elif speedups[i] < 1:
            label = f'{height:.2f}s\n({1/speedups[i]:.1f}× slower)'
        else:
            label = f'{height:.2f}s\n(baseline)'

        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Total Loading Time (seconds)', fontweight='bold')
    ax.set_xlabel('Data Format', fontweight='bold')
    ax.set_title('Experiment 2: Tabular Format Loading Performance\n' +
                 '(Train + Validation)', fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(load_times) * 1.25)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plot4_tabular_loading.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: plot4_tabular_loading.png")

def plot5_training_accuracy():
    """
    PLOT 5: Training Time & Accuracy Comparison (Exp 2)
    Grouped bar chart with line overlay for accuracy
    """
    print("Generating Plot 5: Training Time & Accuracy Comparison...")

    train_results, _ = load_exp2_data()

    formats = ['CSV', 'Parquet', 'LMDB', 'Feather']
    train_times = []
    val_accuracies = []

    for fmt in formats:
        fmt_key = fmt.lower()
        train_times.append(train_results[fmt_key]['train_model_mean'])
        val_accuracies.append(train_results[fmt_key]['val_accuracy'] * 100)

    # Create plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(formats))
    width = 0.6

    # Bar chart for training time
    bars = ax1.bar(x, train_times, width,
                    color=[COLORS[f] for f in formats],
                    edgecolor='black', linewidth=1.2,
                    label='Training Time')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}s',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax1.set_xlabel('Data Format', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(formats)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(train_times) * 1.2)

    # Line plot for accuracy
    ax2 = ax1.twinx()
    line = ax2.plot(x, val_accuracies, 'ro-', linewidth=2.5,
                     markersize=10, label='Validation Accuracy')

    # Add accuracy labels
    for i, acc in enumerate(val_accuracies):
        ax2.text(i, acc + 0.005, f'{acc:.2f}%',
                 ha='center', va='bottom', fontsize=9, fontweight='bold',
                 color='red')

    ax2.set_ylabel('Validation Accuracy (%)', fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(87.0, 87.3)

    # Title
    variance = (max(val_accuracies) - min(val_accuracies))
    ax1.set_title(f'Experiment 2: Training Time & Model Accuracy\n' +
                  f'(Accuracy variance: {variance:.2f}%, Training time <0.3% variance)',
                  fontweight='bold', pad=20)

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plot5_training_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: plot5_training_accuracy.png")

def plot6_end_to_end():
    """
    PLOT 6: End-to-End Pipeline Performance (Exp 2)
    Stacked bar chart showing breakdown: Load + Train + Eval
    """
    print("Generating Plot 6: End-to-End Pipeline Performance...")

    train_results, _ = load_exp2_data()

    formats = ['Parquet', 'Feather', 'CSV', 'LMDB']
    load_times = []
    train_times = []
    eval_times = []

    for fmt in formats:
        fmt_key = fmt.lower()
        load_times.append(train_results[fmt_key]['load_train_mean'] +
                         train_results[fmt_key]['load_val_mean'])
        train_times.append(train_results[fmt_key]['train_model_mean'])
        eval_times.append(train_results[fmt_key]['eval_train_mean'] +
                         train_results[fmt_key]['eval_val_mean'])

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(formats))
    width = 0.6

    bars1 = ax.bar(x, load_times, width, label='Load Time',
                    color='#3498db', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x, train_times, width, bottom=load_times,
                    label='Train Time', color='#e74c3c',
                    edgecolor='black', linewidth=1.2)
    bars3 = ax.bar(x, eval_times, width,
                    bottom=[l+t for l, t in zip(load_times, train_times)],
                    label='Eval Time', color='#2ecc71',
                    edgecolor='black', linewidth=1.2)

    # Add total time labels on top
    totals = [l+t+e for l, t, e in zip(load_times, train_times, eval_times)]
    for i, (bar, total) in enumerate(zip(bars3, totals)):
        height = bar.get_y() + bar.get_height()
        ax.text(i, height, f'{total:.1f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Calculate speedup vs CSV
    csv_total = totals[2]
    speedups = [csv_total / t for t in totals]

    # Add speedup annotations
    for i, speedup in enumerate(speedups):
        if speedup != 1.0:
            label = f'{speedup:.2f}×' if speedup > 1 else f'{1/speedup:.2f}× slower'
            ax.text(i, totals[i] * 0.5, label,
                    ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_ylabel('Time (seconds)', fontweight='bold')
    ax.set_xlabel('Data Format', fontweight='bold')
    ax.set_title('Experiment 2: End-to-End Pipeline Performance\n' +
                 '(Load + Train + Eval breakdown)', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(formats)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plot6_end_to_end.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: plot6_end_to_end.png")

def plot7_scale_impact():
    """
    PLOT 7: Format Impact vs Dataset Scale (Cross-Experiment)
    Line plot showing how format differences scale
    """
    print("Generating Plot 7: Format Impact vs Dataset Scale...")

    # Data from both experiments
    scales = ['60K\n(Images)', '1M\n(Tabular)']
    scale_values = [60000, 1000000]

    # Training time variance (from paper)
    training_variance = [5.3, 0.3]

    # Loading time differences (calculated)
    # Exp1: minimal differences (all ~20-21 samples/sec, no separate loading measured)
    # Exp2: Feather 4.16s vs LMDB 480.93s = 115.6× difference
    loading_variance_percent = [5, 11560]  # Scaled for visualization

    # Storage variance
    # Exp1: CSV 35MB vs LMDB 19070MB = 544×
    # Exp2: Feather 1164MB vs LMDB 11444MB = 9.8×
    storage_variance = [54400, 980]  # Percentages

    fig, ax = plt.subplots(figsize=(10, 6))

    x = [0, 1]

    # Plot lines
    line1 = ax.plot(x, training_variance, 'o-', linewidth=2.5, markersize=10,
                     label='Training Time Variance', color='#e74c3c')
    line2 = ax.plot(x, [5, 115.6], 's-', linewidth=2.5, markersize=10,
                     label='Loading Time Difference (Fastest vs Slowest)',
                     color='#3498db')

    # Add value labels above the points with offset
    for i in range(2):
        offset = training_variance[i] * 1.3  # 30% above the point
        # Offset horizontally to avoid overlap at x=0
        x_offset = -0.04 if i == 0 else 0
        ax.text(i + x_offset, offset, f'{training_variance[i]:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#e74c3c')

    # Loading time labels with horizontal offset to avoid overlap
    ax.text(0 + 0.04, 5 * 1.3, '5.3%', ha='center', va='bottom', fontsize=9,
            fontweight='bold', color='#3498db')
    ax.text(1, 115.6 * 1.3, '115.6× (23× faster)', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#3498db')

    ax.set_ylabel('Performance Variance/Difference', fontweight='bold')
    ax.set_xlabel('Dataset Scale', fontweight='bold')
    ax.set_title('Cross-Experiment: Format Impact vs Dataset Scale\n' +
                 '(Loading differences amplify with scale, training remains neutral)',
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim(0.1, 1000)

    # Add annotations
    ax.annotate('CPU-bound:\nFormat impact minimal',
                xy=(0, 5), xytext=(0.15, 20),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax.annotate('Larger scale:\nI/O differences amplify',
                xy=(1, 115.6), xytext=(0.7, 300),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
                fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plot7_scale_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: plot7_scale_impact.png")

def main():
    """Generate all plots"""
    print("=" * 60)
    print("GENERATING ALL PLOTS FOR FORMAT MATTERS PAPER")
    print("=" * 60)
    print()

    # Generate all plots
    plot1_image_throughput()
    plot2_image_storage()
    plot3_tabular_storage()
    plot4_tabular_loading()
    plot5_training_accuracy()
    plot6_end_to_end()
    plot7_scale_impact()

    print()
    print("=" * 60)
    print(f"ALL PLOTS GENERATED SUCCESSFULLY!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
    print()
    print("Plot files generated:")
    print("  1. plot1_image_throughput.png")
    print("  2. plot2_image_storage.png")
    print("  3. plot3_tabular_storage.png")
    print("  4. plot4_tabular_loading.png")
    print("  5. plot5_training_accuracy.png")
    print("  6. plot6_end_to_end.png")
    print("  7. plot7_scale_impact.png")

if __name__ == '__main__':
    main()
