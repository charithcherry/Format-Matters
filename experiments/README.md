# Format Matters: A Systems Characterization of Data Storage Formats for ML Training

A comprehensive empirical study comparing data storage formats for PyTorch-based ML training pipelines.

## Current Status: Phase 1 Complete - Expanding to Row vs Column Comparison

**Latest Update (Nov 2025)**: Successfully debugged and trained baseline models on all 4 row-oriented formats. Discovered that format choice has minimal impact on small-dataset CPU training (~3.5% performance difference). **Next phase**: Adding column-oriented formats (Parquet, HDF5) to compare row vs column access patterns.

---

## Project Overview

This project systematically evaluates data storage formats for machine learning training, focusing on **systems-level performance characteristics**.

### Phase 1: Row-Oriented Formats (COMPLETED)
- **CSV + files** (baseline): Simple manifest with individual image files
- **WebDataset**: TAR-based sharded format with optional compression
- **TFRecord**: Google's serialized record format
- **LMDB**: Lightning Memory-Mapped Database

### Phase 2: Column-Oriented Formats (COMPLETED)
- **Parquet**: Industry-standard columnar format
- **Arrow/Feather**: In-memory columnar format

---

## Key Findings (Phase 1)

### Baseline Training Results (CIFAR-10, 3 Epochs, CPU)

| Format | Val Accuracy | Throughput | Peak Memory | Disk I/O | Status |
|--------|--------------|------------|-------------|----------|--------|
| **CSV** | 61.41% | 21.28 samples/s | 2439 MB | 0.05 MB/s | Working |
| **TFRecord** | 58.42% | 21.20 samples/s | 2803 MB | 0.05 MB/s | Working |
| **LMDB** | 59.79% | 21.53 samples/s | 2600 MB | 0.00 MB/s | Working |
| **WebDataset** | 43.45% | 20.79 samples/s | 3688 MB | 0.13 MB/s | Has bug |

### Critical Insights

1. **Format Doesn't Matter for Small Datasets + CPU Training**
   - Only 3.5% throughput difference (20.79 - 21.53 samples/s)
   - Training (forward/backward pass) dominates time, not data loading
   - OS caching makes all formats effectively "in-memory"

2. **Three Major Bugs Fixed**
   - **Data duplication**: Case-insensitive filesystem caused 2x duplicate samples (100k instead of 50k)
   - **Insufficient shuffle buffer**: 5k buffer on class-ordered data → increased to 50k
   - **Data ordering**: Files naturally ordered by class → shuffled during build

3. **When Format WOULD Matter**
   - GPU training (500-1000 samples/s → data loading becomes bottleneck)
   - Large datasets (won't fit in RAM → disk I/O matters)
   - Network storage (latency exposes format inefficiencies)
   - Parallel loading (num_workers > 1 → format scalability matters)

4. **Comprehensive Metrics Collection Verified**
   - Throughput, epoch time, CPU utilization, memory usage, disk I/O
   - Detailed time-series logs (~15k samples per run)
   - Build time and storage efficiency tracked

---

## Repository Structure

```
format-matters/
├── README.md                           # This file
├── PROJECT_STATUS.md                   # Detailed status tracking
├── FINAL_ANALYSIS.md                   # Phase 1 results analysis
├── METRICS_AUDIT.md                    # Comprehensive metrics documentation
├── DEBUG_FINDINGS.md                   # Bug investigation details
├── RERUN_INSTRUCTIONS.md              # How to reproduce results
├── env/
│   └── requirements.txt                # Python dependencies
├── data/
│   ├── raw/                            # Original datasets
│   │   ├── cifar10/                   # 50k train, 10k val
│   │   └── imagenet-mini/             # (Optional)
│   └── built/                          # Converted formats
│       └── <dataset>/<format>/<variant>/
├── runs/                               # Experiment results
│   └── <timestamp>/
│       ├── builds/summary.csv          # Build metrics
│       └── train_baselines/
│           ├── summary.csv             # Training summary
│           └── *_metrics.csv           # Detailed time-series
├── notebooks/
│   ├── 00_env_setup_local.ipynb        # Environment setup
│   ├── 01_prepare_datasets.ipynb       # Dataset preparation
│   ├── 02_build_csv_manifest.ipynb     # CSV format builder
│   ├── 03_build_webdataset.ipynb       # WebDataset builder
│   ├── 04_build_tfrecord.ipynb         # TFRecord builder
│   ├── 05_build_lmdb.ipynb             # LMDB builder
│   ├── 10_common_utils.ipynb           # Shared utilities
│   ├── 11_loader_csv.ipynb             # CSV dataloader
│   ├── 12_loader_webdataset.ipynb      # WebDataset dataloader
│   ├── 13_loader_tfrecord.ipynb        # TFRecord dataloader
│   ├── 14_loader_lmdb.ipynb            # LMDB dataloader
│   └── 20_train_baselines.ipynb        # Training experiments
└── scripts/
    ├── FIX_ALL_FORMATS.py              # Bug fix automation
    └── verify_*.py                     # Verification scripts
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- ~5 GB disk space for CIFAR-10 experiments
- 8 GB RAM minimum
- (Optional) NVIDIA GPU for GPU experiments

### Installation

```bash
# Clone repository
cd format-matters

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r env/requirements.txt

# Launch Jupyter
jupyter notebook notebooks/
```

### Running Experiments

```bash
# 1. Prepare data
Run: 01_prepare_datasets.ipynb

# 2. Build formats (run in order)
Run: 02_build_csv_manifest.ipynb
Run: 03_build_webdataset.ipynb
Run: 04_build_tfrecord.ipynb
Run: 05_build_lmdb.ipynb

# 3. Train and compare
Run: 20_train_baselines.ipynb

# 4. View results
Check: runs/<latest>/train_baselines/summary.csv
Read: FINAL_ANALYSIS.md
```

---

## Datasets

### CIFAR-10 (Current)
- **Size**: 50,000 train, 10,000 validation
- **Resolution**: 32×32 RGB
- **Classes**: 10 (airplane, automobile, bird, cat, etc.)
- **Use case**: Format validation, baseline comparison

### ImageNet-mini (Future)
- **Size**: ~35,000 train, ~4,000 validation
- **Resolution**: Variable (resized to 224×224)
- **Use case**: Larger dataset testing

---

## Experimental Design

### Phase 1: Row-Oriented Formats (COMPLETED)

**Formats Tested:**
1. CSV + individual image files
2. WebDataset (TAR shards, 256MB, no compression)
3. TFRecord (shards, 256MB, no compression)
4. LMDB (single database, no compression)

**Training Configuration:**
- Model: ResNet-18
- Dataset: CIFAR-10
- Epochs: 3
- Batch size: 32
- Workers: 0 (single-threaded)
- Device: CPU
- Augmentation: None (for fair comparison)

**Metrics Collected:**
- **Throughput**: Samples/second during training
- **Epoch time**: Total seconds per epoch
- **Validation accuracy**: Model performance
- **CPU utilization**: Mean % during training
- **Memory usage**: Peak RSS in MB
- **Disk I/O**: Read/write MB/s
- **Build time**: Seconds to build format
- **Storage**: Bytes on disk

### Phase 2: Row vs Column Comparison (PLANNED)

**Approach:** Extract rich features from CIFAR-10 images
- Color statistics (6 features)
- Histograms (30 features)
- Texture features (13 features)
- Edge/shape features (10 features)
- ResNet embeddings (512 features)
- ViT embeddings (768 features)
- **Total: ~1,340 features per image**

**Column-Oriented Formats:**
- Parquet (with Snappy compression)
- HDF5 (columnar layout)
- Feather (Arrow format)

**Test Scenarios:**
1. Load all features → Row format should win
2. Load subset of features → Column format should win
3. Feature engineering/selection → Column format should win significantly
4. Add new features → Column format should win (append columns)

---

## Key Metrics

### Build Metrics
- Build time (seconds)
- Disk space (MB)
- File count
- Compression ratio (if applicable)

### Runtime Metrics
- Training throughput (samples/sec)
- Epoch time (seconds)
- First batch latency (seconds)
- CPU utilization (%)
- Memory usage (MB peak)
- Disk I/O rates (MB/s)

### Model Metrics
- Training accuracy
- Validation accuracy
- Training loss
- Validation loss

---

## Technical Stack

- **Framework**: PyTorch 2.3.1
- **Row Formats**: webdataset, tfrecord, lmdb
- **Column Formats**: pyarrow (Parquet), h5py (HDF5)
- **Monitoring**: psutil (CPU/memory/disk)
- **Analysis**: pandas, matplotlib, seaborn
- **Environment**: Python 3.10+

---

## Documentation

- **README.md** (this file): Project overview
- **PROJECT_STATUS.md**: Detailed completion tracking
- **FINAL_ANALYSIS.md**: Phase 1 results and insights
- **METRICS_AUDIT.md**: What metrics are collected and where
- **DEBUG_FINDINGS.md**: Bug investigation and fixes
- **RERUN_INSTRUCTIONS.md**: How to reproduce experiments

---

## Known Issues

### WebDataset Loader Bug
**Status**: Identified but not fixed

**Problem**: `.unbatched().batched()` pattern corrupts data during loading
- Training accuracy: 61.74%
- Validation accuracy: 43.45% (should be ~60%)
- Train/val gap: 18.29% (overfitting signature)

**Proposed Fix**: Remove unbatch/rebatch pattern, use native WebDataset batching

**Location**: `notebooks/12_loader_webdataset.ipynb` lines 47-52

---

## Future Work

### Phase 2: Column-Oriented Formats
1. Extract features from CIFAR-10 images
2. Build Parquet/HDF5/Feather formats
3. Create loaders for column formats
4. Compare row vs column across different access patterns

### Phase 3: Advanced Experiments (Optional)
1. **Multi-worker scaling**: Test num_workers=1,2,4,8
2. **Cold cache analysis**: Clear OS cache between runs
3. **Simulated network storage**: Add artificial latency
4. **Batch size sensitivity**: Test batch=1,8,32,128,512
5. **Compression tradeoffs**: Detailed analysis of compression options
6. **GPU training**: Repeat experiments on GPU hardware

---

## Citation

If you use this work, please cite:

```bibtex
@misc{formatmatters2025,
  title={Format Matters: A Systems Characterization of Data Storage Formats for ML Training},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/format-matters}}
}
```

---

## References

- [WebDataset Documentation](https://github.com/webdataset/webdataset)
- [TFRecord Format Guide](https://www.tensorflow.org/tutorials/load_data/tfrecord)
- [LMDB Documentation](https://lmdb.readthedocs.io/)
- [Apache Parquet Format](https://parquet.apache.org/)
- [HDF5 Documentation](https://www.hdfgroup.org/solutions/hdf5/)
- [PyTorch DataLoader Best Practices](https://pytorch.org/docs/stable/data.html)

---

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
