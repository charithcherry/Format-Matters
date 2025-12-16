# Format Matters: A Systems Characterization of Data Storage Formats for ML Training

A comprehensive empirical study comparing data storage formats for ML training pipelines across image and tabular workloads.

## Status: ALL EXPERIMENTS COMPLETE ✓

Both experiments have been completed and published. See the full paper in `../papers/` for detailed analysis.

---

## Project Overview

This project systematically evaluates data storage formats for machine learning training, focusing on **systems-level performance characteristics** across two distinct workloads:

### Experiment 1: Image Classification (COMPLETED ✓)
**Dataset:** CIFAR-10 (60,000 images)
**Formats Tested:** CSV, WebDataset, TFRecord, LMDB (all row-oriented)
**Key Finding:** At small scale (60K samples) on CPU, format choice has minimal impact on throughput (5.3% variance) due to computation bottlenecks. However, storage efficiency varies dramatically (544× between CSV manifest and LMDB).

### Experiment 2: Tabular ML (COMPLETED ✓)
**Dataset:** American Express Default Prediction (1,000,000 rows, 187 features)
**Formats Tested:** CSV, LMDB (row-oriented) vs Parquet, Feather (columnar)
**Key Finding:** At larger scale (1M rows), columnar formats achieve 2.2-2.7× better compression and 4.8-6.2× faster loading. LMDB catastrophically underperforms (18.6× slower loading). Training time remains format-neutral (<0.3% variance).

---

## Key Findings Summary

### Scale-Dependency of Format Impact
- **Small datasets (60K):** Minimal throughput differences (5.3%), CPU-bound training masks format effects
- **Large datasets (1M):** Substantial loading differences (6.2× speedup with columnar formats) as I/O costs grow
- **Insight:** Format optimization benefits **loading**, not **training**, in CPU-bound scenarios

### Format Performance by Modality

**Image Data (Experiment 1):**
| Format | Throughput | Storage | Best For |
|--------|------------|---------|----------|
| CSV | 21.28 s/s | 35 MB | Development, prototyping |
| LMDB | 21.53 s/s | 19,070 MB | Random access workloads |
| TFRecord | 21.20 s/s | 4,060 MB | TensorFlow ecosystems |
| WebDataset | 20.46 s/s | 4,430 MB | Cloud streaming |

**Tabular Data (Experiment 2):**
| Format | Load Time | Storage | Compression | Best For |
|--------|-----------|---------|-------------|----------|
| **Feather** | 4.16s | 1,164 MB | 2.7× | Preprocessing/analysis |
| **Parquet** | 5.45s | 1,452 MB | 2.2× | **Production ML (recommended)** |
| CSV | 25.91s | 3,200 MB | 1.0× | Development, small scale |
| LMDB | 480.93s | 11,444 MB | 1.1× | **Avoid for bulk ML** |

### Critical Insights

1. **Format-Neutral Model Accuracy**
   - Exp 1: 58-61% validation accuracy (minimal variance)
   - Exp 2: 87.17-87.19% validation accuracy (0.02% variance)
   - Format affects I/O efficiency, NOT ML outcomes

2. **LMDB is Workload-Dependent**
   - Optimized for random key-value lookups, NOT sequential bulk reads
   - 18.6× slower loading for tabular ML (1M rows)
   - 544× storage overhead for images vs CSV manifests
   - **Use only for:** RL experience replay, random sampling, item-level queries
   - **Avoid for:** Full-dataset epoch iterations, sequential training

3. **Columnar Formats Excel for Tabular ML at Scale**
   - Parquet: Best overall (16% faster end-to-end, 2.2× compression)
   - Feather: Fastest loading (6.2×) but 15% slower training
   - Only applicable to structured data, NOT images

4. **CPU-Bound Training Masks Format Effects**
   - Both experiments show <5.3% training time variance
   - CPU utilization: 790-1444% (saturated)
   - Format optimization benefits I/O, not computation

---

## Repository Structure

```
Format-Matters/
├── papers/                             # Published paper and drafts
│   ├── format_matters_1129_paper.md    # Full paper with all findings
│   ├── format_matters_paper.tex        # LaTeX version
│   └── ...
├── docs/                               # Presentation materials
│   ├── presentation_slides.md
│   ├── presentation_script.md
│   └── presentation_qa.md
├── experiments/                        # THIS DIRECTORY
│   ├── README.md                       # This file
│   ├── requirements.txt                # Python dependencies
│   ├── exp1/                           # Image classification experiments
│   │   ├── src/                        # Jupyter notebooks
│   │   │   ├── 00_env_setup_*.ipynb
│   │   │   ├── 01_prepare_datasets.ipynb
│   │   │   ├── 02-05_build_*.ipynb     # Format builders
│   │   │   ├── 10-14_loader_*.ipynb    # Dataloaders
│   │   │   ├── 20_train_baselines.ipynb
│   │   │   ├── 30-31_analysis_*.ipynb
│   │   │   └── 40_decision_guide.ipynb
│   │   └── runs/                       # Experiment results
│   └── exp2/                           # Tabular ML experiments
│       ├── src/                        # Source code
│       │   └── run_exp2.py             # Main experiment script
│       ├── runs/                       # Experiment results (renamed from logs)
│       ├── outputs/                    # Generated plots and analysis
│       └── exp2_1M_run.log             # Execution log
└── plots/                              # Paper figures
    ├── plot1_image_throughput.png
    ├── plot2_image_storage.png
    ├── plot3_tabular_storage.png
    ├── plot4_tabular_loading.png
    ├── plot5_training_accuracy.png
    ├── plot6_end_to_end.png
    └── plot7_scale_impact.png
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- 5-10 GB disk space (depends on experiments)
- 8 GB RAM minimum
- CPU-only (GPU experiments not included)

### Installation

```bash
# Navigate to experiments directory
cd experiments/

# Create virtual environment
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

**Experiment 1 (Image Classification):**
```bash
# Navigate to exp1 source
cd exp1/src/

# Launch Jupyter
jupyter notebook

# Run notebooks in order:
# 1. 00_env_setup_local.ipynb
# 2. 01_prepare_datasets.ipynb
# 3. 02-05_build_*.ipynb (all format builders)
# 4. 10-14_loader_*.ipynb (verify dataloaders)
# 5. 20_train_baselines.ipynb (main experiment)
# 6. 30-31_analysis_*.ipynb (results analysis)

# Results will be in exp1/runs/<timestamp>/
```

**Experiment 2 (Tabular ML):**
```bash
# Navigate to exp2 source
cd exp2/src/

# Run main experiment script
python run_exp2.py

# Results will be in exp2/runs/ and exp2/outputs/
```

---

## Datasets

### Experiment 1: CIFAR-10
- **Size:** 50,000 train + 10,000 validation (60,000 total)
- **Resolution:** 32×32 RGB images
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Download:** Automatic via torchvision.datasets

### Experiment 2: American Express Default Prediction
- **Size:** 1,000,000 rows sampled from full dataset
- **Features:** 187 numeric features (after dropping categoricals)
- **Target:** Binary classification (credit default)
- **Source:** Kaggle competition dataset
- **Preprocessing:** Stratified 80/20 split, median imputation for NaN values

---

## Technical Stack

**Core Libraries:**
- **Framework:** PyTorch 2.8.0 (Exp 1)
- **ML:** scikit-learn (Exp 2)
- **Data Processing:** pandas 2.2.2, numpy 1.26.4, pyarrow 16.1.0

**Format Libraries:**
- **Row-oriented:** webdataset 0.2.86, tfrecord 1.14.6, lmdb 1.5.1
- **Columnar:** pyarrow (Parquet), pyarrow (Feather/Arrow IPC)
- **Compression:** zstandard, lz4

**Monitoring:**
- **System:** psutil 5.9.8 (CPU, memory, disk I/O)
- **Visualization:** matplotlib 3.8.4, seaborn

**Environment:** Python 3.10+, Windows 11, AMD Ryzen 8-core @ 3.3 GHz

---

## Experimental Methodology

### Experiment 1 Configuration
- **Model:** ResNet-18 (pretrained=False)
- **Optimizer:** SGD (lr=0.001, momentum=0.9, weight_decay=1e-4)
- **Training:** 3 epochs, batch_size=64, num_workers=0
- **Transforms:** Resize(224×224), ToTensor(), Normalize(ImageNet stats)
- **Monitoring:** 0.5s intervals for CPU%, memory, disk I/O

### Experiment 2 Configuration
- **Model:** RandomForestClassifier (n_estimators=100, max_depth=10)
- **Data:** 1M rows × 187 features (numeric only)
- **Split:** 80/20 train/validation, stratified
- **Evaluation:** Accuracy, F1-score (weighted)
- **Monitoring:** Same as Exp 1

---

## Actionable Recommendations

### For Small-Scale Development (<100K samples)
→ **Use CSV** (simplicity, debuggability, human-readable)

### For Production Tabular ML (>100K samples)
→ **Use Parquet** (best overall: 16% faster end-to-end + 2.2× compression + industry standard)

### For Iterative ML Workflows (CV, hyperparameter tuning)
→ **Use Parquet** (Feather's faster loading negated by 15% training overhead)

### For Data Preprocessing/Exploration (no training)
→ **Use Feather** (fastest loading + building when training time doesn't matter)

### For Image Pipelines on CPU
→ **Use CSV manifests** (minimal storage, competitive throughput)

### For GPU Training (predicted, not tested)
→ **Use optimized formats** (WebDataset for images, Parquet for tabular) as I/O becomes bottleneck

### LMDB is Workload-Dependent
- **Use for:** Random sampling (RL replay buffers), item-level queries
- **Avoid for:** Sequential bulk ML training, full-dataset epoch iterations

---

## Performance Metrics Collected

### Build Metrics
- Build time (seconds)
- Storage size (MB: data size vs disk size)
- Compression ratio (vs CSV baseline)
- File count

### Runtime Metrics
- **Loading:** Time to load train/val data (seconds)
- **Throughput:** Samples/second during training
- **Epoch time:** Seconds per training epoch
- **CPU utilization:** % (multi-core aggregate)
- **Memory usage:** Peak RSS in MB
- **Disk I/O:** Read/write MB/s

### Model Metrics
- Training accuracy (%)
- Validation accuracy (%)
- F1-score (for tabular only)

---

## Citation

If you use this work, please cite:

```bibtex
@misc{formatmatters2025,
  title={Format Matters: An Empirical Study of Data Storage Format Impact on Machine Learning Training Pipelines},
  author={Your Names},
  year={2025},
  howpublished={Systems for Machine Learning (SysML)}
}
```

---

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/data.html)
- [WebDataset GitHub](https://github.com/webdataset/webdataset)
- [TFRecord Format Guide](https://www.tensorflow.org/tutorials/load_data/tfrecord)
- [LMDB Documentation](https://lmdb.readthedocs.io/)
- [Apache Parquet Format](https://parquet.apache.org/)
- [Apache Arrow Documentation](https://arrow.apache.org/)
- [Kaggle: American Express Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction)

---

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please open a GitHub issue.

---

**Last Updated:** December 2025
**Status:** All experiments complete, paper published
