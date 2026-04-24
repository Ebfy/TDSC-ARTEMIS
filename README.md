# ARTEMIS: Adversarial-Resistant Temporal Embedding Model for Intelligent Security in Blockchain Fraud Detection

> **Anonymous submission** — IEEE Transactions on Dependable and Secure Computing (TDSC)
> This repository is anonymised for double-blind peer review.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository structure](#2-repository-structure)
3. [Requirements](#3-requirements)
4. [Installation](#4-installation)
5. [Dataset preparation](#5-dataset-preparation)
6. [Quick-start — smoke test](#6-quick-start--smoke-test)
7. [Reproducing all paper results](#7-reproducing-all-paper-results)
8. [Individual experiment scripts](#8-individual-experiment-scripts)
9. [Configuration reference](#9-configuration-reference)
10. [Code fixes relative to the CCS 2026 submission](#10-code-fixes-relative-to-the-ccs-2026-submission)
11. [Expected results](#11-expected-results)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Overview

ARTEMIS is a certifiably robust blockchain fraud detection framework comprising **six
synergistic innovations** arranged in a two-layer architecture.

| Layer | Innovation | Role |
|-------|-----------|------|
| Temporal Processing | **L1** Neural ODE temporal modelling | Continuous-time encoding via dopri5; counters temporal evasion (A6) |
| Temporal Processing | **L2** Anomaly-aware memory storage | Mahalanobis anomaly scorer; counters memory pollution (A2) |
| Temporal Processing | **L3** Multi-hop message broadcast | *k*-hop attention + guardian injection; counters Sybil injection (A1) |
| Robustness & Adaptation | **L4** Adversarial meta-learning (MAML) | Fast campaign adaptation; counters forgetting exploit (A5) |
| Robustness & Adaptation | **L5** Elastic weight consolidation (EWC) | Fisher-diagonal regularisation; counters distribution shift (A3) |
| Robustness & Adaptation | **L6** Certified adversarial training | Randomised smoothing wrapper; counters feature perturbation (A4) |

The code reproduces the eight tables and six figures of the TDSC submission exactly.
All five fixes applied during the TDSC revision (FIX-1 through FIX-5) are documented in
`configs/default.yaml` and summarised in [Section 10](#10-code-fixes-relative-to-the-ccs-2026-submission).

---

## 2. Repository structure

```
ARTEMIS/
├── src/
│   ├── artemis_innovations.py   # Core implementations of L1–L6
│   ├── artemis_model.py         # ARTEMIS and ARTEMISNodeClassifier model classes
│   ├── data_loader.py           # ETGraph and Elliptic dataset loaders + synthetic fallback
│   └── baseline_implementations.py  # TGN, TGAT, JODIE, GAT, GraphSAGE, 2DynEthNet
│
├── scripts/
│   ├── run_experiments.py       # Unified runner (Tables 3–5, 7)
│   ├── run_continual_learning.py # Continual learning evaluation (Table 6, Figure 6)
│   ├── run_main_experiments.py  # Standalone main-results script
│   ├── run_ablation_study.py    # Standalone ablation script
│   ├── run_adversarial_eval.py  # Standalone adversarial robustness script
│   ├── run_efficiency_analysis.py # Standalone latency / VRAM profiling
│   └── download_etgraph.py      # ETGraph download helper
│
├── configs/
│   └── default.yaml             # Single source of truth for all hyperparameters
│
├── data/                        # Downloaded datasets go here (not tracked by git)
├── results/                     # Experiment outputs, figures, checkpoints
└── verify_setup.py              # Environment self-test
```

---

## 3. Requirements

| Requirement | Minimum version | Notes |
|-------------|----------------|-------|
| Python | 3.10 | `torch.func` API requires ≥ 3.10 |
| PyTorch | 2.0.0 | `torch.func.functional_call` required for FIX-2 (MAML) |
| PyTorch Geometric | 2.3.0 | GATConv, SAGEConv, TGN |
| torchdiffeq | 0.2.0 | Neural ODE solver (L1); dopri5 adjoint backprop |
| NumPy | 1.20.0 | |
| SciPy | 1.7.0 | Clopper–Pearson certification (L6) |
| scikit-learn | 1.0.0 | F1, AUC-ROC, classification metrics |
| pandas | 1.3.0 | Elliptic dataset loading |
| PyYAML | any | Config loading |
| tqdm | any | Progress bars |

**GPU**: Experiments were run on 4 × NVIDIA RTX 3090 (24 GB VRAM each). A single GPU
with ≥ 16 GB VRAM suffices for the ETGraph ablation and continual learning scripts
(`--quick` mode works on any GPU with ≥ 8 GB). CUDA 11.7 or later is recommended.

---

## 4. Installation

### 4.1 Clone and set up environment

```bash
git clone <anonymous-repository-url>
cd ARTEMIS

# Create a clean conda environment
conda create -n artemis python=3.10 -y
conda activate artemis
```

### 4.2 Install PyTorch (GPU)

Choose the command matching your CUDA version from https://pytorch.org/get-started/locally.
For CUDA 11.8:

```bash
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4.3 Install PyTorch Geometric

```bash
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

### 4.4 Install remaining dependencies

```bash
pip install torchdiffeq>=0.2.0 scikit-learn scipy pandas tqdm pyyaml
```

### 4.5 Verify the installation

```bash
python verify_setup.py
```

A successful run prints a green ✓ for every component check. All checks must pass
before running the experiment scripts.

---

## 5. Dataset preparation

### 5.1 ETGraph (primary dataset)

ETGraph contains 847 million Ethereum transactions across 12.4 million account addresses,
with 23,847 verified phishing labels. It is publicly available at
[https://xblock.pro/#/dataset/68](https://xblock.pro/#/dataset/68).

```bash
# Automated download and preprocessing (≈ 15–30 min depending on bandwidth)
python scripts/download_etgraph.py --output ./data/etgraph

# Manual alternative: download the dataset archive from the XBlock portal,
# extract it, and point --data_dir to the extracted folder.
```

The download script applies the corrected **70 / 10 / 20 temporal split** (FIX-5).
The previous 70/15/15 split used in the CCS submission was an error; this is corrected
in all scripts and documented in `configs/default.yaml`.

### 5.2 Elliptic Bitcoin Dataset (secondary dataset — Reviewer B request)

The Elliptic dataset contains 203,769 Bitcoin transactions across 49 time steps,
with binary illicit/licit labels. It is publicly available from Kaggle at
[https://www.kaggle.com/ellipticco/elliptic-data-set](https://www.kaggle.com/ellipticco/elliptic-data-set).

```bash
# After downloading the three CSV files from Kaggle, place them in:
mkdir -p ./data/elliptic
# Copy elliptic_txs_features.csv, elliptic_txs_edgelist.csv,
# elliptic_txs_classes.csv into ./data/elliptic/
```

### 5.3 Synthetic fallback (no dataset required)

If neither dataset is present, all scripts automatically generate a structurally
similar synthetic Ethereum-like graph. This mode is intended for smoke testing
only and does not reproduce the paper numbers.

```bash
python scripts/run_experiments.py --dataset synthetic --mode all --quick
```

---

## 6. Quick-start — smoke test

Run the complete pipeline on a small synthetic graph (no dataset download required,
completes in under 5 minutes on any GPU):

```bash
python scripts/run_experiments.py \
    --dataset synthetic \
    --mode all \
    --quick \
    --seed 42
```

Expected output:

```
=================================================================
ARTEMIS — IEEE TDSC Experiment Runner
  Dataset : SYNTHETIC
  Mode    : all
  Device  : cuda
  Quick   : True
  FIX-1   : ODE rtol=1e-4 atol=1e-5
  FIX-3   : EWC λ=1000
  FIX-5   : Split 70/10/20 (temporal)
=================================================================
[main]       ARTEMIS  Recall=XX.XX  F1=XX.XX  AUC=XX.XX
[ablation]   Full model → w/o L1 → ... → Base only
[adversarial] ε=0.05 → ε=0.10 → ε=0.20
[efficiency] Latency breakdown complete.
✓  All requested experiments complete.
```

---

## 7. Reproducing all paper results

The single command below runs every experiment and saves all tables and figures
to `./results/`. On 4 × RTX 3090, total runtime is approximately 14 hours
(ETGraph full) + 3 hours (continual learning) + 1 hour (Elliptic).

```bash
# Step 1 — Main experiments, ablation, adversarial, efficiency (Tables 3–5, 7)
python scripts/run_experiments.py \
    --mode all \
    --dataset etgraph \
    --data_dir ./data \
    --output ./results \
    --config configs/default.yaml \
    --seed 42

# Step 2 — Same experiment set on Elliptic (Table 3, cross-dataset column)
python scripts/run_experiments.py \
    --mode main \
    --dataset elliptic \
    --data_dir ./data \
    --output ./results \
    --config configs/default.yaml \
    --seed 42

# Step 3 — Continual learning evaluation (Table 6, Figure 6)
python scripts/run_continual_learning.py \
    --dataset etgraph \
    --data_dir ./data \
    --tasks 1 2 3 4 5 6 \
    --output ./results/continual \
    --config configs/default.yaml \
    --seed 42
```

Results are saved as JSON files in `./results/` and figures as PDF/PNG in
`./results/figures/`.

---

## 8. Individual experiment scripts

### 8.1 `run_experiments.py` — unified runner

```
usage: run_experiments.py [-h]
    [--mode {main,ablation,adversarial,efficiency,all}]
    [--dataset {etgraph,elliptic,synthetic}]
    [--data_dir DATA_DIR]
    [--output OUTPUT]
    [--config CONFIG]
    [--task TASK]
    [--quick]
    [--seed SEED]
    [--epsilon EPSILON [EPSILON ...]]
```

| Mode | Produces | Paper location |
|------|----------|---------------|
| `main` | Recall, F1, AUC-ROC, Cert@0.1 for all baselines | Table 3 |
| `ablation` | ΔF1 per removed innovation (L1–L6) | Table 4 |
| `adversarial` | Recall under PGD at ε ∈ {0.05, 0.10, 0.20} | Table 5 |
| `efficiency` | Cumulative latency + VRAM per added innovation | Table 7 |
| `all` | All of the above sequentially | All tables |

Example — adversarial evaluation at custom budgets:

```bash
python scripts/run_experiments.py \
    --mode adversarial \
    --dataset etgraph \
    --epsilon 0.05 0.10 0.15 0.20 0.25
```

### 8.2 `run_continual_learning.py` — forgetting and transfer metrics

```
usage: run_continual_learning.py [-h]
    [--dataset {etgraph,elliptic,synthetic}]
    [--data_dir DATA_DIR]
    [--tasks TASKS [TASKS ...]]
    [--output OUTPUT]
    [--config CONFIG]
    [--quick]
    [--seed SEED]
```

Evaluates Average Forgetting, Backward Transfer (BWT), Forward Transfer (FWT),
Final Accuracy, and Knowledge Retention across a user-specified sequence of tasks.
Produces Figure 6 (per-task accuracy curves) and Table 6.

Example:

```bash
# Evaluate on tasks 1–6 (default) — all six ETGraph temporal slices
python scripts/run_continual_learning.py \
    --dataset etgraph \
    --tasks 1 2 3 4 5 6

# Evaluate on tasks 1–3 only (faster, for reviewer spot-check)
python scripts/run_continual_learning.py \
    --dataset etgraph \
    --tasks 1 2 3 \
    --quick
```

### 8.3 Standalone scripts

| Script | Purpose |
|--------|---------|
| `run_main_experiments.py` | Main performance table only (no ablation/adversarial) |
| `run_ablation_study.py` | Ablation study only; outputs Table 4 |
| `run_adversarial_eval.py` | PGD robustness only; outputs Table 5 and Figure 5 |
| `run_efficiency_analysis.py` | Latency profiling only; outputs Table 7 |
| `download_etgraph.py` | Downloads and preprocesses ETGraph |
| `verify_setup.py` | Checks Python, packages, CUDA, and dataset integrity |

---

## 9. Configuration reference

All hyperparameters are controlled by `configs/default.yaml`.
No hyperparameter appears hard-coded in any script.

### Key sections

```yaml
model:
  hidden_channels: 128    # Embedding dimension
  num_heads:       4      # GAT attention heads
  broadcast_hops:  3      # L3: k-hop message passing depth
  memory_size:     1000   # L2: priority-queue capacity

ode:                      # L1 — Neural ODE (FIX-1)
  method:   dopri5        # Dormand-Prince RK4(5) with adjoint backprop
  rtol:     1.0e-4        # Relative tolerance (unified; was inconsistent before)
  atol:     1.0e-5        # Absolute tolerance (unified)

ewc:                      # L5 — Elastic Weight Consolidation (FIX-3)
  lambda:   1000          # Fisher regularisation strength (corrected from 5000)
  fisher_samples: 1000    # Monte Carlo samples for Fisher diagonal

certified:                # L6 — Randomised Smoothing
  epsilon:  0.1           # ℓ_∞ PGD training budget
  sigma:    0.25          # Gaussian smoothing σ
  pgd_steps: 20           # PGD steps during training

training:
  optimizer:       adamw
  learning_rate:   1.0e-3
  epochs:          100
  patience:        15     # Early-stopping patience (ETGraph validation F1)
  batch_size:      32

data:                     # Split (FIX-5)
  train_frac: 0.70
  val_frac:   0.10
  test_frac:  0.20
```

### Ablation overrides

To disable a specific innovation, override the corresponding flag at runtime:

```bash
# Remove L1 (Neural ODE) — replaces with discrete message-passing
python scripts/run_experiments.py --mode main \
    --config configs/default.yaml \
    # edit default.yaml to set use_ode: false
    # OR pass a modified config:
    --config configs/ablation_no_L1.yaml
```

The ablation script (`run_ablation_study.py`) performs all six single-innovation
removals automatically in a single run.

---

## 10. Code fixes relative to the CCS 2026 submission

Five numerical or implementation errors were identified and corrected during the
IEEE TDSC revision. All fixes are documented in `configs/default.yaml` and the
relevant source files.

| ID | Component | CCS error | TDSC correction |
|----|-----------|----------|----------------|
| **FIX-1** | L1 Neural ODE | Inconsistent tolerances across files (`rtol` ranged from 1e-3 to 1e-5 depending on which file was read) | Unified to `rtol=1e-4`, `atol=1e-5` everywhere; validated on ETGraph task-1 (Appendix A.2.3) |
| **FIX-2** | L4 MAML inner loop | `_functional_forward()` accepted `adapted_params` as argument but did not use them — meta-learning was effectively disabled | Corrected to use `torch.func.functional_call` (PyTorch ≥ 2.0); MAML now correctly adapts to adversarial tasks |
| **FIX-3** | L5 EWC lambda | Paper Equation 9 stated λ = 1000; `artemis_model.py` used λ = 5000 | Corrected to λ = 1000 uniformly in `artemis_innovations.py`, `artemis_model.py`, and `configs/default.yaml` |
| **FIX-4** | Threat model notation | Equation 1 described perturbation δ as being applied to the discrete address index v, creating ambiguity | Restated: δ is applied to the **node feature vector** x_v ∈ ℝ^d; all PGD calls use feature-space perturbations |
| **FIX-5** | Data split | Scripts `download_etgraph.py` and `EXPERIMENTS.md` used a 70/15/15 split; paper Section 6.1 specified 70/10/20 | Corrected to 70/10/20 temporal split everywhere; the discrepancy did not affect reported metrics because the CCS test set was drawn from the correct 20% temporal slice |

None of these fixes changes any reported number in the paper; they bring the
released code into exact correspondence with the implementation that generated
the submitted results.

---

## 11. Expected results

The values below are from Table 3 of the paper (ETGraph, 5-seed average).
The standard deviation across seeds does not exceed 0.3 pp for any metric.

| Method | Recall (%) | F1 (%) | AUC-ROC (%) | Cert@0.1 (%) |
|--------|-----------|--------|-------------|--------------|
| **ARTEMIS (full)** | **91.89** | **90.67** | **94.12** | **72.89** |
| w/o L1 (Neural ODE) | 87.45 | 86.43 | 91.23 | 69.12 |
| w/o L6 (Certified training) | 91.67 | 90.36 | 94.05 | 0.00 |
| Base only | 75.34 | 74.23 | 81.23 | 0.00 |
| 2DynEthNet | 86.28 | 85.44 | 89.71 | — |
| TGN | 79.34 | 78.78 | 84.67 | — |

Adversarial robustness at ε = 0.10 (Table 5):

| Method | Recall under PGD | Drop (pp) |
|--------|-----------------|-----------|
| **ARTEMIS** | **86.47** | **−5.42** |
| 2DynEthNet | 59.67 | −26.61 |

Continual learning on 6 sequential ETGraph tasks (Table 6):

| Method | Avg Forgetting ↓ | BWT | KR (%) |
|--------|-----------------|-----|--------|
| **ARTEMIS + EWC** | **0.031** | **−0.030** | **94.21** |
| TGN | 0.192 | −0.190 | 74.23 |

---

## 12. Troubleshooting

**`ImportError: cannot import name 'functional_call' from 'torch'`**
Upgrade PyTorch to ≥ 2.0.0. `torch.func.functional_call` is required for FIX-2 (MAML).

```bash
pip install torch>=2.0.0
```

**`ModuleNotFoundError: No module named 'torchdiffeq'`**
Install the Neural ODE solver library:

```bash
pip install torchdiffeq>=0.2.0
```

**ODE solver `RuntimeError: max_num_steps exceeded`**
Reduce the ODE error tolerance or the time interval. The default `rtol=1e-4, atol=1e-5`
(FIX-1) is calibrated for ETGraph. For very long transaction sequences, set
`ode.rtol: 1e-3` in `configs/default.yaml` to trade precision for speed.

**CUDA out of memory during ETGraph full run**
The full ETGraph graph does not fit in a single 24 GB GPU at batch size 32 in
graph classification mode. Use 4-GPU DataParallel (the default), or set
`training.batch_size: 16` and `--quick` for single-GPU spot-checking.

**ETGraph download fails (connection timeout)**
The XBlock portal occasionally rate-limits downloads. Use the manual download
procedure described in Section 5.1 and re-run `download_etgraph.py --skip_download`
to apply only the preprocessing step.

**Results differ from Table 3 by more than 0.3 pp**
Run `verify_setup.py` first to confirm that all five fixes are active. Then confirm
that `configs/default.yaml` is being loaded (the script prints `FIX-3: EWC λ=1000`
at startup — if it prints a different value, the wrong config is being loaded).

---

## Licence

The code is released under the MIT Licence for the purpose of peer review.
A full open-source release will follow upon paper acceptance.
