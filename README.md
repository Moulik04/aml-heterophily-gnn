# Graph-Based Financial Intelligence: Detecting Money Laundering with Heterophily-Aware GNNs

> **DS402 Final Project — Penn State University, April 2026**
> Moulik Jain

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.3%2B-red)](https://pyg.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

Anti-money laundering (AML) detection in financial networks faces three compounding challenges: extreme class imbalance (0.003% fraud rate), relational structure that tabular models cannot exploit, and adversarial actors who deliberately route funds through legitimate accounts.

This project identifies **heterophily**, the tendency of fraudulent nodes to connect to legitimate ones, as the central failure mode of standard GNNs on AML graphs. Standard message-passing architectures (GAT, GraphSAGE) dilute the fraud signal by averaging over predominantly benign neighborhoods. I address this with **H2GCN**, a heterophily-aware architecture that separates ego embeddings from neighbor aggregation, and further introduce a two-stage cascade combining H2GCN node embeddings with XGBoost.

### Key Results

| Model | ROC-AUC | PR-AUC | Recall |
|---|---|---|---|
| Random Forest (tabular) | 0.977 | 0.152 | 0.718 |
| XGBoost (tabular) | 0.972 | 0.511 | 0.581 |
| GraphSAGE | 0.947 | 0.042 | 0.798 |
| GAT | 0.902 | 0.005 | 0.960 |
| TGN | 0.974 | 0.136 | 0.623 |
| H2GCN | 0.960 | 0.113 | **0.903** |
| **H2GCN + XGBoost Cascade** | **0.993** | **0.595** | 0.548 |

> Evaluated on IBM AML HI-Medium test set (400,000 transactions, 0.003% fraud rate).
> GAT generates 187,115 false positives at 96% recall. The cascade reduces this to 46,373, a 4× reduction, while maintaining PR-AUC 119× higher than GAT standalone.

**Elliptic Bitcoin generalization** (zero-shot, no retraining): GAT achieves ROC-AUC = 0.906, PR-AUC = 0.599, Recall = 0.944.

---

## The Core Problem: Heterophily in Financial Graphs

Money laundering is a *structural* phenomenon. Fraudsters layer funds through chains of legitimate intermediary accounts (smurfing, roundtripping, shell companies). This means the neighborhood of a fraudulent node is overwhelmingly benign, standard GNNs that average over neighbor features wash out the fraud signal before classification.

**GAT score distribution:** High fraud probabilities spread broadly to benign transactions, producing ~187K false positives.

**H2GCN score distribution:** Ego-neighbor separation produces a sharp bimodal distribution, fraud scores cluster near 1.0 for true positives, 0.0 for benign transactions.

---

## Architecture

### H2GCN (Heterophily-Aware GCN)

Three design principles adapted from Zhu et al. (2020):

1. **Ego-neighbor separation** — node's own features are projected independently, never mixed with aggregated neighbor features
2. **Multi-hop aggregation** — separate SAGEConv layers aggregate 1-hop and 2-hop neighborhoods
3. **Concatenative combination** — ego embedding and all hop-wise aggregations are concatenated (not averaged)

```
z_v = Linear([ego_v | h_v^(1) | h_v^(2)])
```

This ensures the fraud signal at node `v` cannot be diluted by its typically benign neighborhood.

### Two-Stage Cascade

1. Train H2GCN, extract node embeddings from the trained encoder
2. For each transaction edge (u, v): construct feature vector = [source embedding | destination embedding | raw edge features]
3. Train XGBoost on this enriched feature matrix with `scale_pos_weight` set to training class ratio
4. Select F1-optimal decision threshold on validation set

---

## Novel Features

Three domain-driven structuring-detection features encode smurfing behavior that raw transaction amounts miss:

| Feature | Description |
|---|---|
| `near_threshold` | Binary flag for amounts in [$8,500, $10,000) — just below the US Bank Secrecy Act reporting threshold |
| `structuring_burst` | Normalized per-account count of near-threshold outgoing transfers (clipped at 20, scaled to [0,1]) |
| `same_bank` | Binary flag for intra-bank transfers, which exhibit different laundering patterns than cross-institution transfers |

Two graph structural features:

| Feature | Description |
|---|---|
| `pagerank` | Computed via sparse power-iteration on SciPy CSR matrix — converges in ~40 iterations |
| `kcore_proxy` | min(out_degree, in_degree) — cheap approximation of graph k-coreness for identifying structurally central nodes |

---

## Repository Structure

```
aml-heterophily-gnn/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── data/
│   └── README.md                  # Instructions for downloading IBM AML and Elliptic datasets
│
├── src/
│   ├── data/
│   │   ├── load_ibm.py            # IBM AML HI-Medium loading and preprocessing
│   │   ├── load_elliptic.py       # Elliptic Bitcoin dataset loading
│   │   └── feature_engineering.py # Edge/node feature construction + structuring features
│   │
│   ├── models/
│   │   ├── graphsage.py           # Two-layer SAGEConv baseline
│   │   ├── gat.py                 # GATv2Conv with 4 attention heads
│   │   ├── tgn.py                 # Temporal Graph Network with TGNMemory
│   │   ├── h2gcn.py               # H2GCN — heterophily-aware architecture (proposed)
│   │   └── cascade.py             # Two-stage H2GCN + XGBoost cascade
│   │
│   ├── train.py                   # Training loop with Focal Loss + cosine annealing
│   ├── evaluate.py                # ROC-AUC, PR-AUC, recall, F1 evaluation
│   └── calibration.py             # Platt scaling + isotonic regression
│
├── notebooks/
│   ├── 01_eda.ipynb               # Dataset exploration, class imbalance analysis
│   ├── 02_feature_engineering.ipynb  # Structuring feature construction and validation
│   ├── 03_model_comparison.ipynb  # Full model tournament with ablations
│   └── 04_results_visualization.ipynb  # Score distributions, PR curves, confusion matrices
│
├── results/
│   ├── figures/                   # Score distributions, PR curves, loss curves (Figs 1-6)
│   └── tables/                    # Model comparison tables (Tables I-II equivalent)
│
└── report/
    └── Jain_AML_GNN_DS402.pdf     # Full paper
```

---

## Datasets

### IBM AML HI-Medium
- 2,000,000 transactions between ~500,000 accounts
- Fraud rate: 0.003% (extremely imbalanced)
- Download: [IBM AML Dataset on Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
- Place in `data/ibm/`

### Elliptic Bitcoin Dataset
- 203,769 transaction nodes, 234,355 directed edges
- Illicit rate: 9.76%
- Download: [Elliptic Dataset on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
- Place in `data/elliptic/`

---

## Installation

```bash
git clone https://github.com/Moulik04/aml-heterophily-gnn.git
cd aml-heterophily-gnn
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=2.0.0
torch-geometric>=2.3.0
xgboost>=1.7.0
scikit-learn>=1.2.0
scipy>=1.10.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
```

> GPU training was performed on Bridges2 PSC (NVIDIA V100 32GB). CPU training is supported but significantly slower on the IBM dataset.

---

## Usage

### Feature Engineering
```bash
python src/data/feature_engineering.py --dataset ibm --data_dir data/ibm/
```

### Training

```bash
# Train H2GCN on IBM AML
python src/train.py --model h2gcn --dataset ibm --epochs 15 --batch_size 8192 --lr 1e-3

# Train baseline GAT
python src/train.py --model gat --dataset ibm --epochs 15

# Train on Elliptic Bitcoin
python src/train.py --model gat --dataset elliptic --epochs 15
```

### Two-Stage Cascade
```bash
python src/models/cascade.py --gnn_checkpoint checkpoints/h2gcn_ibm.pt --dataset ibm
```

### Evaluation
```bash
python src/evaluate.py --checkpoint checkpoints/h2gcn_cascade_ibm.pt --dataset ibm
```

---

## Training Details

| Parameter | Value |
|---|---|
| Loss function | Focal Loss (γ = 2) |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| LR scheduler | Cosine annealing with warm restarts |
| Epochs | 15 |
| Batch size | 8,192 edges |
| Graph sampling | Neighborhood sampling with CSR-based subgraph extraction |
| Calibration | Platt scaling / isotonic regression |
| Threshold selection | F1-optimal on validation set |
| Hardware | Bridges2 PSC — NVIDIA V100 32GB |
| Train/Val/Test split | 60/20/20 (stratified, edge-level) |

---

## Results Discussion

The 22× PR-AUC gap between H2GCN (0.113) and GAT (0.005) on identical data and features isolates heterophily as the dominant failure mode. Both models use identical edge-level classification heads and training procedures, the only difference is H2GCN's ego-projection and concatenative multi-hop aggregation.

The cascade's 8.1 percentage point PR-AUC improvement over tabular XGBoost (0.595 vs. 0.511) is attributable entirely to structural information encoded in the H2GCN embeddings, information that raw edge features cannot recover.

---

## Limitations

- IBM dataset is synthetic (calibrated to real banking patterns, but distributional gap to live data is unknown)
- 46,373 false positives per 400K transactions (11.6% FPR) would require additional false-positive reduction for bank-scale deployment
- Pipeline is batch-trained; cannot respond to adversarial concept drift in real time

---

## Future Work

- **Temporal H2GCN**: Combine TGN's temporal memory with H2GCN's ego-separation for both structural and sequence signals
- **Explainability**: GNNExplainer for compliance-ready suspicious activity report justifications (EU AMLD6)
- **Online learning**: Continual learning with drift detection for real-time adaptation
- **Multi-relational graphs**: Heterogeneous GNNs for multiple edge types (wire, ACH, card, cash)

---

## References

1. Hamilton et al. (2017). Inductive representation learning on large graphs. *NeurIPS*.
2. Veličković et al. (2018). Graph attention networks. *ICLR*.
3. Rossi et al. (2020). Temporal graph networks for deep learning on dynamic graphs. *arXiv:2006.10637*.
4. Zhu et al. (2020). Beyond homophily in graph neural networks. *NeurIPS*.
5. Weber et al. (2019). Anti-money laundering in Bitcoin. *KDD Workshop on Anomaly Detection in Finance*.
6. Altman et al. (2023). Realistic synthetic financial transactions for AML models. *NeurIPS Datasets and Benchmarks*.
7. Lin et al. (2017). Focal loss for dense object detection. *ICCV*.
8. Chen & Guestrin (2016). XGBoost. *KDD*.

---

## Citation

```bibtex
@misc{jain2026aml,
  author    = {Moulik Jain},
  title     = {Graph-Based Financial Intelligence: Detecting Money Laundering with Heterophily-Aware Graph Neural Networks},
  year      = {2026},
  school    = {Pennsylvania State University},
  note      = {DS402 Final Project}
}
```

---

*Trained on Bridges2 PSC supercomputing resources at Pittsburgh Supercomputing Center.*
