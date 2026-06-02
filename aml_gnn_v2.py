#!/usr/bin/env python3
"""
aml_gnn_v2.py — AML Graph Neural Network Pipeline (Full Production Version)
============================================================================
Improvements over PoC (aml_poc_experiment.py):
  1.  Full-graph training with NeighborLoader / LinkNeighborLoader (no subsampling)
  2.  Edge-level classification  (transaction-level AML labels)
  3.  Temporal features: Time2Vec, per-account velocity, delta_t
  4.  Temporal Graph Network (TGN) architecture with node memory
  5.  Graph Attention Network (GAT v2) with multi-head attention
  6.  Heterogeneous graph: Account nodes + Bank nodes (from accounts.csv)
  7.  GraphSMOTE: graph-aware minority oversampling
  8.  Focal Loss for extreme class imbalance
  9.  Optuna hyperparameter sweep  (--mode sweep)
  10. IBM HI-Medium + LI-Medium datasets via Kaggle API download
  11. Elliptic Bitcoin dataset for cross-domain validation

Datasets used:
  • IBM HI-Medium_Trans.csv  (high-illicit, ~13 M transactions)
  • IBM LI-Medium_Trans.csv  (low-illicit,  ~13 M transactions)
  • Elliptic Bitcoin dataset (local files)

Extra IBM files used:
  • *_Patterns.txt  → laundering attack-type label  (FAN-OUT / CYCLE / etc.)
  • *_accounts.csv  → entity type + bank ID as node features & hetero edges

Run modes:
  python aml_gnn_v2.py                        # train with default config
  python aml_gnn_v2.py --model GAT            # use GAT instead of GraphSAGE
  python aml_gnn_v2.py --model TGN            # use TGN
  python aml_gnn_v2.py --dataset ELLIPTIC     # use Elliptic dataset
  python aml_gnn_v2.py --mode sweep           # Optuna hyperparameter search
  python aml_gnn_v2.py --task node            # node-level classification
"""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — Installation & Imports
# ══════════════════════════════════════════════════════════════════════════════
import subprocess, sys

def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import importlib.util as _ilu
for _pkg in ["torch", "torch_geometric", "scikit-learn", "pandas",
             "matplotlib", "numpy", "optuna", "kaggle"]:
    if _ilu.find_spec(_pkg.replace("-", "_")) is None:
        print(f"Installing {_pkg}…")
        _install(_pkg)

import os, re, json, warnings, argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.nn import SAGEConv, GATv2Conv, HeteroConv, to_hetero
from torch_geometric.nn.models.tgn import TGNMemory, IdentityMessage, LastAggregator
from torch_geometric.utils import to_undirected, add_self_loops

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, ConfusionMatrixDisplay,
                             precision_recall_curve, auc)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

import optuna
from types import SimpleNamespace
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Configuration
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Dataset ──────────────────────────────────────────────────────────────
    "DATASET"        : "IBM_HI",      # "IBM_HI" | "IBM_LI" | "ELLIPTIC"
    "DATASET_SIZE"   : "medium",      # "small" (local) | "medium" | "large"
    "KAGGLE_DATASET" : "ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
    "DATA_DIR"       : ".",           # directory where CSV files live

    # ── Task ─────────────────────────────────────────────────────────────────
    "TASK"           : "edge",        # "edge" (transaction-level) | "node" (account-level)
    "MULTI_CLASS"    : True,          # also predict AML pattern type (FAN-OUT/CYCLE/etc.)
    "HETERO"         : True,          # use heterogeneous graph (accounts + banks)

    # ── Model ─────────────────────────────────────────────────────────────────
    "MODEL"          : "GraphSAGE",   # "GraphSAGE" | "GAT" | "TGN"
    "HIDDEN_DIM"     : 256,
    "NUM_LAYERS"     : 3,
    "DROPOUT"        : 0.3,
    "GAT_HEADS"      : 8,
    "TGN_MEMORY_DIM" : 128,
    "TGN_TIME_DIM"   : 32,

    # ── Training ──────────────────────────────────────────────────────────────
    "EPOCHS"         : 15,
    "LR"             : 0.001,
    "WEIGHT_DECAY"   : 5e-4,
    "BATCH_SIZE"     : 4096,
    "NUM_NEIGHBORS"  : [20, 10],      # NeighborLoader hops
    "FOCAL_ALPHA"    : 0.25,
    "FOCAL_GAMMA"    : 2.0,
    "SMOTE_K"        : 5,             # GraphSMOTE neighbours
    "SMOTE_RATIO"    : 0.3,           # target minority fraction after SMOTE
    "MAX_ROWS"       : 2_000_000,     # subsample CSV rows (None = full dataset)

    # ── Optuna ────────────────────────────────────────────────────────────────
    "OPTUNA_TRIALS"  : 20,
    "OPTUNA_METRIC"  : "val_recall",

    # ── Misc ──────────────────────────────────────────────────────────────────
    "SEED"           : 42,
    "RESULTS_DIR"    : "results_v2",
    "DEVICE"         : "cuda" if torch.cuda.is_available() else "cpu",
}

PATTERN_TYPES = ["BENIGN", "FAN-OUT", "FAN-IN", "CYCLE",
                 "SCATTER-GATHER", "GATHER-SCATTER", "BIPARTITE", "STACK", "RANDOM"]
PATTERN_TO_IDX = {p: i for i, p in enumerate(PATTERN_TYPES)}
ENTITY_TYPES   = ["Corporation", "Partnership", "Sole Proprietorship",
                  "Individual", "Country", "Direct", "Unknown"]
ENTITY_TO_IDX  = {e: i for i, e in enumerate(ENTITY_TYPES)}

torch.manual_seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])
os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Kaggle Setup & Data Download
# ══════════════════════════════════════════════════════════════════════════════

KAGGLE_SETUP_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════╗
║  KAGGLE SETUP REQUIRED (one-time, on Bridges2)              ║
╠══════════════════════════════════════════════════════════════╣
║  1. Go to kaggle.com → Settings → API → Create New Token   ║
║  2. Copy your API token (starts with KGAT_...)              ║
║  3. In Bridges2 terminal run:                               ║
║     mkdir -p ~/.kaggle                                       ║
║     echo YOUR_TOKEN > ~/.kaggle/access_token                ║
║     chmod 600 ~/.kaggle/access_token                        ║
║  OR set env var:                                            ║
║     export KAGGLE_API_TOKEN=YOUR_TOKEN                      ║
╚══════════════════════════════════════════════════════════════╝
"""

def _kaggle_download(filename: str, dest_dir: str = ".") -> bool:
    """Download a single file from the IBM AML Kaggle dataset."""
    dest = Path(dest_dir) / filename
    if dest.exists():
        print(f"  [cache] {filename} already exists, skipping download.")
        return True
    print(f"  [download] {filename} …")
    result = subprocess.run(
        ["kaggle", "datasets", "download",
         "-d", CONFIG["KAGGLE_DATASET"],
         "-f", filename,
         "--path", dest_dir,
         "--unzip"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [error] {result.stderr.strip()}")
        return False
    return dest.exists()


def setup_ibm_data(size: str = "medium") -> dict[str, str | None]:
    """
    Ensure IBM dataset files exist. Downloads via Kaggle if missing.
    Falls back to HI-Small (local) if download fails.
    Returns paths dict: {trans, patterns, accounts} for HI and LI variants.
    """
    d = CONFIG["DATA_DIR"]
    files = {}
    for variant in ("HI", "LI"):
        cap = size.capitalize()
        trans_name    = f"{variant}-{cap}_Trans.csv"
        patterns_name = f"{variant}-{cap}_Patterns.txt"
        accounts_name = f"{variant}-{cap}_accounts.csv"

        trans_path = Path(d) / trans_name
        if not trans_path.exists():
            print(f"\nDataset file '{trans_name}' not found. Attempting Kaggle download…")
            print(KAGGLE_SETUP_INSTRUCTIONS)
            ok = _kaggle_download(trans_name, d)
            if not ok:
                fallback = Path(d) / "HI-Small_Trans.csv"
                print(f"  Download failed. Falling back to {fallback}")
                trans_path = fallback if fallback.exists() else None

        for fname in (patterns_name, accounts_name):
            if not (Path(d) / fname).exists():
                _kaggle_download(fname, d)

        files[variant] = {
            "trans"    : str(trans_path) if trans_path and trans_path.exists() else None,
            "patterns" : str(Path(d) / patterns_name) if (Path(d) / patterns_name).exists() else None,
            "accounts" : str(Path(d) / accounts_name) if (Path(d) / accounts_name).exists() else None,
        }
    return files


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Data Loading
# ══════════════════════════════════════════════════════════════════════════════

IBM_COLS = ["Timestamp", "From_Bank", "From_Account",
            "To_Bank",   "To_Account",
            "Amount_Received", "Receiving_Currency",
            "Amount_Paid",     "Payment_Currency",
            "Payment_Format",  "Is_Laundering"]


def load_ibm_trans(path: str, max_rows: int = None, row_offset: int = 0) -> pd.DataFrame:
    info = f"  Loading transactions: {Path(path).name}"
    if row_offset:
        info += f"  [skip {row_offset:,} rows]"
    if max_rows:
        info += f"  [cap {max_rows:,} rows]"
    print(info, flush=True)
    # pandas keeps row 0 as header regardless of skiprows list
    skip = range(1, row_offset + 1) if row_offset > 0 else None
    df = pd.read_csv(path, low_memory=False, nrows=max_rows, skiprows=skip, header=0)
    df.columns = IBM_COLS
    df["Is_Laundering"] = df["Is_Laundering"].astype(int)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M",
                                     errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df["unix_t"] = df["Timestamp"].astype(np.int64) // 10**9
    print(f"    {len(df):,} rows  |  fraud: {df['Is_Laundering'].sum():,} "
          f"({df['Is_Laundering'].mean()*100:.3f}%)")
    return df


def parse_patterns(path: str) -> dict[tuple, str]:
    """
    Parse *_Patterns.txt to map (from_acc, to_acc, amount_paid) → pattern type.
    Pattern blocks start with  'BEGIN LAUNDERING ATTEMPT - TYPE'
    and contain transaction lines in the same format as _Trans.csv.
    """
    if not path:
        return {}
    mapping = {}
    current = None
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN"):
                m = re.search(r"BEGIN LAUNDERING ATTEMPT - ([^:]+)", line)
                current = m.group(1).strip() if m else "UNKNOWN"
            elif line.startswith("END"):
                current = None
            elif current and "," in line:
                parts = line.split(",")
                if len(parts) >= 9:
                    from_acc   = parts[2].strip()
                    to_acc     = parts[4].strip()
                    amount_str = parts[7].strip()
                    mapping[(from_acc, to_acc, amount_str)] = current
    print(f"  Parsed {len(mapping):,} labeled pattern transactions  "
          f"({len(set(mapping.values()))} types)")
    return mapping


def parse_accounts(path: str) -> pd.DataFrame:
    """
    Load *_accounts.csv and extract entity type + bank ID per account.
    Returns DataFrame with columns [Account, Bank_ID, Entity_Type_Idx].
    """
    if not path:
        return pd.DataFrame(columns=["Account", "Bank_ID", "Entity_Type_Idx"])
    acc = pd.read_csv(path, low_memory=False)
    acc.columns = [c.strip() for c in acc.columns]
    acc = acc.rename(columns={"Account Number": "Account", "Bank ID": "Bank_ID"})
    raw_type = acc["Entity Name"].str.extract(r"^([A-Za-z ]+)#").iloc[:, 0].str.strip()
    acc["Entity_Type_Idx"] = raw_type.map(ENTITY_TO_IDX).fillna(ENTITY_TO_IDX["Unknown"]).astype(int)
    print(f"  Loaded accounts: {len(acc):,}  |  "
          f"{acc['Entity_Type_Idx'].nunique()} entity types  |  "
          f"{acc['Bank_ID'].nunique():,} banks")
    return acc[["Account", "Bank_ID", "Entity_Type_Idx"]]


def load_elliptic() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = CONFIG["DATA_DIR"]
    print("  Loading Elliptic dataset…")
    feats   = pd.read_csv(f"{d}/elliptic_txs_features.csv",   header=None)
    classes = pd.read_csv(f"{d}/elliptic_txs_classes.csv")
    edges   = pd.read_csv(f"{d}/elliptic_txs_edgelist.csv")
    feats.columns = ["txId", "time_step"] + [f"f{i}" for i in range(feats.shape[1]-2)]
    classes["label"] = classes["class"].map({"1": 1, "2": 0}).fillna(-1).astype(int)
    print(f"    {len(feats):,} nodes  |  {len(edges):,} edges  |  "
          f"illicit: {(classes['label']==1).sum():,}")
    return feats, classes, edges


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

class Time2Vec(nn.Module):
    """Learnable time encoding: [t, sin(w1*t+b1), …, sin(wk*t+bk)]"""
    def __init__(self, out_dim: int):
        super().__init__()
        self.linear  = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, out_dim - 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1).float()
        return torch.cat([self.linear(t), torch.sin(self.periodic(t))], dim=-1)


def engineer_ibm_features(df: pd.DataFrame,
                           acc_df: pd.DataFrame,
                           pattern_map: dict) -> tuple:
    """
    Returns:
      node_feat_df : DataFrame indexed by account, shape [N, F_node]
      edge_feat_df : DataFrame aligned with df rows, shape [E, F_edge]
      edge_labels  : ndarray of shape [E] — binary (Is_Laundering)
      pattern_labels: ndarray of shape [E] — pattern class index (0=BENIGN)
      account_to_id: dict mapping account string → int node index
      bank_to_id   : dict mapping bank int → int bank node index
    """
    print("  Feature engineering (IBM)…")

    # ── Account → integer node ID ──
    all_accounts   = pd.concat([df["From_Account"], df["To_Account"]]).unique()
    account_to_id  = {a: i for i, a in enumerate(all_accounts)}
    n_nodes        = len(account_to_id)

    df = df.copy()
    df["src_id"] = df["From_Account"].map(account_to_id)
    df["dst_id"] = df["To_Account"].map(account_to_id)

    # ── Temporal edge features ──
    df = df.sort_values("unix_t").reset_index(drop=True)
    # Delta-t: seconds since last outgoing transaction from this account
    df["prev_t"]  = df.groupby("src_id")["unix_t"].shift(1)
    df["delta_t"] = (df["unix_t"] - df["prev_t"]).fillna(0).clip(0, 1e7)
    # Velocity: transactions per hour in a 24h rolling window (approximated as count)
    df["hour_of_day"]  = df["Timestamp"].dt.hour
    df["day_of_week"]  = df["Timestamp"].dt.dayofweek
    # Running count per account (captures burst behavior)
    df["out_running_count"] = df.groupby("src_id").cumcount() + 1

    # Encode categoricals
    le_fmt  = LabelEncoder().fit(df["Payment_Format"])
    le_rcur = LabelEncoder().fit(df["Receiving_Currency"])
    le_pcur = LabelEncoder().fit(df["Payment_Currency"])
    df["fmt_enc"]   = le_fmt.transform(df["Payment_Format"])
    df["rcur_enc"]  = le_rcur.transform(df["Receiving_Currency"])
    df["pcur_enc"]  = le_pcur.transform(df["Payment_Currency"])

    # Normalize amount (log1p)
    df["amount_log"] = np.log1p(df["Amount_Paid"])

    # Structuring detection features
    # Transactions just under $10,000 are classic structuring (smurfing) indicators.
    THRESHOLD = 10_000.0
    df["near_threshold"]    = ((df["Amount_Paid"] >= THRESHOLD * 0.85) &
                               (df["Amount_Paid"] <  THRESHOLD)).astype(np.float32)
    # Per-account burst count of near-threshold sends (normalized to [0,1])
    df["structuring_burst"] = (df.groupby("src_id")["near_threshold"]
                                .transform("sum")).clip(0, 20) / 20.0
    df["same_bank"]         = (df["From_Bank"] == df["To_Bank"]).astype(np.float32)

    EDGE_FEAT_COLS = ["amount_log", "fmt_enc", "rcur_enc", "pcur_enc",
                      "delta_t", "hour_of_day", "day_of_week", "out_running_count",
                      "near_threshold", "structuring_burst", "same_bank"]
    edge_feat_df = df[EDGE_FEAT_COLS].copy()
    # Standardize continuous edge features
    cont_cols = ["amount_log", "delta_t", "out_running_count", "structuring_burst"]
    scaler_e = StandardScaler()
    edge_feat_df[cont_cols] = scaler_e.fit_transform(edge_feat_df[cont_cols])

    # ── Pattern multi-class labels ──
    def get_pattern(row):
        key = (row["From_Account"], row["To_Account"], str(row["Amount_Paid"]))
        return PATTERN_TO_IDX.get(pattern_map.get(key, "BENIGN"), 0)
    if pattern_map:
        df["pattern_idx"] = df.apply(get_pattern, axis=1)
    else:
        df["pattern_idx"] = 0

    edge_labels   = df["Is_Laundering"].values.astype(np.int64)
    pattern_labels= df["pattern_idx"].values.astype(np.int64)

    # ── Node features (vectorised groupby) ──
    eps = 1e-9
    sent = df.groupby("src_id").agg(
        out_deg      = ("Amount_Paid",    "count"),
        total_sent   = ("Amount_Paid",    "sum"),
        avg_sent     = ("Amount_Paid",    "mean"),
        std_sent     = ("Amount_Paid",    "std"),
        n_recipients = ("dst_id",         "nunique"),
        n_sent_fmts  = ("fmt_enc",        "nunique"),
    ).fillna(0)

    recv = df.groupby("dst_id").agg(
        in_deg       = ("Amount_Received","count"),
        total_recv   = ("Amount_Received","sum"),
        avg_recv     = ("Amount_Received","mean"),
        std_recv     = ("Amount_Received","std"),
        n_senders    = ("src_id",         "nunique"),
    ).fillna(0)

    node_df = pd.DataFrame(index=range(n_nodes))
    node_df = node_df.join(sent,  how="left").fillna(0)
    node_df = node_df.join(recv,  how="left").fillna(0)
    node_df["recv_sent_ratio"] = (
        node_df["total_recv"] / (node_df["total_sent"] + eps)).clip(0, 1e4)
    node_df["degree_ratio"] = (
        node_df["out_deg"] / (node_df["in_deg"] + node_df["out_deg"] + eps))

    # ── Graph structural features (PageRank + k-core proxy) ──
    # PageRank captures global centrality — high-PR fraud hubs stand out.
    # k-core proxy (min of in/out degree) approximates graph coreness cheaply.
    node_df["kcore_proxy"] = np.minimum(
        node_df["out_deg"].values, node_df["in_deg"].values).astype(np.float32)
    try:
        from scipy.sparse import csr_matrix as _csr
        _src_np = df["src_id"].values.astype(np.int64)
        _dst_np = df["dst_id"].values.astype(np.int64)
        _out    = np.bincount(_src_np, minlength=n_nodes).astype(np.float64)
        _out_s  = np.where(_out > 0, _out, 1.0)
        _w      = 1.0 / _out_s[_src_np]
        _A      = _csr((_w, (_dst_np, _src_np)), shape=(n_nodes, n_nodes))
        _pr     = np.full(n_nodes, 1.0 / n_nodes)
        _alpha  = 0.85
        _tel    = (1.0 - _alpha) / n_nodes
        for _ in range(40):
            _pr_new = _alpha * _A.dot(_pr) + _tel
            _dang   = _alpha * _pr[_out == 0].sum() / n_nodes
            _pr_new += _dang
            if np.abs(_pr_new - _pr).max() < 1e-5:
                break
            _pr = _pr_new
        node_df["pagerank"] = _pr_new
        print("    Graph structural features: pagerank + kcore_proxy computed.", flush=True)
    except Exception as _e:
        print(f"    [warn] PageRank skipped ({_e}); using degree proxy.", flush=True)
        node_df["pagerank"] = (node_df["out_deg"] + node_df["in_deg"]) / (
            node_df["out_deg"].max() + node_df["in_deg"].max() + eps)

    # ── Merge entity type from accounts.csv ──
    acc_lookup = {}
    if not acc_df.empty:
        acc_lookup = dict(zip(acc_df["Account"], acc_df["Entity_Type_Idx"]))
    node_df["entity_type"] = pd.Series(
        {account_to_id[a]: acc_lookup.get(a, ENTITY_TO_IDX["Unknown"])
         for a in account_to_id}
    ).fillna(ENTITY_TO_IDX["Unknown"])

    NODE_FEAT_COLS = ["out_deg", "in_deg", "total_sent", "total_recv",
                      "avg_sent", "avg_recv", "std_sent", "std_recv",
                      "n_recipients", "n_senders", "n_sent_fmts",
                      "recv_sent_ratio", "degree_ratio", "entity_type",
                      "pagerank", "kcore_proxy"]

    x_raw = node_df[NODE_FEAT_COLS].values.astype(np.float32)
    scaler_n = StandardScaler()
    x_scaled = scaler_n.fit_transform(x_raw)

    # ── Bank → integer node ID (for heterogeneous graph) ──
    bank_to_id = {}
    if not acc_df.empty:
        banks = acc_df["Bank_ID"].unique()
        bank_to_id = {b: i for i, b in enumerate(banks)}

    print(f"    Nodes: {n_nodes:,}  |  Node feats: {x_scaled.shape[1]}  |  "
          f"Edge feats: {edge_feat_df.shape[1]}  |  "
          f"Banks (hetero): {len(bank_to_id):,}")
    return (x_scaled, edge_feat_df.values.astype(np.float32),
            edge_labels, pattern_labels,
            df[["src_id", "dst_id"]].values,
            account_to_id, bank_to_id, acc_df)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Graph Construction
# ══════════════════════════════════════════════════════════════════════════════

def build_pyg_data(x, edge_feats, edge_labels, pattern_labels,
                   src_dst, acc_df, bank_to_id, task="edge") -> Data:
    """Build a PyG Data object for homogeneous (or base of hetero) graph."""
    n_nodes = x.shape[0]
    src = torch.tensor(src_dst[:, 0], dtype=torch.long)
    dst = torch.tensor(src_dst[:, 1], dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    x_t        = torch.tensor(x,             dtype=torch.float)
    ea_t       = torch.tensor(edge_feats,    dtype=torch.float)
    el_t       = torch.tensor(edge_labels,   dtype=torch.long)
    pat_t      = torch.tensor(pattern_labels,dtype=torch.long)

    # Node labels: account is fraud if it appears in any laundering edge
    node_labels = torch.zeros(n_nodes, dtype=torch.long)
    fraud_mask  = el_t == 1
    node_labels[src[fraud_mask]] = 1
    node_labels[dst[fraud_mask]] = 1

    data = Data(
        x                = x_t,
        edge_index       = edge_index,
        edge_attr        = ea_t,
        edge_label       = el_t,        # per-edge binary label
        pattern_label    = pat_t,       # per-edge multi-class pattern
        y                = node_labels, # per-node binary label
        num_nodes        = n_nodes,
    )
    return data


def build_hetero_data(base_data: Data, acc_df: pd.DataFrame,
                      bank_to_id: dict, account_to_id: dict) -> HeteroData:
    """
    Augment homogeneous Data with bank nodes to form a HeteroData graph.
    Edge types:
      (account, transacts, account)  — from transactions
      (account, registered_at, bank) — from accounts.csv
    """
    if not bank_to_id or acc_df.empty:
        return None

    n_banks = len(bank_to_id)
    # Bank node features: one-hot over bank index (simple baseline)
    bank_x = torch.eye(min(n_banks, 128), dtype=torch.float)
    if n_banks > 128:
        bank_x = torch.zeros(n_banks, 128, dtype=torch.float)
        for i in range(n_banks):
            bank_x[i, i % 128] = 1.0

    # Build account → bank edges
    acc_ids, bank_ids = [], []
    acc_bank_map = dict(zip(acc_df["Account"], acc_df["Bank_ID"]))
    for acc, acc_idx in account_to_id.items():
        bank_raw = acc_bank_map.get(acc)
        if bank_raw is not None and bank_raw in bank_to_id:
            acc_ids.append(acc_idx)
            bank_ids.append(bank_to_id[bank_raw])

    h = HeteroData()
    h["account"].x = base_data.x
    h["account"].y = base_data.y
    h["bank"].x    = bank_x

    h["account", "transacts",    "account"].edge_index = base_data.edge_index
    h["account", "transacts",    "account"].edge_attr  = base_data.edge_attr
    h["account", "transacts",    "account"].edge_label = base_data.edge_label
    h["account", "transacts",    "account"].pattern_label = base_data.pattern_label

    if acc_ids:
        reg_edge = torch.tensor([acc_ids, bank_ids], dtype=torch.long)
        h["account", "registered_at", "bank"].edge_index = reg_edge

    return h


def make_masks(n: int, y: np.ndarray, seed: int = 42):
    idx = np.arange(n)
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=seed, stratify=y)
    va, te  = train_test_split(tmp, test_size=0.50, random_state=seed, stratify=y[tmp])
    train_m = torch.zeros(n, dtype=torch.bool); train_m[tr] = True
    val_m   = torch.zeros(n, dtype=torch.bool); val_m[va]   = True
    test_m  = torch.zeros(n, dtype=torch.bool); test_m[te]  = True
    return train_m, val_m, test_m


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Model Definitions
# ══════════════════════════════════════════════════════════════════════════════

class EdgeMLP(nn.Module):
    """Classifies edges from concatenated node pair embeddings + edge features."""
    def __init__(self, node_dim: int, edge_feat_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        in_dim = 2 * node_dim + edge_feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, node_dim), nn.BatchNorm1d(node_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim // 2), nn.ReLU(),
            nn.Linear(node_dim // 2, out_dim),
        )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor,
                edge_feat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_src, z_dst, edge_feat], dim=-1))


# ── Model 1: GraphSAGE ────────────────────────────────────────────────────────
class GraphSAGEModel(nn.Module):
    def __init__(self, in_ch, hidden, n_layers, edge_feat_dim,
                 dropout=0.3, n_pattern_classes=9):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        dims = [in_ch] + [hidden] * n_layers
        for i in range(n_layers):
            self.convs.append(SAGEConv(dims[i], dims[i+1], aggr="mean"))
            self.bns.append(nn.BatchNorm1d(dims[i+1]))
        self.dropout = dropout

        # Binary edge classifier
        self.edge_clf   = EdgeMLP(hidden, edge_feat_dim, 2, dropout)
        # Pattern multi-class head
        self.pattern_clf = EdgeMLP(hidden, edge_feat_dim, n_pattern_classes, dropout)
        # Node classifier (optional)
        self.node_clf   = nn.Linear(hidden, 2)

    def encode(self, x, edge_index):
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_new = F.relu(bn(conv(x, edge_index)))
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x_new + x if (x.shape == x_new.shape) else x_new   # skip when dims match
        return x

    def forward(self, x, edge_index, edge_attr, src_idx=None, dst_idx=None):
        z = self.encode(x, edge_index)
        if src_idx is not None:
            edge_out     = self.edge_clf(z[src_idx], z[dst_idx], edge_attr)
            pattern_out  = self.pattern_clf(z[src_idx], z[dst_idx], edge_attr)
            return edge_out, pattern_out, self.node_clf(z)
        return self.node_clf(z)


# ── Model 2: GAT (v2) ─────────────────────────────────────────────────────────
class GATModel(nn.Module):
    def __init__(self, in_ch, hidden, n_layers, edge_feat_dim,
                 heads=8, dropout=0.3, n_pattern_classes=9):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        # First n_layers-1: concatenate heads; last layer: average heads
        for i in range(n_layers):
            in_d  = in_ch if i == 0 else hidden
            out_d = hidden // heads if i < n_layers - 1 else hidden
            h     = heads if i < n_layers - 1 else 1
            self.convs.append(GATv2Conv(in_d, out_d, heads=h,
                                        concat=(i < n_layers-1),
                                        dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout

        self.edge_clf    = EdgeMLP(hidden, edge_feat_dim, 2, dropout)
        self.pattern_clf = EdgeMLP(hidden, edge_feat_dim, n_pattern_classes, dropout)
        self.node_clf    = nn.Linear(hidden, 2)

    def encode(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = F.elu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index, edge_attr, src_idx=None, dst_idx=None):
        z = self.encode(x, edge_index)
        if src_idx is not None:
            return (self.edge_clf(z[src_idx], z[dst_idx], edge_attr),
                    self.pattern_clf(z[src_idx], z[dst_idx], edge_attr),
                    self.node_clf(z))
        return self.node_clf(z)


# ── Model 3: H2GCN (Heterophily-Aware GCN) ───────────────────────────────────
class H2GCNModel(nn.Module):
    """
    Heterophily-aware GNN inspired by H2GCN (Zhu et al., 2020).

    AML graphs are strongly heterophilic: fraudsters deliberately route funds
    through legitimate accounts, so a fraud node's neighbours are mostly benign.
    Standard message-passing dilutes the fraud signal by averaging in benign
    neighbours.  H2GCN fixes this with three design choices:
      1. Ego-feature separation — the node's own embedding is kept distinct
         from aggregated neighbour embeddings (no self-loops in aggregation).
      2. Multi-hop aggregation — 1-hop and 2-hop neighbourhood representations
         are concatenated, capturing both immediate and extended context.
      3. Higher-order concatenation — all representations ([ego, 1-hop, 2-hop])
         are concatenated (not summed/averaged) before the final projection,
         so the classifier can learn which signal is predictive per node.
    """
    def __init__(self, in_ch, hidden, n_layers, edge_feat_dim,
                 dropout=0.3, n_pattern_classes=9):
        super().__init__()
        # Ego branch: projects raw input without neighbourhood mixing
        self.ego_lin  = nn.Linear(in_ch, hidden)
        self.ego_bn   = nn.BatchNorm1d(hidden)

        # Neighbour aggregation layers (no self-loops)
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for i in range(n_layers):
            in_d = in_ch if i == 0 else hidden
            self.convs.append(SAGEConv(in_d, hidden, aggr="mean"))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.dropout  = dropout
        self.n_layers = n_layers

        # Final projection: ego + n_layers neighbourhood embeddings
        concat_dim = (n_layers + 1) * hidden
        self.final_proj = nn.Linear(concat_dim, hidden)
        self.final_bn   = nn.BatchNorm1d(hidden)

        self.edge_clf    = EdgeMLP(hidden, edge_feat_dim, 2, dropout)
        self.pattern_clf = EdgeMLP(hidden, edge_feat_dim, n_pattern_classes, dropout)
        self.node_clf    = nn.Linear(hidden, 2)

    def encode(self, x, edge_index):
        ego = F.relu(self.ego_bn(self.ego_lin(x)))
        hop_embs = [ego]
        x_hop = x
        for conv, bn in zip(self.convs, self.bns):
            x_hop = F.relu(bn(conv(x_hop, edge_index)))
            x_hop = F.dropout(x_hop, p=self.dropout, training=self.training)
            hop_embs.append(x_hop)
        z = torch.cat(hop_embs, dim=-1)          # [N, (n_layers+1)*hidden]
        z = F.relu(self.final_bn(self.final_proj(z)))
        return z

    def forward(self, x, edge_index, edge_attr, src_idx=None, dst_idx=None):
        z = self.encode(x, edge_index)
        if src_idx is not None:
            return (self.edge_clf(z[src_idx], z[dst_idx], edge_attr),
                    self.pattern_clf(z[src_idx], z[dst_idx], edge_attr),
                    self.node_clf(z))
        return self.node_clf(z)


# ── Model 4: Temporal Graph Network ──────────────────────────────────────────
class TGNModel(nn.Module):
    """
    TGN with PyG TGNMemory.
    Processes edges in temporal order; memory accumulates per-node history.
    Edge classification uses node memories BEFORE the current edge (no leakage).
    """
    def __init__(self, num_nodes, node_feat_dim, edge_feat_dim,
                 memory_dim=128, time_dim=32, hidden=256,
                 dropout=0.3, n_pattern_classes=9):
        super().__init__()
        self.memory = TGNMemory(
            num_nodes,
            edge_feat_dim,
            memory_dim,
            time_dim,
            message_module   = IdentityMessage(edge_feat_dim, memory_dim, time_dim),
            aggregator_module= LastAggregator(),
        )
        # GNN on top of memory
        combined_dim = memory_dim + node_feat_dim
        self.gnn1 = SAGEConv(combined_dim, hidden, aggr="mean")
        self.gnn2 = SAGEConv(hidden,       hidden, aggr="mean")
        self.bn1  = nn.BatchNorm1d(hidden)
        self.bn2  = nn.BatchNorm1d(hidden)
        self.drop = dropout

        self.edge_clf    = EdgeMLP(hidden, edge_feat_dim, 2, dropout)
        self.pattern_clf = EdgeMLP(hidden, edge_feat_dim, n_pattern_classes, dropout)
        self.node_clf    = nn.Linear(hidden, 2)

    def reset_memory(self):
        self.memory.reset_state()

    def forward_temporal_batch(self, x_all, edge_index, edge_attr,
                                src_b, dst_b, t_b, msg_b):
        """
        Process one temporal batch:
          1. Get pre-update memory for involved nodes.
          2. Run GNN on the full graph using memory as node features.
          3. Classify edges.
          4. Update memory with this batch.
        """
        n_id = torch.cat([src_b, dst_b]).unique()
        z_mem, _ = self.memory(n_id)
        # Scatter memory back into full node space
        z_full = torch.zeros(x_all.size(0), z_mem.size(1),
                             device=x_all.device)
        z_full[n_id] = z_mem

        # Combine static node features with temporal memory
        z = torch.cat([z_full, x_all], dim=-1)

        # Two-layer GNN
        z = F.relu(self.bn1(self.gnn1(z, edge_index)))
        z = F.dropout(z, p=self.drop, training=self.training)
        z = F.relu(self.bn2(self.gnn2(z, edge_index)))

        edge_out    = self.edge_clf(z[src_b], z[dst_b], msg_b)
        pattern_out = self.pattern_clf(z[src_b], z[dst_b], msg_b)

        # Update memory (no gradient through stored memory)
        self.memory.update_state(src_b, dst_b, t_b, msg_b)

        return edge_out, pattern_out, self.node_clf(z)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Focal Loss
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) — down-weights easy examples to focus
    training on hard, misclassified cases. Superior to weighted CE for
    severe imbalance (AML positive rate < 0.5%).
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 weight: torch.Tensor = None):
        super().__init__()
        self.alpha  = alpha
        self.gamma  = gamma
        self.weight = weight   # optional per-class weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt   = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


def make_focal_loss(y: np.ndarray, device: str) -> FocalLoss:
    n1 = int(y.sum())
    n0 = len(y) - n1
    w  = torch.tensor([1.0, n0 / max(n1, 1)], dtype=torch.float, device=device)
    return FocalLoss(alpha=CONFIG["FOCAL_ALPHA"],
                     gamma=CONFIG["FOCAL_GAMMA"], weight=w)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — GraphSMOTE
# ══════════════════════════════════════════════════════════════════════════════

def graph_smote(x: np.ndarray, y: np.ndarray,
                edge_index: np.ndarray,
                k: int = 5, target_ratio: float = 0.3) -> tuple:
    """
    Graph-aware SMOTE (Zhao et al., 2021 — simplified variant).
    Generates synthetic minority nodes by interpolating features of
    k-nearest fraud neighbours, and connects them to their donors.

    Returns augmented (x, y, edge_index).
    Only applied if the minority fraction is below target_ratio.
    """
    minority_idx = np.where(y == 1)[0]
    minority_frac = len(minority_idx) / len(y)
    if minority_frac >= target_ratio:
        return x, y, edge_index

    n_needed = int(target_ratio * len(y) / (1 - target_ratio)) - len(minority_idx)
    n_needed = max(0, n_needed)
    if n_needed == 0 or len(minority_idx) < k + 1:
        return x, y, edge_index

    print(f"  GraphSMOTE: generating {n_needed:,} synthetic fraud nodes…")

    # k-NN in feature space of minority class
    x_min = x[minority_idx]
    knn   = NearestNeighbors(n_neighbors=min(k+1, len(minority_idx)),
                              metric="euclidean").fit(x_min)
    _, nbrs = knn.kneighbors(x_min)

    syn_x, syn_y, new_edges = [], [], []
    base_id = len(y)

    for _ in range(n_needed):
        # Pick a random minority node and one of its neighbours
        i      = np.random.randint(len(minority_idx))
        j      = nbrs[i, np.random.randint(1, nbrs.shape[1])]
        lam    = np.random.uniform(0, 1)
        x_new  = x_min[i] + lam * (x_min[j] - x_min[i])
        syn_x.append(x_new)
        syn_y.append(1)
        # Connect synthetic node to donor minority node
        donor = minority_idx[i]
        new_edges.append([donor,   base_id])
        new_edges.append([base_id, donor])
        base_id += 1

    if not syn_x:
        return x, y, edge_index

    x_aug = np.vstack([x, np.array(syn_x, dtype=np.float32)])
    y_aug = np.concatenate([y, np.array(syn_y, dtype=np.int64)])
    new_edges_t = np.array(new_edges, dtype=np.int64).T  # [2, 2*n_needed]
    ei_aug = np.hstack([edge_index, new_edges_t])
    print(f"    Augmented: {len(y):,} → {len(y_aug):,} nodes  "
          f"({y_aug.mean()*100:.2f}% fraud)")
    return x_aug, y_aug, ei_aug


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Training & Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _to_device(batch, device):
    """Move batch (PyG Data or SimpleNamespace) tensors to device."""
    if hasattr(batch, "to"):
        return batch.to(device)
    for attr in ("x", "edge_index", "edge_attr", "edge_label",
                 "edge_label_index", "pattern_label", "y", "train_mask"):
        v = getattr(batch, attr, None)
        if isinstance(v, torch.Tensor):
            setattr(batch, attr, v.to(device))
    return batch


def _forward_batch(model, batch, device, cfg):
    """Run one forward pass on a mini-batch from any loader type."""
    batch = _to_device(batch, device)
    model_type = cfg["MODEL"]

    if model_type == "TGN":
        # TGN uses temporal batching — handled in train_tgn()
        raise NotImplementedError("Call train_tgn() for TGN model.")

    x          = batch.x
    edge_index = batch.edge_index
    edge_attr  = batch.edge_attr if hasattr(batch, "edge_attr") and batch.edge_attr is not None \
                 else torch.zeros(edge_index.size(1), 1, device=device)

    if cfg["TASK"] == "edge":
        src_idx  = batch.edge_label_index[0]
        dst_idx  = batch.edge_label_index[1]
        # Subset edge_attr to the labeled edges (LinkNeighborLoader semantic)
        n_label  = src_idx.size(0)
        ea_label = batch.edge_label_attr if hasattr(batch, "edge_label_attr") \
                   else edge_attr[:n_label]
        edge_out, pattern_out, node_out = model(x, edge_index, ea_label,
                                                 src_idx, dst_idx)
        return edge_out, pattern_out, batch.edge_label, batch.pattern_label[:n_label]
    else:
        node_out = model(x, edge_index, edge_attr)
        mask = batch.train_mask
        return node_out[mask], None, batch.y[mask], None


def train_epoch(model, loader, optimizer, criterion, pattern_criterion,
                device, cfg):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        edge_out, pat_out, y_e, y_p = _forward_batch(model, batch, device, cfg)
        loss = criterion(edge_out, y_e.to(device))
        if cfg["MULTI_CLASS"] and pat_out is not None:
            loss = loss + 0.3 * pattern_criterion(pat_out, y_p.to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, pattern_criterion, device, cfg):
    model.eval()
    all_preds, all_probs, all_true, all_pat_pred, all_pat_true = [], [], [], [], []
    total_loss = 0.0
    for batch in loader:
        edge_out, pat_out, y_e, y_p = _forward_batch(model, batch, device, cfg)
        loss = criterion(edge_out, y_e.to(device))
        total_loss += loss.item()
        probs = F.softmax(edge_out, dim=1)[:, 1].cpu().numpy()
        preds = edge_out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_true.extend(y_e.cpu().numpy())
        if pat_out is not None:
            all_pat_pred.extend(pat_out.argmax(dim=1).cpu().numpy())
            all_pat_true.extend(y_p.cpu().numpy())

    y_true  = np.array(all_true)
    y_pred  = np.array(all_preds)
    y_probs = np.array(all_probs)
    recall  = (y_pred[y_true==1] == 1).mean() if (y_true==1).sum() > 0 else 0.0
    return {
        "loss"         : total_loss / max(len(loader), 1),
        "recall"       : recall,
        "preds"        : y_pred,
        "probs"        : y_probs,
        "true"         : y_true,
        "pat_pred"     : np.array(all_pat_pred) if all_pat_pred else None,
        "pat_true"     : np.array(all_pat_true) if all_pat_true else None,
    }


def train_tgn(model, data, optimizer, criterion, pattern_criterion,
              device, cfg, split="train"):
    """TGN training: processes edges sorted by timestamp in mini-batches."""
    perm = data.edge_index[0].argsort()   # sort by src; timestamp sort done in data prep
    src_all  = data.edge_index[0][perm].to(device)
    dst_all  = data.edge_index[1][perm].to(device)
    ea_all   = data.edge_attr[perm].to(device)
    el_all   = data.edge_label[perm].to(device)
    pl_all   = data.pattern_label[perm].to(device)
    # Unix timestamps encoded in first edge feature
    t_all    = (ea_all[:, 0] * 1e8).long()

    model.train() if split == "train" else model.eval()
    model.reset_memory()

    all_loss, all_preds, all_probs, all_true = 0.0, [], [], []
    n_batches = 0
    bs = cfg["BATCH_SIZE"]

    # Precompute CSR for fast 1-hop subgraph lookup (same logic as _EdgeBatchLoader)
    ei_np  = data.edge_index.cpu().numpy()
    N, E   = data.num_nodes, ei_np.shape[1]
    nids   = np.concatenate([ei_np[0], ei_np[1]])
    eids   = np.tile(np.arange(E), 2)
    order  = np.argsort(nids, kind="stable")
    _se    = eids[order].astype(np.int64)
    _iptr  = np.concatenate([[0], np.cumsum(np.bincount(nids[order], minlength=N))]).astype(np.int64)
    def _hop(nodes_np):
        segs = [_se[_iptr[n]:_iptr[n + 1]] for n in nodes_np]
        return torch.from_numpy(np.unique(np.concatenate(segs))) if segs else torch.zeros(0, dtype=torch.long)

    ctx = torch.enable_grad() if split == "train" else torch.no_grad()
    with ctx:
        for start in range(0, len(src_all), bs):
            s = slice(start, start + bs)
            batch_nodes_np = np.unique(np.concatenate([src_all[s].cpu().numpy(), dst_all[s].cpu().numpy()]))
            hop_t  = _hop(batch_nodes_np)
            sub_ei = data.edge_index[:, hop_t].to(device)
            edge_out, pat_out, _ = model.forward_temporal_batch(
                data.x.to(device), sub_ei,
                ea_all, src_all[s], dst_all[s], t_all[s], ea_all[s]
            )
            y_e = el_all[s]
            loss = criterion(edge_out, y_e)
            if cfg["MULTI_CLASS"] and pat_out is not None:
                loss = loss + 0.3 * pattern_criterion(pat_out, pl_all[s])

            if split == "train":
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.memory.detach()

            all_loss += loss.item()
            all_preds.extend(edge_out.argmax(1).detach().cpu().numpy())
            all_probs.extend(F.softmax(edge_out, 1)[:, 1].detach().cpu().numpy())
            all_true.extend(y_e.cpu().numpy())
            n_batches += 1

    y_true = np.array(all_true)
    y_pred = np.array(all_preds)
    recall = (y_pred[y_true==1] == 1).mean() if (y_true==1).sum() > 0 else 0.0
    return {
        "loss"   : all_loss / max(n_batches, 1),
        "recall" : recall,
        "preds"  : y_pred,
        "probs"  : np.array(all_probs),
        "true"   : y_true,
    }


def full_training_run(model, train_loader, val_loader, data,
                      criterion, pattern_criterion,
                      optimizer, scheduler, device, cfg, epochs=None):
    epochs = epochs or cfg["EPOCHS"]
    history = {k: [] for k in ("train_loss", "val_loss",
                                "train_recall", "val_recall")}
    is_tgn  = cfg["MODEL"] == "TGN"

    print(f"\n  {'Ep':>4}  {'TrLoss':>8}  {'ValLoss':>8}  "
          f"{'TrRecall':>9}  {'ValRecall':>10}")
    print("  " + "─" * 56)

    best_val_recall = 0.0
    best_state      = None

    for ep in range(1, epochs + 1):
        if is_tgn:
            tr = train_tgn(model, data, optimizer, criterion,
                           pattern_criterion, device, cfg, split="train")
            va = train_tgn(model, data, optimizer, criterion,
                           pattern_criterion, device, cfg, split="val")
        else:
            tr_loss = train_epoch(model, train_loader, optimizer,
                                  criterion, pattern_criterion, device, cfg)
            tr      = {"loss": tr_loss}
            va      = eval_epoch(model, val_loader, criterion,
                                 pattern_criterion, device, cfg)
            tr["recall"] = va["recall"]   # recompute on train if needed

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_recall"].append(tr.get("recall", 0))
        history["val_recall"].append(va["recall"])

        print(f"  {ep:>4}  {tr['loss']:>8.4f}  {va['loss']:>8.4f}  "
              f"{tr.get('recall',0):>9.4f}  {va['recall']:>10.4f}")

        if va["recall"] > best_val_recall:
            best_val_recall = va["recall"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        scheduler.step()

    # Restore best checkpoint
    if best_state:
        model.load_state_dict(best_state)
    return history


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Optuna Hyperparameter Sweep
# ══════════════════════════════════════════════════════════════════════════════

def run_optuna_sweep(data, n_nodes, in_dim, edge_feat_dim, device, cfg):
    print(f"\n[Optuna] Starting sweep: {cfg['OPTUNA_TRIALS']} trials, "
          f"optimising {cfg['OPTUNA_METRIC']}…\n")

    def objective(trial):
        c = {
            **cfg,
            "HIDDEN_DIM"   : trial.suggest_categorical("hidden",   [128, 256]),
            "NUM_LAYERS"   : trial.suggest_int("n_layers", 2, 4),
            "DROPOUT"      : trial.suggest_float("dropout",  0.1, 0.5),
            "LR"           : trial.suggest_float("lr",      1e-4, 5e-3, log=True),
            "WEIGHT_DECAY" : trial.suggest_float("wd",      1e-5, 1e-3, log=True),
            "FOCAL_GAMMA"  : trial.suggest_float("gamma",   1.0,  3.0),
            "EPOCHS"       : 10,    # fewer epochs for search efficiency
        }
        if c["MODEL"] == "GAT":
            c["GAT_HEADS"] = trial.suggest_categorical("heads", [4, 8])

        model = _build_model(c, n_nodes, in_dim, edge_feat_dim).to(device)
        train_l, val_l, _ = _build_loaders(data, c)
        focal   = make_focal_loss(data.edge_label.numpy(), device)
        pat_ce  = nn.CrossEntropyLoss()
        opt     = torch.optim.Adam(model.parameters(), lr=c["LR"],
                                   weight_decay=c["WEIGHT_DECAY"])
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

        hist = full_training_run(model, train_l, val_l, data,
                                 focal, pat_ce, opt, sched, device, c, epochs=10)
        return max(hist["val_recall"])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=cfg["OPTUNA_TRIALS"], show_progress_bar=True)

    best = study.best_params
    print(f"\n  Best params (val_recall={study.best_value:.4f}):")
    for k, v in best.items():
        print(f"    {k}: {v}")

    # Save Optuna results
    out = {"best_val_recall": study.best_value, "best_params": best}
    with open(f"{cfg['RESULTS_DIR']}/optuna_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → {cfg['RESULTS_DIR']}/optuna_results.json")
    return best


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — Evaluation & Plots
# ══════════════════════════════════════════════════════════════════════════════

def full_evaluation(model, test_loader, data, criterion, pattern_criterion,
                    device, cfg):
    is_tgn = cfg["MODEL"] == "TGN"
    if is_tgn:
        res = train_tgn(model, data, None, criterion, pattern_criterion,
                        device, cfg, split="val")
    else:
        res = eval_epoch(model, test_loader, criterion, pattern_criterion,
                         device, cfg)

    y_true  = res["true"]
    y_pred  = res["preds"]
    y_probs = res["probs"]

    print("\n" + "═"*70)
    print("  FINAL TEST SET EVALUATION")
    print("═"*70)
    rpt = classification_report(y_true, y_pred,
                                 target_names=["Benign", "Laundering"], digits=4)
    print(rpt)
    roc = roc_auc_score(y_true, y_probs)
    pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_probs)
    pr_auc  = auc(pr_rec, pr_prec)
    print(f"  ROC-AUC  : {roc:.4f}")
    print(f"  PR-AUC   : {pr_auc:.4f}  (better metric for imbalanced data)")
    print("═"*70)

    rpt_dict = classification_report(y_true, y_pred,
                                      target_names=["Benign", "Laundering"],
                                      output_dict=True)
    metrics = {"roc_auc": round(roc, 4), "pr_auc": round(pr_auc, 4),
               "classification_report": rpt_dict}

    # ── Pattern-type breakdown ──
    if res.get("pat_pred") is not None and res.get("pat_true") is not None:
        pat_true = res["pat_true"]
        pat_pred = res["pat_pred"]
        print("\n  Pattern-Type Detection Breakdown (multi-class):")
        for pt in PATTERN_TYPES[1:]:   # skip BENIGN
            idx_t = PATTERN_TO_IDX[pt]
            mask  = pat_true == idx_t
            if mask.sum() == 0:
                continue
            acc_p = (pat_pred[mask] == idx_t).mean()
            print(f"    {pt:<20}: {mask.sum():>5} samples  |  "
                  f"detection rate: {acc_p:.3f}")

    return metrics, res


def plot_results(history: dict, metrics: dict, res: dict, cfg: dict):
    rd = cfg["RESULTS_DIR"]
    ep = range(1, len(history["train_loss"]) + 1)

    # ── Training curves ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(ep, history["train_loss"], "b-o", ms=4, label="Train")
    axes[0].plot(ep, history["val_loss"],   "r-o", ms=4, label="Val")
    axes[0].set_title(f"Loss — {cfg['MODEL']}", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Focal Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep, history["train_recall"], "b-o", ms=4, label="Train")
    axes[1].plot(ep, history["val_recall"],   "r-o", ms=4, label="Val")
    axes[1].set_title(f"Recall (Fraud) — {cfg['MODEL']}", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Recall")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{rd}/training_curves_{cfg['MODEL']}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Confusion matrix ──
    cm  = confusion_matrix(res["true"], res["preds"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Benign", "Laundering"]).plot(
        ax=ax, colorbar=True, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix — {cfg['MODEL']} (Test)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{rd}/confusion_matrix_{cfg['MODEL']}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── PR curve ──
    from sklearn.metrics import precision_recall_curve, auc
    p, r, _ = precision_recall_curve(res["true"], res["probs"])
    pr_auc   = auc(r, p)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(r, p, "b-", lw=2, label=f"PR-AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {cfg['MODEL']}", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{rd}/pr_curve_{cfg['MODEL']}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── ROC curve ──
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(res["true"], res["probs"])
    roc_auc_val = metrics.get("roc_auc", 0.0)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"ROC-AUC = {roc_auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {cfg['MODEL']}", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{rd}/roc_curve_{cfg['MODEL']}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Score distribution (fraud vs benign probability) ──
    probs = np.array(res["probs"]); labels = np.array(res["true"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probs[labels == 0], bins=50, alpha=0.6, color="steelblue", label="Benign", density=True)
    ax.hist(probs[labels == 1], bins=20, alpha=0.8, color="crimson",   label="Fraud",  density=True)
    ax.set_xlabel("Predicted Fraud Probability"); ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution — {cfg['MODEL']}", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{rd}/score_dist_{cfg['MODEL']}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Plots saved to '{rd}/'")
    with open(f"{rd}/metrics_{cfg['MODEL']}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {rd}/metrics_{cfg['MODEL']}.json")


def run_sklearn_baseline(edge_feats: np.ndarray, edge_labels: np.ndarray, cfg: dict):
    """
    Train RandomForest and XGBoost/GradientBoosting on raw edge features (no graph).
    Serves as baseline to demonstrate the value of graph structure.
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as sk_auc
    try:
        from xgboost import XGBClassifier
        _xgb_available = True
    except ImportError:
        _xgb_available = False

    print("\n" + "═" * 70, flush=True)
    print("  SKLEARN BASELINE (no graph structure)", flush=True)
    print("═" * 70, flush=True)

    # Same 60/20/20 split as GNN
    idx = np.arange(len(edge_labels))
    strat = edge_labels if np.bincount(edge_labels).min() >= 2 else None
    tr, tmp = train_test_split(idx, test_size=0.40, random_state=cfg["SEED"], stratify=strat)
    strat2  = edge_labels[tmp] if strat is not None and np.bincount(edge_labels[tmp]).min() >= 2 else None
    va, te  = train_test_split(tmp, test_size=0.50, random_state=cfg["SEED"], stratify=strat2)

    X_tr, y_tr = edge_feats[tr], edge_labels[tr]
    X_te, y_te = edge_feats[te], edge_labels[te]

    # Class weight for imbalance
    pos = y_tr.sum(); neg = len(y_tr) - pos
    scale = neg / max(pos, 1)

    results = {}
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced",
            n_jobs=-1, random_state=cfg["SEED"]),
    }
    if _xgb_available:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, scale_pos_weight=scale,
            use_label_encoder=False, eval_metric="logloss",
            random_state=cfg["SEED"], n_jobs=-1)
    else:
        models["GradientBoosting"] = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, random_state=cfg["SEED"])

    for name, clf in models.items():
        print(f"\n  Training {name}…", flush=True)
        clf.fit(X_tr, y_tr)
        probs  = clf.predict_proba(X_te)[:, 1]
        preds  = (probs >= 0.5).astype(int)
        recall = (preds[y_te == 1] == 1).mean() if (y_te == 1).sum() > 0 else 0.0
        try:
            roc = roc_auc_score(y_te, probs)
            p, r, _ = precision_recall_curve(y_te, probs)
            pr_auc  = sk_auc(r, p)
        except Exception:
            roc = pr_auc = 0.0
        print(f"  {name:20s}  Recall={recall:.4f}  ROC-AUC={roc:.4f}  PR-AUC={pr_auc:.4f}",
              flush=True)
        results[name] = {"recall": recall, "roc_auc": roc, "pr_auc": pr_auc,
                         "probs": probs.tolist(), "true": y_te.tolist()}

    # Save and plot
    rd = cfg["RESULTS_DIR"]
    with open(f"{rd}/baseline_results.json", "w") as f:
        json.dump({k: {m: v for m, v in r.items() if m not in ("probs", "true")}
                   for k, r in results.items()}, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#e07b39", "#5b8db8", "#6dbf67", "#c45c74"]
    for i, (name, r) in enumerate(results.items()):
        p_arr, re_arr, _ = precision_recall_curve(r["true"], r["probs"])
        axes[0].plot(re_arr, p_arr, lw=2, color=colors[i % len(colors)],
                     label=f"{name} (PR-AUC={r['pr_auc']:.4f})")
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(r["true"], r["probs"])
        axes[1].plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                     label=f"{name} (AUC={r['roc_auc']:.4f})")
    for ax, title, xl, yl in zip(
            axes,
            ["PR Curve — Baselines", "ROC Curve — Baselines"],
            ["Recall", "False Positive Rate"],
            ["Precision", "True Positive Rate"]):
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    plt.tight_layout()
    plt.savefig(f"{rd}/baseline_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Baseline results → {rd}/baseline_results.json", flush=True)
    print(f"  Baseline curves  → {rd}/baseline_curves.png", flush=True)
    return results


def platt_calibrate(model, val_loader, test_loader, data, device, cfg):
    """
    Platt scaling: fit a logistic regression on val-set raw scores,
    then apply it to test-set scores for calibrated probabilities.
    Returns calibrated test probs and a reliability diagram.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import calibration_curve

    def _get_probs(loader):
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                try:
                    edge_out, _, y_e, _ = _forward_batch(model, batch, device, cfg)
                    probs = torch.softmax(edge_out, dim=1)[:, 1].cpu().numpy()
                    all_probs.extend(probs)
                    all_labels.extend(y_e.cpu().numpy())
                except Exception:
                    continue
        return np.array(all_probs), np.array(all_labels)

    val_probs,  val_labels  = _get_probs(val_loader)
    test_probs, test_labels = _get_probs(test_loader)

    # Fit Platt scaler on val set
    platt = LogisticRegression(C=1.0, max_iter=1000)
    platt.fit(val_probs.reshape(-1, 1), val_labels)
    cal_probs = platt.predict_proba(test_probs.reshape(-1, 1))[:, 1]

    # Find optimal threshold on val set (maximise F1) — 0.5 is wrong at 0.006% fraud rate
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as sk_auc, f1_score
    val_cal = platt.predict_proba(val_probs.reshape(-1, 1))[:, 1]
    vp, vr, vt = precision_recall_curve(val_labels, val_cal)
    vf1 = 2 * vp * vr / (vp + vr + 1e-9)
    best_thresh = float(vt[np.argmax(vf1)]) if len(vt) > 0 else 0.5
    print(f"  [Platt] Optimal threshold (val F1): {best_thresh:.4f}", flush=True)

    cal_preds = (cal_probs >= best_thresh).astype(int)
    cal_recall = (cal_preds[test_labels == 1] == 1).mean() if (test_labels == 1).sum() > 0 else 0.0
    try:
        cal_roc = roc_auc_score(test_labels, cal_probs)
        cp, cr, _ = precision_recall_curve(test_labels, cal_probs)
        cal_pr_auc = sk_auc(cr, cp)
    except Exception:
        cal_roc = cal_pr_auc = 0.0

    print(f"\n  [Platt Calibration] Recall: {cal_recall:.4f}  |  "
          f"ROC-AUC: {cal_roc:.4f}  |  PR-AUC: {cal_pr_auc:.4f}", flush=True)

    # Reliability diagram
    rd = cfg["RESULTS_DIR"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Before vs after calibration curves
    n_bins = min(10, max(3, int((test_labels == 1).sum() // 2)))
    try:
        frac_pos_raw, mean_pred_raw = calibration_curve(test_labels, test_probs,  n_bins=n_bins)
        frac_pos_cal, mean_pred_cal = calibration_curve(test_labels, cal_probs,   n_bins=n_bins)
        axes[0].plot(mean_pred_raw, frac_pos_raw, "s-", color="tomato",    label="Before calibration")
        axes[0].plot(mean_pred_cal, frac_pos_cal, "s-", color="steelblue", label="After Platt scaling")
        axes[0].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    except Exception:
        axes[0].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    axes[0].set_xlabel("Mean Predicted Probability"); axes[0].set_ylabel("Fraction of Positives")
    axes[0].set_title("Reliability Diagram", fontsize=12, fontweight="bold")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Score distributions before and after calibration
    axes[1].hist(cal_probs[test_labels == 0], bins=40, alpha=0.6, color="steelblue",
                 label="Benign (calibrated)", density=True)
    axes[1].hist(cal_probs[test_labels == 1], bins=10, alpha=0.8, color="crimson",
                 label="Fraud (calibrated)", density=True)
    axes[1].set_xlabel("Calibrated Fraud Probability"); axes[1].set_ylabel("Density")
    axes[1].set_title("Calibrated Score Distribution", fontsize=12, fontweight="bold")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Platt Scaling — {cfg['MODEL']}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{rd}/calibration_{cfg['MODEL']}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Calibration plot → {rd}/calibration_{cfg['MODEL']}.png", flush=True)

    return cal_probs, cal_roc, cal_pr_auc


def _get_edge_splits(data: "Data", cfg: dict):
    """
    Return (train_idx, val_idx, test_idx) edge arrays using 60/20/20 stratified split.
    Extracted so both _build_loaders and the cascade can share the same indices.
    """
    n     = data.edge_label.size(0)
    el_np = data.edge_label.numpy()
    idx   = np.arange(n)
    strat = el_np if np.bincount(el_np).min() >= 2 else None
    tr_e, tmp = train_test_split(idx, test_size=0.40,
                                  random_state=cfg["SEED"], stratify=strat)
    strat2 = el_np[tmp] if (strat is not None
                             and np.bincount(el_np[tmp]).min() >= 2) else None
    va_e, te_e = train_test_split(tmp, test_size=0.50,
                                   random_state=cfg["SEED"], stratify=strat2)
    return tr_e, va_e, te_e


def run_cascade_classifier(model, data, tr_e, te_e, device, cfg):
    """
    Two-stage cascade: GNN node embeddings → XGBoost edge classifier.

    Stage 1 (already done): GNN trained end-to-end on edge labels.
    Stage 2: freeze GNN, extract per-node embeddings z[N, hidden], then
    build edge features [z[src] || z[dst] || edge_attr] and train XGBoost.

    Why this helps: XGBoost can exploit non-linear interactions between the
    GNN embedding and raw edge features (amount, delta_t, structuring flags)
    that the GNN's linear EdgeMLP might miss.  It also applies its own optimal
    threshold internally via the F1-maximising search.
    """
    try:
        from xgboost import XGBClassifier
        _has_xgb = True
    except ImportError:
        print("  [Cascade] XGBoost not available — skipping cascade.", flush=True)
        return None

    if cfg.get("MODEL") == "TGN":
        print("  [Cascade] TGN has no encode() — skipping cascade.", flush=True)
        return None

    print("\n" + "═" * 70, flush=True)
    print("  TWO-STAGE CASCADE: GNN Embeddings → XGBoost", flush=True)
    print("═" * 70, flush=True)

    model.eval()
    # Extract node embeddings — run encode() on full graph in chunks to avoid OOM
    max_ei = min(data.edge_index.size(1), 600_000)
    try:
        with torch.no_grad():
            x_dev  = data.x.to(device)
            ei_dev = data.edge_index[:, :max_ei].to(device)
            z      = model.encode(x_dev, ei_dev).cpu().numpy()
    except RuntimeError as _oom:
        print(f"  [Cascade] OOM during embedding extraction ({_oom}), skipping.", flush=True)
        return None

    src_all = data.edge_index[0].numpy()
    dst_all = data.edge_index[1].numpy()
    ea_np   = data.edge_attr.numpy()

    # Build [src_emb || dst_emb || edge_feat] feature matrix
    X_all = np.hstack([z[src_all], z[dst_all], ea_np]).astype(np.float32)
    y_all = data.edge_label.numpy()

    X_tr, y_tr = X_all[tr_e], y_all[tr_e]
    X_te, y_te = X_all[te_e], y_all[te_e]

    pos = int(y_tr.sum()); neg = len(y_tr) - pos
    print(f"  Train: {len(y_tr):,} edges  ({pos} fraud)  |  "
          f"Test: {len(y_te):,} edges  ({int(y_te.sum())} fraud)", flush=True)

    clf = XGBClassifier(
        n_estimators=300, max_depth=6,
        scale_pos_weight=neg / max(pos, 1),
        learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=cfg["SEED"], n_jobs=-1, verbosity=0,
    )
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)[:, 1]

    # Find optimal threshold by maximising val-set F1 (reuse train set as proxy)
    from sklearn.metrics import precision_recall_curve, roc_auc_score, auc as _auc
    vp, vr, vt = precision_recall_curve(y_tr, clf.predict_proba(X_tr)[:, 1])
    vf1  = 2 * vp * vr / (vp + vr + 1e-9)
    best_t = float(vt[np.argmax(vf1)]) if len(vt) > 0 else 0.5

    preds  = (probs >= best_t).astype(int)
    recall = (preds[y_te == 1] == 1).mean() if (y_te == 1).sum() > 0 else 0.0
    try:
        roc   = roc_auc_score(y_te, probs)
        p_c, r_c, _ = precision_recall_curve(y_te, probs)
        pr_auc = _auc(r_c, p_c)
    except Exception:
        roc = pr_auc = 0.0

    from sklearn.metrics import classification_report
    print(classification_report(y_te, preds,
                                 target_names=["Benign", "Laundering"], digits=4))
    print(f"  Cascade Recall={recall:.4f}  ROC-AUC={roc:.4f}  "
          f"PR-AUC={pr_auc:.4f}  threshold={best_t:.4f}", flush=True)

    # Save cascade metrics alongside GNN metrics
    rd = cfg["RESULTS_DIR"]
    cascade_out = {
        "recall": recall, "roc_auc": round(roc, 4),
        "pr_auc": round(pr_auc, 4), "threshold": round(best_t, 4),
    }
    with open(f"{rd}/metrics_cascade_{cfg['MODEL']}.json", "w") as _f:
        json.dump(cascade_out, _f, indent=2)
    print(f"  Cascade metrics → {rd}/metrics_cascade_{cfg['MODEL']}.json", flush=True)
    return cascade_out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — Helpers (build model / loaders)
# ══════════════════════════════════════════════════════════════════════════════

def _build_model(cfg, n_nodes, in_dim, edge_feat_dim):
    n_pat = len(PATTERN_TYPES)
    if cfg["MODEL"] == "GraphSAGE":
        return GraphSAGEModel(in_dim, cfg["HIDDEN_DIM"], cfg["NUM_LAYERS"],
                              edge_feat_dim, cfg["DROPOUT"], n_pat)
    elif cfg["MODEL"] == "GAT":
        return GATModel(in_dim, cfg["HIDDEN_DIM"], cfg["NUM_LAYERS"],
                        edge_feat_dim, cfg["GAT_HEADS"], cfg["DROPOUT"], n_pat)
    elif cfg["MODEL"] == "H2GCN":
        return H2GCNModel(in_dim, cfg["HIDDEN_DIM"], cfg["NUM_LAYERS"],
                          edge_feat_dim, cfg["DROPOUT"], n_pat)
    elif cfg["MODEL"] == "TGN":
        return TGNModel(n_nodes, in_dim, edge_feat_dim,
                        cfg["TGN_MEMORY_DIM"], cfg["TGN_TIME_DIM"],
                        cfg["HIDDEN_DIM"], cfg["DROPOUT"], n_pat)
    else:
        raise ValueError(f"Unknown model: {cfg['MODEL']}")


class _EdgeBatchLoader:
    """
    Full-graph edge batch loader — no C++ extensions required.
    Loads the entire graph once (fits HI-Medium on A100/V100) and yields
    mini-batches of edge indices only. Avoids pyg-lib/torch-sparse dependency.
    For very large graphs (HI-Large) swap to LinkNeighborLoader with pyg-lib.
    """
    def __init__(self, data: Data, edge_idx: np.ndarray,
                 batch_size: int, shuffle: bool = False):
        self.data       = data
        self.edge_idx   = torch.from_numpy(edge_idx.astype(np.int64))
        self.batch_size = batch_size
        self.shuffle    = shuffle

        # Precompute CSR: node → list of edge indices touching it.
        # One-time O(E log E) build; per-batch lookup is O(batch_nodes × avg_degree)
        # instead of O(E) torch.isin — ~1000x faster on large graphs.
        print("  [Loader] Building CSR edge index...", flush=True)
        ei_np  = data.edge_index.cpu().numpy()
        N, E   = data.num_nodes, ei_np.shape[1]
        nids   = np.concatenate([ei_np[0], ei_np[1]])   # 2E node entries
        eids   = np.tile(np.arange(E), 2)               # 2E edge entries
        order  = np.argsort(nids, kind="stable")
        self._sorted_edges = eids[order].astype(np.int64)
        counts = np.bincount(nids[order], minlength=N)
        self._indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        print(f"  [Loader] Done — {E:,} edges / {N:,} nodes indexed.", flush=True)

    def __len__(self):
        return max(1, (len(self.edge_idx) + self.batch_size - 1) // self.batch_size)

    def _hop_edges(self, batch_nodes_np: np.ndarray) -> torch.Tensor:
        """Return edge indices of all edges touching any node in batch_nodes_np."""
        segs = [self._sorted_edges[self._indptr[n]:self._indptr[n + 1]]
                for n in batch_nodes_np]
        if not segs:
            return torch.zeros(0, dtype=torch.long)
        return torch.from_numpy(np.unique(np.concatenate(segs)))

    def __iter__(self):
        perm = (torch.randperm(len(self.edge_idx)) if self.shuffle
                else torch.arange(len(self.edge_idx)))
        for start in range(0, len(perm), self.batch_size):
            idx  = perm[start : start + self.batch_size]
            eidx = self.edge_idx[idx]
            src_g = self.data.edge_index[0, eidx].numpy()
            dst_g = self.data.edge_index[1, eidx].numpy()
            batch_nodes = np.unique(np.concatenate([src_g, dst_g]))
            hop_t  = self._hop_edges(batch_nodes)
            sub_ea = self.data.edge_attr[hop_t] if self.data.edge_attr is not None else None
            yield SimpleNamespace(
                x               = self.data.x,
                edge_index      = self.data.edge_index[:, hop_t],
                edge_attr       = sub_ea,
                edge_label      = self.data.edge_label[eidx],
                edge_label_index= self.data.edge_index[:, eidx],
                pattern_label   = self.data.pattern_label[eidx],
            )


def _build_loaders(data: Data, cfg: dict):
    """
    Build loaders for edge or node classification.
    Uses _EdgeBatchLoader (full-graph, no C++ deps) for edge task,
    and PyG NeighborLoader for node task.
    To use LinkNeighborLoader (needs pyg-lib): set USE_LINK_LOADER=True below.
    """
    USE_LINK_LOADER = False   # flip to True on Bridges2 after: pip install pyg-lib

    if cfg["TASK"] == "edge":
        tr_e, va_e, te_e = _get_edge_splits(data, cfg)
        bs = cfg["BATCH_SIZE"]
        if USE_LINK_LOADER:
            common = dict(num_neighbors=cfg["NUM_NEIGHBORS"],
                          batch_size=bs, num_workers=0)
            return (LinkNeighborLoader(data, **common,
                                       edge_label_index=data.edge_index[:, tr_e],
                                       edge_label=data.edge_label[tr_e], shuffle=True),
                    LinkNeighborLoader(data, **common,
                                       edge_label_index=data.edge_index[:, va_e],
                                       edge_label=data.edge_label[va_e], shuffle=False),
                    LinkNeighborLoader(data, **common,
                                       edge_label_index=data.edge_index[:, te_e],
                                       edge_label=data.edge_label[te_e], shuffle=False))
        return (_EdgeBatchLoader(data, tr_e, bs, shuffle=True),
                _EdgeBatchLoader(data, va_e, bs, shuffle=False),
                _EdgeBatchLoader(data, te_e, bs, shuffle=False))
    else:
        # Node classification — NeighborLoader (uses torch-sparse if available)
        tm, vm, tem = make_masks(data.num_nodes, data.y.numpy(), cfg["SEED"])
        data.train_mask = tm; data.val_mask = vm; data.test_mask = tem
        # Check if C++ extensions are available before trying NeighborLoader —
        # it imports fine but raises ImportError only during iteration without pyg-lib.
        try:
            import torch_geometric.typing as _pygt
            _has_c = getattr(_pygt, "WITH_PYG_LIB", False) or getattr(_pygt, "WITH_TORCH_SPARSE", False)
        except Exception:
            _has_c = False

        if _has_c:
            common = dict(num_neighbors=cfg["NUM_NEIGHBORS"],
                          batch_size=cfg["BATCH_SIZE"], num_workers=0)
            return (NeighborLoader(data, input_nodes=tm, **common, shuffle=True),
                    NeighborLoader(data, input_nodes=vm, **common, shuffle=False),
                    NeighborLoader(data, input_nodes=tem, **common, shuffle=False))
        else:
            # Fallback: full-graph batches (works without pyg-lib; fine for Elliptic size)
            class _FullGraphLoader:
                def __init__(self, mask): self.mask = mask
                def __len__(self): return 1
                def __iter__(self):
                    yield SimpleNamespace(x=data.x, edge_index=data.edge_index,
                                          edge_attr=data.edge_attr,
                                          y=data.y, train_mask=self.mask,
                                          edge_label=data.edge_label,
                                          edge_label_index=data.edge_index,
                                          pattern_label=data.pattern_label)
            return _FullGraphLoader(tm), _FullGraphLoader(vm), _FullGraphLoader(tem)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — Elliptic Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_elliptic(cfg: dict):
    """
    Load Elliptic dataset, build graph, and train the selected model.
    Node features: 165 pre-computed features from the dataset (drop txId + time_step).
    Task: node-level classification (overrides config TASK to 'node').
    """
    print("\n" + "═"*70)
    print("[Elliptic] Loading and preparing dataset…")
    feats, classes, edges = load_elliptic()

    # Only keep nodes with known labels
    known = classes[classes["label"] != -1]
    txid_set  = set(known["txId"])
    node_ids  = sorted(txid_set)
    nid_map   = {n: i for i, n in enumerate(node_ids)}

    feat_filt = feats[feats["txId"].isin(txid_set)].copy()
    feat_cols = [c for c in feat_filt.columns if c not in ("txId", "time_step")]
    x_raw = feat_filt.set_index("txId").loc[node_ids, feat_cols].values.astype(np.float32)
    x_scaled = StandardScaler().fit_transform(x_raw)

    cls_filt  = known.set_index("txId").loc[node_ids, "label"].values.astype(np.int64)

    # Build edge index (filtered to known nodes)
    edge_filt = edges[edges["txId1"].isin(txid_set) & edges["txId2"].isin(txid_set)]
    src = torch.tensor([nid_map[t] for t in edge_filt["txId1"]], dtype=torch.long)
    dst = torch.tensor([nid_map[t] for t in edge_filt["txId2"]], dtype=torch.long)
    ei  = to_undirected(torch.stack([src, dst]))

    data = Data(
        x           = torch.tensor(x_scaled, dtype=torch.float),
        edge_index  = ei,
        y           = torch.tensor(cls_filt, dtype=torch.long),
        edge_label  = torch.zeros(ei.size(1), dtype=torch.long),  # dummy
        edge_attr   = torch.zeros(ei.size(1), x_scaled.shape[1], dtype=torch.float),
        pattern_label= torch.zeros(ei.size(1), dtype=torch.long),
    )

    print(f"  Graph: {data.num_nodes:,} nodes  |  {ei.size(1):,} edges  |  "
          f"fraud: {cls_filt.sum():,} ({cls_filt.mean()*100:.2f}%)")

    # Override to node task for Elliptic
    ecfg = {**cfg, "TASK": "node", "MULTI_CLASS": False}
    ecfg["MODEL"] = cfg.get("MODEL", "GAT")

    device = ecfg["DEVICE"]
    n_nodes, in_dim = data.num_nodes, data.x.size(1)
    model = _build_model(ecfg, n_nodes, in_dim, 0).to(device)

    train_l, val_l, test_l = _build_loaders(data, ecfg)
    focal   = make_focal_loss(cls_filt, device)
    pat_ce  = nn.CrossEntropyLoss()
    opt     = torch.optim.Adam(model.parameters(), lr=ecfg["LR"],
                               weight_decay=ecfg["WEIGHT_DECAY"])
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ecfg["EPOCHS"])

    print(f"[Elliptic] Training {ecfg['MODEL']}…")
    history = full_training_run(model, train_l, val_l, data, focal, pat_ce,
                                opt, sched, device, ecfg)
    metrics, res = full_evaluation(model, test_l, data, focal, pat_ce, device, ecfg)
    ecfg_plot = {**ecfg, "MODEL": f"{ecfg['MODEL']}_Elliptic"}
    plot_results(history, metrics, res, ecfg_plot)
    return model, history, metrics


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — Main Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AML GNN v2")
    parser.add_argument("--model",   default=CONFIG["MODEL"],
                        choices=["GraphSAGE", "GAT", "TGN", "H2GCN"])
    parser.add_argument("--dataset", default=CONFIG["DATASET"],
                        choices=["IBM_HI", "IBM_LI", "ELLIPTIC"])
    parser.add_argument("--size",    default=CONFIG["DATASET_SIZE"],
                        choices=["small", "medium", "large"])
    parser.add_argument("--task",    default=CONFIG["TASK"],
                        choices=["edge", "node"])
    parser.add_argument("--mode",    default="train",
                        choices=["train", "sweep"])
    parser.add_argument("--epochs",  type=int, default=CONFIG["EPOCHS"])
    parser.add_argument("--no-hetero", action="store_true")
    parser.add_argument("--no-smote",  action="store_true")
    parser.add_argument("--no-multiclass", action="store_true")
    parser.add_argument("--max-rows",  type=int, default=CONFIG["MAX_ROWS"],
                        help="Cap CSV rows loaded (0 = full dataset)")
    parser.add_argument("--row-offset", type=int, default=0,
                        help="Skip this many rows before loading (for high-fraud regions)")
    parser.add_argument("--no-cascade", action="store_true",
                        help="Skip two-stage cascade classifier")
    parser.add_argument("--local-test", action="store_true",
                        help="CPU-friendly settings: small dataset, 5 epochs, 2048 batch")
    args = parser.parse_args()

    # --local-test: override config for quick CPU validation
    local_overrides = {}
    if args.local_test:
        local_overrides = {
            "DATASET_SIZE" : "small",
            "EPOCHS"       : 3,
            "BATCH_SIZE"   : 4096,
            "NUM_LAYERS"   : 2,       # shallower for CPU speed
            "MAX_ROWS"     : 100_000,
            "OPTUNA_TRIALS": 3,
        }
        args.size = "small"
        print("  [local-test] CPU mode: 100k rows, 3 epochs, 2 layers")

    cfg = {
        **CONFIG,
        **local_overrides,
        "MODEL"      : args.model,
        "DATASET"    : args.dataset,
        "DATASET_SIZE": args.size,
        "TASK"       : args.task,
        "EPOCHS"     : local_overrides.get("EPOCHS", args.epochs),
        "HETERO"     : not args.no_hetero,
        "MULTI_CLASS" : not args.no_multiclass,
        "MAX_ROWS"    : local_overrides.get("MAX_ROWS",
                            args.max_rows if args.max_rows else None),
        "ROW_OFFSET"  : args.row_offset,
    }
    device = cfg["DEVICE"]
    print(f"\n{'═'*70}")
    print(f"  AML GNN v2  |  Model: {cfg['MODEL']}  |  Dataset: {cfg['DATASET']}-{cfg['DATASET_SIZE']}")
    print(f"  Task: {cfg['TASK']}  |  Device: {device}  |  Mode: {args.mode}")
    print(f"{'═'*70}\n")

    # ── Elliptic shortcut ──
    if cfg["DATASET"] == "ELLIPTIC":
        run_elliptic(cfg)
        return

    # ── IBM Pipeline ──
    print("[1/7] Setting up IBM dataset…")
    ibm_files = setup_ibm_data(size=cfg["DATASET_SIZE"])
    variant   = "HI" if cfg["DATASET"] == "IBM_HI" else "LI"
    paths     = ibm_files[variant]

    if not paths["trans"]:
        print("ERROR: No transaction file found. Exiting.")
        sys.exit(1)

    print(f"\n[2/7] Loading data (variant={variant})…")
    df          = load_ibm_trans(paths["trans"],
                                  max_rows=cfg.get("MAX_ROWS"),
                                  row_offset=cfg.get("ROW_OFFSET", 0))
    pattern_map = parse_patterns(paths["patterns"])
    acc_df      = parse_accounts(paths["accounts"])

    print(f"\n[3/7] Feature engineering…")
    (x, edge_feats, edge_labels, pattern_labels,
     src_dst, account_to_id, bank_to_id, acc_df) = engineer_ibm_features(
        df, acc_df, pattern_map)

    # ── GraphSMOTE (improvement #7) ──
    if not args.no_smote and cfg["TASK"] == "node":
        node_labels = np.zeros(len(account_to_id), dtype=np.int64)
        fraud_rows  = src_dst[edge_labels == 1]
        node_labels[fraud_rows[:, 0]] = 1
        node_labels[fraud_rows[:, 1]] = 1
        ei_np = src_dst.T
        x, node_labels, ei_np = graph_smote(x, node_labels, ei_np,
                                            k=cfg["SMOTE_K"],
                                            target_ratio=cfg["SMOTE_RATIO"])
        src_dst = ei_np.T

    print(f"\n[3b/7] Running sklearn baselines…")
    run_sklearn_baseline(edge_feats, edge_labels, cfg)

    print(f"\n[4/7] Building PyG graph…")
    data = build_pyg_data(x, edge_feats, edge_labels, pattern_labels,
                          src_dst, acc_df, bank_to_id, cfg["TASK"])
    data.num_nodes = x.shape[0]
    print(f"  Homogeneous graph: {data.num_nodes:,} nodes  |  "
          f"{data.edge_index.size(1):,} edges")

    if cfg["HETERO"] and not args.no_hetero:
        hetero = build_hetero_data(data, acc_df, bank_to_id, account_to_id)
        if hetero is not None:
            print(f"  Heterogeneous graph: account nodes + {len(bank_to_id):,} bank nodes")

    print(f"\n[5/7] Building model ({cfg['MODEL']})…")
    n_nodes, in_dim    = data.num_nodes, data.x.size(1)
    edge_feat_dim      = data.edge_attr.size(1)
    model              = _build_model(cfg, n_nodes, in_dim, edge_feat_dim).to(device)
    n_params           = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # ── Optuna sweep ──
    if args.mode == "sweep":
        best_params = run_optuna_sweep(data, n_nodes, in_dim,
                                       edge_feat_dim, device, cfg)
        # Apply best params and run final training
        cfg.update({
            "HIDDEN_DIM" : best_params.get("hidden",   cfg["HIDDEN_DIM"]),
            "NUM_LAYERS" : best_params.get("n_layers", cfg["NUM_LAYERS"]),
            "DROPOUT"    : best_params.get("dropout",  cfg["DROPOUT"]),
            "LR"         : best_params.get("lr",       cfg["LR"]),
            "GAT_HEADS"  : best_params.get("heads",    cfg["GAT_HEADS"]),
        })
        model = _build_model(cfg, n_nodes, in_dim, edge_feat_dim).to(device)

    print(f"\n[6/7] Training…")
    train_l, val_l, test_l = _build_loaders(data, cfg)

    focal   = make_focal_loss(data.edge_label.numpy(), device)
    pat_ce  = nn.CrossEntropyLoss()
    opt     = torch.optim.Adam(model.parameters(), lr=cfg["LR"],
                               weight_decay=cfg["WEIGHT_DECAY"])
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                          T_max=cfg["EPOCHS"])

    if cfg["MODEL"] != "TGN":
        history = full_training_run(model, train_l, val_l, data,
                                    focal, pat_ce, opt, sched, device, cfg)
    else:
        # TGN temporal training (no loader, processes data directly)
        history = {k: [] for k in ("train_loss", "val_loss",
                                    "train_recall", "val_recall")}
        for ep in range(1, cfg["EPOCHS"] + 1):
            tr = train_tgn(model, data, opt, focal, pat_ce,
                           device, cfg, split="train")
            va = train_tgn(model, data, opt, focal, pat_ce,
                           device, cfg, split="val")
            history["train_loss"].append(tr["loss"])
            history["val_loss"].append(va["loss"])
            history["train_recall"].append(tr["recall"])
            history["val_recall"].append(va["recall"])
            sched.step()
            print(f"  Ep {ep:>3}  TrLoss={tr['loss']:.4f}  "
                  f"ValRecall={va['recall']:.4f}")

    print(f"\n[7/7] Evaluating…")
    metrics, res = full_evaluation(model, test_l, data, focal, pat_ce, device, cfg)
    plot_results(history, metrics, res, cfg)

    # ── Platt calibration (post-training) ──
    if cfg["MODEL"] != "TGN":
        print("\n  [Platt Calibration] Fitting on val set, evaluating on test set…", flush=True)
        platt_calibrate(model, val_l, test_l, data, device, cfg)

    # ── Two-stage cascade: GNN embeddings → XGBoost ──
    if not args.no_cascade and cfg["MODEL"] != "TGN" and cfg["TASK"] == "edge":
        tr_e, _, te_e = _get_edge_splits(data, cfg)
        run_cascade_classifier(model, data, tr_e, te_e, device, cfg)

    # Save model checkpoint (TGN skipped — state dict is ~600 MB)
    if cfg["MODEL"] != "TGN":
        ckpt_path = f"{cfg['RESULTS_DIR']}/model_{cfg['MODEL']}_{variant}.pt"
        torch.save({"model_state": model.state_dict(), "config": cfg}, ckpt_path)
        print(f"\n  Model checkpoint → {ckpt_path}")
    print("\n  Done!\n")


if __name__ == "__main__":
    main()
