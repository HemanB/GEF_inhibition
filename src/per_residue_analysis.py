#!/usr/bin/env python3
"""Per-residue analysis: statistics + interpretable attention models.

Layer 1: Correlation-based position importance (ITSN only — fixed length)
Layer 2: MLP contribution model (sum of per-residue contributions → binding)
Layer 3: Interpretable attention sweep (score-value decomposition)
  - Sweeps entropy penalty and architecture for accuracy–interpretability tradeoff
  - Produces Pareto frontier and selects best interpretable model
  - Generates per-residue attention heatmaps
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT / "data" / "raw" / "GEF_inhibitors_modeling_data.csv"
RESULTS = PROJECT / "results"
FEATURE_DIR = PROJECT / "data" / "processed" / "features"
FIG_DIR = RESULTS / "figures"
STAT_DIR = RESULTS / "statistical_analysis"
MODEL_DIR = RESULTS / "models"
VAL_DIR = RESULTS / "validation"
PYMOL_DIR = RESULTS / "pymol_scripts"

for d in [FIG_DIR, STAT_DIR, MODEL_DIR, VAL_DIR, PYMOL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Feature columns (non-metadata)
META_COLS = {"inhibitor_id", "class", "binding_mean", "binding_z", "scaffold_position",
             "relative_position", "distance_to_junction"}
AA_COLS = [f"aa_{aa}" for aa in sorted(
    "ALA ARG ASN ASP CYS GLU GLN GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())]


def get_feature_cols(df):
    return [c for c in df.columns if c not in META_COLS and c not in AA_COLS]


def class_normalize_binding(df):
    """Z-score binding within class to remove class-mean effect."""
    df = df.copy()
    for cls in df["class"].unique():
        mask = df["class"] == cls
        vals = df.loc[mask, "binding_mean"]
        if vals.std() > 0:
            df.loc[mask, "binding_z"] = (vals - vals.mean()) / vals.std()
        else:
            df.loc[mask, "binding_z"] = 0.0
    return df


# ═══════════════════════════════════════════════════════════════════════
# LAYER 1: Statistical Analysis
# ═══════════════════════════════════════════════════════════════════════

def layer1_statistics(df: pd.DataFrame):
    """Position-level correlation analysis for ITSN (all scaffolds = 151 aa)."""
    print("\n" + "=" * 60)
    print("LAYER 1: Statistical Analysis")
    print("=" * 60)

    feat_cols = get_feature_cols(df)
    results = []

    itsn = df[df["class"] == "ITSN"].copy()
    itsn_designs = itsn["inhibitor_id"].unique()
    n_itsn = len(itsn_designs)
    print(f"\nITSN positional analysis (n={n_itsn} designs, 151 positions):")

    for pos in range(1, 152):
        pos_data = itsn[itsn["scaffold_position"] == pos]
        if len(pos_data) < n_itsn:
            continue
        binding = pos_data["binding_z"].values if "binding_z" in pos_data.columns else pos_data["binding_mean"].values
        for feat in feat_cols:
            vals = pos_data[feat].values
            if np.std(vals) == 0:
                continue
            rho, p = stats.spearmanr(vals, binding)
            if not np.isnan(rho):
                results.append({
                    "class": "ITSN", "position": pos, "feature": feat,
                    "spearman_rho": round(rho, 4), "p_value": round(p, 6),
                })

    print("\nCross-design feature analysis (all designs, all positions):")
    for feat in feat_cols:
        design_feat = df.groupby("inhibitor_id")[feat].mean()
        design_binding = df.groupby("inhibitor_id")["binding_mean"].first()
        common = design_feat.index.intersection(design_binding.index)
        if len(common) < 5:
            continue
        rho, p = stats.spearmanr(design_feat[common], design_binding[common])
        if abs(rho) > 0.3 and not np.isnan(rho):
            print(f"  {feat:40s} ρ={rho:+.3f} p={p:.4f}")

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df.to_csv(STAT_DIR / "position_binding_correlations.csv", index=False)
        sig = results_df[results_df["p_value"] < 0.1].sort_values("p_value")
        sig.to_csv(STAT_DIR / "significant_associations.csv", index=False)
        print(f"\nSignificant associations (p<0.1): {len(sig)}")
        if len(sig) > 0:
            print(sig.head(20).to_string(index=False))
            pos_importance = sig.groupby("position")["spearman_rho"].apply(
                lambda x: np.sum(np.abs(x))).sort_values(ascending=False)
            print(f"\nTop positions by summed |ρ|:")
            for pos, imp in pos_importance.head(15).items():
                n_sig = len(sig[sig["position"] == pos])
                print(f"  Position {pos:>3d}: Σ|ρ|={imp:.2f} ({n_sig} sig features)")

    return results_df


# ═══════════════════════════════════════════════════════════════════════
# LAYER 2: MLP Contribution Model
# ═══════════════════════════════════════════════════════════════════════

class ResidueContributionModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x, mask=None):
        contributions = self.mlp(x).squeeze(-1)
        if mask is not None:
            contributions = contributions.masked_fill(mask, 0.0)
        predicted = contributions.sum(dim=1, keepdim=True)
        return predicted, contributions


# ═══════════════════════════════════════════════════════════════════════
# LAYER 3: Interpretable Attention (Score-Value Decomposition)
# ═══════════════════════════════════════════════════════════════════════

class InterpretableAttention(nn.Module):
    """Score-value attention for interpretable position importance.

    score_net: determines which positions to attend → attention weights
    value_net: determines each position's scalar contribution
    prediction = Σ softmax(score / temperature) * value

    This forces information through the attention bottleneck: the model
    cannot bypass attention by encoding predictions into a rich hidden
    representation, because values are scalar.
    """
    def __init__(self, feature_dim, hidden_dim=8, temperature=1.0):
        super().__init__()
        if hidden_dim == 0:
            # Fully linear scoring — most interpretable
            self.score_net = nn.Linear(feature_dim, 1)
        else:
            # Small nonlinear scoring
            self.score_net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
        self.value_net = nn.Linear(feature_dim, 1)
        self.temperature = temperature

    def forward(self, x, mask=None):
        scores = self.score_net(x).squeeze(-1) / self.temperature
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        values = self.value_net(x).squeeze(-1)
        if mask is not None:
            values = values.masked_fill(mask, 0.0)
        predicted = (attn * values).sum(dim=-1, keepdim=True)
        return predicted, attn


# ═══════════════════════════════════════════════════════════════════════
# LAYER 4: Minimal Transformer (Single-Layer Self-Attention)
# ═══════════════════════════════════════════════════════════════════════

class MinimalTransformer(nn.Module):
    """Single-layer self-attention: each residue sees all others.

    Same aggregation as MLP (sum of per-residue contributions), so the
    ONLY difference is that contributions are context-enriched via
    self-attention. Fair comparison for "does residue-residue context help?"
    """
    def __init__(self, feature_dim, hidden_dim=16, n_heads=2, dropout=0.1):
        super().__init__()
        self.encoder = nn.Linear(feature_dim, hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.last_self_attn = None  # stored for extraction

    def forward(self, x, mask=None):
        encoded = F.relu(self.encoder(x))
        attn_out, attn_weights = self.self_attn(
            encoded, encoded, encoded,
            key_padding_mask=mask, average_attn_weights=False)
        self.last_self_attn = attn_weights.detach()  # (batch, n_heads, seq, seq)
        context = self.norm(encoded + self.drop(attn_out))
        contributions = self.value_head(context).squeeze(-1)
        if mask is not None:
            contributions = contributions.masked_fill(mask, 0.0)
        predicted = contributions.sum(dim=1, keepdim=True)
        return predicted, contributions


# ═══════════════════════════════════════════════════════════════════════
# Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════

def prepare_data(df: pd.DataFrame, designs_df: pd.DataFrame):
    """Prepare per-design residue feature tensors."""
    feat_cols = get_feature_cols(df)
    all_feat_cols = AA_COLS + feat_cols
    all_feat_cols = [c for c in all_feat_cols if c in df.columns and df[c].std() > 0]

    design_data = {}
    for inh_id in df["inhibitor_id"].unique():
        sub = df[df["inhibitor_id"] == inh_id].sort_values("scaffold_position")
        X = sub[all_feat_cols].values.astype(np.float32)
        binding_z = sub["binding_z"].iloc[0] if "binding_z" in sub.columns else 0.0
        binding_raw = sub["binding_mean"].iloc[0]
        design_data[inh_id] = {
            "features": X,
            "binding_z": float(binding_z),
            "binding_raw": float(binding_raw),
            "n_residues": len(sub),
            "positions": sub["scaffold_position"].values,
            "class": sub["class"].iloc[0],
        }

    return design_data, all_feat_cols


def run_loo_cv(design_data: dict, ModelClass, feat_dim: int,
               n_seeds: int = 10, n_epochs: int = 500, lr: float = 0.003,
               weight_decay: float = 0.01, use_binding_z: bool = True,
               entropy_weight: float = 0.0, collect_self_attn: bool = False):
    """LOO-CV at complex level with multiple seeds. Accumulates importance across seeds.

    If collect_self_attn=True, also collects self-attention matrices from
    models that have a .last_self_attn attribute (e.g. MinimalTransformer).
    Returns (predictions, importances, self_attn_dict) when True.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    design_ids = list(design_data.keys())
    n = len(design_ids)
    max_len = max(d["n_residues"] for d in design_data.values())

    def pad(features, max_len):
        n_res = features.shape[0]
        padded = np.zeros((max_len, features.shape[1]), dtype=np.float32)
        padded[:n_res] = features
        mask = np.ones(max_len, dtype=bool)
        mask[:n_res] = False
        return padded, mask

    all_feats = np.vstack([d["features"] for d in design_data.values()])
    scaler = StandardScaler()
    scaler.fit(all_feats)

    binding_key = "binding_z" if use_binding_z else "binding_raw"

    all_predictions = np.zeros((n_seeds, n))
    all_importances = {}
    all_self_attn = {} if collect_self_attn else None

    for seed in range(n_seeds):
        torch.manual_seed(seed * 42)
        np.random.seed(seed * 42)

        for loo_idx in range(n):
            test_id = design_ids[loo_idx]
            train_ids = [d for i, d in enumerate(design_ids) if i != loo_idx]

            train_X, train_mask, train_y = [], [], []
            for tid in train_ids:
                scaled = scaler.transform(design_data[tid]["features"])
                pf, pm = pad(scaled, max_len)
                train_X.append(pf)
                train_mask.append(pm)
                train_y.append(design_data[tid][binding_key])

            train_X = torch.tensor(np.stack(train_X)).to(device)
            train_mask = torch.tensor(np.stack(train_mask)).to(device)
            train_y = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1).to(device)

            test_scaled = scaler.transform(design_data[test_id]["features"])
            test_pf, test_pm = pad(test_scaled, max_len)
            test_X = torch.tensor(test_pf).unsqueeze(0).to(device)
            test_mask = torch.tensor(test_pm).unsqueeze(0).to(device)

            model = ModelClass(feat_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.MSELoss()

            model.train()
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                pred, importance = model(train_X, train_mask)
                loss = criterion(pred, train_y)

                if entropy_weight > 0 and importance.dim() == 2:
                    entropy = -torch.sum(importance * torch.log(importance + 1e-9), dim=-1)
                    loss = loss + entropy_weight * entropy.mean()

                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                pred, importance = model(test_X, test_mask)
                all_predictions[seed, loo_idx] = pred.cpu().item()

                n_real = design_data[test_id]["n_residues"]
                imp = importance.squeeze().cpu().numpy()[:n_real]
                if test_id not in all_importances:
                    all_importances[test_id] = np.zeros_like(imp)
                all_importances[test_id] += imp / n_seeds

                # Collect self-attention matrices (transformer only)
                if collect_self_attn and hasattr(model, 'last_self_attn') \
                        and model.last_self_attn is not None:
                    sa = model.last_self_attn.squeeze(0).cpu().numpy()  # (n_heads, seq, seq)
                    sa_avg = sa.mean(axis=0)[:n_real, :n_real]  # avg heads, trim padding
                    if test_id not in all_self_attn:
                        all_self_attn[test_id] = np.zeros_like(sa_avg)
                    all_self_attn[test_id] += sa_avg / n_seeds

    mean_preds = all_predictions.mean(axis=0)
    if collect_self_attn:
        return mean_preds, all_importances, all_self_attn
    return mean_preds, all_importances


def evaluate_predictions(design_data, predictions, design_ids, label,
                         use_binding_z=True):
    """Evaluate LOO predictions."""
    binding_key = "binding_z" if use_binding_z else "binding_raw"
    actual = np.array([design_data[d][binding_key] for d in design_ids])

    rho, p_rho = stats.spearmanr(actual, predictions)
    r, p_r = stats.pearsonr(actual, predictions)

    within_class = {}
    for cls in ["ITSN", "Vav"]:
        cls_mask = [design_data[d]["class"] == cls for d in design_ids]
        cls_actual = actual[cls_mask]
        cls_pred = predictions[cls_mask]
        if len(cls_actual) >= 4:
            rho_c, _ = stats.spearmanr(cls_actual, cls_pred)
            within_class[cls] = rho_c
        else:
            within_class[cls] = np.nan

    print(f"\n  {label}:")
    print(f"    Overall: ρ={rho:.3f} (p={p_rho:.3f}), r={r:.3f}")
    print(f"    Within-ITSN ρ: {within_class.get('ITSN', np.nan):.3f}")
    print(f"    Within-Vav ρ:  {within_class.get('Vav', np.nan):.3f}")

    return {
        "label": label,
        "spearman_rho": round(float(rho), 4),
        "pearson_r": round(float(r), 4),
        "within_itsn_rho": round(float(within_class.get("ITSN", np.nan)), 4),
        "within_vav_rho": round(float(within_class.get("Vav", np.nan)), 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# Attention Interpretability Metrics
# ═══════════════════════════════════════════════════════════════════════

def attention_statistics(attn_dict):
    """Compute interpretability metrics for attention weights."""
    ginis, entropies, maxes, top10s = [], [], [], []

    for inh_id, attn in attn_dict.items():
        n = len(attn)
        if n == 0:
            continue

        # Gini coefficient
        sorted_a = np.sort(attn)
        idx = np.arange(1, n + 1)
        gini = (2 * np.sum(idx * sorted_a) / (n * np.sum(sorted_a))) - (n + 1) / n
        ginis.append(max(0, gini))

        # Normalized entropy (0 = delta, 1 = uniform)
        ent = -np.sum(attn * np.log(attn + 1e-12))
        entropies.append(ent / np.log(n))

        # Max attention
        maxes.append(float(attn.max()))

        # Top-10% share (fraction of total attention in top 10% of positions)
        k = max(1, n // 10)
        top10s.append(float(np.sort(attn)[-k:].sum()))

    return {
        "gini": round(np.mean(ginis), 4),
        "norm_entropy": round(np.mean(entropies), 4),
        "max_attn": round(np.mean(maxes), 5),
        "top10_share": round(np.mean(top10s), 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# Attention Sweep
# ═══════════════════════════════════════════════════════════════════════

SWEEP_CONFIGS = [
    # Linear scoring, entropy sweep
    {"name": "linear_e0.0",   "hidden_dim": 0, "entropy_weight": 0.0,  "temperature": 1.0},
    {"name": "linear_e0.1",   "hidden_dim": 0, "entropy_weight": 0.1,  "temperature": 1.0},
    {"name": "linear_e0.3",   "hidden_dim": 0, "entropy_weight": 0.3,  "temperature": 1.0},
    {"name": "linear_e0.5",   "hidden_dim": 0, "entropy_weight": 0.5,  "temperature": 1.0},
    {"name": "linear_e1.0",   "hidden_dim": 0, "entropy_weight": 1.0,  "temperature": 1.0},
    # MLP(8) scoring, entropy sweep
    {"name": "mlp8_e0.0",     "hidden_dim": 8, "entropy_weight": 0.0,  "temperature": 1.0},
    {"name": "mlp8_e0.1",     "hidden_dim": 8, "entropy_weight": 0.1,  "temperature": 1.0},
    {"name": "mlp8_e0.3",     "hidden_dim": 8, "entropy_weight": 0.3,  "temperature": 1.0},
    {"name": "mlp8_e0.5",     "hidden_dim": 8, "entropy_weight": 0.5,  "temperature": 1.0},
    {"name": "mlp8_e1.0",     "hidden_dim": 8, "entropy_weight": 1.0,  "temperature": 1.0},
    # Temperature sweep at moderate entropy
    {"name": "mlp8_e0.3_t0.5", "hidden_dim": 8, "entropy_weight": 0.3, "temperature": 0.5},
    {"name": "mlp8_e0.3_t0.1", "hidden_dim": 8, "entropy_weight": 0.3, "temperature": 0.1},
]


def run_attention_sweep(design_data, feat_dim, configs=None,
                        n_seeds=5, n_epochs=300, lr=0.003):
    """Run LOO-CV for multiple attention configs. Returns results table + cached weights."""
    if configs is None:
        configs = SWEEP_CONFIGS

    results = []
    attn_cache = {}
    design_ids = list(design_data.keys())

    for cfg in configs:
        name = cfg["name"]
        print(f"\n  Config: {name} (h={cfg['hidden_dim']}, "
              f"ent={cfg['entropy_weight']}, T={cfg['temperature']})")

        # Avoid lambda late-binding bug with default argument
        h, t = cfg["hidden_dim"], cfg["temperature"]
        model_factory = lambda fd, _h=h, _t=t: InterpretableAttention(
            fd, hidden_dim=_h, temperature=_t)

        # Count params
        dummy = model_factory(feat_dim)
        n_params = sum(p.numel() for p in dummy.parameters())

        preds, importances = run_loo_cv(
            design_data, model_factory, feat_dim,
            n_seeds=n_seeds, n_epochs=n_epochs, lr=lr,
            entropy_weight=cfg["entropy_weight"],
        )

        metrics = evaluate_predictions(design_data, preds, design_ids, name)
        attn_stats = attention_statistics(importances)
        attn_cache[name] = importances

        result = {**cfg, **metrics, **attn_stats, "n_params": n_params}
        results.append(result)

        print(f"    Params={n_params}, Gini={attn_stats['gini']:.3f}, "
              f"MaxAttn={attn_stats['max_attn']:.5f}, "
              f"Top10%={attn_stats['top10_share']:.3f}")

    return results, attn_cache


def select_best_config(sweep_results, min_gini=0.15):
    """Select best model from Pareto frontier: highest ρ with meaningful attention focus."""
    # Among configs with Gini above threshold, pick highest ρ
    focused = [r for r in sweep_results if r["gini"] >= min_gini]
    if focused:
        best = max(focused, key=lambda r: r["spearman_rho"])
        print(f"\n  Selected: {best['name']} (ρ={best['spearman_rho']:.3f}, "
              f"Gini={best['gini']:.3f})")
        return best

    # Fallback: pick highest Gini
    best = max(sweep_results, key=lambda r: r["gini"])
    print(f"\n  Fallback (no Gini>{min_gini}): {best['name']} "
          f"(ρ={best['spearman_rho']:.3f}, Gini={best['gini']:.3f})")
    return best


# ═══════════════════════════════════════════════════════════════════════
# Transformer Sweep
# ═══════════════════════════════════════════════════════════════════════

TRANSFORMER_CONFIGS = [
    # Small model (h=8, 1 head), dropout sweep
    {"name": "tf_h8_nh1_d0.1",     "hidden_dim": 8,  "n_heads": 1, "dropout": 0.1, "wd": 0.01},
    {"name": "tf_h8_nh1_d0.3",     "hidden_dim": 8,  "n_heads": 1, "dropout": 0.3, "wd": 0.01},
    {"name": "tf_h8_nh1_d0.5",     "hidden_dim": 8,  "n_heads": 1, "dropout": 0.5, "wd": 0.01},
    {"name": "tf_h8_nh1_d0.3_wd5", "hidden_dim": 8,  "n_heads": 1, "dropout": 0.3, "wd": 0.05},
    {"name": "tf_h8_nh1_d0.5_wd1", "hidden_dim": 8,  "n_heads": 1, "dropout": 0.5, "wd": 0.1},
    # Medium model (h=16, 2 heads)
    {"name": "tf_h16_nh2_d0.1",     "hidden_dim": 16, "n_heads": 2, "dropout": 0.1, "wd": 0.01},
    {"name": "tf_h16_nh2_d0.3",     "hidden_dim": 16, "n_heads": 2, "dropout": 0.3, "wd": 0.01},
    {"name": "tf_h16_nh2_d0.5",     "hidden_dim": 16, "n_heads": 2, "dropout": 0.5, "wd": 0.01},
    {"name": "tf_h16_nh2_d0.3_wd5", "hidden_dim": 16, "n_heads": 2, "dropout": 0.3, "wd": 0.05},
    {"name": "tf_h16_nh4_d0.3_wd5", "hidden_dim": 16, "n_heads": 4, "dropout": 0.3, "wd": 0.05},
]


def run_transformer_sweep(design_data, feat_dim, configs=None,
                          n_seeds=5, n_epochs=300, lr=0.003):
    """Sweep transformer configs. Returns results + self-attention for best."""
    if configs is None:
        configs = TRANSFORMER_CONFIGS

    results = []
    contrib_cache = {}
    self_attn_cache = {}
    design_ids = list(design_data.keys())

    for cfg in configs:
        name = cfg["name"]
        h, nh, do = cfg["hidden_dim"], cfg["n_heads"], cfg["dropout"]
        wd = cfg["wd"]
        print(f"\n  Config: {name} (h={h}, heads={nh}, drop={do}, wd={wd})")

        model_factory = lambda fd, _h=h, _nh=nh, _do=do: MinimalTransformer(
            fd, hidden_dim=_h, n_heads=_nh, dropout=_do)

        dummy = model_factory(feat_dim)
        n_params = sum(p.numel() for p in dummy.parameters())

        preds, contribs, self_attn = run_loo_cv(
            design_data, model_factory, feat_dim,
            n_seeds=n_seeds, n_epochs=n_epochs, lr=lr,
            weight_decay=wd, collect_self_attn=True,
        )

        metrics = evaluate_predictions(design_data, preds, design_ids, name)
        contrib_cache[name] = contribs
        self_attn_cache[name] = self_attn

        result = {**cfg, **metrics, "n_params": n_params}
        results.append(result)
        print(f"    Params={n_params}")

    return results, contrib_cache, self_attn_cache


def plot_self_attention_heatmap(self_attn_dict, design_data, designs_df, label):
    """Plot N×N self-attention for 3 example ITSN designs (best/median/worst)."""
    itsn = designs_df[designs_df["class"] == "ITSN"]
    itsn_ids = [d["inhibitor_id"] for _, d in itsn.iterrows()
                if d["inhibitor_id"] in self_attn_dict]
    if not itsn_ids:
        return

    # Sort by binding_z, pick best/median/worst
    bz = [(inh_id, design_data[inh_id]["binding_z"]) for inh_id in itsn_ids]
    bz.sort(key=lambda x: x[1])
    picks = [bz[0], bz[len(bz) // 2], bz[-1]]  # worst, median, best

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (inh_id, binding_z) in zip(axes, picks):
        sa = self_attn_dict[inh_id]
        n = sa.shape[0]
        im = ax.imshow(sa, cmap="viridis", aspect="equal", interpolation="nearest")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        short = inh_id.replace("ITSN_RFD1_", "")
        ax.set_title(f"{short} (z={binding_z:.1f})\n{n}×{n} self-attention")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(f"Residue–Residue Self-Attention ({label})", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"self_attention_heatmap_{label}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / f'self_attention_heatmap_{label}.png'}")


def plot_transformer_vs_mlp(mlp_metrics, tf_results, best_tf_metrics):
    """Bar chart: MLP vs transformer configs."""
    fig, ax = plt.subplots(figsize=(12, 5))

    names = ["MLP"] + [r["name"].replace("tf_", "") for r in tf_results]
    overall = [mlp_metrics["spearman_rho"]] + [r["spearman_rho"] for r in tf_results]
    itsn = [mlp_metrics["within_itsn_rho"]] + [r["within_itsn_rho"] for r in tf_results]
    vav = [mlp_metrics["within_vav_rho"]] + [r["within_vav_rho"] for r in tf_results]

    x = np.arange(len(names))
    w = 0.25
    ax.bar(x - w, overall, w, label="Overall ρ", color="#9C27B0")
    ax.bar(x, itsn, w, label="ITSN ρ", color="#2196F3")
    ax.bar(x + w, vav, w, label="Vav ρ", color="#FF9800")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("MLP (no context) vs Transformer (residue-residue context)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "transformer_vs_mlp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'transformer_vs_mlp.png'}")


def print_transformer_table(mlp_metrics, tf_results):
    """Print comparison table: MLP vs all transformer configs."""
    print("\n" + "=" * 80)
    print("TRANSFORMER vs MLP COMPARISON")
    print("=" * 80)
    print(f"{'Config':<25} {'Params':>6} {'ρ':>7} {'ITSN ρ':>8} {'Vav ρ':>7}")
    print("-" * 55)
    print(f"{'MLP (no context)':<25} {'737':>6} {mlp_metrics['spearman_rho']:>7.3f} "
          f"{mlp_metrics['within_itsn_rho']:>8.3f} {mlp_metrics['within_vav_rho']:>7.3f}")
    print("-" * 55)
    for r in tf_results:
        print(f"{r['name']:<25} {r['n_params']:>6d} {r['spearman_rho']:>7.3f} "
              f"{r['within_itsn_rho']:>8.3f} {r['within_vav_rho']:>7.3f}")


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════

def plot_pareto(sweep_results):
    """Pareto plot: prediction accuracy vs attention focus (Gini)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in sweep_results:
        color = "#2196F3" if r["hidden_dim"] == 0 else "#FF9800"
        marker = "o" if r["temperature"] == 1.0 else "s"
        ax.scatter(r["spearman_rho"], r["gini"], c=color, marker=marker,
                   s=100, edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(f"e={r['entropy_weight']}", (r["spearman_rho"], r["gini"]),
                    fontsize=7, textcoords="offset points", xytext=(5, 5))

    ax.set_xlabel("Overall Spearman ρ (prediction accuracy)")
    ax.set_ylabel("Attention Gini (0=uniform, 1=one-hot)")
    ax.set_title("Accuracy–Interpretability Tradeoff")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
               markersize=10, label="Linear score"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF9800",
               markersize=10, label="MLP(8) score"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markersize=10, label="Low temperature"),
    ]
    ax.legend(handles=legend_elements, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "pareto_accuracy_interpretability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'pareto_accuracy_interpretability.png'}")


def plot_attention_heatmap(design_data, attn_dict, designs_df, label):
    """Attention heatmap: designs × scaffold positions."""
    for cls in ["ITSN", "Vav"]:
        cls_designs = designs_df[designs_df["class"] == cls]
        cls_ids = [d["inhibitor_id"] for _, d in cls_designs.iterrows()
                   if d["inhibitor_id"] in attn_dict]
        if not cls_ids:
            continue

        max_len = max(len(attn_dict[inh_id]) for inh_id in cls_ids)
        mat = np.full((len(cls_ids), max_len), np.nan)
        for i, inh_id in enumerate(cls_ids):
            attn = attn_dict[inh_id]
            mat[i, :len(attn)] = attn

        # Sort by binding_z
        binding_z = [design_data[inh_id]["binding_z"] for inh_id in cls_ids]
        sort_idx = np.argsort(binding_z)
        mat = mat[sort_idx]
        sorted_ids = [cls_ids[i] for i in sort_idx]
        sorted_bz = [binding_z[i] for i in sort_idx]

        if cls == "ITSN":
            y_labels = [f"{inh.replace('ITSN_RFD1_', '')} (z={bz:.1f})"
                        for inh, bz in zip(sorted_ids, sorted_bz)]
        else:
            y_labels = [f"{inh.replace('Vav_denovo_', 'V')} (z={bz:.1f})"
                        for inh, bz in zip(sorted_ids, sorted_bz)]

        fig, ax = plt.subplots(figsize=(max(14, max_len * 0.12), len(cls_ids) * 0.8 + 2))

        # Use a mask for NaN values
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_bad("white")
        im = ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest",
                       vmin=0, vmax=np.nanpercentile(mat, 99))
        ax.set_yticks(range(len(cls_ids)))
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_xlabel("Scaffold Position")
        ax.set_title(f"Per-Residue Attention — {cls} ({label})")

        # Mark junction region (scaffold_length from designs data)
        for i, inh_id in enumerate(sorted_ids):
            design = designs_df[designs_df["inhibitor_id"] == inh_id].iloc[0]
            junc = int(design["scaffold_length"])
            ax.axvline(x=junc - 0.5, color="cyan", linewidth=0.8, alpha=0.6)

        plt.colorbar(im, ax=ax, label="Attention Weight", shrink=0.8)
        plt.tight_layout()
        fig.savefig(FIG_DIR / f"attention_heatmap_{cls}_{label}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {FIG_DIR / f'attention_heatmap_{cls}_{label}.png'}")


def plot_attention_profile(design_data, attn_dict, designs_df, label):
    """Mean attention profile by position (ITSN overlay + mean)."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    for ax_idx, cls in enumerate(["ITSN", "Vav"]):
        ax = axes[ax_idx]
        cls_designs = designs_df[designs_df["class"] == cls]
        cls_ids = [d["inhibitor_id"] for _, d in cls_designs.iterrows()
                   if d["inhibitor_id"] in attn_dict]
        if not cls_ids:
            continue

        for inh_id in cls_ids:
            attn = attn_dict[inh_id]
            positions = design_data[inh_id]["positions"]
            bz = design_data[inh_id]["binding_z"]
            short = inh_id.replace("ITSN_RFD1_", "").replace("Vav_denovo_", "V")
            ax.plot(positions, attn, alpha=0.4, linewidth=0.8,
                    label=f"{short} (z={bz:.1f})")

        # Mean profile (per-position average, normalizing by count at each position)
        max_pos = max(len(attn_dict[inh_id]) for inh_id in cls_ids)
        mean_attn = np.zeros(max_pos)
        count_attn = np.zeros(max_pos)
        for inh_id in cls_ids:
            attn = attn_dict[inh_id]
            mean_attn[:len(attn)] += attn
            count_attn[:len(attn)] += 1
        mean_attn = np.divide(mean_attn, count_attn, out=np.zeros_like(mean_attn),
                              where=count_attn > 0)
        valid = count_attn > 0
        ax.plot(np.arange(1, max_pos + 1)[valid], mean_attn[valid],
                "k-", linewidth=2.5, label="Mean")

        # Uniform reference (use median scaffold length for this class)
        median_len = int(np.median([len(attn_dict[inh_id]) for inh_id in cls_ids]))
        ax.axhline(y=1.0 / median_len, color="gray", linestyle="--", alpha=0.5,
                   label=f"Uniform (1/{median_len})")

        # Junction
        junc = int(cls_designs.iloc[0]["scaffold_length"])
        ax.axvline(x=junc + 0.5, color="red", linestyle="--", linewidth=1.5,
                   label="DH junction")

        ax.set_xlabel("Scaffold Position")
        ax.set_ylabel("Attention Weight")
        ax.set_title(f"{cls} Per-Residue Attention Profile ({label})")
        ax.legend(fontsize=7, ncol=4 if cls == "ITSN" else 3)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(FIG_DIR / f"attention_profile_{label}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / f'attention_profile_{label}.png'}")


def plot_sweep_table(sweep_results):
    """Print and save sweep comparison table."""
    print("\n" + "=" * 100)
    print("ATTENTION SWEEP RESULTS")
    print("=" * 100)
    header = (f"{'Config':<22} {'Params':>6} {'ρ':>7} {'ITSN ρ':>8} {'Vav ρ':>7} "
              f"{'Gini':>6} {'Entropy':>8} {'MaxAttn':>8} {'Top10%':>7}")
    print(header)
    print("-" * 100)
    for r in sweep_results:
        print(f"{r['name']:<22} {r['n_params']:>6d} {r['spearman_rho']:>7.3f} "
              f"{r['within_itsn_rho']:>8.3f} {r['within_vav_rho']:>7.3f} "
              f"{r['gini']:>6.3f} {r['norm_entropy']:>8.4f} "
              f"{r['max_attn']:>8.5f} {r['top10_share']:>7.3f}")


def plot_comparison_bar(mlp_metrics, best_attn_metrics):
    """Bar chart comparing MLP vs best attention model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = [mlp_metrics["label"], best_attn_metrics["label"]]
    itsn = [mlp_metrics["within_itsn_rho"], best_attn_metrics["within_itsn_rho"]]
    vav = [mlp_metrics["within_vav_rho"], best_attn_metrics["within_vav_rho"]]

    x = np.arange(len(methods))
    w = 0.35
    ax.bar(x - w / 2, itsn, w, label="ITSN ρ", color="#2196F3")
    ax.bar(x + w / 2, vav, w, label="Vav ρ", color="#FF9800")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Within-class Spearman ρ")
    ax.set_title("MLP vs Best Interpretable Attention")
    ax.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════

def validate_hotspot_overlap(design_data, importances, designs_df, label):
    """Check if high-importance scaffold residues are near scaffold-DH junction."""
    print(f"\n  Hotspot validation ({label}):")
    results = []

    for _, design in designs_df.iterrows():
        inh_id = design["inhibitor_id"]
        if inh_id not in importances:
            continue

        imp = importances[inh_id]
        scaffold_len = int(design["scaffold_length"])
        positions = design_data[inh_id]["positions"]

        threshold = np.percentile(np.abs(imp), 80)
        top_positions = positions[np.abs(imp) >= threshold]

        junction = scaffold_len
        near_junction = sum(1 for p in top_positions if abs(p - junction) <= 10)
        total_near = sum(1 for p in positions if abs(p - junction) <= 10)
        expected = len(top_positions) * total_near / len(positions)
        enrichment = near_junction / max(expected, 0.1)

        results.append({
            "inhibitor_id": inh_id, "n_top": len(top_positions),
            "near_junction": near_junction, "expected": round(expected, 1),
            "enrichment": round(enrichment, 2),
        })
        status = "ENRICH" if enrichment > 1.5 else "ok" if enrichment > 1.0 else "low"
        print(f"    {inh_id:<35s} top_near_junction={near_junction}/{len(top_positions)} "
              f"(expected {expected:.1f}) [{status}]")

    return pd.DataFrame(results)


def generate_importance_pymol(design_data, importances, designs_df, label):
    """PyMOL scripts colored by attention weight."""
    with open(RESULTS / "best_models.json") as f:
        best_models = json.load(f)

    for _, design in designs_df.iterrows():
        inh_id = design["inhibitor_id"]
        if inh_id not in importances:
            continue

        imp = np.abs(importances[inh_id])
        positions = design_data[inh_id]["positions"]
        max_imp = imp.max() if imp.max() > 0 else 1.0

        bm = best_models[inh_id]
        cif_name = f"{bm['prefix']}_model_{bm['model_idx']}.cif"

        lines = [
            f"# PyMOL attention ({label}) for {inh_id}",
            f"load {cif_name}, {inh_id.replace('-', '_')}",
            f"color gray80, {inh_id.replace('-', '_')}",
            f"color palegreen, chain B", "",
        ]

        for pos, val in zip(positions, imp):
            norm = val / max_imp
            r = min(1.0, norm * 2)
            g = max(0, 1.0 - norm * 2)
            lines.append(f"set_color imp_{pos}, [{r:.2f}, {g:.2f}, 0.0]")
            lines.append(f"color imp_{pos}, chain A and resi {pos}")

        lines.extend(["", "# Switch regions",
                       "color marine, chain B and resi 29-42",
                       "color forest, chain B and resi 62-68"])

        with open(VAL_DIR / f"{inh_id}_{label}.pml", "w") as f:
            f.write("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Per-Residue Analysis Pipeline (Interpretability Focus)")
    print("=" * 60)

    df = pd.read_csv(FEATURE_DIR / "per_residue_features.csv")
    designs_df = pd.read_csv(DATA_CSV)
    df = class_normalize_binding(df)
    print(f"Loaded {len(df)} residue observations across {df['inhibitor_id'].nunique()} designs")

    # ── Layer 1: Statistics ───────────────────────────────────────
    layer1_statistics(df)

    # ── Prepare data for ML ───────────────────────────────────────
    design_data, feat_cols = prepare_data(df, designs_df)
    feat_dim = len(feat_cols)
    design_ids = list(design_data.keys())
    print(f"\nFeature dim: {feat_dim}")

    # ── Layer 2: MLP Contribution Model ───────────────────────────
    print("\n" + "=" * 60)
    print("LAYER 2: MLP Contribution Model")
    print("=" * 60)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device_str}")

    mlp_preds, mlp_imp = run_loo_cv(
        design_data, lambda fd: ResidueContributionModel(fd, hidden_dim=32),
        feat_dim, n_seeds=10, n_epochs=500, lr=0.003,
    )
    mlp_metrics = evaluate_predictions(design_data, mlp_preds, design_ids, "MLP")
    np.savez(MODEL_DIR / "mlp_contributions.npz", **mlp_imp)

    # ── Layer 3: Interpretable Attention Sweep ────────────────────
    print("\n" + "=" * 60)
    print("LAYER 3: Interpretable Attention Sweep")
    print("=" * 60)
    print(f"  Using device: {device_str}")
    print(f"  Sweeping {len(SWEEP_CONFIGS)} configurations...")

    sweep_results, attn_cache = run_attention_sweep(
        design_data, feat_dim, SWEEP_CONFIGS,
        n_seeds=5, n_epochs=300, lr=0.003,
    )

    # ── Select best interpretable model ───────────────────────────
    print("\n" + "=" * 60)
    print("MODEL SELECTION (Pareto Frontier)")
    print("=" * 60)
    best_cfg = select_best_config(sweep_results)
    best_name = best_cfg["name"]
    best_attn = attn_cache[best_name]

    # Rerun best model with more seeds for stable attention weights
    print(f"\n  Rerunning {best_name} with 10 seeds, 500 epochs for final weights...")
    h, t = best_cfg["hidden_dim"], best_cfg["temperature"]
    best_preds, best_attn_final = run_loo_cv(
        design_data,
        lambda fd, _h=h, _t=t: InterpretableAttention(fd, hidden_dim=_h, temperature=_t),
        feat_dim, n_seeds=10, n_epochs=500, lr=0.003,
        entropy_weight=best_cfg["entropy_weight"],
    )
    best_metrics = evaluate_predictions(design_data, best_preds, design_ids,
                                         f"Best ({best_name})")
    best_attn_stats = attention_statistics(best_attn_final)
    print(f"    Final Gini={best_attn_stats['gini']:.3f}, "
          f"MaxAttn={best_attn_stats['max_attn']:.5f}, "
          f"Top10%={best_attn_stats['top10_share']:.3f}")

    # ── Layer 4: Transformer (Self-Attention) Sweep ─────────────
    print("\n" + "=" * 60)
    print("LAYER 4: Transformer Sweep (does residue-residue context help?)")
    print("=" * 60)
    print(f"  Using device: {device_str}")
    print(f"  Sweeping {len(TRANSFORMER_CONFIGS)} configurations...")

    tf_results, tf_contribs, tf_self_attn = run_transformer_sweep(
        design_data, feat_dim, TRANSFORMER_CONFIGS,
        n_seeds=5, n_epochs=300, lr=0.003,
    )

    # Select best transformer: highest overall ρ
    best_tf = max(tf_results, key=lambda r: r["spearman_rho"])
    best_tf_name = best_tf["name"]
    print(f"\n  Best transformer: {best_tf_name} (ρ={best_tf['spearman_rho']:.3f})")

    # Rerun best transformer with more seeds
    print(f"  Rerunning {best_tf_name} with 10 seeds, 500 epochs...")
    _h, _nh, _do, _wd = best_tf["hidden_dim"], best_tf["n_heads"], best_tf["dropout"], best_tf["wd"]
    best_tf_preds, best_tf_contribs, best_tf_sa = run_loo_cv(
        design_data,
        lambda fd, h=_h, nh=_nh, do=_do: MinimalTransformer(fd, hidden_dim=h, n_heads=nh, dropout=do),
        feat_dim, n_seeds=10, n_epochs=500, lr=0.003,
        weight_decay=_wd, collect_self_attn=True,
    )
    best_tf_metrics = evaluate_predictions(design_data, best_tf_preds, design_ids,
                                            f"Best TF ({best_tf_name})")

    # ── Save ──────────────────────────────────────────────────────
    np.savez(MODEL_DIR / "best_attention_weights.npz", **best_attn_final)
    np.savez(MODEL_DIR / "best_transformer_contribs.npz", **best_tf_contribs)
    np.savez(MODEL_DIR / "best_transformer_self_attn.npz",
             **{k: v for k, v in best_tf_sa.items()})

    for cfg_name, attn_dict in attn_cache.items():
        np.savez(MODEL_DIR / f"attn_{cfg_name}.npz", **attn_dict)

    sweep_df = pd.DataFrame(sweep_results)
    sweep_df.to_csv(MODEL_DIR / "attention_sweep_results.csv", index=False)
    tf_df = pd.DataFrame(tf_results)
    tf_df.to_csv(MODEL_DIR / "transformer_sweep_results.csv", index=False)

    # Predictions CSV
    pred_rows = []
    for i, inh_id in enumerate(design_ids):
        pred_rows.append({
            "inhibitor_id": inh_id,
            "class": design_data[inh_id]["class"],
            "actual_z": design_data[inh_id]["binding_z"],
            "actual_raw": design_data[inh_id]["binding_raw"],
            "pred_mlp": float(mlp_preds[i]),
            "pred_best_attn": float(best_preds[i]),
            "pred_best_tf": float(best_tf_preds[i]),
        })
    pd.DataFrame(pred_rows).to_csv(MODEL_DIR / "cv_predictions.csv", index=False)

    # ── Visualize ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VISUALIZATION & VALIDATION")
    print("=" * 60)

    # Sweep results tables
    plot_sweep_table(sweep_results)
    print_transformer_table(mlp_metrics, tf_results)

    # Pareto plot (attention sweep)
    plot_pareto(sweep_results)

    # Attention heatmaps and profiles
    plot_attention_heatmap(design_data, best_attn_final, designs_df, best_name)
    plot_attention_profile(design_data, best_attn_final, designs_df, best_name)

    # Transformer visualizations
    plot_transformer_vs_mlp(mlp_metrics, tf_results, best_tf_metrics)
    plot_self_attention_heatmap(best_tf_sa, design_data, designs_df, best_tf_name)

    # Validation
    validate_hotspot_overlap(design_data, mlp_imp, designs_df, "MLP")
    validate_hotspot_overlap(design_data, best_attn_final, designs_df, best_name)
    validate_hotspot_overlap(design_data, best_tf_contribs, designs_df, best_tf_name)

    # PyMOL scripts
    generate_importance_pymol(design_data, best_attn_final, designs_df, best_name)
    generate_importance_pymol(design_data, best_tf_contribs, designs_df, best_tf_name)
    generate_importance_pymol(design_data, mlp_imp, designs_df, "MLP")

    # Comparison
    plot_comparison_bar(mlp_metrics, best_metrics)

    # ── Final summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Method':<30} {'ρ':>7} {'ITSN ρ':>8} {'Vav ρ':>7} {'Gini':>6}")
    print("-" * 60)
    print(f"{'MLP (no context)':<30} {mlp_metrics['spearman_rho']:>7.3f} "
          f"{mlp_metrics['within_itsn_rho']:>8.3f} {mlp_metrics['within_vav_rho']:>7.3f} "
          f"{'N/A':>6}")
    print(f"{'Best Attention (score-val)':<30} {best_metrics['spearman_rho']:>7.3f} "
          f"{best_metrics['within_itsn_rho']:>8.3f} {best_metrics['within_vav_rho']:>7.3f} "
          f"{best_attn_stats['gini']:>6.3f}")
    print(f"{'Best Transformer (context)':<30} {best_tf_metrics['spearman_rho']:>7.3f} "
          f"{best_tf_metrics['within_itsn_rho']:>8.3f} {best_tf_metrics['within_vav_rho']:>7.3f} "
          f"{'N/A':>6}")

    # Key question answered
    mlp_rho = mlp_metrics["spearman_rho"]
    tf_rho = best_tf_metrics["spearman_rho"]
    delta = tf_rho - mlp_rho
    print(f"\n  Context effect: Δρ = {delta:+.3f} "
          f"({'context helps' if delta > 0.05 else 'context hurts' if delta < -0.05 else 'marginal'})")


if __name__ == "__main__":
    main()
