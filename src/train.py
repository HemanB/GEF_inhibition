#!/usr/bin/env python3
"""Phase 5: LOO-CV Training for the attention-based model."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy import stats

from model import ContactAttentionModel

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT / "data" / "raw" / "GEF_inhibitors_modeling_data.csv"
RESULTS = PROJECT / "results"
CONTACT_DIR = PROJECT / "data" / "processed" / "contact_maps"
FEATURE_DIR = PROJECT / "data" / "processed" / "features"
MODEL_DIR = RESULTS / "model_outputs"
ATTN_DIR = RESULTS / "attention_weights"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ATTN_DIR.mkdir(parents=True, exist_ok=True)

# GTPase regions (1-indexed)
SWITCH_I = set(range(29, 43))
SWITCH_II = set(range(62, 69))

# Contact type encoding
CONTACT_TYPES = ["AA", "AC", "AP", "CC", "CP", "PP"]


def prepare_contact_features(inh_id: str, scaffold_len: int) -> np.ndarray:
    """Build per-contact feature matrix for one design.

    Features (15 dims):
    - contact_type one-hot (6)
    - distance (1)
    - AF3 contact_prob placeholder (1) - from full_data if available
    - AF3 PAE placeholder (1)
    - is_scaffold (1)
    - GTPase region one-hot (3): switch_I, switch_II, other
    - burial placeholders (2)
    """
    contact_file = CONTACT_DIR / f"{inh_id}_contacts.json"
    with open(contact_file) as f:
        contacts = json.load(f)

    # Load AF3 full data for contact_probs and PAE
    bm_path = RESULTS / "best_models.json"
    af3_cp = {}
    af3_pae = {}
    if bm_path.exists():
        with open(bm_path) as f:
            best_models = json.load(f)
        bm = best_models.get(inh_id, {})
        if bm:
            dirname = bm["dirname"]
            prefix = bm["prefix"]
            model_idx = bm["model_idx"]
            fd_path = (PROJECT / "data" / "af3_server_outputs" / dirname /
                       f"{prefix}_full_data_{model_idx}.json")
            if fd_path.exists():
                with open(fd_path) as f:
                    fd = json.load(f)
                cp_matrix = np.array(fd["contact_probs"])
                pae_matrix = np.array(fd["pae"])
                token_chain_ids = fd["token_chain_ids"]
                chain_a_idx = [i for i in range(len(token_chain_ids))
                               if token_chain_ids[i] == "A"]
                chain_b_idx = [i for i in range(len(token_chain_ids))
                               if token_chain_ids[i] == "B"]
                cp_AB = cp_matrix[np.ix_(chain_a_idx, chain_b_idx)]
                pae_AB = pae_matrix[np.ix_(chain_a_idx, chain_b_idx)]
                for c in contacts:
                    a_tok = c["res_a_idx"] - 1
                    b_tok = c["res_b_idx"] - 1
                    if a_tok < cp_AB.shape[0] and b_tok < cp_AB.shape[1]:
                        af3_cp[(c["res_a_idx"], c["res_b_idx"])] = cp_AB[a_tok, b_tok]
                        af3_pae[(c["res_a_idx"], c["res_b_idx"])] = pae_AB[a_tok, b_tok]

    # Filter to contacts at 8A (to include scaffold contacts)
    contacts_8 = [c for c in contacts if c["at_8_0"]]

    if not contacts_8:
        return np.zeros((1, 15), dtype=np.float32), contacts_8

    features = []
    for c in contacts_8:
        feat = []
        # Contact type one-hot (6)
        ct = c["contact_type"]
        for t in CONTACT_TYPES:
            feat.append(1.0 if ct == t else 0.0)

        # Distance (1) - normalized
        feat.append(c["distance"] / 8.0)

        # AF3 contact prob (1)
        key = (c["res_a_idx"], c["res_b_idx"])
        feat.append(float(af3_cp.get(key, 0.0)))

        # AF3 PAE (1) - normalized
        feat.append(float(af3_pae.get(key, 10.0)) / 30.0)

        # Is scaffold (1)
        feat.append(1.0 if c["res_a_idx"] <= scaffold_len else 0.0)

        # GTPase region one-hot (3)
        b_idx = c["res_b_idx"]
        feat.append(1.0 if b_idx in SWITCH_I else 0.0)
        feat.append(1.0 if b_idx in SWITCH_II else 0.0)
        feat.append(1.0 if b_idx not in SWITCH_I and b_idx not in SWITCH_II else 0.0)

        # Burial placeholders (2)
        feat.append(0.5)  # res_a burial placeholder
        feat.append(0.5)  # res_b burial placeholder

        features.append(feat)

    return np.array(features, dtype=np.float32), contacts_8


def run_loo_cv(design_ids: list[str], binding_values: np.ndarray,
               scaffold_lens: dict, label: str = "all",
               n_seeds: int = 10, n_epochs: int = 300,
               lr: float = 0.005, weight_decay: float = 0.01):
    """Run leave-one-out cross-validation with multiple seeds."""
    n = len(design_ids)
    print(f"\n{'='*50}")
    print(f"LOO-CV: {label} (n={n}), {n_seeds} seeds")
    print(f"{'='*50}")

    # Prepare all contact features
    all_contact_feats = {}
    all_contacts = {}
    for inh_id in design_ids:
        feats, contacts = prepare_contact_features(inh_id, scaffold_lens[inh_id])
        all_contact_feats[inh_id] = feats
        all_contacts[inh_id] = contacts

    # Pad to same length
    max_contacts = max(f.shape[0] for f in all_contact_feats.values())
    input_dim = list(all_contact_feats.values())[0].shape[1]

    def pad_features(feats, max_len):
        n_contacts = feats.shape[0]
        if n_contacts >= max_len:
            return feats[:max_len], np.zeros(max_len, dtype=bool)
        padded = np.zeros((max_len, feats.shape[1]), dtype=np.float32)
        padded[:n_contacts] = feats
        mask = np.ones(max_len, dtype=bool)
        mask[:n_contacts] = False
        return padded, mask

    # Normalize binding values
    binding_mean_val = binding_values.mean()
    binding_std_val = binding_values.std()
    binding_norm = (binding_values - binding_mean_val) / binding_std_val

    # LOO predictions across seeds
    all_predictions = np.zeros((n_seeds, n))
    all_attention_weights = {}

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        predictions = np.zeros(n)

        for loo_idx in range(n):
            # Split
            test_id = design_ids[loo_idx]
            train_ids = [d for i, d in enumerate(design_ids) if i != loo_idx]
            train_binding = np.array([binding_norm[i] for i in range(n) if i != loo_idx])

            # Prepare training data
            train_feats = []
            train_masks = []
            for tid in train_ids:
                pf, pm = pad_features(all_contact_feats[tid], max_contacts)
                train_feats.append(pf)
                train_masks.append(pm)

            train_X = torch.tensor(np.stack(train_feats))
            train_mask = torch.tensor(np.stack(train_masks))
            train_y = torch.tensor(train_binding, dtype=torch.float32).unsqueeze(1)

            # Test data
            test_pf, test_pm = pad_features(all_contact_feats[test_id], max_contacts)
            test_X = torch.tensor(test_pf).unsqueeze(0)
            test_mask = torch.tensor(test_pm).unsqueeze(0)

            # Model
            model = ContactAttentionModel(
                input_dim=input_dim, hidden_dim=16,
                n_heads=4, dropout=0.1
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                         weight_decay=weight_decay)
            criterion = nn.MSELoss()

            # Train
            model.train()
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                pred, _ = model(train_X, train_mask)
                loss = criterion(pred, train_y)
                loss.backward()
                optimizer.step()

            # Predict
            model.eval()
            with torch.no_grad():
                pred, attn = model(test_X, test_mask)
                pred_val = pred.item() * binding_std_val + binding_mean_val
                predictions[loo_idx] = pred_val

                # Save attention weights for last seed
                if seed == n_seeds - 1:
                    n_real = all_contact_feats[test_id].shape[0]
                    attn_np = attn.squeeze().numpy()[:n_real]
                    all_attention_weights[test_id] = attn_np

        all_predictions[seed] = predictions

    # Average predictions across seeds
    mean_predictions = all_predictions.mean(axis=0)

    # Save attention weights
    for inh_id, attn_w in all_attention_weights.items():
        np.save(ATTN_DIR / f"{inh_id}_attention.npy", attn_w)

    return mean_predictions, all_attention_weights


def run_linear_baseline(design_ids: list[str], binding_values: np.ndarray,
                         feature_df: pd.DataFrame, label: str = "all"):
    """Run Ridge regression LOO-CV baseline."""
    n = len(design_ids)
    predictions = np.zeros(n)

    # Select numeric features
    feat_cols = [c for c in feature_df.columns
                 if c not in ("inhibitor_id", "class", "binding_mean", "target_gtpase")]
    X = feature_df.set_index("inhibitor_id").loc[design_ids, feat_cols].values

    # Fill NaN with 0
    X = np.nan_to_num(X, nan=0.0)

    for loo_idx in range(n):
        X_train = np.delete(X, loo_idx, axis=0)
        y_train = np.delete(binding_values, loo_idx)
        X_test = X[loo_idx:loo_idx+1]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train)
        predictions[loo_idx] = model.predict(X_test_s)[0]

    return predictions


def evaluate(actual: np.ndarray, predicted: np.ndarray, label: str) -> dict:
    """Compute evaluation metrics."""
    r, p = stats.pearsonr(actual, predicted)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    rho, _ = stats.spearmanr(actual, predicted)

    metrics = {
        "label": label,
        "n": len(actual),
        "pearson_r": round(float(r), 4),
        "pearson_p": round(float(p), 4),
        "spearman_rho": round(float(rho), 4),
        "r2": round(float(r2), 4),
        "mae": round(float(mae), 1),
        "rmse": round(float(rmse), 1),
    }
    print(f"\n  {label}: r={r:.3f} (p={p:.3f}), R²={r2:.3f}, "
          f"MAE={mae:.0f}, RMSE={rmse:.0f}, ρ={rho:.3f}")
    return metrics


def main():
    print("=" * 60)
    print("Phase 5: Attention-Based Model Training")
    print("=" * 60)

    designs = pd.read_csv(DATA_CSV)
    feature_df = pd.read_csv(FEATURE_DIR / "all_features.csv")

    # Prepare data subsets
    itsn = designs[designs["class"] == "ITSN"]
    vav = designs[designs["class"] == "Vav"]

    subsets = {
        "ITSN": (itsn["inhibitor_id"].tolist(), itsn["binding_mean"].values.astype(float)),
        "Vav": (vav["inhibitor_id"].tolist(), vav["binding_mean"].values.astype(float)),
        "Combined": (designs["inhibitor_id"].tolist(), designs["binding_mean"].values.astype(float)),
    }

    scaffold_lens = dict(zip(designs["inhibitor_id"],
                             designs["scaffold_length"].astype(int)))

    all_metrics = []
    all_predictions_rows = []

    for subset_name, (ids, binding) in subsets.items():
        print(f"\n\n{'#'*50}")
        print(f"# Subset: {subset_name} (n={len(ids)})")
        print(f"{'#'*50}")

        # Model parameters
        model_info = ContactAttentionModel(input_dim=15)
        print(f"Model parameters: {model_info.count_parameters()}")

        # Attention model LOO-CV
        attn_preds, attn_weights = run_loo_cv(
            ids, binding, scaffold_lens,
            label=subset_name,
            n_seeds=10, n_epochs=300
        )
        metrics_attn = evaluate(binding, attn_preds, f"{subset_name}_attention")
        all_metrics.append(metrics_attn)

        # Linear baseline
        linear_preds = run_linear_baseline(ids, binding, feature_df, subset_name)
        metrics_linear = evaluate(binding, linear_preds, f"{subset_name}_linear")
        all_metrics.append(metrics_linear)

        # Save predictions
        for i, inh_id in enumerate(ids):
            all_predictions_rows.append({
                "inhibitor_id": inh_id,
                "subset": subset_name,
                "actual": float(binding[i]),
                "predicted_attention": float(attn_preds[i]),
                "predicted_linear": float(linear_preds[i]),
            })

    # Save results
    pred_df = pd.DataFrame(all_predictions_rows)
    pred_df.to_csv(MODEL_DIR / "loo_predictions.csv", index=False)

    with open(MODEL_DIR / "loo_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n\nSaved predictions: {MODEL_DIR / 'loo_predictions.csv'}")
    print(f"Saved metrics: {MODEL_DIR / 'loo_metrics.json'}")

    # Print summary
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    print(f"{'Subset':<25} {'Model':<12} {'r':>6} {'R²':>6} {'MAE':>8}")
    print("-" * 60)
    for m in all_metrics:
        parts = m["label"].rsplit("_", 1)
        subset = parts[0]
        model_type = parts[1]
        print(f"{subset:<25} {model_type:<12} {m['pearson_r']:>6.3f} {m['r2']:>6.3f} {m['mae']:>8.0f}")


if __name__ == "__main__":
    main()
