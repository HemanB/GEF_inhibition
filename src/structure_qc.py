#!/usr/bin/env python3
"""Phase 1: Structure QC & Model Selection for AF3 predictions."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT / "data" / "raw" / "GEF_inhibitors_modeling_data.csv"
AF3_DIR = PROJECT / "data" / "af3_server_outputs"
RESULTS = PROJECT / "results"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def inhibitor_to_dirname(inhibitor_id: str, gtpase: str) -> str:
    """Convert inhibitor_id + gtpase to AF3 output directory name."""
    return f"{inhibitor_id}_{gtpase}".lower().replace("-", "_")


def load_design_info() -> list[dict]:
    """Load design metadata from CSV."""
    with open(DATA_CSV) as f:
        return list(csv.DictReader(f))


def parse_summary_confidences(design_dir: Path, prefix: str) -> list[dict]:
    """Parse all 5 summary_confidences JSONs for one design."""
    records = []
    for model_idx in range(5):
        fname = f"{prefix}_summary_confidences_{model_idx}.json"
        fpath = design_dir / fname
        if not fpath.exists():
            print(f"  WARNING: missing {fpath}")
            continue
        with open(fpath) as f:
            sc = json.load(f)
        records.append({
            "model_idx": model_idx,
            "iptm": sc.get("iptm", np.nan),
            "ptm": sc.get("ptm", np.nan),
            "ranking_score": sc.get("ranking_score", np.nan),
            "has_clash": sc.get("has_clash", 0.0),
            "fraction_disordered": sc.get("fraction_disordered", np.nan),
            "chain_iptm_A": sc.get("chain_iptm", [np.nan, np.nan])[0],
            "chain_iptm_B": sc.get("chain_iptm", [np.nan, np.nan])[1],
            "chain_ptm_A": sc.get("chain_ptm", [np.nan, np.nan])[0],
            "chain_ptm_B": sc.get("chain_ptm", [np.nan, np.nan])[1],
            "cross_chain_pae_AB": sc.get("chain_pair_pae_min", [[np.nan, np.nan]])[0][1]
                if len(sc.get("chain_pair_pae_min", [])) > 0 else np.nan,
            "cross_chain_pae_BA": sc.get("chain_pair_pae_min", [[], [np.nan, np.nan]])[1][0]
                if len(sc.get("chain_pair_pae_min", [])) > 1 else np.nan,
        })
    return records


def extract_interface_plddt(full_data: dict, n_inhib: int) -> dict:
    """Extract interface pLDDT from full_data JSON.

    Chain A = inhibitor (tokens 0..n_inhib-1), Chain B = GTPase.
    Interface = residues with cross-chain contact_prob > 0.5.
    """
    contact_probs = np.array(full_data["contact_probs"])
    atom_plddts = np.array(full_data["atom_plddts"])
    token_chain_ids = full_data["token_chain_ids"]
    token_res_ids = full_data["token_res_ids"]

    n_tokens = len(token_chain_ids)
    chain_a_idx = [i for i in range(n_tokens) if token_chain_ids[i] == "A"]
    chain_b_idx = [i for i in range(n_tokens) if token_chain_ids[i] == "B"]

    # Cross-chain PAE block
    pae = np.array(full_data["pae"])
    pae_AB = pae[np.ix_(chain_a_idx, chain_b_idx)]

    # Cross-chain contact probs
    cp_AB = contact_probs[np.ix_(chain_a_idx, chain_b_idx)]

    # Interface residues: those with any cross-chain contact_prob > 0.5
    inhib_interface = np.any(cp_AB > 0.5, axis=1)
    gtpase_interface = np.any(cp_AB > 0.5, axis=0)

    # Per-token pLDDT (mean of atom plddts per token)
    # Map atoms to tokens via chain_ids
    atom_chain_ids = full_data["atom_chain_ids"]
    # Build per-token pLDDT by averaging atom plddts
    # Atoms are ordered by token, need to group them
    token_plddts = []
    atom_idx = 0
    for tok_idx in range(n_tokens):
        tok_chain = token_chain_ids[tok_idx]
        tok_res = token_res_ids[tok_idx]
        atom_group = []
        while atom_idx < len(atom_chain_ids) and atom_chain_ids[atom_idx] == tok_chain:
            # Check if this atom belongs to current residue
            # Atoms are sequential within chains
            atom_group.append(atom_plddts[atom_idx])
            atom_idx += 1
            # Rough grouping: standard amino acid has ~8-14 heavy atoms
            # We'll use all atoms until chain changes
            if atom_idx < len(atom_chain_ids) and atom_chain_ids[atom_idx] == tok_chain:
                # Keep going in same chain
                pass
            else:
                break
        if atom_group:
            token_plddts.append(np.mean(atom_group))
        else:
            token_plddts.append(np.nan)

    # Simpler approach: use atom_plddts directly grouped by chain
    # Reset and do it properly
    token_plddts_a = []
    token_plddts_b = []

    # Count atoms per chain
    a_atoms = [i for i, c in enumerate(atom_chain_ids) if c == "A"]
    b_atoms = [i for i, c in enumerate(atom_chain_ids) if c == "B"]

    n_a_tokens = len(chain_a_idx)
    n_b_tokens = len(chain_b_idx)

    # Approximate: distribute atoms evenly across tokens per chain
    if a_atoms:
        a_plddts = atom_plddts[a_atoms]
        atoms_per_tok = len(a_atoms) / n_a_tokens
        for i in range(n_a_tokens):
            start = int(i * atoms_per_tok)
            end = int((i + 1) * atoms_per_tok)
            token_plddts_a.append(np.mean(a_plddts[start:end]))

    if b_atoms:
        b_plddts = atom_plddts[b_atoms]
        atoms_per_tok = len(b_atoms) / n_b_tokens
        for i in range(n_b_tokens):
            start = int(i * atoms_per_tok)
            end = int((i + 1) * atoms_per_tok)
            token_plddts_b.append(np.mean(b_plddts[start:end]))

    token_plddts_a = np.array(token_plddts_a)
    token_plddts_b = np.array(token_plddts_b)

    # Interface pLDDT
    if np.any(inhib_interface):
        iface_plddt_A = float(np.mean(token_plddts_a[inhib_interface]))
    else:
        iface_plddt_A = float(np.mean(token_plddts_a))

    if np.any(gtpase_interface):
        iface_plddt_B = float(np.mean(token_plddts_b[gtpase_interface]))
    else:
        iface_plddt_B = float(np.mean(token_plddts_b))

    return {
        "interface_plddt_A": round(iface_plddt_A, 2),
        "interface_plddt_B": round(iface_plddt_B, 2),
        "mean_cross_pae": round(float(np.mean(pae_AB)), 2),
        "n_interface_res_A": int(np.sum(inhib_interface)),
        "n_interface_res_B": int(np.sum(gtpase_interface)),
    }


def run_qc() -> tuple[pd.DataFrame, dict]:
    """Run full QC analysis. Returns (qc_df, best_models)."""
    designs = load_design_info()
    all_records = []
    best_models = {}

    for design in designs:
        inh_id = design["inhibitor_id"]
        gtpase = design["target_gtpase"]
        cls = design["class"]
        dirname = inhibitor_to_dirname(inh_id, gtpase)
        design_dir = AF3_DIR / dirname
        prefix = f"fold_{dirname}"

        if not design_dir.exists():
            print(f"WARNING: directory not found for {inh_id}: {design_dir}")
            continue

        # Parse all 5 models
        model_records = parse_summary_confidences(design_dir, prefix)

        # Find best model
        best_idx = max(range(len(model_records)),
                       key=lambda i: model_records[i]["ranking_score"])
        best_models[inh_id] = {
            "model_idx": model_records[best_idx]["model_idx"],
            "ranking_score": model_records[best_idx]["ranking_score"],
            "dirname": dirname,
            "prefix": prefix,
        }

        # Extract interface pLDDT for best model
        full_data_path = design_dir / f"{prefix}_full_data_{best_idx}.json"
        iface_metrics = {}
        if full_data_path.exists():
            with open(full_data_path) as f:
                full_data = json.load(f)
            n_inhib = int(design["total_length"])
            iface_metrics = extract_interface_plddt(full_data, n_inhib)

        for rec in model_records:
            row = {
                "inhibitor_id": inh_id,
                "class": cls,
                "target_gtpase": gtpase,
                "is_best": rec["model_idx"] == best_idx,
                **rec,
            }
            if rec["model_idx"] == best_idx:
                row.update(iface_metrics)
            all_records.append(row)

    qc_df = pd.DataFrame(all_records)
    return qc_df, best_models


def plot_qc_summary(qc_df: pd.DataFrame):
    """Generate QC summary figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. iPTM distribution by design
    ax = axes[0, 0]
    best = qc_df[qc_df["is_best"]]
    colors = ["#2196F3" if c == "ITSN" else "#FF9800" for c in best["class"]]
    ax.bar(range(len(best)), best["iptm"].values, color=colors)
    ax.set_xticks(range(len(best)))
    ax.set_xticklabels(best["inhibitor_id"].values, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("iPTM")
    ax.set_title("iPTM (Best Model per Design)")
    ax.axhline(y=0.85, color="gray", linestyle="--", alpha=0.5, label="0.85 threshold")
    ax.legend(fontsize=8)

    # 2. Ranking score distribution
    ax = axes[0, 1]
    for cls, color in [("ITSN", "#2196F3"), ("Vav", "#FF9800")]:
        subset = qc_df[qc_df["class"] == cls]
        ax.scatter(subset["iptm"], subset["ranking_score"], c=color,
                   alpha=0.6, label=cls, edgecolors="k", linewidths=0.3)
    ax.set_xlabel("iPTM")
    ax.set_ylabel("Ranking Score")
    ax.set_title("iPTM vs Ranking Score (All Models)")
    ax.legend()

    # 3. Cross-chain PAE for best models
    ax = axes[1, 0]
    ax.bar(range(len(best)), best["cross_chain_pae_AB"].values, color=colors)
    ax.set_xticks(range(len(best)))
    ax.set_xticklabels(best["inhibitor_id"].values, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Min Cross-Chain PAE (A→B)")
    ax.set_title("Cross-Chain PAE (Best Models)")

    # 4. Interface pLDDT for best models
    ax = axes[1, 1]
    if "interface_plddt_A" in best.columns:
        x = np.arange(len(best))
        w = 0.35
        ax.bar(x - w/2, best["interface_plddt_A"].values, w, label="Inhibitor", color=colors, alpha=0.7)
        ax.bar(x + w/2, best["interface_plddt_B"].values, w, label="GTPase", color="gray", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(best["inhibitor_id"].values, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Interface pLDDT")
        ax.set_title("Interface pLDDT (Best Models)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "qc_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'qc_summary.png'}")


def main():
    print("=" * 60)
    print("Phase 1: Structure QC & Model Selection")
    print("=" * 60)

    qc_df, best_models = run_qc()

    # Save QC report
    qc_path = RESULTS / "qc_report.csv"
    qc_df.to_csv(qc_path, index=False)
    print(f"\nSaved QC report: {qc_path} ({len(qc_df)} rows)")

    # Save best models
    bm_path = RESULTS / "best_models.json"
    with open(bm_path, "w") as f:
        json.dump(best_models, f, indent=2)
    print(f"Saved best models: {bm_path}")

    # Print summary
    best = qc_df[qc_df["is_best"]]
    print(f"\n{'Design':<35} {'iPTM':>6} {'PTM':>6} {'Rank':>6} {'Clash':>6} {'PAE_AB':>7}")
    print("-" * 75)
    for _, row in best.iterrows():
        print(f"{row['inhibitor_id']:<35} {row['iptm']:>6.3f} {row['ptm']:>6.3f} "
              f"{row['ranking_score']:>6.3f} {row['has_clash']:>6.1f} "
              f"{row['cross_chain_pae_AB']:>7.2f}")

    print(f"\niPTM range: {best['iptm'].min():.3f} - {best['iptm'].max():.3f}")
    print(f"All models clash-free: {(qc_df['has_clash'] == 0).all()}")

    # Plot
    plot_qc_summary(qc_df)

    return qc_df, best_models


if __name__ == "__main__":
    main()
