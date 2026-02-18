#!/usr/bin/env python3
"""Phase 6: Visualization & Validation of attention-based analysis."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT / "data" / "raw" / "GEF_inhibitors_modeling_data.csv"
RESULTS = PROJECT / "results"
FIG_DIR = RESULTS / "figures"
CONTACT_DIR = PROJECT / "data" / "processed" / "contact_maps"
ATTN_DIR = RESULTS / "attention_weights"
MODEL_DIR = RESULTS / "model_outputs"
PYMOL_DIR = RESULTS / "pymol_scripts"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PYMOL_DIR.mkdir(parents=True, exist_ok=True)

SWITCH_I = set(range(29, 43))
SWITCH_II = set(range(62, 69))


def plot_attention_heatmap(inh_id: str, contacts: list[dict],
                           attn_weights: np.ndarray, scaffold_len: int,
                           hotspots: list[int], n_inhib: int, n_gtpase: int,
                           cls: str):
    """Plot attention heatmap (inhibitor x GTPase) with annotations."""
    contacts_8 = [c for c in contacts if c["at_8_0"]]

    if len(contacts_8) == 0 or len(attn_weights) == 0:
        return

    # Build attention map matrix
    attn_map = np.zeros((n_inhib, n_gtpase))
    n_real = min(len(attn_weights), len(contacts_8))
    for i in range(n_real):
        c = contacts_8[i]
        a_idx = c["res_a_idx"] - 1  # 0-indexed
        b_idx = c["res_b_idx"] - 1
        if 0 <= a_idx < n_inhib and 0 <= b_idx < n_gtpase:
            attn_map[a_idx, b_idx] += attn_weights[i]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Custom colormap
    cmap = LinearSegmentedColormap.from_list("attn", ["white", "#FFF3E0", "#FF9800", "#E65100", "#B71C1C"])

    # Plot heatmap (only show relevant range)
    # Find range with non-zero attention
    row_mask = attn_map.sum(axis=1) > 0
    col_mask = attn_map.sum(axis=0) > 0

    if row_mask.any() and col_mask.any():
        row_min = max(0, np.where(row_mask)[0][0] - 5)
        row_max = min(n_inhib, np.where(row_mask)[0][-1] + 6)
        col_min = max(0, np.where(col_mask)[0][0] - 5)
        col_max = min(n_gtpase, np.where(col_mask)[0][-1] + 6)
    else:
        row_min, row_max = 0, n_inhib
        col_min, col_max = 0, n_gtpase

    sub_map = attn_map[row_min:row_max, col_min:col_max]

    im = ax.imshow(sub_map.T, aspect="auto", cmap=cmap,
                   extent=[row_min+1, row_max+1, col_max+1, col_min+1],
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Attention Weight", shrink=0.8)

    # Annotate switch regions
    for switch_res in SWITCH_I:
        if col_min < switch_res - 1 < col_max:
            ax.axhline(y=switch_res, color="#2196F3", linewidth=0.5, alpha=0.6)
    for switch_res in SWITCH_II:
        if col_min < switch_res - 1 < col_max:
            ax.axhline(y=switch_res, color="#4CAF50", linewidth=0.5, alpha=0.6)

    # Annotate scaffold boundary
    ax.axvline(x=scaffold_len + 0.5, color="red", linewidth=1.5, linestyle="--", alpha=0.8)

    # Annotate hotspots
    for hp in hotspots:
        if row_min < hp - 1 < row_max:
            ax.axvline(x=hp, color="magenta", linewidth=0.8, linestyle=":", alpha=0.8)

    # Labels
    ax.set_xlabel(f"Inhibitor Position (scaffold ← | → DH domain)", fontsize=11)
    ax.set_ylabel(f"GTPase Position ({'Cdc42' if cls == 'ITSN' else 'Rac1'})", fontsize=11)
    ax.set_title(f"Attention Heatmap: {inh_id}", fontsize=13, fontweight="bold")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#2196F3", alpha=0.6, label="Switch I (29-42)"),
        mpatches.Patch(facecolor="#4CAF50", alpha=0.6, label="Switch II (62-68)"),
        plt.Line2D([0], [0], color="red", linestyle="--", label=f"Scaffold boundary ({scaffold_len})"),
        plt.Line2D([0], [0], color="magenta", linestyle=":", label="Hotspot positions"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / f"attention_heatmap_{inh_id}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residue_importance(inh_id: str, contacts: list[dict],
                            attn_weights: np.ndarray, scaffold_len: int,
                            hotspots: list[int], n_inhib: int):
    """Plot per-residue importance (summed attention) for inhibitor."""
    contacts_8 = [c for c in contacts if c["at_8_0"]]
    n_real = min(len(attn_weights), len(contacts_8))

    # Sum attention per inhibitor residue
    res_importance = np.zeros(n_inhib)
    for i in range(n_real):
        c = contacts_8[i]
        a_idx = c["res_a_idx"] - 1
        if 0 <= a_idx < n_inhib:
            res_importance[a_idx] += attn_weights[i]

    fig, ax = plt.subplots(figsize=(14, 4))

    # Color by region
    colors = []
    for i in range(n_inhib):
        if i < scaffold_len:
            colors.append("#FF9800")  # scaffold
        else:
            colors.append("#2196F3")  # DH domain

    ax.bar(range(1, n_inhib + 1), res_importance, color=colors, width=1.0, edgecolor="none")

    # Mark hotspots
    for hp in hotspots:
        if 0 < hp <= n_inhib and res_importance[hp - 1] > 0:
            ax.annotate(f"{hp}", (hp, res_importance[hp - 1]),
                       fontsize=7, ha="center", va="bottom", color="red")

    ax.axvline(x=scaffold_len + 0.5, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Inhibitor Position")
    ax.set_ylabel("Summed Attention")
    ax.set_title(f"Per-Residue Importance: {inh_id}")

    legend = [
        mpatches.Patch(color="#FF9800", label="Scaffold"),
        mpatches.Patch(color="#2196F3", label="DH Domain"),
    ]
    ax.legend(handles=legend, fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / f"residue_importance_{inh_id}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_contact_type_importance(all_contacts: dict, all_attn: dict):
    """Plot importance by contact type across all designs."""
    TYPES = ["AA", "AC", "AP", "CC", "CP", "PP"]
    type_attention = {t: [] for t in TYPES}

    for inh_id, contacts in all_contacts.items():
        if inh_id not in all_attn:
            continue
        attn = all_attn[inh_id]
        contacts_8 = [c for c in contacts if c["at_8_0"]]
        n_real = min(len(attn), len(contacts_8))

        for i in range(n_real):
            ct = contacts_8[i]["contact_type"]
            if ct in type_attention:
                type_attention[ct].append(attn[i])

    fig, ax = plt.subplots(figsize=(8, 5))
    means = [np.mean(type_attention[t]) if type_attention[t] else 0 for t in TYPES]
    stds = [np.std(type_attention[t]) if type_attention[t] else 0 for t in TYPES]
    counts = [len(type_attention[t]) for t in TYPES]

    bars = ax.bar(TYPES, means, yerr=stds, capsize=3, color=sns.color_palette("Set2", len(TYPES)))
    ax.set_xlabel("Contact Type")
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title("Contact Type Importance (Averaged Across All Designs)")

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"n={count}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "contact_type_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(pred_df: pd.DataFrame):
    """Plot predicted vs actual binding values."""
    subsets = pred_df["subset"].unique()

    fig, axes = plt.subplots(1, len(subsets), figsize=(6 * len(subsets), 5))
    if len(subsets) == 1:
        axes = [axes]

    for ax, subset in zip(axes, subsets):
        sub = pred_df[pred_df["subset"] == subset]
        actual = sub["actual"].values
        pred_attn = sub["predicted_attention"].values
        pred_linear = sub["predicted_linear"].values

        ax.scatter(actual, pred_attn, c="#E65100", s=60, zorder=3,
                   label="Attention", edgecolors="k", linewidths=0.5)
        ax.scatter(actual, pred_linear, c="#2196F3", s=60, zorder=2, marker="^",
                   label="Linear", edgecolors="k", linewidths=0.5)

        # Identity line
        all_vals = np.concatenate([actual, pred_attn, pred_linear])
        lims = [min(all_vals) * 0.8, max(all_vals) * 1.2]
        ax.plot(lims, lims, "k--", alpha=0.3, zorder=1)

        ax.set_xlabel("Actual Binding")
        ax.set_ylabel("Predicted Binding")
        ax.set_title(f"{subset} (n={len(sub)})")
        ax.legend(fontsize=9)

        # Add labels for each point
        for _, row in sub.iterrows():
            label = row["inhibitor_id"].replace("ITSN_RFD1_", "").replace("Vav_denovo_", "V")
            ax.annotate(label, (row["actual"], row["predicted_attention"]),
                       fontsize=6, alpha=0.7, xytext=(3, 3),
                       textcoords="offset points")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "prediction_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def validate_hotspots(all_contacts: dict, all_attn: dict, designs: pd.DataFrame):
    """Check if hotspot positions receive above-average attention."""
    print("\n" + "=" * 60)
    print("HOTSPOT VALIDATION")
    print("=" * 60)

    results = []

    for _, design in designs.iterrows():
        inh_id = design["inhibitor_id"]
        if inh_id not in all_attn:
            continue

        hotspots = [int(design[f"hotspot_{i}"]) for i in range(1, 5)]
        contacts = all_contacts[inh_id]
        attn = all_attn[inh_id]
        contacts_8 = [c for c in contacts if c["at_8_0"]]
        n_real = min(len(attn), len(contacts_8))

        # Attention per inhibitor position
        pos_attn = {}
        for i in range(n_real):
            pos = contacts_8[i]["res_a_idx"]
            pos_attn[pos] = pos_attn.get(pos, 0) + attn[i]

        if not pos_attn:
            continue

        mean_attn = np.mean(list(pos_attn.values()))

        hotspot_attn = [pos_attn.get(hp, 0) for hp in hotspots]
        n_above_avg = sum(1 for a in hotspot_attn if a > mean_attn)
        mean_hotspot_attn = np.mean(hotspot_attn)

        result = {
            "inhibitor_id": inh_id,
            "class": design["class"],
            "mean_attention": round(mean_attn, 6),
            "mean_hotspot_attention": round(mean_hotspot_attn, 6),
            "hotspots_above_avg": n_above_avg,
            "n_hotspots": len(hotspots),
            "enrichment": round(mean_hotspot_attn / mean_attn, 2) if mean_attn > 0 else 0,
        }
        results.append(result)

        status = "PASS" if n_above_avg >= 2 else "WEAK" if n_above_avg >= 1 else "FAIL"
        print(f"  {inh_id:<35} hotspot enrich={result['enrichment']:.1f}x "
              f"({n_above_avg}/{len(hotspots)} above avg) [{status}]")

    val_df = pd.DataFrame(results)
    val_df.to_csv(RESULTS / "hotspot_validation.csv", index=False)

    # Summary
    if results:
        mean_enrich = np.mean([r["enrichment"] for r in results])
        n_pass = sum(1 for r in results if r["hotspots_above_avg"] >= 2)
        print(f"\n  Overall: mean enrichment = {mean_enrich:.1f}x, "
              f"{n_pass}/{len(results)} designs pass (>=2 hotspots above avg)")

    return val_df


def generate_pymol_scripts(all_contacts: dict, all_attn: dict, designs: pd.DataFrame):
    """Generate PyMOL coloring scripts based on attention weights."""
    for _, design in designs.iterrows():
        inh_id = design["inhibitor_id"]
        if inh_id not in all_attn:
            continue

        contacts = all_contacts[inh_id]
        attn = all_attn[inh_id]
        contacts_8 = [c for c in contacts if c["at_8_0"]]
        n_real = min(len(attn), len(contacts_8))
        scaffold_len = int(design["scaffold_length"])

        # Per-residue attention
        res_attn_a = {}
        res_attn_b = {}
        for i in range(n_real):
            c = contacts_8[i]
            res_attn_a[c["res_a_idx"]] = res_attn_a.get(c["res_a_idx"], 0) + attn[i]
            res_attn_b[c["res_b_idx"]] = res_attn_b.get(c["res_b_idx"], 0) + attn[i]

        # Normalize
        max_a = max(res_attn_a.values()) if res_attn_a else 1
        max_b = max(res_attn_b.values()) if res_attn_b else 1

        bm_path = RESULTS / "best_models.json"
        with open(bm_path) as f:
            bm = json.load(f)[inh_id]
        dirname = bm["dirname"]
        prefix = bm["prefix"]
        model_idx = bm["model_idx"]
        cif_name = f"{prefix}_model_{model_idx}.cif"

        lines = [
            f"# PyMOL attention coloring script for {inh_id}",
            f"# Load: load {cif_name}",
            f"load {cif_name}, {inh_id.replace('-', '_')}",
            f"color gray80, {inh_id.replace('-', '_')}",
            f"color palegreen, chain B",
            "",
            "# Color by attention weight (red = high attention)",
        ]

        for res, val in sorted(res_attn_a.items()):
            norm_val = val / max_a
            r = min(1.0, norm_val * 2)
            g = max(0, 1.0 - norm_val * 2)
            lines.append(f"set_color attn_a_{res}, [{r:.2f}, {g:.2f}, 0.0]")
            lines.append(f"color attn_a_{res}, chain A and resi {res}")

        for res, val in sorted(res_attn_b.items()):
            norm_val = val / max_b
            r = min(1.0, norm_val * 2)
            g = max(0, 1.0 - norm_val * 2)
            lines.append(f"set_color attn_b_{res}, [{r:.2f}, {g:.2f}, 0.0]")
            lines.append(f"color attn_b_{res}, chain B and resi {res}")

        # Highlight hotspots
        hotspots = [int(design[f"hotspot_{i}"]) for i in range(1, 5)]
        lines.append("")
        lines.append("# Hotspot positions (spheres)")
        for hp in hotspots:
            lines.append(f"show spheres, chain A and resi {hp} and name CA")

        # Show switch regions
        lines.append("")
        lines.append("# Switch regions")
        lines.append("color marine, chain B and resi 29-42  # Switch I")
        lines.append("color forest, chain B and resi 62-68  # Switch II")

        # Scaffold boundary
        lines.append(f"")
        lines.append(f"# Scaffold boundary at residue {scaffold_len}")
        lines.append(f"select scaffold, chain A and resi 1-{scaffold_len}")
        lines.append(f"select dh_domain, chain A and resi {scaffold_len+1}-{int(design['total_length'])}")

        pml_path = PYMOL_DIR / f"{inh_id}.pml"
        with open(pml_path, "w") as f:
            f.write("\n".join(lines))

    print(f"Generated {len(list(PYMOL_DIR.glob('*.pml')))} PyMOL scripts in {PYMOL_DIR}")


def plot_feature_correlations():
    """Plot correlation heatmap of top features vs binding."""
    feat_path = PROJECT / "data" / "processed" / "features" / "all_features.csv"
    if not feat_path.exists():
        return

    df = pd.read_csv(feat_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {"binding_mean"}
    feat_cols = [c for c in numeric_cols if c not in exclude and df[c].std() > 0]

    if not feat_cols or "binding_mean" not in df.columns:
        return

    # Compute correlations
    corrs = {}
    for c in feat_cols:
        if df[c].isna().sum() == 0:
            from scipy import stats
            r, _ = stats.pearsonr(df[c], df["binding_mean"])
            corrs[c] = r

    # Sort by absolute correlation
    sorted_feats = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:20]

    if not sorted_feats:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    feat_names = [f[0] for f in sorted_feats]
    feat_vals = [f[1] for f in sorted_feats]
    colors = ["#E65100" if v > 0 else "#2196F3" for v in feat_vals]

    ax.barh(range(len(feat_names)), feat_vals, color=colors)
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, fontsize=8)
    ax.set_xlabel("Pearson r with binding_mean")
    ax.set_title("Top Feature Correlations with Binding")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "feature_correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 60)
    print("Phase 6: Visualization & Validation")
    print("=" * 60)

    designs = pd.read_csv(DATA_CSV)

    # Load all contacts and attention weights
    all_contacts = {}
    all_attn = {}

    for _, design in designs.iterrows():
        inh_id = design["inhibitor_id"]

        # Contacts
        contact_file = CONTACT_DIR / f"{inh_id}_contacts.json"
        if contact_file.exists():
            with open(contact_file) as f:
                all_contacts[inh_id] = json.load(f)

        # Attention weights
        attn_file = ATTN_DIR / f"{inh_id}_attention.npy"
        if attn_file.exists():
            all_attn[inh_id] = np.load(attn_file)

    print(f"Loaded {len(all_contacts)} contact sets, {len(all_attn)} attention weight sets")

    # 1. Attention heatmaps
    print("\nGenerating attention heatmaps...")
    for _, design in designs.iterrows():
        inh_id = design["inhibitor_id"]
        if inh_id not in all_attn or inh_id not in all_contacts:
            continue

        hotspots = [int(design[f"hotspot_{i}"]) for i in range(1, 5)]
        plot_attention_heatmap(
            inh_id, all_contacts[inh_id], all_attn[inh_id],
            int(design["scaffold_length"]), hotspots,
            int(design["total_length"]),
            191 if design["target_gtpase"] == "Cdc42" else 192,
            design["class"]
        )

        plot_residue_importance(
            inh_id, all_contacts[inh_id], all_attn[inh_id],
            int(design["scaffold_length"]), hotspots,
            int(design["total_length"])
        )

    # 2. Contact type importance
    print("Generating contact type importance chart...")
    plot_contact_type_importance(all_contacts, all_attn)

    # 3. Prediction scatter
    pred_path = MODEL_DIR / "loo_predictions.csv"
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        print("Generating prediction scatter plots...")
        plot_predictions(pred_df)

    # 4. Feature correlations
    print("Generating feature correlation chart...")
    plot_feature_correlations()

    # 5. Hotspot validation
    validate_hotspots(all_contacts, all_attn, designs)

    # 6. PyMOL scripts
    print("\nGenerating PyMOL scripts...")
    generate_pymol_scripts(all_contacts, all_attn, designs)

    # Summary of all generated files
    print(f"\n{'='*60}")
    print("GENERATED FILES")
    print(f"{'='*60}")
    for d in [FIG_DIR, PYMOL_DIR]:
        files = sorted(d.glob("*"))
        print(f"\n{d.relative_to(PROJECT)}/")
        for f in files:
            print(f"  {f.name}")


if __name__ == "__main__":
    main()
