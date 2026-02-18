#!/usr/bin/env python3
"""Phase 2: Contact Extraction & PRODIGY Features from AF3 structures."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from Bio.PDB.MMCIFParser import MMCIFParser

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT / "data" / "raw" / "GEF_inhibitors_modeling_data.csv"
AF3_DIR = PROJECT / "data" / "af3_server_outputs"
RESULTS = PROJECT / "results"
CONTACT_DIR = PROJECT / "data" / "processed" / "contact_maps"
FEATURE_DIR = PROJECT / "data" / "processed" / "features"
CONTACT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# ── PRODIGY residue classification ─────────────────────────────────────
CHARGED = {"ARG", "LYS", "HIS", "ASP", "GLU"}
POLAR = {"SER", "THR", "ASN", "GLN", "TYR"}
APOLAR = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "GLY", "CYS"}

def classify_residue(resname: str) -> str:
    """Classify residue as charged/polar/apolar."""
    rn = resname.upper().strip()
    if rn in CHARGED:
        return "charged"
    elif rn in POLAR:
        return "polar"
    elif rn in APOLAR:
        return "apolar"
    return "apolar"  # default for non-standard


def contact_type(class_a: str, class_b: str) -> str:
    """Return PRODIGY IC type code (CC, CP, CA, PP, PA, AA)."""
    mapping = {"charged": "C", "polar": "P", "apolar": "A"}
    pair = sorted([mapping[class_a], mapping[class_b]])
    return "".join(pair)


def parse_structure(cif_path: Path):
    """Parse mmCIF file with BioPython."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("complex", str(cif_path))
    return structure[0]  # first model


def get_cb_or_ca(residue):
    """Get CB atom (or CA for GLY)."""
    if residue.get_resname() == "GLY":
        if "CA" in residue:
            return residue["CA"]
    else:
        if "CB" in residue:
            return residue["CB"]
    if "CA" in residue:
        return residue["CA"]
    return None


def extract_contacts(model, cutoffs=(5.5, 8.0)):
    """Extract inter-chain contacts at multiple cutoffs.

    Returns list of contact dicts with distance and residue info.
    Chain A = inhibitor, Chain B = GTPase.
    """
    chains = list(model.get_chains())
    if len(chains) < 2:
        raise ValueError(f"Expected 2 chains, got {len(chains)}")

    chain_a = chains[0]  # inhibitor
    chain_b = chains[1]  # GTPase

    # Build residue lists (skip hetero atoms)
    res_a = [r for r in chain_a.get_residues() if r.id[0] == " "]
    res_b = [r for r in chain_b.get_residues() if r.id[0] == " "]

    max_cutoff = max(cutoffs)
    contacts = []

    for ra in res_a:
        cb_a = get_cb_or_ca(ra)
        if cb_a is None:
            continue
        for rb in res_b:
            cb_b = get_cb_or_ca(rb)
            if cb_b is None:
                continue
            dist = cb_a - cb_b  # BioPython distance
            if dist <= max_cutoff:
                class_a = classify_residue(ra.get_resname())
                class_b = classify_residue(rb.get_resname())
                ct = contact_type(class_a, class_b)

                contacts.append({
                    "res_a_idx": ra.id[1],
                    "res_a_name": ra.get_resname(),
                    "res_a_class": class_a,
                    "res_b_idx": rb.id[1],
                    "res_b_name": rb.get_resname(),
                    "res_b_class": class_b,
                    "distance": round(float(dist), 3),
                    "contact_type": ct,
                    "at_5_5": bool(dist <= 5.5),
                    "at_8_0": bool(dist <= 8.0),
                })

    return contacts, len(res_a), len(res_b)


def compute_prodigy_features(contacts: list[dict], n_res_a: int, n_res_b: int,
                              scaffold_len: int) -> dict:
    """Compute 9 PRODIGY features + scaffold-specific counts."""
    # Filter to 5.5A contacts
    c55 = [c for c in contacts if c["at_5_5"]]

    # IC counts by type
    type_counts = {"AA": 0, "AP": 0, "AC": 0, "PP": 0, "PA": 0, "CC": 0}
    for c in c55:
        ct = c["contact_type"]
        # Normalize: PA and AP are same, etc.
        if ct in type_counts:
            type_counts[ct] += 1
        else:
            # Should not happen with our classification
            type_counts[ct] = type_counts.get(ct, 0) + 1

    # Interface residues
    iface_a = set(c["res_a_idx"] for c in c55)
    iface_b = set(c["res_b_idx"] for c in c55)

    # NIS = non-interacting surface residues
    # Approximate: all residues not in interface
    nis_a = set(range(1, n_res_a + 1)) - iface_a
    nis_b = set(range(1, n_res_b + 1)) - iface_b

    # For NIS composition, we need residue names
    # Build residue name maps from contacts (or structure)
    # We'll use a simpler approach: count from all contacts
    all_res_a_class = {}
    all_res_b_class = {}
    for c in contacts:
        all_res_a_class[c["res_a_idx"]] = c["res_a_class"]
        all_res_b_class[c["res_b_idx"]] = c["res_b_class"]

    # NIS class counts (only for residues we've seen in extended contacts)
    total_nis = len(nis_a) + len(nis_b)
    if total_nis > 0:
        nis_apolar = sum(1 for r in nis_a if all_res_a_class.get(r, "") == "apolar") + \
                     sum(1 for r in nis_b if all_res_b_class.get(r, "") == "apolar")
        nis_charged = sum(1 for r in nis_a if all_res_a_class.get(r, "") == "charged") + \
                      sum(1 for r in nis_b if all_res_b_class.get(r, "") == "charged")
        nis_polar = sum(1 for r in nis_a if all_res_a_class.get(r, "") == "polar") + \
                    sum(1 for r in nis_b if all_res_b_class.get(r, "") == "polar")
        # NIS composition as percentages of total surface (approximated)
        pct_nis_apolar = nis_apolar / total_nis * 100
        pct_nis_charged = nis_charged / total_nis * 100
        pct_nis_polar = nis_polar / total_nis * 100
    else:
        pct_nis_apolar = pct_nis_charged = pct_nis_polar = 0.0

    # Scaffold vs DH contacts
    scaffold_contacts_55 = [c for c in c55 if c["res_a_idx"] <= scaffold_len]
    dh_contacts_55 = [c for c in c55 if c["res_a_idx"] > scaffold_len]
    scaffold_contacts_8 = [c for c in contacts if c["at_8_0"] and c["res_a_idx"] <= scaffold_len]
    dh_contacts_8 = [c for c in contacts if c["at_8_0"] and c["res_a_idx"] > scaffold_len]

    features = {
        "IC_charged_charged": type_counts["CC"],
        "IC_charged_polar": type_counts["CP"],
        "IC_charged_apolar": type_counts["AC"],
        "IC_polar_polar": type_counts["PP"],
        "IC_polar_apolar": type_counts["AP"],
        "IC_apolar_apolar": type_counts["AA"],
        "pct_NIS_apolar": round(pct_nis_apolar, 2),
        "pct_NIS_charged": round(pct_nis_charged, 2),
        "pct_NIS_polar": round(pct_nis_polar, 2),
        "total_contacts_5.5": len(c55),
        "total_contacts_8.0": len([c for c in contacts if c["at_8_0"]]),
        "n_interface_res_A": len(iface_a),
        "n_interface_res_B": len(iface_b),
        "scaffold_contacts_5.5": len(scaffold_contacts_55),
        "scaffold_contacts_8.0": len(scaffold_contacts_8),
        "dh_contacts_5.5": len(dh_contacts_55),
        "dh_contacts_8.0": len(dh_contacts_8),
    }
    return features


def extract_af3_contact_features(design_dir: Path, prefix: str, model_idx: int,
                                  contacts: list[dict], n_inhib: int, n_gtpase: int) -> dict:
    """Extract AF3 contact_probs and PAE as features for scaffold contacts."""
    full_data_path = design_dir / f"{prefix}_full_data_{model_idx}.json"
    if not full_data_path.exists():
        return {"mean_af3_contact_prob_scaffold": np.nan, "mean_af3_pae_scaffold": np.nan}

    with open(full_data_path) as f:
        fd = json.load(f)

    contact_probs = np.array(fd["contact_probs"])
    pae = np.array(fd["pae"])
    token_chain_ids = fd["token_chain_ids"]

    chain_a_idx = [i for i in range(len(token_chain_ids)) if token_chain_ids[i] == "A"]
    chain_b_idx = [i for i in range(len(token_chain_ids)) if token_chain_ids[i] == "B"]

    cp_AB = contact_probs[np.ix_(chain_a_idx, chain_b_idx)]
    pae_AB = pae[np.ix_(chain_a_idx, chain_b_idx)]

    # Get scaffold rows (0-indexed: scaffold_len = n rows from top)
    scaffold_len = n_inhib - len(chain_b_idx)  # approximate
    # Actually: chain A = inhibitor (n_inhib tokens), chain B = gtpase
    # scaffold = first scaffold_len tokens of chain A

    # For each contact, get AF3 metrics
    scaffold_cp = []
    scaffold_pae = []

    for c in contacts:
        if c["at_8_0"]:
            a_tok = c["res_a_idx"] - 1  # 0-indexed
            b_tok = c["res_b_idx"] - 1  # 0-indexed
            if a_tok < cp_AB.shape[0] and b_tok < cp_AB.shape[1]:
                if c["res_a_idx"] <= n_inhib:  # scaffold region
                    scaffold_cp.append(cp_AB[a_tok, b_tok])
                    scaffold_pae.append(pae_AB[a_tok, b_tok])

    return {
        "mean_af3_contact_prob_scaffold": round(float(np.mean(scaffold_cp)), 4) if scaffold_cp else 0.0,
        "mean_af3_pae_scaffold": round(float(np.mean(scaffold_pae)), 4) if scaffold_pae else 0.0,
        "mean_af3_contact_prob_all": round(float(np.mean(cp_AB)), 4),
        "mean_af3_pae_cross": round(float(np.mean(pae_AB)), 4),
    }


def build_contact_map(contacts: list[dict], n_res_a: int, n_res_b: int, cutoff: float = 5.5) -> np.ndarray:
    """Build binary contact map matrix."""
    cmap = np.zeros((n_res_a, n_res_b), dtype=np.float32)
    for c in contacts:
        if c["distance"] <= cutoff:
            a_idx = c["res_a_idx"] - 1  # 0-indexed
            b_idx = c["res_b_idx"] - 1
            if 0 <= a_idx < n_res_a and 0 <= b_idx < n_res_b:
                cmap[a_idx, b_idx] = 1.0
    return cmap


def main():
    print("=" * 60)
    print("Phase 2: Contact Extraction & PRODIGY Features")
    print("=" * 60)

    # Load best models
    with open(RESULTS / "best_models.json") as f:
        best_models = json.load(f)

    # Load design info
    designs = pd.read_csv(DATA_CSV)

    all_features = []

    for _, design in designs.iterrows():
        inh_id = design["inhibitor_id"]
        gtpase = design["target_gtpase"]
        scaffold_len = int(design["scaffold_length"])
        total_len = int(design["total_length"])

        bm = best_models[inh_id]
        dirname = bm["dirname"]
        prefix = bm["prefix"]
        model_idx = bm["model_idx"]

        design_dir = AF3_DIR / dirname
        cif_path = design_dir / f"{prefix}_model_{model_idx}.cif"

        print(f"\nProcessing {inh_id} (model {model_idx})...")

        # Parse structure
        model = parse_structure(cif_path)

        # Extract contacts
        contacts, n_res_a, n_res_b = extract_contacts(model)
        print(f"  Residues: A={n_res_a}, B={n_res_b}")
        print(f"  Contacts: {sum(1 for c in contacts if c['at_5_5'])} (5.5A), "
              f"{sum(1 for c in contacts if c['at_8_0'])} (8.0A)")

        # Save contact list
        contact_file = CONTACT_DIR / f"{inh_id}_contacts.json"
        with open(contact_file, "w") as f:
            json.dump(contacts, f)

        # Save contact map
        cmap = build_contact_map(contacts, n_res_a, n_res_b, cutoff=5.5)
        np.save(CONTACT_DIR / f"{inh_id}_contact_map.npy", cmap)

        # Compute PRODIGY features
        prodigy = compute_prodigy_features(contacts, n_res_a, n_res_b, scaffold_len)

        # Extract AF3 features
        af3_feats = extract_af3_contact_features(
            design_dir, prefix, model_idx, contacts, total_len, n_res_b
        )

        # Combine
        row = {
            "inhibitor_id": inh_id,
            "class": design["class"],
            "binding_mean": design["binding_mean"],
            **prodigy,
            **af3_feats,
        }
        all_features.append(row)

        scaffold_c = prodigy["scaffold_contacts_5.5"]
        dh_c = prodigy["dh_contacts_5.5"]
        print(f"  Scaffold contacts (5.5A): {scaffold_c}, DH contacts (5.5A): {dh_c}")

    # Save feature table
    feat_df = pd.DataFrame(all_features)
    feat_path = FEATURE_DIR / "prodigy_features.csv"
    feat_df.to_csv(feat_path, index=False)
    print(f"\nSaved PRODIGY features: {feat_path}")
    print(feat_df.to_string())

    return feat_df


if __name__ == "__main__":
    main()
