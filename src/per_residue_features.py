#!/usr/bin/env python3
"""Per-residue feature extraction for scaffold positions.

Reframes the problem: instead of 15 complex-level observations,
we extract features for each scaffold residue (~2,050 total).
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBIO

try:
    import freesasa
except ImportError:
    freesasa = None

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT / "data" / "raw" / "GEF_inhibitors_modeling_data.csv"
AF3_DIR = PROJECT / "data" / "af3_server_outputs"
RESULTS = PROJECT / "results"
FEATURE_DIR = PROJECT / "data" / "processed" / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

MKDSSP = "/cwork/hsb26/envs/gef/bin/mkdssp"

# ── Residue properties ─────────────────────────────────────────────────
AA_LIST = sorted("ALA ARG ASN ASP CYS GLU GLN GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
CHARGED = {"ARG", "LYS", "HIS", "ASP", "GLU"}
POLAR = {"SER", "THR", "ASN", "GLN", "TYR"}
AROMATIC = {"PHE", "TYR", "TRP", "HIS"}

# GTPase regions
SWITCH_I = set(range(29, 43))
SWITCH_II = set(range(62, 69))


def parse_structure(cif_path: Path):
    parser = MMCIFParser(QUIET=True)
    return parser.get_structure("complex", str(cif_path))


def run_dssp(structure) -> dict:
    """Run DSSP and return {(chain, resid_str): ss_type} mapping."""
    io = PDBIO()
    io.set_structure(structure)
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        io.save(f)
        pdb_path = f.name

    ss_map = {}
    try:
        result = subprocess.run([MKDSSP, pdb_path], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            in_section = False
            for line in result.stdout.split("\n"):
                if line.strip().startswith("#") and "RESIDUE" in line:
                    in_section = True
                    continue
                if in_section and len(line) > 16:
                    try:
                        resnum = line[5:10].strip()
                        chain = line[11].strip()
                        ss = line[16]
                        if ss in ("H", "G", "I"):
                            ss_map[(chain, resnum)] = "H"
                        elif ss in ("E", "B"):
                            ss_map[(chain, resnum)] = "E"
                        else:
                            ss_map[(chain, resnum)] = "C"
                    except (IndexError, ValueError):
                        continue
    except Exception:
        pass
    finally:
        Path(pdb_path).unlink(missing_ok=True)
    return ss_map


def compute_sasa_per_residue(structure) -> dict:
    """Compute per-residue SASA. Returns {(chain, resnum_str): sasa}."""
    io = PDBIO()
    io.set_structure(structure)
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        io.save(f)
        pdb_path = f.name

    residue_sasa = {}
    try:
        struct_fs = freesasa.Structure(pdb_path)
        result = freesasa.calc(struct_fs)
        for i in range(struct_fs.nAtoms()):
            chain = struct_fs.chainLabel(i)
            resi = struct_fs.residueNumber(i)
            key = (chain, resi)
            residue_sasa[key] = residue_sasa.get(key, 0) + result.atomArea(i)
    except Exception as e:
        print(f"  SASA warning: {e}")
    finally:
        Path(pdb_path).unlink(missing_ok=True)
    return residue_sasa


def extract_per_residue_features(inh_id: str, design: pd.Series) -> list[dict]:
    """Extract features for each scaffold residue in one design."""
    scaffold_len = int(design["scaffold_length"])
    total_len = int(design["total_length"])
    cls = design["class"]
    gtpase = design["target_gtpase"]
    hotspots = [int(design[f"hotspot_{i}"]) for i in range(1, 5)]
    binding = float(design["binding_mean"])

    # Load best model info
    with open(RESULTS / "best_models.json") as f:
        bm_info = json.load(f)[inh_id]
    dirname = bm_info["dirname"]
    prefix = bm_info["prefix"]
    model_idx = bm_info["model_idx"]

    design_dir = AF3_DIR / dirname
    cif_path = design_dir / f"{prefix}_model_{model_idx}.cif"

    # ── Parse structure ───────────────────────────────────────────
    structure = parse_structure(cif_path)
    model = structure[0]
    chains = list(model.get_chains())
    chain_a = chains[0]
    chain_b = chains[1]
    res_a = [r for r in chain_a.get_residues() if r.id[0] == " "]
    res_b = [r for r in chain_b.get_residues() if r.id[0] == " "]

    # ── Load AF3 full data ────────────────────────────────────────
    fd_path = design_dir / f"{prefix}_full_data_{model_idx}.json"
    with open(fd_path) as f:
        fd = json.load(f)
    cp_matrix = np.array(fd["contact_probs"])
    pae_matrix = np.array(fd["pae"])
    atom_plddts = np.array(fd["atom_plddts"])
    token_chain_ids = fd["token_chain_ids"]
    token_res_ids = fd["token_res_ids"]

    a_tok_idx = [i for i, c in enumerate(token_chain_ids) if c == "A"]
    b_tok_idx = [i for i, c in enumerate(token_chain_ids) if c == "B"]

    cp_AB = cp_matrix[np.ix_(a_tok_idx, b_tok_idx)]  # (n_inhib, n_gtpase)
    pae_AB = pae_matrix[np.ix_(a_tok_idx, b_tok_idx)]
    pae_AA = pae_matrix[np.ix_(a_tok_idx, a_tok_idx)]

    # Per-token pLDDT (average atom plddts per chain)
    a_atoms = [i for i, c in enumerate(fd["atom_chain_ids"]) if c == "A"]
    b_atoms = [i for i, c in enumerate(fd["atom_chain_ids"]) if c == "B"]
    a_plddts = atom_plddts[a_atoms]
    n_a_tokens = len(a_tok_idx)
    atoms_per_tok = len(a_atoms) / n_a_tokens
    token_plddt = np.array([
        np.mean(a_plddts[int(i * atoms_per_tok):int((i + 1) * atoms_per_tok)])
        for i in range(n_a_tokens)
    ])

    # ── Contacts at 8A ───────────────────────────────────────────
    # Build per-scaffold-residue contact info from structure
    scaffold_res = res_a[:scaffold_len]
    res_b_list = list(res_b)

    def get_cb(residue):
        if residue.get_resname() == "GLY":
            return residue["CA"] if "CA" in residue else None
        return residue["CB"] if "CB" in residue else (residue["CA"] if "CA" in residue else None)

    # ── DSSP ──────────────────────────────────────────────────────
    ss_map = run_dssp(structure)

    # ── SASA ──────────────────────────────────────────────────────
    sasa_map = compute_sasa_per_residue(structure) if freesasa else {}

    # ── Hotspot GTPase contacts ───────────────────────────────────
    # Which GTPase residues do DH hotspot positions contact?
    hotspot_gtpase_residues = set()
    for hp_pos in hotspots:
        hp_res = None
        for r in res_a:
            if r.id[1] == hp_pos:
                hp_res = r
                break
        if hp_res is None:
            continue
        hp_cb = get_cb(hp_res)
        if hp_cb is None:
            continue
        for rb in res_b_list:
            rb_cb = get_cb(rb)
            if rb_cb and (hp_cb - rb_cb) <= 10.0:
                hotspot_gtpase_residues.add(rb.id[1])

    # ── Extract features per scaffold residue ─────────────────────
    records = []
    for i, res in enumerate(scaffold_res):
        pos = res.id[1]  # 1-indexed
        resname = res.get_resname()
        cb = get_cb(res)

        feat = {
            "inhibitor_id": inh_id,
            "class": cls,
            "binding_mean": binding,
            "scaffold_position": pos,
            "relative_position": (pos - 1) / max(scaffold_len - 1, 1),
            "distance_to_junction": scaffold_len - pos,
        }

        # AA identity one-hot
        for aa in AA_LIST:
            feat[f"aa_{aa}"] = 1.0 if resname == aa else 0.0

        # AA properties
        feat["is_charged"] = 1.0 if resname in CHARGED else 0.0
        feat["is_polar"] = 1.0 if resname in POLAR else 0.0
        feat["is_aromatic"] = 1.0 if resname in AROMATIC else 0.0

        # Contacts to GTPase at various cutoffs
        n_contacts_5 = 0
        n_contacts_8 = 0
        n_contacts_switch_I = 0
        n_contacts_switch_II = 0
        n_contacts_hotspot_interface = 0
        min_dist_gtpase = 999.0
        contact_distances = []

        if cb is not None:
            for rb in res_b_list:
                rb_cb = get_cb(rb)
                if rb_cb is None:
                    continue
                dist = cb - rb_cb
                if dist <= 12.0:
                    contact_distances.append(dist)
                    if dist <= 5.5:
                        n_contacts_5 += 1
                    if dist <= 8.0:
                        n_contacts_8 += 1
                        if rb.id[1] in SWITCH_I:
                            n_contacts_switch_I += 1
                        if rb.id[1] in SWITCH_II:
                            n_contacts_switch_II += 1
                        if rb.id[1] in hotspot_gtpase_residues:
                            n_contacts_hotspot_interface += 1
                    min_dist_gtpase = min(min_dist_gtpase, dist)

        feat["n_contacts_5.5"] = n_contacts_5
        feat["n_contacts_8.0"] = n_contacts_8
        feat["n_contacts_switch_I"] = n_contacts_switch_I
        feat["n_contacts_switch_II"] = n_contacts_switch_II
        feat["n_contacts_hotspot_iface"] = n_contacts_hotspot_interface
        feat["min_dist_gtpase"] = min_dist_gtpase if min_dist_gtpase < 999 else 50.0
        feat["mean_contact_dist"] = np.mean(contact_distances) if contact_distances else 50.0

        # AF3 confidence signals
        tok_i = i  # 0-indexed token for scaffold residue
        if tok_i < cp_AB.shape[0]:
            feat["af3_max_cp_to_gtpase"] = float(cp_AB[tok_i].max())
            feat["af3_mean_cp_to_gtpase"] = float(cp_AB[tok_i].mean())
            feat["af3_mean_pae_to_gtpase"] = float(pae_AB[tok_i].mean())
            feat["af3_min_pae_to_gtpase"] = float(pae_AB[tok_i].min())
            feat["af3_plddt"] = float(token_plddt[tok_i])

            # PAE to hotspot DH positions
            hotspot_tok_indices = [hp - 1 for hp in hotspots if hp - 1 < len(a_tok_idx)]
            if hotspot_tok_indices:
                pae_to_hotspots = [pae_AA[tok_i, ht] for ht in hotspot_tok_indices]
                feat["af3_mean_pae_to_hotspots"] = float(np.mean(pae_to_hotspots))
            else:
                feat["af3_mean_pae_to_hotspots"] = 30.0

            # PAE to switch regions
            switch_tok_indices = [j for j, rid in enumerate(token_res_ids)
                                  if token_chain_ids[j] == "B" and rid in SWITCH_I | SWITCH_II]
            # Map to b_tok_idx offsets
            switch_b_offsets = []
            for j in range(len(b_tok_idx)):
                b_resid = token_res_ids[b_tok_idx[j]]
                if b_resid in SWITCH_I or b_resid in SWITCH_II:
                    switch_b_offsets.append(j)
            if switch_b_offsets:
                feat["af3_mean_pae_to_switches"] = float(np.mean(pae_AB[tok_i, switch_b_offsets]))
            else:
                feat["af3_mean_pae_to_switches"] = 30.0
        else:
            feat["af3_max_cp_to_gtpase"] = 0.0
            feat["af3_mean_cp_to_gtpase"] = 0.0
            feat["af3_mean_pae_to_gtpase"] = 30.0
            feat["af3_min_pae_to_gtpase"] = 30.0
            feat["af3_plddt"] = 50.0
            feat["af3_mean_pae_to_hotspots"] = 30.0
            feat["af3_mean_pae_to_switches"] = 30.0

        # Secondary structure
        ss = ss_map.get(("A", str(pos)), "C")
        feat["ss_helix"] = 1.0 if ss == "H" else 0.0
        feat["ss_sheet"] = 1.0 if ss == "E" else 0.0
        feat["ss_coil"] = 1.0 if ss == "C" else 0.0

        # SASA
        sasa = sasa_map.get(("A", str(pos)), 0.0)
        feat["sasa"] = sasa

        records.append(feat)

    return records


def main():
    print("=" * 60)
    print("Per-Residue Feature Extraction")
    print("=" * 60)

    designs = pd.read_csv(DATA_CSV)
    all_records = []

    for _, design in designs.iterrows():
        inh_id = design["inhibitor_id"]
        scaffold_len = int(design["scaffold_length"])
        print(f"  {inh_id} (scaffold={scaffold_len} residues)...")

        records = extract_per_residue_features(inh_id, design)
        all_records.extend(records)
        print(f"    → {len(records)} residue features extracted")

    df = pd.DataFrame(all_records)

    # Save
    out_path = FEATURE_DIR / "per_residue_features.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"Shape: {df.shape}")
    print(f"NaN counts: {df.isna().sum().sum()}")

    # Summary stats
    meta_cols = ["inhibitor_id", "class", "binding_mean", "scaffold_position"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    print(f"Feature columns: {len(feat_cols)}")
    print(f"\nDesigns: {df['inhibitor_id'].nunique()}")
    print(f"Total residues: {len(df)}")
    print(f"  ITSN: {len(df[df['class'] == 'ITSN'])}")
    print(f"  Vav:  {len(df[df['class'] == 'Vav'])}")

    return df


if __name__ == "__main__":
    main()
