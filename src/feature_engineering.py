#!/usr/bin/env python3
"""Phase 3: Extended Feature Engineering from AF3 structures."""

import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
CONTACT_DIR = PROJECT / "data" / "processed" / "contact_maps"
FEATURE_DIR = PROJECT / "data" / "processed" / "features"

MKDSSP = "/cwork/hsb26/envs/gef/bin/mkdssp"

# ── GTPase switch regions ──────────────────────────────────────────────
SWITCH_I = set(range(29, 43))   # residues 29-42
SWITCH_II = set(range(62, 69))  # residues 62-68

# ── Residue properties ─────────────────────────────────────────────────
POSITIVE = {"ARG", "LYS"}
NEGATIVE = {"ASP", "GLU"}
AROMATIC = {"PHE", "TYR", "TRP", "HIS"}
HBOND_DONORS = {"ARG", "LYS", "HIS", "ASN", "GLN", "SER", "THR", "TRP", "TYR"}
HBOND_ACCEPTORS = {"ASP", "GLU", "ASN", "GLN", "SER", "THR", "HIS", "TYR"}


def parse_structure(cif_path: Path):
    """Parse mmCIF file."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("complex", str(cif_path))
    return structure


def get_residues_by_chain(model):
    """Get standard residues per chain."""
    chains = list(model.get_chains())
    res_a = [r for r in chains[0].get_residues() if r.id[0] == " "]
    res_b = [r for r in chains[1].get_residues() if r.id[0] == " "]
    return res_a, res_b


# ── Tier 2: GTPase-specific features ──────────────────────────────────

def compute_switch_features(contacts: list[dict], hotspots: list[int]) -> dict:
    """Count contacts to switch I/II and hotspot positions."""
    c55 = [c for c in contacts if c["at_5_5"]]

    contacts_switch_I = sum(1 for c in c55 if c["res_b_idx"] in SWITCH_I)
    contacts_switch_II = sum(1 for c in c55 if c["res_b_idx"] in SWITCH_II)

    # Typed switch contacts
    switch_contacts = [c for c in c55 if c["res_b_idx"] in SWITCH_I | SWITCH_II]
    ic_charged_switches = sum(1 for c in switch_contacts
                              if c["res_a_class"] == "charged" or c["res_b_class"] == "charged")
    ic_polar_switches = sum(1 for c in switch_contacts
                            if c["res_a_class"] == "polar" or c["res_b_class"] == "polar")
    ic_apolar_switches = sum(1 for c in switch_contacts
                             if c["res_a_class"] == "apolar" and c["res_b_class"] == "apolar")

    # Hotspot contacts (inhibitor positions)
    hotspot_set = set(hotspots)
    contacts_hotspots = sum(1 for c in c55 if c["res_a_idx"] in hotspot_set)

    return {
        "contacts_switch_I": contacts_switch_I,
        "contacts_switch_II": contacts_switch_II,
        "IC_charged_to_switches": ic_charged_switches,
        "IC_polar_to_switches": ic_polar_switches,
        "IC_apolar_to_switches": ic_apolar_switches,
        "contacts_to_hotspots": contacts_hotspots,
    }


# ── Tier 3: Hydrogen bond analysis ────────────────────────────────────

def compute_hbond_features(model, contacts: list[dict]) -> dict:
    """Estimate H-bonds from heavy-atom distances (no H in AF3).

    D-A distance < 3.5A with donor N/O and acceptor O/N.
    """
    chains = list(model.get_chains())
    chain_a = chains[0]
    chain_b = chains[1]

    res_a = {r.id[1]: r for r in chain_a.get_residues() if r.id[0] == " "}
    res_b = {r.id[1]: r for r in chain_b.get_residues() if r.id[0] == " "}

    donor_atoms = {"N", "NE", "NH1", "NH2", "NZ", "ND1", "ND2", "NE1", "NE2",
                   "OG", "OG1", "OH"}
    acceptor_atoms = {"O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH",
                      "ND1", "NE2", "SD"}

    hbonds = []
    # Check inter-chain atom pairs for H-bond-like geometry
    c55 = [c for c in contacts if c["at_5_5"]]
    checked_pairs = set()

    for c in c55:
        ra = res_a.get(c["res_a_idx"])
        rb = res_b.get(c["res_b_idx"])
        if ra is None or rb is None:
            continue

        pair_key = (c["res_a_idx"], c["res_b_idx"])
        if pair_key in checked_pairs:
            continue
        checked_pairs.add(pair_key)

        for atom_a in ra.get_atoms():
            if atom_a.get_name() not in donor_atoms | acceptor_atoms:
                continue
            for atom_b in rb.get_atoms():
                if atom_b.get_name() not in donor_atoms | acceptor_atoms:
                    continue
                dist = atom_a - atom_b
                if dist < 3.5:
                    is_da = (atom_a.get_name() in donor_atoms and
                             atom_b.get_name() in acceptor_atoms)
                    is_ad = (atom_a.get_name() in acceptor_atoms and
                             atom_b.get_name() in donor_atoms)
                    if is_da or is_ad:
                        hbonds.append({
                            "res_a": c["res_a_idx"],
                            "res_b": c["res_b_idx"],
                            "dist": float(dist),
                            "atom_a": atom_a.get_name(),
                            "atom_b": atom_b.get_name(),
                        })

    # Geometry score: closer to 2.8-3.2A is better
    geo_scores = []
    for hb in hbonds:
        d = hb["dist"]
        if 2.8 <= d <= 3.2:
            geo_scores.append(1.0)
        elif d < 2.8:
            geo_scores.append(max(0, 1.0 - (2.8 - d) / 0.5))
        else:
            geo_scores.append(max(0, 1.0 - (d - 3.2) / 0.3))

    return {
        "hbond_count": len(hbonds),
        "hbond_geometry_score": round(float(np.mean(geo_scores)), 3) if geo_scores else 0.0,
        "hbond_mean_dist": round(float(np.mean([h["dist"] for h in hbonds])), 3) if hbonds else 0.0,
    }


# ── Tier 4: Interface area & packing ──────────────────────────────────

def structure_to_pdb_string(structure):
    """Convert BioPython structure to PDB string for freesasa."""
    io = PDBIO()
    io.set_structure(structure)
    import io as stdlib_io
    string_io = stdlib_io.StringIO()
    io.save(string_io)
    return string_io.getvalue()


def compute_sasa_features(structure, model) -> dict:
    """Compute interface area and SASA-based features using freesasa."""
    if freesasa is None:
        return {"interface_area": np.nan, "delta_sasa": np.nan}

    # Write structure to temporary PDB
    io = PDBIO()
    io.set_structure(structure)

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        io.save(f)
        pdb_path = f.name

    try:
        # Compute SASA of complex
        struct_complex = freesasa.Structure(pdb_path)
        result_complex = freesasa.calc(struct_complex)
        area_complex = result_complex.totalArea()

        # Compute SASA of each chain in isolation by writing separate PDB files
        io_pdb = PDBIO()
        chain_areas = {}
        chains_list = list(model.get_chains())
        for chain in chains_list:
            with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as cf:
                class ChainSelect:
                    def __init__(self, chain_id):
                        self.chain_id = chain_id
                    def accept_model(self, m): return True
                    def accept_chain(self, c): return c.id == self.chain_id
                    def accept_residue(self, r): return True
                    def accept_atom(self, a): return True
                io_pdb.set_structure(structure)
                io_pdb.save(cf, select=ChainSelect(chain.id))
                chain_pdb = cf.name
            try:
                chain_struct = freesasa.Structure(chain_pdb)
                chain_result = freesasa.calc(chain_struct)
                chain_areas[chain.id] = chain_result.totalArea()
            finally:
                Path(chain_pdb).unlink(missing_ok=True)

        # Interface area = (SASA_A_alone + SASA_B_alone - SASA_complex) / 2
        sasa_a = chain_areas.get(chains_list[0].id, 0.0)
        sasa_b = chain_areas.get(chains_list[1].id, 0.0)
        interface_area = (sasa_a + sasa_b - area_complex) / 2.0

    except Exception as e:
        print(f"  freesasa warning: {e}")
        interface_area = np.nan
        area_complex = np.nan
    finally:
        Path(pdb_path).unlink(missing_ok=True)

    return {
        "interface_area": round(float(interface_area), 1) if not np.isnan(interface_area) else np.nan,
    }


def compute_packing_features(contacts: list[dict], res_a: list, res_b: list) -> dict:
    """Compute packing density and interface planarity."""
    c55 = [c for c in contacts if c["at_5_5"]]
    if not c55:
        return {"packing_density": 0.0, "interface_planarity": np.nan}

    # Packing density: contacts per interface residue
    iface_res = set()
    for c in c55:
        iface_res.add(("A", c["res_a_idx"]))
        iface_res.add(("B", c["res_b_idx"]))
    packing = len(c55) / max(len(iface_res), 1)

    # Interface planarity via PCA on interface CB atoms
    iface_coords = []
    for r in res_a:
        if r.id[1] in {c["res_a_idx"] for c in c55}:
            atom = r["CB"] if "CB" in r else (r["CA"] if "CA" in r else None)
            if atom:
                iface_coords.append(atom.get_vector().get_array())
    for r in res_b:
        if r.id[1] in {c["res_b_idx"] for c in c55}:
            atom = r["CB"] if "CB" in r else (r["CA"] if "CA" in r else None)
            if atom:
                iface_coords.append(atom.get_vector().get_array())

    planarity = np.nan
    if len(iface_coords) >= 3:
        coords = np.array(iface_coords)
        coords_centered = coords - coords.mean(axis=0)
        _, s, _ = np.linalg.svd(coords_centered, full_matrices=False)
        # Planarity: ratio of smallest to largest singular value
        # Lower = more planar
        planarity = float(s[2] / s[0]) if s[0] > 0 else np.nan

    return {
        "packing_density": round(packing, 3),
        "interface_planarity": round(planarity, 4) if not np.isnan(planarity) else np.nan,
    }


# ── Tier 5: Electrostatics ────────────────────────────────────────────

def compute_electrostatic_features(contacts: list[dict], res_a: list, res_b: list) -> dict:
    """Compute salt bridges, cation-pi, pi-pi interactions."""
    res_a_map = {r.id[1]: r for r in res_a}
    res_b_map = {r.id[1]: r for r in res_b}

    salt_bridges = 0
    cation_pi = 0
    pi_pi = 0

    c55 = [c for c in contacts if c["at_5_5"]]

    for c in c55:
        rn_a = c["res_a_name"]
        rn_b = c["res_b_name"]

        # Salt bridges: opposite charges < 4A
        if c["distance"] < 4.0:
            if (rn_a in POSITIVE and rn_b in NEGATIVE) or \
               (rn_a in NEGATIVE and rn_b in POSITIVE):
                salt_bridges += 1

        # Cation-pi
        if (rn_a in POSITIVE and rn_b in AROMATIC) or \
           (rn_a in AROMATIC and rn_b in POSITIVE):
            if c["distance"] < 6.0:
                cation_pi += 1

        # Pi-pi stacking
        if rn_a in AROMATIC and rn_b in AROMATIC:
            if c["distance"] < 7.0:
                pi_pi += 1

    return {
        "salt_bridge_count": salt_bridges,
        "cation_pi_count": cation_pi,
        "pi_pi_count": pi_pi,
    }


# ── Tier 6: SASA-based burial ─────────────────────────────────────────

def compute_burial_features(structure, contacts: list[dict]) -> dict:
    """Compute core/rim residue classification from SASA."""
    if freesasa is None:
        return {"core_residue_count": np.nan, "rim_residue_count": np.nan}

    io = PDBIO()
    io.set_structure(structure)

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        io.save(f)
        pdb_path = f.name

    try:
        struct_fs = freesasa.Structure(pdb_path)
        result = freesasa.calc(struct_fs)

        c55 = [c for c in contacts if c["at_5_5"]]
        iface_res = set()
        for c in c55:
            iface_res.add(c["res_a_idx"])
            iface_res.add(c["res_b_idx"])

        # Get per-residue SASA
        n_atoms = struct_fs.nAtoms()
        residue_sasa = {}
        for i in range(n_atoms):
            resi = struct_fs.residueNumber(i)
            chain = struct_fs.chainLabel(i)
            key = (chain, resi)
            residue_sasa[key] = residue_sasa.get(key, 0) + result.atomArea(i)

        # Core: interface residue with relative SASA < 25%
        # Rim: interface residue with relative SASA >= 25%
        core = 0
        rim = 0
        for res_idx in iface_res:
            # Check both chains
            sasa = 0
            for chain in ["A", "B"]:
                s = residue_sasa.get((chain, str(res_idx)), 0)
                sasa += s
            if sasa < 10:  # Low SASA = buried = core
                core += 1
            else:
                rim += 1

    except Exception as e:
        print(f"  burial warning: {e}")
        core = rim = 0
    finally:
        Path(pdb_path).unlink(missing_ok=True)

    return {
        "core_residue_count": core,
        "rim_residue_count": rim,
    }


# ── Tier 7: Secondary structure ───────────────────────────────────────

def compute_ss_features(structure, contacts: list[dict]) -> dict:
    """Compute interface secondary structure composition via DSSP."""
    io = PDBIO()
    io.set_structure(structure)

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        io.save(f)
        pdb_path = f.name

    ss_map = {}  # (chain, resid) -> SS type
    try:
        result = subprocess.run(
            [MKDSSP, pdb_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            in_residue_section = False
            for line in result.stdout.split("\n"):
                if line.strip().startswith("#"):
                    if "RESIDUE" in line:
                        in_residue_section = True
                    continue
                if in_residue_section and len(line) > 16:
                    try:
                        resnum = line[5:10].strip()
                        chain = line[11].strip()
                        ss = line[16]
                        if ss in ("H", "G", "I"):
                            ss_type = "helix"
                        elif ss in ("E", "B"):
                            ss_type = "sheet"
                        else:
                            ss_type = "coil"
                        ss_map[(chain, resnum)] = ss_type
                    except (IndexError, ValueError):
                        continue
    except Exception as e:
        print(f"  DSSP warning: {e}")
    finally:
        Path(pdb_path).unlink(missing_ok=True)

    # Interface SS composition
    c55 = [c for c in contacts if c["at_5_5"]]
    iface_ss = []
    for c in c55:
        for chain, res_idx in [("A", c["res_a_idx"]), ("B", c["res_b_idx"])]:
            ss = ss_map.get((chain, str(res_idx)), "coil")
            iface_ss.append(ss)

    total = max(len(iface_ss), 1)
    helix_pct = sum(1 for s in iface_ss if s == "helix") / total * 100
    sheet_pct = sum(1 for s in iface_ss if s == "sheet") / total * 100
    coil_pct = sum(1 for s in iface_ss if s == "coil") / total * 100

    return {
        "interface_helix_pct": round(helix_pct, 1),
        "interface_sheet_pct": round(sheet_pct, 1),
        "interface_coil_pct": round(coil_pct, 1),
    }


# ── Tier 12: Distance distribution ────────────────────────────────────

def compute_distance_features(contacts: list[dict]) -> dict:
    """Bin contacts by distance."""
    dists = [c["distance"] for c in contacts]

    if not dists:
        return {
            "contact_distance_mean": np.nan,
            "contact_distance_std": np.nan,
            "contacts_bin_0_4": 0,
            "contacts_bin_4_5.5": 0,
            "contacts_bin_5.5_8": 0,
            "contacts_bin_8_12": 0,
        }

    return {
        "contact_distance_mean": round(float(np.mean(dists)), 3),
        "contact_distance_std": round(float(np.std(dists)), 3),
        "contacts_bin_0_4": sum(1 for d in dists if d < 4.0),
        "contacts_bin_4_5.5": sum(1 for d in dists if 4.0 <= d < 5.5),
        "contacts_bin_5.5_8": sum(1 for d in dists if 5.5 <= d < 8.0),
        "contacts_bin_8_12": sum(1 for d in dists if 8.0 <= d <= 12.0),
    }


def main():
    print("=" * 60)
    print("Phase 3: Extended Feature Engineering")
    print("=" * 60)

    # Load best models and design info
    with open(RESULTS / "best_models.json") as f:
        best_models = json.load(f)
    designs = pd.read_csv(DATA_CSV)

    # Load PRODIGY features as base
    prodigy_df = pd.read_csv(FEATURE_DIR / "prodigy_features.csv")

    all_features = []

    for _, design in designs.iterrows():
        inh_id = design["inhibitor_id"]
        gtpase = design["target_gtpase"]
        scaffold_len = int(design["scaffold_length"])

        bm = best_models[inh_id]
        dirname = bm["dirname"]
        prefix = bm["prefix"]
        model_idx = bm["model_idx"]

        design_dir = AF3_DIR / dirname
        cif_path = design_dir / f"{prefix}_model_{model_idx}.cif"

        print(f"\nProcessing {inh_id}...")

        # Parse structure
        structure = parse_structure(cif_path)
        model = structure[0]
        res_a, res_b = get_residues_by_chain(model)

        # Load contacts
        contact_file = CONTACT_DIR / f"{inh_id}_contacts.json"
        with open(contact_file) as f:
            contacts = json.load(f)

        # Hotspot positions from CSV
        hotspots = [int(design[f"hotspot_{i}"]) for i in range(1, 5)]

        features = {"inhibitor_id": inh_id}

        # Tier 2: Switch/hotspot features
        features.update(compute_switch_features(contacts, hotspots))

        # Tier 3: H-bond features
        features.update(compute_hbond_features(model, contacts))

        # Tier 4: Packing features
        features.update(compute_sasa_features(structure, model))
        features.update(compute_packing_features(contacts, res_a, res_b))

        # Tier 5: Electrostatics
        features.update(compute_electrostatic_features(contacts, res_a, res_b))

        # Tier 6: Burial
        features.update(compute_burial_features(structure, contacts))

        # Tier 7: Secondary structure
        features.update(compute_ss_features(structure, contacts))

        # Tier 12: Distance bins
        features.update(compute_distance_features(contacts))

        all_features.append(features)
        print(f"  Features: {len(features)} computed")

    # Merge with PRODIGY features
    ext_df = pd.DataFrame(all_features)
    merged = prodigy_df.merge(ext_df, on="inhibitor_id", how="left")

    # Save
    out_path = FEATURE_DIR / "all_features.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved all features: {out_path}")
    print(f"Shape: {merged.shape}")
    print(f"NaN counts:\n{merged.isna().sum()[merged.isna().sum() > 0]}")

    # Quick sanity: Pearson r of some features vs binding
    from scipy import stats
    print("\nFeature correlations with binding_mean:")
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ("binding_mean",) or merged[col].isna().any():
            continue
        if merged[col].std() == 0:
            continue
        r, p = stats.pearsonr(merged[col], merged["binding_mean"])
        if abs(r) > 0.3:
            print(f"  {col:40s} r={r:+.3f}  p={p:.3f}")

    return merged


if __name__ == "__main__":
    main()
