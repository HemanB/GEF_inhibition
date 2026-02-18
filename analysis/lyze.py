#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json, os
from pathlib import Path
import subprocess
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
from Bio.PDB import PDBParser
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from ripser import ripser
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# 1) USalign wrapper
# ---------------------------------------------------------------------------
def run_usalign(
    ref_pdb: str | Path,
    model_pdb: str | Path,
    *,
    multimer_mode: bool = False,
    fast: bool = True,
    ter: str = "0",
) -> Dict[str, float | str | None]:
    cmd = ["USalign", str(ref_pdb), str(model_pdb)]
    if multimer_mode:
        cmd += ["-mm", "1", "-ter", ter]
    if fast:
        cmd.append("-fast")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"USalign error for {model_pdb}: {res.stderr}")
        return {"USalign_RMSD": None, "USalign_TM": None, "full_output": res.stderr}
    out = res.stdout
    rmsd = tm = None
    for line in out.splitlines():
        if rmsd is None and "RMSD" in line:
            sep = "=" if "=" in line else ":" if ":" in line else None
            if sep:
                for part in line.split(","):
                    if "RMSD" in part:
                        try:
                            rmsd = float(part.split(sep)[1].split()[0])
                        except ValueError:
                            pass
        if tm is None and "TM-score" in line:
            sep = "=" if "=" in line else ":" if ":" in line else None
            if sep:
                try:
                    tm = float(line.split("TM-score" + sep)[1].split()[0])
                except ValueError:
                    pass
    if rmsd is None or tm is None:
        print(f"Warning: could not parse RMSD/TM-score for {model_pdb}")
    return {"USalign_RMSD": rmsd, "USalign_TM": tm, "full_output": out}

# ---------------------------------------------------------------------------
# 2) pLDDT extraction
# ---------------------------------------------------------------------------
def extract_plddt_scores(af_base_dir: str | Path) -> pd.DataFrame:
    parser = PDBParser(QUIET=True)
    records: List[Tuple[str, str, str, str, float, int]] = []
    for pdb_path in Path(af_base_dir).rglob("ranked_*.pdb"):
        try:
            struct = parser.get_structure("af", pdb_path)
            bfs = [atom.get_bfactor() for atom in struct.get_atoms()]
            avg = float(np.mean(bfs)) if bfs else np.nan
        except Exception as e:
            print(f"Error parsing {pdb_path}: {e}")
            avg = np.nan
        try:
            rank = int(pdb_path.stem.split("_")[1])
        except:
            rank = -1
        name = f"{pdb_path.parents[1].name}/{pdb_path.parent.name}"
        records.append((name, pdb_path.parent.name, pdb_path.stem, pdb_path.name, avg, rank))
    return pd.DataFrame(
        records,
        columns=["Name", "Object_Name", "Model_Key", "Ranked_File", "pLDDT", "Rank"],
    )

# ---------------------------------------------------------------------------
# 3) persistent-homology H1-loop
# ---------------------------------------------------------------------------
def mean_h1_lifetime(pdb_file: str | Path, distance_threshold: float = 8.0) -> float:
    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure("model", pdb_file)
        ca_coords = [
            res["CA"].coord
            for res in struct.get_residues()
            if "CA" in res and res.id[0] == " "
        ]
    except Exception as e:
        print(f"PH parse error {pdb_file}: {e}")
        return np.nan
    if len(ca_coords) < 2:
        return np.nan
    coords = np.array(ca_coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    dmat = np.linalg.norm(diff, axis=-1)
    dgms = ripser(dmat, distance_matrix=True).get("dgms", [])
    if len(dgms) < 2 or dgms[1].size == 0:
        return 0.0
    lifetimes = dgms[1][:, 1] - dgms[1][:, 0]
    return float(np.mean(lifetimes))

# ---------------------------------------------------------------------------
# 4) pTM/ipTM extraction
# ---------------------------------------------------------------------------
def extract_ptm_iptm(ranking_debug_filepath: str | Path, model_identifier: str) -> Tuple[float, float]:
    pTM = np.nan
    ipTM = np.nan
    if not os.path.isfile(ranking_debug_filepath):
        return pTM, ipTM
    try:
        data = json.load(open(ranking_debug_filepath))
        scores = data.get("iptm+ptm", {})
        if model_identifier in scores:
            pTM = float(scores[model_identifier]) * 100
            ipTM = pTM
    except Exception as e:
        print(f"pTM parse error {ranking_debug_filepath}: {e}")
    return pTM, ipTM

# ---------------------------------------------------------------------------
# 5) average PAE extraction
# ---------------------------------------------------------------------------
def extract_average_pae(pae_json_filepath: str | Path) -> float:
    v = np.nan
    if not os.path.isfile(pae_json_filepath):
        return v
    try:
        data = json.load(open(pae_json_filepath))
        matrix = None
        if isinstance(data, dict):
            matrix = data.get("predicted_aligned_error")
        elif isinstance(data, list) and data:
            matrix = data[0].get("predicted_aligned_error")
        if isinstance(matrix, list):
            arr = np.array(matrix)
            v = float(np.mean(arr))
    except Exception as e:
        print(f"PAE parse error {pae_json_filepath}: {e}")
    return v

# ---------------------------------------------------------------------------
# 6) clash score
# ---------------------------------------------------------------------------
def calculate_clash_score(pdb_filepath: str | Path, distance_threshold: float = 2.0) -> float:
    if not os.path.isfile(pdb_filepath):
        return np.nan
    parser = PDBParser(QUIET=True)
    try:
        atoms = [a for a in parser.get_structure('s', pdb_filepath).get_atoms() if a.element != 'H']
    except Exception:
        return np.nan
    coords = np.array([a.get_coord() for a in atoms])
    resids = [a.get_parent().get_id()[1] for a in atoms]
    pairs = cKDTree(coords).query_pairs(r=distance_threshold)
    return float(sum(1 for i, j in pairs if resids[i] != resids[j]))

# ---------------------------------------------------------------------------
# 7) scatter plot
# ---------------------------------------------------------------------------
def scatter_plot(df: pd.DataFrame, x: str, y: str, out_png: Path, title: str, ylab: str):
    fig = px.scatter(
        df, x=x, y=y,
        color=df["Rank"].astype(str),
        hover_data={x:":.2f", y:":.2f"},
        title=title,
        labels={x: x, y: ylab},
    )
    fig.update_layout(template="plotly_white")
    out_png.write_bytes(fig.to_image(format="png", scale=2))
    print(f"Saved {out_png}")

# ---------------------------------------------------------------------------
# 8) PDF builder
# ---------------------------------------------------------------------------
def build_pdf(
    merged_df: pd.DataFrame,
    pdf_path: Path,
    plots: Dict[str, Path],
    af_dir: str | Path,
):
    doc = SimpleDocTemplate(str(pdf_path), pagesize=landscape(letter))
    styles = getSampleStyleSheet()
    elems = []
    elems.append(Paragraph(f"Analysis Report<br/><i>AF dir:</i> {af_dir}", styles["Title"]))
    elems.append(Spacer(1, 12))

    hdr = [
        "#","Name","Reference","Model","pLDDT","RMSD","TM","Mean H1 Lt",
        "pTM","ipTM","avg_PAE","Clash"
    ]
    data = [hdr]
    ranked = merged_df.sort_values("Rank_Score", ascending=False).reset_index(drop=True)
    for i, row in ranked.iterrows():
        data.append([
            i+1,
            row["Name"], row["Reference"], row["Model"],
            f"{row['pLDDT']:.2f}", f"{row['rmsd']:.2f}", f"{row['TM score']:.4f}", f"{row['H1_Lifetime']:.3f}",
            f"{row['pTM']:.2f}", f"{row['ipTM']:.2f}", f"{row['avg_PAE']:.2f}", f"{row['Clash']:.0f}"
        ])

    tbl = Table(data, repeatRows=1, colWidths=[25,100,60,70,45,45,55,55,45,45,50,40])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#4F81BD")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.whitesmoke),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("GRID",(0,0),(-1,-1),0.5,colors.black),
    ]))
    elems.append(tbl)
    elems.append(Spacer(1, 24))

    for title, path in plots.items():
        elems.append(Paragraph(title, styles["Heading2"]))
        elems.append(Spacer(1, 6))
        elems.append(RLImage(str(path), width=6.5*inch, height=4.5*inch))
        elems.append(Spacer(1, 18))

    doc.build(elems)
    print(f"PDF written to {pdf_path}")

# ---------------------------------------------------------------------------
# 9) Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(args: argparse.Namespace):
    out_dir = Path(args.output_dir); out_dir.mkdir(exist_ok=True, parents=True)
    print("[1/4] Extracting pLDDT scores …")
    plddt_df = extract_plddt_scores(args.af_base_dir)
    plddt_df.to_csv(out_dir / args.plddt_tsv, sep="\t", index=False)

    print("[2/4] Running USalign comparisons …")
    us_rows: List[Tuple] = []
    for ref in Path(args.rf_base_dir).glob("*.pdb"):
        refn = ref.stem
        for mod in Path(args.af_base_dir).rglob(f"{refn}*/ranked_*.pdb"):
            rk = int(mod.stem.split("_")[1])
            nm = f"{mod.parents[1].name}/{mod.parent.name}"
            met = run_usalign(ref, mod, multimer_mode=True)
            us_rows.append((nm, refn, mod.stem, rk, met["USalign_RMSD"], met["USalign_TM"], str(mod)))
    us_df = pd.DataFrame(
        us_rows,
        columns=["Name","Reference","Model","Rank","rmsd","TM score","PDB_Path"],
    )
    us_df.to_csv(out_dir / args.rmsd_tsv, sep="\t", index=False)

    print("[3/4] Merging + computing extras …")
    merged = (
        plddt_df
            .merge(us_df, on=["Name","Rank"], how="inner")
            .dropna(subset=["pLDDT","rmsd","TM score"])
            .reset_index(drop=True)
    )
    merged["H1_Lifetime"] = [mean_h1_lifetime(p) for p in merged["PDB_Path"]]

    # compute PAE, pTM/ipTM, clash
    extras = {"pTM": [], "ipTM": [], "avg_PAE": [], "Clash": []}
    for _, r in merged.iterrows():
        p = Path(r["PDB_Path"])
        idx = int(p.stem.split("_")[1])
        model_id = f"model_{idx+1}_multimer_v3_pred_0"

        ptm_json  = p.parent / "ranking_debug.json"
        ptm, iptm = extract_ptm_iptm(ptm_json, model_id)

        pae_json  = p.parent / f"pae_{model_id}.json"
        avg_pae   = extract_average_pae(pae_json)

        clash     = calculate_clash_score(p)

        extras["pTM"].append(ptm)
        extras["ipTM"].append(iptm)
        extras["avg_PAE"].append(avg_pae)
        extras["Clash"].append(clash)

    for k, v in extras.items():
        merged[k] = v

    merged["Rank_Score"] = (merged["pLDDT"]/100 + merged["TM score"]) / 2.0
    merged.to_csv(out_dir / args.final_merged_tsv, sep="\t", index=False)

    print("[4/4] Creating scatter plots …")
    plots = {
        "pLDDT vs TM score": out_dir / args.scatter1,
        "TM score vs RMSD":   out_dir / args.scatter2,
        "pLDDT vs RMSD":      out_dir / args.scatter3,
    }
    scatter_plot(merged, "pLDDT", "TM score", out_dir / args.scatter1, "pLDDT vs TM","TM score")
    scatter_plot(merged, "TM score", "rmsd",   out_dir / args.scatter2, "TM vs RMSD","RMSD (Å)")
    scatter_plot(merged, "pLDDT", "rmsd",      out_dir / args.scatter3, "pLDDT vs RMSD","RMSD (Å)")

    if args.generate_pdf:
        build_pdf(merged, out_dir / args.pdf_name, plots, args.af_base_dir)

# ---------------------------------------------------------------------------
# 10) CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Full Analysis pipeline with extended metrics")
    p.add_argument("--rf_base_dir",    required=True)
    p.add_argument("--af_base_dir",    required=True)
    p.add_argument("--output_dir",     required=True)
    p.add_argument("--plddt_tsv",      default="plddt.tsv")
    p.add_argument("--rmsd_tsv",       default="rmsd.tsv")
    p.add_argument("--final_merged_tsv",default="final_merged.tsv")
    p.add_argument("--scatter1",       default="scatter_plddt_vs_tm.png")
    p.add_argument("--scatter2",       default="scatter_tm_vs_rmsd.png")
    p.add_argument("--scatter3",       default="scatter_plddt_vs_rmsd.png")
    p.add_argument("--pdf_name",       default="analysis_report.pdf")
    p.add_argument("--generate_pdf",   action="store_true")
    args = p.parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()
