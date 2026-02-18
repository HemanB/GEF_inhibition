#!/usr/bin/env python3
"""Generate AlphaFold 3 Server JSON inputs for all inhibitor-GTPase pairs."""

import csv
import json
from pathlib import Path

# GTPase sequences (UniProt canonical)
GTPASE_SEQUENCES = {
    "Cdc42": (  # P60953, human CDC42, 191 aa
        "MQTIKCVVVGDGAVGKTCLLISYTTNKFPSEYVPTVFDNYAVTVMIGGEPYTLGLFDTAG"
        "QEDYDRLRPLSYPQTDVFLVCFSVVSPSSFENVKEKWVPEITHHCPKTPFLLVGTQIDLR"
        "DDPSTIEKLAKNKQKPITPETAEKLARDLKAVKYVECSALTQKGLKNVFDEAILAALEPPE"
        "PKKSRRCVLL"
    ),
    "Rac1": (  # P63000, human RAC1, 192 aa
        "MQAIKCVVVGDGAVGKTCLLISYTTNAFPGEYIPTVFDNYSANVMVDGKPVNLGLWDTAG"
        "QEDYDRLRPLSYPQTDVFLICFSLVSPASFENVRAKWYPEVRHHCPNTPIILVGTKLDLRD"
        "DKDTIEKLKEKKLTPITYPQGLAMAKEIGAVKYLECSALTQRGLKTVFDEAIRAVLCPPPV"
        "KKRKRKCLLL"
    ),
}

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "GEF_inhibitors_modeling_data.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "af3_inputs"


def make_af3_server_json(name: str, inhibitor_seq: str, gtpase_name: str) -> list[dict]:
    """Create an AF3 server-format JSON (top-level array) for a 2-chain complex."""
    return [
        {
            "name": name,
            "modelSeeds": [],
            "dialect": "alphafoldserver",
            "version": 1,
            "sequences": [
                {"proteinChain": {"sequence": inhibitor_seq, "count": 1}},
                {"proteinChain": {"sequence": GTPASE_SEQUENCES[gtpase_name], "count": 1}},
            ],
        }
    ]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            inh_id = row["inhibitor_id"]
            gtpase = row["target_gtpase"]
            seq = row["sequence"]

            job_name = f"{inh_id}_{gtpase}"
            payload = make_af3_server_json(job_name, seq, gtpase)

            out_path = OUT_DIR / f"{job_name}.json"
            with open(out_path, "w") as jf:
                json.dump(payload, jf, indent=2)

            print(f"  {out_path.name}  ({len(seq)} + {len(GTPASE_SEQUENCES[gtpase])} aa)")

    print(f"\nGenerated {sum(1 for _ in OUT_DIR.glob('*.json'))} JSON files in {OUT_DIR}")


if __name__ == "__main__":
    main()
