#!/usr/bin/env python3
"""Phase 4: Network Features from residue interaction graphs."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import networkx as nx
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT / "data" / "raw" / "GEF_inhibitors_modeling_data.csv"
RESULTS = PROJECT / "results"
CONTACT_DIR = PROJECT / "data" / "processed" / "contact_maps"
FEATURE_DIR = PROJECT / "data" / "processed" / "features"

# Switch regions on GTPase
SWITCH_I = set(range(29, 43))
SWITCH_II = set(range(62, 69))


def build_contact_graph(contacts: list[dict], cutoff: float = 8.0) -> nx.Graph:
    """Build residue interaction graph from contacts.

    Nodes are (chain, resid) tuples. Edges connect contacting residues.
    """
    G = nx.Graph()
    for c in contacts:
        if c["distance"] <= cutoff:
            node_a = ("A", c["res_a_idx"])
            node_b = ("B", c["res_b_idx"])
            if not G.has_node(node_a):
                G.add_node(node_a, chain="A", resid=c["res_a_idx"],
                           resname=c["res_a_name"], res_class=c["res_a_class"])
            if not G.has_node(node_b):
                G.add_node(node_b, chain="B", resid=c["res_b_idx"],
                           resname=c["res_b_name"], res_class=c["res_b_class"])
            if G.has_edge(node_a, node_b):
                # Keep minimum distance
                if c["distance"] < G[node_a][node_b]["distance"]:
                    G[node_a][node_b]["distance"] = c["distance"]
            else:
                G.add_edge(node_a, node_b,
                           distance=c["distance"],
                           contact_type=c["contact_type"])
    return G


def compute_network_features(G: nx.Graph, scaffold_len: int,
                              hotspots: list[int]) -> dict:
    """Compute graph-theoretic features from contact network."""
    if G.number_of_nodes() == 0:
        return {
            "network_nodes": 0,
            "network_edges": 0,
            "betweenness_max": 0.0,
            "betweenness_mean": 0.0,
            "degree_mean": 0.0,
            "degree_max": 0,
            "clustering_coeff": 0.0,
            "community_count": 0,
            "shortest_path_scaffold_switchI": np.nan,
            "shortest_path_scaffold_switchII": np.nan,
            "shortest_path_scaffold_hotspot": np.nan,
        }

    features = {
        "network_nodes": G.number_of_nodes(),
        "network_edges": G.number_of_edges(),
    }

    # Betweenness centrality
    bc = nx.betweenness_centrality(G)
    features["betweenness_max"] = round(max(bc.values()), 4) if bc else 0.0
    features["betweenness_mean"] = round(float(np.mean(list(bc.values()))), 4) if bc else 0.0

    # Degree statistics
    degrees = [d for _, d in G.degree()]
    features["degree_mean"] = round(float(np.mean(degrees)), 2) if degrees else 0.0
    features["degree_max"] = max(degrees) if degrees else 0

    # Clustering coefficient
    features["clustering_coeff"] = round(nx.average_clustering(G), 4)

    # Community detection (greedy modularity)
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
        features["community_count"] = len(communities)
    except Exception:
        features["community_count"] = 1

    # Shortest paths from scaffold interface residues to switch regions and hotspots
    scaffold_nodes = [n for n in G.nodes() if n[0] == "A" and n[1] <= scaffold_len]
    switch_I_nodes = [n for n in G.nodes() if n[0] == "B" and n[1] in SWITCH_I]
    switch_II_nodes = [n for n in G.nodes() if n[0] == "B" and n[1] in SWITCH_II]
    hotspot_nodes = [n for n in G.nodes() if n[0] == "A" and n[1] in set(hotspots)]

    def min_shortest_path(sources, targets):
        if not sources or not targets:
            return np.nan
        min_path = float("inf")
        for s in sources:
            for t in targets:
                try:
                    path_len = nx.shortest_path_length(G, s, t)
                    min_path = min(min_path, path_len)
                except nx.NetworkXNoPath:
                    continue
        return min_path if min_path < float("inf") else np.nan

    features["shortest_path_scaffold_switchI"] = min_shortest_path(scaffold_nodes, switch_I_nodes)
    features["shortest_path_scaffold_switchII"] = min_shortest_path(scaffold_nodes, switch_II_nodes)
    features["shortest_path_scaffold_hotspot"] = min_shortest_path(scaffold_nodes, hotspot_nodes)

    return features


def main():
    print("=" * 60)
    print("Phase 4: Network Features")
    print("=" * 60)

    designs = pd.read_csv(DATA_CSV)
    all_features = []

    for _, design in designs.iterrows():
        inh_id = design["inhibitor_id"]
        scaffold_len = int(design["scaffold_length"])
        hotspots = [int(design[f"hotspot_{i}"]) for i in range(1, 5)]

        # Load contacts
        contact_file = CONTACT_DIR / f"{inh_id}_contacts.json"
        if not contact_file.exists():
            print(f"  WARNING: contacts not found for {inh_id}")
            continue

        with open(contact_file) as f:
            contacts = json.load(f)

        print(f"\n{inh_id}:")

        # Build graph at 8A cutoff
        G = build_contact_graph(contacts, cutoff=8.0)
        print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Compute features
        features = compute_network_features(G, scaffold_len, hotspots)
        features["inhibitor_id"] = inh_id
        all_features.append(features)

        print(f"  Betweenness max: {features['betweenness_max']:.4f}")
        print(f"  Communities: {features['community_count']}")
        sp_si = features['shortest_path_scaffold_switchI']
        sp_sii = features['shortest_path_scaffold_switchII']
        print(f"  Scaffold→SwitchI path: {sp_si}, Scaffold→SwitchII path: {sp_sii}")

    # Save and merge
    net_df = pd.DataFrame(all_features)
    net_path = FEATURE_DIR / "network_features.csv"
    net_df.to_csv(net_path, index=False)
    print(f"\nSaved network features: {net_path}")

    # Merge into all_features.csv
    all_feat_path = FEATURE_DIR / "all_features.csv"
    if all_feat_path.exists():
        all_df = pd.read_csv(all_feat_path)
        # Drop any existing network columns to avoid duplication
        net_cols = [c for c in net_df.columns if c != "inhibitor_id"]
        all_df = all_df.drop(columns=[c for c in net_cols if c in all_df.columns], errors="ignore")
        merged = all_df.merge(net_df, on="inhibitor_id", how="left")
        merged.to_csv(all_feat_path, index=False)
        print(f"Merged into {all_feat_path}: {merged.shape}")

    return net_df


if __name__ == "__main__":
    main()
