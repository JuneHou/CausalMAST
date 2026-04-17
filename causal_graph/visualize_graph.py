"""
visualize_graph.py — Visualize the CAPRI pruned causal graph with bootstrap stability weights.

Nodes are arranged by topological level (left to right).
Edge color and thickness encode bootstrap stability (frequency).
Edges in the CAPRI pruned graph are shown; stability from edge_stability.json.

Usage:
    python visualize_graph.py
    python visualize_graph.py --capri outputs/capri_graph.json --out outputs/causal_graph.png
"""

import argparse
import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np


CATEGORY_LABELS = {
    "1.1": "1.1\nDisobey Task Spec",
    "1.2": "1.2\nDisobey Role Spec",
    "1.3": "1.3\nStep Repetition",
    "1.4": "1.4\nLoss of Conv History",
    "1.5": "1.5\nUnaware of Termination",
    "2.1": "2.1\nConversation Reset",
    "2.2": "2.2\nFail to Clarify",
    "2.3": "2.3\nTask Derailment",
    "2.4": "2.4\nInfo Withholding",
    "2.6": "2.6\nAction-Reasoning\nMismatch",
    "3.1": "3.1\nPremature Termination",
    "3.2": "3.2\nWeak Verification",
    "3.3": "3.3\nNo/Incorrect\nVerification",
}

# Color nodes by category group
GROUP_COLORS = {
    "1": "#4C72B0",   # Task Compliance — blue
    "2": "#DD8452",   # Multi-Agent Coordination — orange
    "3": "#55A868",   # Output Verification — green
}


def load_data(capri_path: str, stability_path: str) -> tuple:
    with open(capri_path) as f:
        capri = json.load(f)
    with open(stability_path) as f:
        stab = json.load(f)

    capri_edges = {(e["a"], e["b"]) for e in capri["edges"]}

    stability = {}
    for e in stab["edges"]:
        stability[(e["a"], e["b"])] = e["frequency"]

    return capri_edges, stability


def build_graph(capri_edges: set, stability: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for a, b in capri_edges:
        w = stability.get((a, b), 0.0)
        G.add_edge(a, b, weight=w)
    return G


def topological_layout(G: nx.DiGraph, hierarchy_path: str = None) -> dict:
    """
    Position nodes by topological level (x) and spread vertically (y).
    If hierarchy_path provided, use precomputed levels; else compute from graph.
    """
    if hierarchy_path:
        try:
            with open(hierarchy_path) as f:
                h = json.load(f)
            levels = {}
            for level_key, nodes in h["levels"].items():
                lvl = int(level_key.split("_")[1])
                for n in nodes:
                    levels[n] = lvl
        except Exception:
            levels = None
    else:
        levels = None

    if levels is None:
        # Fallback: longest path layering
        for node in G.nodes():
            G.nodes[node]["layer"] = 0
        try:
            for node in nx.topological_sort(G):
                for succ in G.successors(node):
                    G.nodes[succ]["layer"] = max(
                        G.nodes[succ].get("layer", 0),
                        G.nodes[node].get("layer", 0) + 1
                    )
        except nx.NetworkXUnfeasible:
            pass
        levels = {n: G.nodes[n].get("layer", 0) for n in G.nodes()}

    # Group nodes by level
    from collections import defaultdict
    by_level = defaultdict(list)
    for node, lvl in levels.items():
        by_level[lvl].append(node)

    pos = {}
    x_gap = 3.5
    for lvl, nodes in by_level.items():
        nodes_sorted = sorted(nodes)
        n = len(nodes_sorted)
        y_positions = np.linspace(-(n - 1) / 2.0, (n - 1) / 2.0, n)
        for node, y in zip(nodes_sorted, y_positions):
            pos[node] = (lvl * x_gap, y * 1.8)

    return pos


def draw(G: nx.DiGraph, pos: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    # Node colors by group
    node_colors = []
    for node in G.nodes():
        group = node.split(".")[0]
        node_colors.append(GROUP_COLORS.get(group, "#999999"))

    # Edge weights for styling
    edges = list(G.edges(data=True))
    weights = [d["weight"] for _, _, d in edges]
    w_min, w_max = (min(weights), max(weights)) if weights else (0, 1)

    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    edge_colors = [cmap(norm(w)) for w in weights]
    edge_widths = [1.5 + 4.0 * (w - w_min) / max(w_max - w_min, 1e-6) for w in weights]

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowsize=18,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08",
        min_source_margin=30,
        min_target_margin=30,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=2200,
        alpha=0.92,
        linewidths=1.5,
        edgecolors="white",
    )

    # Node labels (short names)
    labels = {n: CATEGORY_LABELS.get(n, n) for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=7.5,
        font_color="white",
        font_weight="bold",
    )

    # Edge weight labels
    edge_labels = {(a, b): f"{d['weight']:.2f}" for a, b, d in edges}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax,
        font_size=7,
        font_color="#333333",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, ec="none"),
        label_pos=0.35,
    )

    # Colorbar for edge stability
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=30)
    cbar.set_label("Bootstrap Stability", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Level annotations — placed above the topmost node at each x column
    x_positions = sorted(set(x for x, y in pos.values()))
    level_names = ["Level 0\n(Root causes)", "Level 1\n(Intermediate)", "Level 2\n(Downstream)"]
    y_max = max(y for _, y in pos.values())
    for i, x in enumerate(x_positions):
        label = level_names[i] if i < len(level_names) else f"Level {i}"
        ax.text(x, y_max + 1.1, label, ha="center", va="bottom", fontsize=9,
                color="#555555", style="italic",
                transform=ax.transData)

    # Group legend
    legend_handles = [
        mpatches.Patch(color=GROUP_COLORS["1"], label="Group 1: Task Compliance"),
        mpatches.Patch(color=GROUP_COLORS["2"], label="Group 2: Multi-Agent Coordination"),
        mpatches.Patch(color=GROUP_COLORS["3"], label="Group 3: Output Verification"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
              framealpha=0.85, edgecolor="#cccccc")

    ax.set_title(
        "MAST-AG2 Causal Error Graph (CAPRI pruned)\n"
        "Edge weight = bootstrap stability  |  Direction = causal precedence",
        fontsize=13, fontweight="bold", pad=16,
    )
    ax.axis("off")

    # Expand y-axis to prevent level labels being clipped
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo - 0.5, y_hi + 1.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"✓ Saved to {out_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Visualize CAPRI causal graph")
    ap.add_argument("--capri", default="outputs/capri_graph.json")
    ap.add_argument("--stability", default="outputs/edge_stability.json")
    ap.add_argument("--hierarchy", default="outputs/hierarchy_levels.json",
                    help="Optional: precomputed hierarchy levels for layout")
    ap.add_argument("--out", default="outputs/causal_graph.png")
    args = ap.parse_args()

    capri_edges, stability = load_data(args.capri, args.stability)
    G = build_graph(capri_edges, stability)
    pos = topological_layout(G, args.hierarchy if args.hierarchy else None)
    draw(G, pos, args.out)

    print(f"\nGraph summary:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"\nEdges by stability (descending):")
    for a, b, d in sorted(G.edges(data=True), key=lambda x: -x[2]["weight"]):
        print(f"  {a} → {b}  ({d['weight']:.2f})")


if __name__ == "__main__":
    main()
