"""
visualize_graphs.py — Visualize Suppes and CAPRI causal graphs.

Produces two side-by-side figures saved as:
    outputs/graph_suppes.png
    outputs/graph_capri.png
    outputs/graph_both.png   (side-by-side comparison)

Edge weight shown is pr_delta = P(B|A) - P(B|~A) from the Suppes screen.
CAPRI edges are a BIC-pruned subset of Suppes edges; their pr_delta is
looked up from the Suppes edge list.

Usage (from causal_graph/):
    conda run -n /data/wang/junh/envs/causal python visualize_graphs.py
    python visualize_graphs.py  --suppes outputs/suppes_graph.json \
                                --capri   outputs/capri_graph.json \
                                --out_dir outputs
"""

import json
import argparse
import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Node metadata
# ---------------------------------------------------------------------------
NODE_LABELS = {
    "1.1": "1.1\nDisobey Task\nSpec.",
    "1.2": "1.2\nDisobey Role\nSpec.",
    "1.3": "1.3\nStep\nRepetition",
    "1.4": "1.4\nLoss of Conv.\nHistory",
    "1.5": "1.5\nUnaware of\nTermination",
    "2.1": "2.1\nConversation\nReset",
    "2.2": "2.2\nFail to Ask\nClarification",
    "2.3": "2.3\nTask\nDerailment",
    "2.4": "2.4\nInfo.\nWithholding",
    "2.6": "2.6\nAction-Reasoning\nMismatch",
    "3.1": "3.1\nPremature\nTermination",
    "3.2": "3.2\nWeak\nVerification",
    "3.3": "3.3\nNo/Incorrect\nVerification",
}

# Colors per top-level MAST group
GROUP_COLORS = {
    "1": "#4C9BE8",   # blue  — Task Compliance
    "2": "#E87C4C",   # orange — Multi-Agent Coordination
    "3": "#5CBF7A",   # green  — Output Verification
}

GROUP_NAMES = {
    "1": "Task Compliance Errors",
    "2": "Multi-Agent Coordination Errors",
    "3": "Output Verification Errors",
}

ALL_NODES = list(NODE_LABELS.keys())


# ---------------------------------------------------------------------------
# Fixed layout: 3-column grid by group, y-spaced evenly within group
# ---------------------------------------------------------------------------
def make_positions():
    layout = {
        "1": ["1.1", "1.2", "1.3", "1.4", "1.5"],
        "2": ["2.1", "2.2", "2.3", "2.4", "2.6"],
        "3": ["3.1", "3.2", "3.3"],
    }
    col_x = {"1": 0.0, "2": 1.0, "3": 2.0}
    pos = {}
    for grp, nodes in layout.items():
        n = len(nodes)
        # center the group vertically
        ys = np.linspace(0.9, 0.1, n)
        for node, y in zip(nodes, ys):
            pos[node] = (col_x[grp], y)
    return pos


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_suppes(path):
    with open(path) as f:
        data = json.load(f)
    # Build lookup: (a, b) -> edge dict
    lookup = {}
    for e in data["edges"]:
        lookup[(e["a"], e["b"])] = e
    return data["edges"], lookup


def load_capri(path, suppes_lookup):
    with open(path) as f:
        data = json.load(f)
    edges = []
    for e in data["edges"]:
        key = (e["a"], e["b"])
        pr_delta = suppes_lookup[key]["pr_delta"] if key in suppes_lookup else 0.0
        edges.append({"a": e["a"], "b": e["b"], "pr_delta": pr_delta})
    return edges


# ---------------------------------------------------------------------------
# Build NetworkX graph
# ---------------------------------------------------------------------------
def build_graph(edges, weight_key="pr_delta"):
    G = nx.DiGraph()
    G.add_nodes_from(ALL_NODES)
    for e in edges:
        G.add_edge(e["a"], e["b"], weight=e[weight_key])
    return G


# ---------------------------------------------------------------------------
# Draw one graph
# ---------------------------------------------------------------------------
def draw_graph(ax, G, pos, title, weight_range=(0.05, 0.87),
               show_edge_labels=True, label_threshold=0.0):
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(-0.08, 1.05)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    # Column header labels
    col_headers = [
        (0.0, 1.02, "Group 1\nTask Compliance", GROUP_COLORS["1"]),
        (1.0, 1.02, "Group 2\nCoordination",    GROUP_COLORS["2"]),
        (2.0, 1.02, "Group 3\nVerification",     GROUP_COLORS["3"]),
    ]
    for cx, cy, label, color in col_headers:
        ax.text(cx, cy, label, ha="center", va="bottom", fontsize=8,
                color=color, fontweight="bold")

    # Nodes
    node_colors = [GROUP_COLORS[n[0]] for n in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=1400,
        alpha=0.92,
    )

    # Node labels (short)
    short_labels = {n: n for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels=short_labels, ax=ax,
        font_size=9, font_weight="bold", font_color="white",
    )

    # Edges: color and thickness by weight
    edges_data = [(u, v, d["weight"]) for u, v, d in G.edges(data=True)]
    if not edges_data:
        return

    weights = [w for _, _, w in edges_data]
    w_min, w_max = weight_range
    cmap = plt.cm.YlOrRd

    # Draw edges one at a time to vary color/width
    for u, v, w in edges_data:
        norm_w = (w - w_min) / (w_max - w_min + 1e-9)
        norm_w = max(0.0, min(1.0, norm_w))
        color = cmap(0.25 + norm_w * 0.7)
        width = 1.0 + norm_w * 4.5

        # Slightly curve edges to reduce overlap
        rad = 0.15 if (v, u) in G.edges() else 0.08
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], ax=ax,
            edge_color=[color],
            width=width,
            arrows=True,
            arrowsize=18,
            connectionstyle=f"arc3,rad={rad}",
            min_source_margin=22,
            min_target_margin=22,
        )

    # Edge weight labels (pr_delta) for edges above threshold
    if show_edge_labels:
        edge_labels = {
            (u, v): f"{w:.2f}"
            for u, v, w in edges_data
            if w >= label_threshold
        }
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, ax=ax,
            font_size=6.5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, ec="none"),
            label_pos=0.35,
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=mcolors.Normalize(vmin=w_min, vmax=w_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=25)
    cbar.set_label("pr_delta  P(B|A) − P(B|¬A)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# Legend patch
# ---------------------------------------------------------------------------
def add_legend(ax):
    patches = [
        mpatches.Patch(color=GROUP_COLORS[g], label=GROUP_NAMES[g])
        for g in ["1", "2", "3"]
    ]
    ax.legend(handles=patches, loc="lower center",
              fontsize=8, framealpha=0.85, ncol=3,
              bbox_to_anchor=(0.5, -0.04))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suppes",  default="outputs/suppes_graph.json")
    parser.add_argument("--capri",   default="outputs/capri_graph.json")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    suppes_edges, suppes_lookup = load_suppes(args.suppes)
    capri_edges = load_capri(args.capri, suppes_lookup)

    G_suppes = build_graph(suppes_edges)
    G_capri  = build_graph(capri_edges)

    pos = make_positions()

    all_weights = [e["pr_delta"] for e in suppes_edges]
    w_range = (min(all_weights), max(all_weights))

    # ---- Individual: Suppes ----
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_graph(ax, G_suppes, pos,
               title=f"Suppes Probabilistic Causation Graph  ({len(suppes_edges)} edges)",
               weight_range=w_range,
               show_edge_labels=False)   # too many edges — skip labels
    add_legend(ax)
    fig.tight_layout()
    out_s = os.path.join(args.out_dir, "graph_suppes.png")
    fig.savefig(out_s, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_s}")

    # ---- Individual: CAPRI ----
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_graph(ax, G_capri, pos,
               title=f"CAPRI Pruned DAG  ({len(capri_edges)} edges, BIC criterion)",
               weight_range=w_range,
               show_edge_labels=True,
               label_threshold=0.0)
    add_legend(ax)
    fig.tight_layout()
    out_c = os.path.join(args.out_dir, "graph_capri.png")
    fig.savefig(out_c, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_c}")

    # ---- Side-by-side comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    draw_graph(axes[0], G_suppes, pos,
               title=f"Suppes Graph  ({len(suppes_edges)} edges)",
               weight_range=w_range,
               show_edge_labels=False)
    draw_graph(axes[1], G_capri, pos,
               title=f"CAPRI Pruned DAG  ({len(capri_edges)} edges)",
               weight_range=w_range,
               show_edge_labels=True,
               label_threshold=0.0)
    add_legend(axes[1])

    fig.suptitle(
        "MAST-AG2 Causal Error Graphs\n"
        "Edge weight = pr_delta [P(B|A) − P(B|¬A)]  |  Nodes colored by MAST error group",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    out_b = os.path.join(args.out_dir, "graph_both.png")
    fig.savefig(out_b, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_b}")

    # ---- Print edge summary ----
    print(f"\nSuppes edges ({len(suppes_edges)}) — sorted by pr_delta:")
    for e in sorted(suppes_edges, key=lambda x: -x["pr_delta"])[:10]:
        print(f"  {e['a']} → {e['b']}  pr_delta={e['pr_delta']:.4f}  "
              f"P(B|A)={e['p_b_given_a']:.3f}  P(B|¬A)={e['p_b_given_not_a']:.3f}")
    print(f"\nCAPRI edges ({len(capri_edges)}) — sorted by pr_delta:")
    for e in sorted(capri_edges, key=lambda x: -x["pr_delta"]):
        src = suppes_lookup.get((e["a"], e["b"]), {})
        print(f"  {e['a']} → {e['b']}  pr_delta={e['pr_delta']:.4f}  "
              f"P(B|A)={src.get('p_b_given_a', 0):.3f}  "
              f"P(B|¬A)={src.get('p_b_given_not_a', 0):.3f}")


if __name__ == "__main__":
    main()
