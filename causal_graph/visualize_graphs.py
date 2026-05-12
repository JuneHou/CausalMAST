"""
visualize_graphs.py — Visualize MAST validated causal graph (and optionally
the corr ≥ τ ablation graph).

Default output: `outputs/graph_causal.png` — intervention-validated causal
graph, weights = |Δ(A→B)|. This is what the +CG / +GI+SI causal-only
inference conditions inject into prompts.

Optional corr ≥ τ graph: pass `--corr_threshold τ` (and `--suppes`) to
additionally render `outputs/graph_corr_t<τ>.png` showing the edge set used
in the corr-threshold ablation — Suppes edges with geomean ≥ τ unioned with
the validated causal edges, weighted by geomean.

Usage (from causal_graph/):

    # default — causal-only figure
    python visualize_graphs.py \
        --effect_edges outputs/interventions/effect_edges.json \
        --out_dir      outputs

    # also render corr ≥ 0.2 ablation graph (for appendix)
    python visualize_graphs.py \
        --effect_edges  outputs/interventions/effect_edges.json \
        --suppes        outputs/suppes_graph.json \
        --corr_threshold 0.2 \
        --out_dir       outputs
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

# Descriptive names (no numeric prefix), used as node labels in the plot.
# Derived from NODE_LABELS values which are formatted "<id>\n<name>".
NODE_NAMES = {k: v.split("\n", 1)[1] for k, v in NODE_LABELS.items()}


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
def load_validated_causal(effect_edges_path):
    """Causal-only graph from intervention validation.
    Weight = |Δ(A→B)|. Only validated=True edges are returned."""
    with open(effect_edges_path) as f:
        data = json.load(f)
    out = []
    for v in data["edges"].values():
        if not v.get("validated"):
            continue
        d = v.get("delta")
        if d is None:
            continue
        out.append({"a": v["a"], "b": v["b"], "weight": abs(d)})
    return out


def load_corr_graph(suppes_path, validated_pairs, corr_threshold):
    """Reproduce the corr ≥ τ injection set used at inference:
    Suppes edges with geomean ≥ τ ∪ validated causal edges, weighted by geomean.
    """
    with open(suppes_path) as f:
        data = json.load(f)
    out = []
    seen = set()
    for e in data["edges"]:
        geomean = (max(0.0, e.get("p_b_given_a", 0.0)) *
                   max(0.0, e.get("pr_delta", 0.0))) ** 0.5
        if geomean >= corr_threshold:
            out.append({"a": e["a"], "b": e["b"], "weight": geomean})
            seen.add((e["a"], e["b"]))
    suppes_lookup = {(e["a"], e["b"]): e for e in data["edges"]}
    for (a, b) in validated_pairs:
        if (a, b) in seen:
            continue
        s = suppes_lookup.get((a, b))
        if s is None:
            continue
        geomean = (max(0.0, s.get("p_b_given_a", 0.0)) *
                   max(0.0, s.get("pr_delta", 0.0))) ** 0.5
        out.append({"a": a, "b": b, "weight": geomean})
    return out


# ---------------------------------------------------------------------------
# Build NetworkX graph
# ---------------------------------------------------------------------------
def build_graph(edges, weight_key="weight"):
    G = nx.DiGraph()
    # Only add nodes that are endpoints of edges. Pass --show_isolated to
    # include the full ALL_NODES taxonomy as isolated nodes.
    for e in edges:
        G.add_edge(e["a"], e["b"], weight=e[weight_key])
    return G


# ---------------------------------------------------------------------------
# Draw one graph
# ---------------------------------------------------------------------------
def draw_graph(ax, G, pos, title, weight_range=(0.05, 0.87),
               weight_label="weight",
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
        node_size=2400,
        alpha=0.92,
    )

    # Node labels — descriptive error-type names, not numeric IDs
    name_labels = {n: NODE_NAMES.get(n, n) for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels=name_labels, ax=ax,
        font_size=6.5, font_weight="bold", font_color="white",
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
    cbar.set_label(weight_label, fontsize=8)
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
    parser.add_argument("--effect_edges", default="outputs/interventions/effect_edges.json",
                        help="effect_edges.json (drives the causal-only plot)")
    parser.add_argument("--suppes",       default=None,
                        help="suppes_graph.json — only required if --corr_threshold is set")
    parser.add_argument("--corr_threshold", type=float, default=None,
                        help="If set, also render the corr ≥ τ ablation graph "
                             "(Suppes edges with geomean ≥ τ ∪ validated causal). "
                             "Requires --suppes.")
    parser.add_argument("--show_isolated", action="store_true",
                        help="Include all 13 MAST taxonomy categories as nodes "
                             "(isolated if no validated edge connects them).")
    parser.add_argument("--out_dir",      default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    causal_edges = load_validated_causal(args.effect_edges)
    G_causal = build_graph(causal_edges)

    corr_edges = None
    G_corr = None
    if args.corr_threshold is not None:
        if not args.suppes:
            parser.error("--corr_threshold requires --suppes")
        validated_pairs = {(e["a"], e["b"]) for e in causal_edges}
        corr_edges = load_corr_graph(args.suppes, validated_pairs, args.corr_threshold)
        G_corr = build_graph(corr_edges)

    # Optionally include the full taxonomy as isolated nodes
    if args.show_isolated:
        G_causal.add_nodes_from(ALL_NODES)
        if G_corr is not None:
            G_corr.add_nodes_from(ALL_NODES)

    pos = make_positions()

    causal_weights = [e["weight"] for e in causal_edges]
    causal_range = (min(causal_weights), max(causal_weights)) if causal_weights else (0, 1)

    CAUSAL_LABEL = r"|Δ(A→B)|  (intervention-validated effect)"

    # ---- Validated causal-only graph (default) ----
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_graph(ax, G_causal, pos,
               title=f"Validated Causal Graph  ({len(causal_edges)} edges, weight=|Δ|)",
               weight_range=causal_range,
               weight_label=CAUSAL_LABEL,
               show_edge_labels=True,
               label_threshold=0.0)
    add_legend(ax)
    fig.tight_layout()
    out_c = os.path.join(args.out_dir, "graph_causal.png")
    fig.savefig(out_c, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_c}  ({len(causal_edges)} validated edges, weight = |Δ|)")

    # ---- Optional: corr ≥ τ ablation graph ----
    if corr_edges is not None:
        corr_weights = [e["weight"] for e in corr_edges]
        corr_range = (min(corr_weights), max(corr_weights)) if corr_weights else (0, 1)
        SUPPES_LABEL = r"geomean = $\sqrt{P(B|A)\cdot \Delta\mathrm{PR}}$"

        fig, ax = plt.subplots(figsize=(9, 7))
        draw_graph(ax, G_corr, pos,
                   title=(f"Corr ≥ {args.corr_threshold} Ablation Graph  "
                          f"({len(corr_edges)} edges, weight=geomean)"),
                   weight_range=corr_range,
                   weight_label=SUPPES_LABEL,
                   show_edge_labels=False)
        add_legend(ax)
        fig.tight_layout()
        tag = f"{args.corr_threshold}".replace(".", "p")
        out_corr = os.path.join(args.out_dir, f"graph_corr_t{tag}.png")
        fig.savefig(out_corr, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_corr}  ({len(corr_edges)} edges; "
              f"Suppes geomean ≥ {args.corr_threshold} ∪ validated)")

    # ---- Print edge summary ----
    print(f"\nValidated causal edges ({len(causal_edges)}) — sorted by |Δ|:")
    for e in sorted(causal_edges, key=lambda x: -x["weight"]):
        print(f"  {e['a']} → {e['b']}  |Δ|={e['weight']:.4f}")


if __name__ == "__main__":
    main()
