import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path(__file__).parent
out_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

def cnr_for_edge(G, u, v):
    nu = set(G.neighbors(u))
    nv = set(G.neighbors(v))
    # remove endpoints from neighbor sets if present
    nu.discard(v); nv.discard(u)
    common = len(nu & nv)
    denom = len(nu | nv)
    return common / denom if denom > 0 else 0.0, common

def edge_local_density(G, u, v):
    # A simple, stable proxy for local density around the edge:
    # average of endpoint clustering coefficients
    cu = nx.clustering(G, u)
    cv = nx.clustering(G, v)
    return 0.5 * (cu + cv)

def classify_edges(G, alpha=0.0, use_ld=False):
    """
    Returns dict edge->type in {'bond','silk','local_bridge','global_bridge'}.
    HETA-like: use CNR thresholds (quantiles) + bridge notions.
    LD-HETA-like: use adjusted CNR = CNR / ( (density+eps)^alpha ).
    """
    eps = 1e-6
    cnr_vals = {}
    adj_vals = {}
    commons = {}
    densities = {}
    for u, v in G.edges():
        c, common = cnr_for_edge(G, u, v)
        d = edge_local_density(G, u, v)
        cnr_vals[(u, v)] = c
        commons[(u, v)] = common
        densities[(u, v)] = d
        if use_ld:
            adj = c / ((d + eps) ** alpha)
            adj_vals[(u, v)] = adj

    vals = np.array(list(adj_vals.values() if use_ld else cnr_vals.values()))
    if len(vals) == 0:
        return {}, cnr_vals, adj_vals, densities

    # Thresholds via quantiles (robust, no fabricated fixed numbers)
    q_bond = np.quantile(vals, 0.70)  # top 30% as bond
    q_silk = np.quantile(vals, 0.30)  # bottom 30% as silk

    bridges = set(nx.bridges(G))  # undirected bridges
    # Normalize bridge orientation in tuples
    bridges = set((min(u,v), max(u,v)) for u,v in bridges)

    edge_type = {}
    for u, v in G.edges():
        key = (u, v)
        key_norm = (min(u,v), max(u,v))
        val = (adj_vals[key] if use_ld else cnr_vals[key])

        if key_norm in bridges:
            et = "global_bridge"
        else:
            # local bridge proxy: no common neighbors (distance without edge >2 in classic def)
            if commons[key] == 0:
                et = "local_bridge"
            else:
                if val >= q_bond:
                    et = "bond"
                elif val <= q_silk:
                    et = "silk"
                else:
                    # middle region: assign by closer side, default to bond-ish vs silk-ish
                    et = "bond" if (val - q_silk) >= (q_bond - val) else "silk"
        edge_type[key] = et

    return edge_type, cnr_vals, adj_vals, densities

def stable_layout(G, seed=7):
    # Increase k to spread out nodes further
    return nx.spring_layout(G, seed=seed, k=1.5 / np.sqrt(len(G) or 1))

def draw_edge_types(G, pos, edge_type, title, filepath):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_title(title)
    ax.axis("off")

    # Draw nodes (smaller and slightly transparent to reduce crowding)
    nx.draw_networkx_nodes(G, pos, node_size=25, ax=ax, alpha=0.7, node_color="#444444")
    
    # Group edges by type
    groups = {"bond": [], "silk": [], "local_bridge": [], "global_bridge": []}
    for (u, v), t in edge_type.items():
        groups[t].append((u, v))

    # Define a clear, color-coded style map (Unified width and solid lines)
    style_map = {
        "bond": dict(style="solid", width=1.2, alpha=0.8, color="#1f77b4"),          # Blue
        "silk": dict(style="solid", width=1.2, alpha=0.3, color="#cccccc"),         # Light Gray
        "local_bridge": dict(style="solid", width=1.2, alpha=0.9, color="#ff7f0e"),  # Orange
        "global_bridge": dict(style="solid", width=1.2, alpha=1.0, color="#d62728"), # Red
    }

    for t, edgelist in groups.items():
        if not edgelist:
            continue
        st = style_map[t]
        nx.draw_networkx_edges(
            G, pos, edgelist=edgelist,
            ax=ax, style=st["style"], width=st["width"], alpha=st["alpha"],
            edge_color=st["color"]
        )

    # Legend with color indicators (Unified solid lines)
    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], color="#1f77b4", linestyle="solid", label="bond"),
        mlines.Line2D([], [], color="#cccccc", linestyle="solid", label="silk"),
        mlines.Line2D([], [], color="#ff7f0e", linestyle="solid", label="local bridge"),
        mlines.Line2D([], [], color="#d62728", linestyle="solid", label="global bridge"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=220)
    plt.close()

# ---------- Experiment 1: Heterogeneous-density SBM structure (HETA-like edge types) ----------
sizes = [70, 70]
p_in_high = 0.18
p_in_low = 0.05
p_between = 0.01
P = [[p_in_high, p_between],
     [p_between, p_in_low]]
G1 = nx.stochastic_block_model(sizes, P, seed=1)
pos1 = stable_layout(G1, seed=11)

edge_type1, cnr1, adj1, dens1 = classify_edges(G1, use_ld=False)
fig1_path = out_dir / "fig1_sbm_heta_edge_types.png"
draw_edge_types(G1, pos1, edge_type1, "Fig 1. Heterogeneous-density SBM (HETA-like edge types)", fig1_path)

# ---------- Experiment 2: Toy network with density gradient + bridges ----------
# Build dense cluster A and sparse cluster B, then a few bridge edges
A = nx.erdos_renyi_graph(55, 0.22, seed=2)
B = nx.erdos_renyi_graph(45, 0.06, seed=3)
# relabel B nodes
B = nx.relabel_nodes(B, {n: n+1000 for n in B.nodes()})
G2 = nx.compose(A, B)

# Add explicit cross edges (bridges) between selected nodes
bridge_pairs = [(0, 1000), (5, 1005), (10, 1010)]
G2.add_edges_from(bridge_pairs)

pos2 = stable_layout(G2, seed=13)
edge_type2, cnr2, adj2, dens2 = classify_edges(G2, use_ld=False)
fig2_path = out_dir / "fig2_toy_network_structure.png"
draw_edge_types(G2, pos2, edge_type2, "Fig 2. Toy network (density gradient + bridging edges)", fig2_path)

# ---------- Experiment 3: CNR distribution vs LD-adjusted CNR distribution ----------
# Use G1 as reference network
alpha = 1.0
_, cnr_vals, adj_vals, _ = classify_edges(G1, alpha=alpha, use_ld=True)
cnr_arr = np.array(list(cnr1.values()))
adj_arr = np.array(list(adj_vals.values()))

plt.figure(figsize=(8, 5))
plt.title(f"Fig 3. CNR vs LD-adjusted CNR distributions (alpha={alpha})")
plt.xlabel("value")
plt.ylabel("count")
plt.hist(cnr_arr, bins=30, histtype="step", linewidth=1.8, label="CNR")
plt.hist(adj_arr, bins=30, histtype="step", linewidth=1.8, linestyle="dashed", label="Adjusted CNR")
plt.legend(frameon=True)
plt.tight_layout()
fig3_path = out_dir / "fig3_cnr_distribution_comparison.png"
plt.savefig(fig3_path, dpi=220)
plt.close()

# ---------- Experiment 4: Local density vs CNR scatter ----------
# Edge density proxy: average clustering at endpoints, from G1
dens_arr = []
cnr_arr2 = []
for (u, v) in G1.edges():
    c, _ = cnr_for_edge(G1, u, v)
    d = edge_local_density(G1, u, v)
    cnr_arr2.append(c)
    dens_arr.append(d)
dens_arr = np.array(dens_arr)
cnr_arr2 = np.array(cnr_arr2)

plt.figure(figsize=(7.5, 5.5))
plt.title("Fig 4. Local density proxy vs CNR (SBM)")
plt.xlabel("Local density proxy (avg clustering of endpoints)")
plt.ylabel("CNR")
plt.scatter(dens_arr, cnr_arr2, s=12, alpha=0.6)
plt.tight_layout()
fig4_path = out_dir / "fig4_local_density_vs_cnr.png"
plt.savefig(fig4_path, dpi=220)
plt.close()

# ---------- Experiment 5: Spatial distribution of four edge types (LD-HETA-like) ----------
alpha5 = 1.2
edge_type5, _, _, _ = classify_edges(G1, alpha=alpha5, use_ld=True)
fig5_path = out_dir / "fig5_edge_types_spatial_ldheta.png"
draw_edge_types(G1, pos1, edge_type5, f"Fig 5. Edge-type spatial map (LD-HETA-like, alpha={alpha5})", fig5_path)

# ---------- Experiment 6: Parameter sensitivity (alpha sweep) ----------
alphas = np.linspace(0.0, 2.0, 11)
types = ["bond", "silk", "local_bridge", "global_bridge"]
proportions = {t: [] for t in types}

for a in alphas:
    et, *_ = classify_edges(G1, alpha=float(a), use_ld=True)
    counts = {t: 0 for t in types}
    for t in et.values():
        counts[t] += 1
    m = max(len(et), 1)
    for t in types:
        proportions[t].append(counts[t] / m)

plt.figure(figsize=(8, 5.5))
plt.title("Fig 6. LD-HETA parameter sensitivity (alpha sweep)")
plt.xlabel("alpha")
plt.ylabel("Proportion of edges")
for t in types:
    plt.plot(alphas, proportions[t], label=t)
plt.legend(frameon=True)
plt.tight_layout()
fig6_path = out_dir / "fig6_parameter_sensitivity_alpha.png"
plt.savefig(fig6_path, dpi=220)
plt.close()

print(fig1_path, fig2_path, fig3_path, fig4_path, fig5_path, fig6_path)
