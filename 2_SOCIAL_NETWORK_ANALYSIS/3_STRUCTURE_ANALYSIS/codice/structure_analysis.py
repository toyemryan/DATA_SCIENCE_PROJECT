"""Visualizzazione e analisi delle strutture – GitHub Social Network (MUSAE)
Connettività e punti critici, k-core, clique, small-world e modelli nulli, rich-club,
community detection (Louvain, Leiden, Label Propagation, Infomap, Fast Greedy), visualizzazione della rete.

uv run --python 3.12 --with python-igraph --with leidenalg --with pandas --with pyarrow \\
        --with matplotlib --with scipy --with tabulate python structure_analysis.py
Opzioni: --recompute  ricalcola layout e partizioni anche se presenti in cache/
"""
import argparse
import random
import time
from collections import Counter
from pathlib import Path

import igraph as ig
import leidenalg as la
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

#########################
# CONFIGURAZIONE
#########################
ROOT = Path(__file__).resolve().parent.parent            # 3_STRUCTURE_ANALYSIS/
RAW = ROOT.parent / "data" / "raw"
CACHE = ROOT / "cache"
FIG_DIR = ROOT / "figures"
REPORT_MD = ROOT / "structure_results.md"
LAYOUT_NPY = CACHE / "layout_drl.npy"
PARTITIONS = CACHE / "partitions.parquet"
SEED = 42
N_NULL = 5                       # repliche per ciascun modello nullo
N_SOURCES = 300                  # sorgenti BFS per stimare la distanza media
TOP_COMM = 15                    # community descritte in dettaglio
LEIDEN_GAMMAS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

C_WEB, C_ML, C_GREY, C_INK = "#2a78d6", "#eb6834", "#898781", "#2b2a27"
C_LIGHT = "#c9c8c2"
CLASS_NAME = {0: "Web", 1: "ML"}
CLASS_COLOR = {0: C_WEB, 1: C_ML}
CAT8 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
# divergente blu (Web) – grigio – arancio (ML), centrato sulla quota ML del grafo
DIV_WEB_ML = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#e4e3df", "#f2a27f", "#c04a1c"])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "axes.edgecolor": C_GREY, "axes.labelcolor": C_INK,
    "xtick.color": "#52514e", "ytick.color": "#52514e", "font.size": 10,
})


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print(f"  figura {name}")


def fmt(x):
    if isinstance(x, (bool, np.bool_)):
        return "sì" if x else "no"
    if isinstance(x, (int, np.integer)) or (isinstance(x, (float, np.floating)) and float(x).is_integer()):
        return f"{int(x):,}".replace(",", " ")
    if isinstance(x, (float, np.floating)):
        return f"{x:.4g}"
    return str(x)


def md_table(df, index=False):
    return df.map(fmt).to_markdown(index=index, disable_numparse=True) + "\n"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    ig.set_random_number_generator(random)


#########################
# CARICAMENTO
#########################
def load_graph():
    edges = pd.read_csv(RAW / "musae_git_edges.csv")
    target = pd.read_csv(RAW / "musae_git_target.csv").sort_values("id").reset_index(drop=True)
    g = ig.Graph(n=len(target), edges=edges[["id_1", "id_2"]].values.tolist(), directed=False)
    g.simplify()
    g.vs["name"] = target["name"].tolist()
    g.vs["ml"] = target["ml_target"].tolist()
    return g


#########################
# FUNZIONI DI SUPPORTO
#########################
def mean_distance_sampled(h, sources):
    """Distanza media tra coppie connesse, stimata con BFS da un campione di sorgenti."""
    tot, cnt = 0.0, 0
    for i in range(0, len(sources), 50):
        d = np.array(h.distances(source=sources[i:i + 50]), dtype=float)
        d = d[np.isfinite(d) & (d > 0)]
        tot += d.sum(); cnt += d.size
    return tot / cnt


def rich_club(deg, edges, ks):
    """phi(k) = 2 E_k / (N_k (N_k - 1)) sui nodi di grado > k (Zhou & Mondragón)."""
    deg = np.asarray(deg)
    emin = np.minimum(deg[edges[:, 0]], deg[edges[:, 1]])
    emin_sorted = np.sort(emin)
    deg_sorted = np.sort(deg)
    out = []
    for k in ks:
        nk = len(deg) - np.searchsorted(deg_sorted, k, side="right")
        ek = len(emin) - np.searchsorted(emin_sorted, k, side="right")
        out.append(2 * ek / (nk * (nk - 1)) if nk > 1 else np.nan)
    return np.array(out)


def community_table(g, memb, top):
    """Statistiche per community: dimensione, quota ML, densità interna, conduttanza, hub."""
    memb = np.asarray(memb)
    el = np.array(g.get_edgelist())
    deg = np.array(g.degree())
    ml = np.array(g.vs["ml"])
    names = np.array(g.vs["name"])
    cu, cv = memb[el[:, 0]], memb[el[:, 1]]
    internal = Counter(cu[cu == cv])
    vol = pd.Series(deg).groupby(memb).sum()
    rows = []
    for c in pd.Series(memb).value_counts().index[:top]:
        nodes = np.where(memb == c)[0]
        n_c, e_in = len(nodes), internal.get(c, 0)
        cut = vol[c] - 2 * e_in
        hub = nodes[np.argmax(deg[nodes])]
        rows.append((c, n_c, n_c / g.vcount(), ml[nodes].mean(), e_in,
                     2 * e_in / (n_c * (n_c - 1)) if n_c > 1 else np.nan,
                     cut / min(vol[c], 2 * g.ecount() - vol[c]), names[hub], deg[hub]))
    return pd.DataFrame(rows, columns=["Community", "Nodi", "Quota nodi", "Quota ML", "Archi interni",
                                       "Densità interna", "Conduttanza", "Hub", "Grado hub"])


#########################
# MAIN
#########################
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    seed_all(SEED)
    CACHE.mkdir(parents=True, exist_ok=True)

    g = load_graph()
    n, m = g.vcount(), g.ecount()
    deg = np.array(g.degree())
    ml = np.array(g.vs["ml"])
    base_ml = ml.mean()
    el = np.array(g.get_edgelist())
    print(f"Grafo: {n} nodi, {m} archi")

    report = ["# Visualizzazione e analisi delle strutture – GitHub Social Network\n\n",
              f"Grafo non orientato: {fmt(n)} nodi, {fmt(m)} archi. "
              "Script: `codice/structure_analysis.py` · Figure: `figures/` · Cache: `cache/`.\n\n"]

    #########################
    # 1. CONNETTIVITÀ E PUNTI CRITICI
    #########################
    print("1. Connettività")
    comps = g.connected_components()
    aps = g.articulation_points()
    bridges = g.bridges()
    bic = g.biconnected_components()
    bic_sizes = sorted((len(c) for c in bic), reverse=True)
    leaves = int((deg == 1).sum())
    # quanti nodi si staccano rimuovendo ciascun punto di articolazione (top 10 per grado)
    ap_rows = []
    for v in sorted(aps, key=lambda v: -deg[v])[:10]:
        h = g.copy(); h.delete_vertices(v)
        sz = sorted(h.connected_components().sizes(), reverse=True)
        ap_rows.append((g.vs[v]["name"], CLASS_NAME[ml[v]], deg[v], len(sz) - 1, n - 1 - sz[0]))
    conn = pd.DataFrame([
        ("Componenti connesse", len(comps)),
        ("Punti di articolazione", len(aps)),
        ("Punti di articolazione (% nodi)", f"{len(aps) / n:.2%}"),
        ("Bridge (archi ponte)", len(bridges)),
        ("Nodi foglia (grado 1)", leaves),
        ("Bridge che non toccano una foglia", sum(1 for e in bridges if min(deg[list(g.es[e].tuple)]) > 1)),
        ("Componenti biconnesse", len(bic)),
        ("Nodi nella componente biconnessa più grande", bic_sizes[0]),
        ("Seconda componente biconnessa (nodi)", bic_sizes[1]),
    ], columns=["Misura", "Valore"])
    report += ["## 1. Connettività e punti critici\n\n", md_table(conn), "\n",
               "**Punti di articolazione con grado più alto** (nodi che, rimossi, disconnettono il grafo):\n\n",
               md_table(pd.DataFrame(ap_rows, columns=["Nome", "Classe", "Grado", "Nuove componenti",
                                                       "Nodi staccati"])), "\n"]

    #########################
    # 2. K-CORE DECOMPOSITION
    #########################
    print("2. k-core")
    core = np.array(g.coreness())
    kmax = core.max()
    ks = np.arange(1, kmax + 1)
    core_size = np.array([(core >= k).sum() for k in ks])
    shell = pd.DataFrame({"core": core, "ml": ml, "deg": deg})
    shell_tab = shell.groupby("core").agg(nodi=("ml", "size"), quota_ml=("ml", "mean"), grado_medio=("deg", "mean"))
    inner = g.induced_subgraph(np.where(core == kmax)[0].tolist())
    inner_ml = np.mean(inner.vs["ml"])
    ml_core = pd.Series(core[ml == 1]).describe()
    web_core = pd.Series(core[ml == 0]).describe()
    sel = [1, 2, 3, 5, 10, 15, 20, 25, 30]
    report += ["\n## 2. Decomposizione k-core\n\n",
               f"- Coreness massima (degenerazione del grafo): **k_max = {kmax}**; il {kmax}-core contiene "
               f"{inner.vcount()} nodi e {inner.ecount()} archi (densità {inner.density():.3f}), "
               f"quota ML = {inner_ml:.1%}.\n",
               f"- Coreness mediana: Web = {web_core['50%']:.0f}, ML = {ml_core['50%']:.0f}; "
               f"Spearman(coreness, grado) = {stats.spearmanr(core, deg)[0]:.3f}.\n\n",
               "Dimensione del k-core (nodi con coreness ≥ k) e composizione per classe:\n\n",
               md_table(pd.DataFrame({
                   "k": sel, "Nodi nel k-core": [core_size[k - 1] for k in sel],
                   "Quota del grafo": [core_size[k - 1] / n for k in sel],
                   "Quota ML nel k-core": [ml[core >= k].mean() for k in sel]})), "\n"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    w = shell_tab["nodi"] * (1 - shell_tab["quota_ml"])
    mlc = shell_tab["nodi"] * shell_tab["quota_ml"]
    ax.bar(shell_tab.index, w, color=C_WEB, width=0.8, label="Web")
    ax.bar(shell_tab.index, mlc, bottom=w, color=C_ML, width=0.8, label="ML", edgecolor="white", linewidth=0.5)
    ax.set_yscale("log"); ax.set_xlabel("Coreness (k-shell)"); ax.set_ylabel("Nodi nella shell")
    ax.set_title("Dimensione delle k-shell", loc="left"); ax.legend(frameon=False); ax.grid(axis="x", visible=False)
    ax = axes[1]
    ax.plot(ks, [ml[core >= k].mean() for k in ks], color=C_ML, linewidth=2, label="Quota ML nel k-core")
    ax.axhline(base_ml, color=C_GREY, linestyle="--", linewidth=1)
    ax.text(1, base_ml + 0.008, f"quota ML nel grafo = {base_ml:.1%}", fontsize=8, color="#52514e")
    ax.set_xlabel("k"); ax.set_ylabel("Quota ML"); ax.set_ylim(0, base_ml + 0.08)
    ax.set_title("Composizione del k-core al crescere di k", loc="left")
    fig.tight_layout()
    save(fig, "01_kcore.png")

    #########################
    # 3. CLIQUE
    #########################
    print("3. Clique")
    t0 = time.time()
    cliques = g.maximal_cliques(min=3)
    clique_num = max(map(len, cliques))
    csize = Counter(map(len, cliques))
    comp_rows = []
    for size in (3, 5, 10, 15):
        cl = [c for c in cliques if len(c) >= size]
        frac_ml = np.array([ml[list(c)].mean() for c in cl])
        comp_rows.append((f"≥ {size}", len(cl), (frac_ml == 0).mean(), (frac_ml == 1).mean(),
                          ((frac_ml > 0) & (frac_ml < 1)).mean()))
    # probabilità attesa di clique "pure" se le etichette fossero assegnate a caso (stessa dimensione)
    maxcl = [c for c in cliques if len(c) == clique_num]
    maxcl_union = sorted(set().union(*map(set, maxcl)))
    maxcl_common = set.intersection(*map(set, maxcl))
    report += ["\n## 3. Clique\n\n",
               f"- Clique massimali con almeno 3 nodi: **{fmt(len(cliques))}** (calcolo in {time.time() - t0:.1f}s).\n",
               f"- Numero di clique ω(G) = **{clique_num}** ({len(maxcl)} clique di dimensione massima).\n",
               f"- Le {len(maxcl)} clique massime coinvolgono {len(maxcl_union)} nodi distinti e ne condividono "
               f"{len(maxcl_common)}: sono varianti dello stesso gruppo densissimo (quota ML = "
               f"{ml[maxcl_union].mean():.1%}; es. {', '.join(sorted(g.vs[maxcl_union[:5]]['name'], key=str.lower))}).\n\n",
               "Composizione per classe delle clique massimali (quota di clique interamente Web, interamente ML, miste):\n\n",
               md_table(pd.DataFrame(comp_rows, columns=["Dimensione", "Clique", "Solo Web", "Solo ML", "Miste"])), "\n"]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    sz = sorted(csize)
    ax.bar(sz, [csize[s] for s in sz], color=C_WEB, width=0.75)
    ax.set_yscale("log"); ax.set_xlabel("Dimensione della clique massimale"); ax.set_ylabel("Numero di clique")
    ax.set_title(f"Distribuzione delle clique massimali (ω = {clique_num})", loc="left")
    ax.grid(axis="x", visible=False)
    save(fig, "02_clique.png")

    #########################
    # 4. SMALL-WORLD E MODELLI NULLI
    #########################
    print("4. Small-world e modelli nulli")
    rng = np.random.default_rng(SEED)
    sources = rng.choice(n, N_SOURCES, replace=False).tolist()

    def props(h):
        return (h.transitivity_undirected(), h.transitivity_avglocal_undirected(mode="zero"),
                mean_distance_sampled(h, sources), h.assortativity_degree(directed=False),
                max(h.connected_components().sizes()) / h.vcount())

    real = props(g)
    null = {"Erdős–Rényi G(n,m)": [], "Configuration model (grado preservato)": []}
    for r in range(N_NULL):
        null["Erdős–Rényi G(n,m)"].append(props(ig.Graph.Erdos_Renyi(n=n, m=m)))
        null["Configuration model (grado preservato)"].append(props(ig.Graph.Degree_Sequence(deg.tolist(), method="vl")))
    cols_sw = ["Transitività", "Clustering medio", "Distanza media", "Assortatività grado", "Quota comp. gigante"]
    sw_rows = [("GitHub (reale)",) + real]
    for k, v in null.items():
        v = np.array(v)
        sw_rows.append((k,) + tuple(f"{mu:.4g} ± {sd:.1g}" for mu, sd in zip(v.mean(0), v.std(0))))
    er = np.array(null["Erdős–Rényi G(n,m)"]).mean(0)
    cm = np.array(null["Configuration model (grado preservato)"]).mean(0)
    sigma_er = (real[1] / er[1]) / (real[2] / er[2])
    sigma_cm = (real[1] / cm[1]) / (real[2] / cm[2])

    # clustering gerarchico C(k) ~ k^-beta
    cc_local = np.array(g.transitivity_local_undirected(mode="zero"))
    bins = np.unique(np.logspace(np.log10(5), np.log10(deg.max() + 1), 20).astype(int))
    dfc = pd.DataFrame({"k": deg, "c": cc_local})
    dfc = dfc[dfc["k"] >= 5]
    dfc["bin"] = pd.cut(dfc["k"], bins, right=False)
    ck = dfc.groupby("bin", observed=True).agg(k=("k", "mean"), c=("c", "mean"))
    ck = ck[ck["c"] > 0]
    slope, intercept, r_val, _, _ = stats.linregress(np.log10(ck["k"]), np.log10(ck["c"]))

    report += ["\n## 4. Small-world e confronto con modelli nulli\n\n",
               f"Medie su {N_NULL} repliche (± dev. std). Distanza media stimata con BFS da {N_SOURCES} sorgenti "
               "(le stesse per tutti i grafi); per ER solo tra coppie connesse. Configuration model generato "
               "con l'algoritmo di Viger–Latapy (grafo semplice e connesso, stessa sequenza dei gradi).\n\n",
               md_table(pd.DataFrame(sw_rows, columns=["Grafo"] + cols_sw)), "\n",
               f"- Coefficiente small-world σ = (C/C_r)/(L/L_r): rispetto a ER = **{sigma_er:.1f}**, rispetto al "
               f"configuration model = **{sigma_cm:.2f}** (σ > 1 → small-world).\n",
               f"- Il clustering medio reale è {real[1] / er[1]:.0f}× quello di ER e {real[1] / cm[1]:.1f}× quello "
               "del configuration model: il clustering **non** è spiegato dalla sola distribuzione dei gradi.\n",
               f"- Clustering gerarchico: C(k) ∝ k^(−β) con β = {-slope:.2f} (R² = {r_val**2:.2f}, k ≥ 5); "
               "β ≈ 1 è la firma dei modelli gerarchici (Ravasz–Barabási).\n"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    labels = ["Clustering medio", "Transitività", "Distanza media"]
    idx = [1, 0, 2]
    x = np.arange(len(labels)); w = 0.26
    for j, (lab, vals, col) in enumerate([("GitHub", real, C_INK), ("Erdős–Rényi", er, C_GREY),
                                          ("Configuration model", cm, C_WEB)]):
        vv = [vals[i] / real[i] for i in idx]
        ax.bar(x + (j - 1) * w, vv, w, color=col, label=lab)
    ax.set_yscale("log"); ax.set_xticks(x, labels); ax.set_ylabel("Valore / valore reale (log)")
    ax.axhline(1, color=C_INK, linewidth=0.8)
    ax.set_title("Grafo reale vs modelli nulli", loc="left", pad=22)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    ax.scatter(dfc["k"], dfc["c"].clip(lower=1e-4), s=3, color=C_LIGHT, alpha=0.3, linewidths=0, rasterized=True)
    ax.plot(ck["k"], ck["c"], "o", color=C_WEB, markersize=5, label="Media per classe di grado")
    kk = np.logspace(np.log10(ck["k"].min()), np.log10(ck["k"].max()), 50)
    ax.plot(kk, 10 ** intercept * kk ** slope, color=C_ML, linewidth=2, label=f"Fit C(k) ∝ k^−{-slope:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-4, 1.2)
    ax.set_xlabel("Grado k"); ax.set_ylabel("C(k)")
    ax.set_title("Clustering gerarchico", loc="left"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "03_small_world.png")

    #########################
    # 5. RICH-CLUB
    #########################
    print("5. Rich-club")
    rk = np.unique(np.logspace(0, np.log10(np.sort(deg)[-20]), 40).astype(int))
    phi = rich_club(deg, el, rk)
    phi_rand = []
    for r in range(N_NULL):
        h = g.copy()
        h.rewire(n=10 * m)        # rewiring che preserva i gradi
        phi_rand.append(rich_club(deg, np.array(h.get_edgelist()), rk))
    phi_rand = np.array(phi_rand).mean(0)
    rho = phi / phi_rand
    rc = pd.DataFrame({"k": rk, "Nodi con grado > k": [(deg > k).sum() for k in rk], "φ(k)": phi,
                       "φ_rand(k)": phi_rand, "ρ(k) = φ/φ_rand": rho})
    report += ["\n## 5. Rich-club\n\n",
               "φ(k) = densità del sottografo dei nodi con grado > k; ρ(k) normalizzato rispetto a "
               f"{N_NULL} grafi ricablati che preservano i gradi (ρ > 1 → gli hub si connettono tra loro più del caso).\n\n",
               md_table(rc.iloc[::4]), "\n",
               f"- ρ massimo = {np.nanmax(rho):.2f} a k = {rk[np.nanargmax(rho)]}; ρ minimo = {np.nanmin(rho):.2f} "
               f"a k = {rk[np.nanargmin(rho)]}.\n"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    ax.plot(rk, phi, color=C_INK, linewidth=2, label="GitHub")
    ax.plot(rk, phi_rand, color=C_GREY, linewidth=2, linestyle="--", label="Ricablato (gradi preservati)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("k"); ax.set_ylabel("φ(k)")
    ax.set_title("Coefficiente rich-club", loc="left"); ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    ax.plot(rk, rho, color=C_WEB, linewidth=2)
    ax.axhline(1, color=C_INK, linewidth=0.8, linestyle="--")
    ax.set_xscale("log"); ax.set_xlabel("k"); ax.set_ylabel("ρ(k)")
    ax.set_title("Rich-club normalizzato", loc="left")
    fig.tight_layout()
    save(fig, "04_rich_club.png")

    #########################
    # 6. COMMUNITY DETECTION
    #########################
    print("6. Community detection")
    if PARTITIONS.exists() and not args.recompute:
        part = pd.read_parquet(PARTITIONS)
        timing = part.attrs.get("timing", {})
    else:
        part, timing = pd.DataFrame(index=range(n)), {}
        algos = {
            "Louvain": lambda: g.community_multilevel().membership,
            "Leiden": lambda: la.find_partition(g, la.ModularityVertexPartition, seed=SEED).membership,
            "Label Propagation": lambda: g.community_label_propagation().membership,
            "Infomap": lambda: g.community_infomap(trials=5).membership,
            "Fast Greedy (CNM)": lambda: g.community_fastgreedy().as_clustering().membership,
        }
        for name, f in algos.items():
            seed_all(SEED)
            t0 = time.time()
            part[name] = f()
            timing[name] = time.time() - t0
            print(f"  {name}: {part[name].nunique()} community in {timing[name]:.1f}s")
        part.attrs["timing"] = timing
        part.to_parquet(PARTITIONS)
    algos = list(part.columns)

    rows = []
    for a in algos:
        memb = part[a].tolist()
        sizes = pd.Series(memb).value_counts()
        rows.append((a, len(sizes), g.modularity(memb), sizes.iloc[0] / n, (sizes >= 10).sum(),
                     ig.compare_communities(memb, ml.tolist(), method="nmi"),
                     ig.compare_communities(memb, ml.tolist(), method="adjusted_rand"),
                     timing.get(a, np.nan)))
    algo_tab = pd.DataFrame(rows, columns=["Algoritmo", "Community", "Modularità Q", "Quota community max",
                                           "Community ≥ 10 nodi", "NMI vs classe", "ARI vs classe", "Tempo (s)"])
    nmi = pd.DataFrame([[ig.compare_communities(part[a].tolist(), part[b].tolist(), method="nmi") for b in algos]
                        for a in algos], index=algos, columns=algos)

    # sweep della risoluzione di Leiden (modularità generalizzata, Reichardt–Bornholdt)
    sweep = []
    for gam in LEIDEN_GAMMAS:
        p = la.find_partition(g, la.RBConfigurationVertexPartition, resolution_parameter=gam, seed=SEED)
        sizes = pd.Series(p.membership).value_counts()
        sweep.append((gam, len(sizes), (sizes >= 10).sum(), g.modularity(p.membership), sizes.iloc[0] / n,
                      ig.compare_communities(p.membership, ml.tolist(), method="nmi")))
    sweep = pd.DataFrame(sweep, columns=["γ", "Community", "Community ≥ 10 nodi", "Modularità Q (γ=1)",
                                         "Quota community max", "NMI vs classe"])

    report += ["\n## 6. Community detection\n\n",
               "### 6.1 Confronto tra algoritmi\n\n", md_table(algo_tab), "\n",
               "NMI tra le partizioni dei diversi algoritmi:\n\n", md_table(nmi, index=True), "\n",
               "### 6.2 Leiden: effetto della risoluzione γ\n\n",
               "Funzione di qualità RB-configuration: γ < 1 favorisce poche community grandi, γ > 1 molte piccole. "
               "La modularità è sempre riportata con γ = 1 per confrontabilità.\n\n", md_table(sweep), "\n"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    y = np.arange(len(algos))
    ax.barh(y, algo_tab["Modularità Q"], color=C_WEB, height=0.55)
    for yi, (q, nc) in enumerate(zip(algo_tab["Modularità Q"], algo_tab["Community"])):
        ax.text(q + 0.005, yi, f"Q = {q:.3f}  ({nc} community)", va="center", fontsize=8)
    ax.set_yticks(y, algos); ax.invert_yaxis(); ax.set_xlim(0, 0.65)
    ax.set_xlabel("Modularità Q"); ax.grid(axis="y", visible=False)
    ax.set_title("Modularità per algoritmo", loc="left")
    ax = axes[1]
    im = ax.imshow(nmi.values, cmap=SEQ_BLUE, vmin=0, vmax=1)
    short = [a.split(" (")[0] for a in algos]
    ax.set_xticks(y, short, rotation=30, ha="right"); ax.set_yticks(y, short); ax.grid(False)
    for i in range(len(algos)):
        for j in range(len(algos)):
            v = nmi.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else C_INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("NMI tra partizioni", loc="left")
    fig.tight_layout()
    save(fig, "05_community_algoritmi.png")

    # dettaglio sulla partizione Leiden
    memb = part["Leiden"].values
    ctab = community_table(g, memb, TOP_COMM)
    sizes = pd.Series(memb).value_counts()
    ml_share_all = pd.Series(ml).groupby(memb).mean()
    # entropia media pesata della classe nelle community vs entropia globale
    H = lambda p: -(p * np.log2(p) + (1 - p) * np.log2(1 - p)) if 0 < p < 1 else 0.0
    h_cond = sum(sizes[c] / n * H(ml_share_all[c]) for c in sizes.index)
    report += [f"### 6.3 Partizione Leiden: le {TOP_COMM} community più grandi\n\n",
               "Conduttanza = archi uscenti / min(volume interno, volume esterno): più è bassa, più la community "
               "è separata dal resto della rete.\n\n",
               md_table(ctab.drop(columns="Community").rename_axis("Rank").reset_index().assign(Rank=lambda d: d["Rank"] + 1)),
               "\n",
               f"- Le prime {TOP_COMM} community coprono il {ctab['Quota nodi'].sum():.1%} dei nodi.\n",
               f"- Entropia della classe: globale = {H(base_ml):.3f} bit, condizionata alla community = "
               f"{h_cond:.3f} bit → la community spiega il {1 - h_cond / H(base_ml):.1%} dell'incertezza "
               "sulla classe (informazione mutua normalizzata sulla classe).\n",
               f"- Community con quota ML > 50%: {(ml_share_all[sizes.index[sizes >= 10]] > 0.5).sum()} "
               f"(tra quelle con ≥ 10 nodi), che contengono il "
               f"{ml[np.isin(memb, [c for c in sizes.index[sizes >= 10] if ml_share_all[c] > 0.5])].sum() / ml.sum():.1%} "
               "degli sviluppatori ML.\n"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    t = ctab.iloc[::-1]
    lbl = [f"#{i}  {h}" for i, h in zip(range(len(t), 0, -1), t["Hub"])]
    w_ = t["Nodi"] * (1 - t["Quota ML"]); m_ = t["Nodi"] * t["Quota ML"]
    ax.barh(lbl, w_, color=C_WEB, height=0.65, label="Web")
    ax.barh(lbl, m_, left=w_, color=C_ML, height=0.65, label="ML", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Nodi"); ax.tick_params(axis="y", labelsize=8); ax.grid(axis="y", visible=False)
    ax.set_title(f"Top {TOP_COMM} community Leiden (etichetta = hub)", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax = axes[1]
    ax.loglog(np.arange(1, len(sizes) + 1), sizes.values, "o-", color=C_WEB, markersize=3, linewidth=1.5)
    ax.set_xlabel("Rank della community"); ax.set_ylabel("Nodi")
    ax.set_title("Dimensione vs rank (Leiden)", loc="left")
    fig.tight_layout()
    save(fig, "06_community_leiden.png")

    # grafo delle community (quotient graph)
    top_ids = ctab["Community"].tolist()
    cu, cv = memb[el[:, 0]], memb[el[:, 1]]
    mask = np.isin(cu, top_ids) & np.isin(cv, top_ids) & (cu != cv)
    inter = Counter(map(tuple, np.sort(np.c_[cu[mask], cv[mask]], axis=1)))
    pos_of = {c: i for i, c in enumerate(top_ids)}
    q = ig.Graph(n=len(top_ids), edges=[(pos_of[a], pos_of[b]) for a, b in inter])
    q.es["weight"] = list(inter.values())
    lay = np.array(q.layout_fruchterman_reingold(weights="weight", niter=3000))
    fig, ax = plt.subplots(figsize=(8, 7))
    wmax = max(inter.values())
    segs = [lay[[pos_of[a], pos_of[b]]] for a, b in inter]
    ax.add_collection(LineCollection(segs, linewidths=[0.3 + 6 * (v / wmax) for v in inter.values()],
                                     colors=C_LIGHT, alpha=0.8, zorder=1))
    shares = ctab["Quota ML"].values
    norm = matplotlib.colors.TwoSlopeNorm(vmin=0, vcenter=base_ml, vmax=1)
    sc = ax.scatter(lay[:, 0], lay[:, 1], s=40 + 2500 * ctab["Nodi"] / ctab["Nodi"].max(), c=shares,
                    cmap=DIV_WEB_ML, norm=norm, edgecolors=C_INK, linewidths=0.6, zorder=2)
    for i in range(len(top_ids)):
        ax.annotate(f"{i + 1}", lay[i], fontsize=8, ha="center", va="center", zorder=3, color=C_INK,
                    fontweight="bold")
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(f"Quota ML (centro = {base_ml:.0%}, quota nel grafo)")
    ax.set_axis_off(); ax.margins(0.12)
    ax.set_title(f"Grafo delle top {TOP_COMM} community Leiden (numero = rank, vedi tabella 6.3)\n"
                 "Dimensione ∝ nodi, spessore ∝ archi tra community",
                 loc="left", fontsize=10)
    save(fig, "07_grafo_community.png")

    #########################
    # 7. VISUALIZZAZIONE DELL'INTERA RETE
    #########################
    print("7. Visualizzazione della rete (layout DrL)")
    if LAYOUT_NPY.exists() and not args.recompute:
        lay = np.load(LAYOUT_NPY)
    else:
        seed_all(SEED)
        t0 = time.time()
        lay = np.array(g.layout_drl())
        print(f"  layout DrL in {time.time() - t0:.0f}s")
        np.save(LAYOUT_NPY, lay)
    sub_e = el[rng.choice(m, 60000, replace=False)]
    order = np.argsort(deg)                   # hub disegnati sopra
    size = 0.3 + 40 * (deg[order] / deg.max()) ** 0.5
    top8 = sizes.index[:8].tolist()
    ccol = np.array([CAT8[top8.index(c)] if c in top8 else C_LIGHT for c in memb])
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, colors, title in [(axes[0], np.where(ml == 1, C_ML, C_WEB), "Colore = classe"),
                              (axes[1], ccol, "Colore = community Leiden (prime 8)")]:
        ax.add_collection(LineCollection(lay[sub_e], colors=C_LIGHT, linewidths=0.1, alpha=0.15,
                                         zorder=1, rasterized=True))
        ax.scatter(lay[order, 0], lay[order, 1], s=size, c=colors[order], linewidths=0, alpha=0.7,
                   zorder=2, rasterized=True)
        ax.set_aspect("equal"); ax.set_axis_off(); ax.autoscale()
        ax.set_title(title, loc="left")
    axes[0].legend([plt.Line2D([], [], marker="o", ls="", color=c) for c in (C_WEB, C_ML)], ["Web", "ML"],
                   frameon=False, loc="lower left")
    axes[1].legend([plt.Line2D([], [], marker="o", ls="", color=CAT8[i]) for i in range(8)] +
                   [plt.Line2D([], [], marker="o", ls="", color=C_LIGHT)],
                   [f"#{i + 1} ({sizes.iloc[i]} nodi)" for i in range(8)] + ["Altre"],
                   frameon=False, loc="lower left", fontsize=7, ncol=3)
    fig.suptitle(f"Rete GitHub completa – layout DrL ({fmt(n)} nodi, campione di 60 000 archi; "
                 "dimensione ∝ √grado)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "08_rete_completa.png")

    #########################
    # 8. SINTESI
    #########################
    best = algo_tab.loc[algo_tab["Modularità Q"].idxmax()]
    lp = algo_tab.set_index("Algoritmo").loc["Label Propagation"]
    report += ["\n## 7. Sintesi\n\n",
               f"- **Connettività**: grafo connesso ma con {len(aps) / n:.1%} di punti di articolazione e "
               f"{fmt(len(bridges))} bridge, quasi tutti verso nodi foglia ({fmt(leaves)} nodi di grado 1): periferia "
               "ad albero appesa a un nucleo biconnesso di "
               f"{bic_sizes[0] / n:.1%} dei nodi.\n",
               f"- **Nucleo**: degenerazione k_max = {kmax}; il nucleo è quasi solo Web "
               f"(quota ML {inner_ml:.1%} vs {base_ml:.1%}), coerente con la centralità.\n",
               f"- **Small-world**: distanza media {real[2]:.2f} (persino minore di ER, {er[2]:.2f}) con clustering {real[1] / er[1]:.0f}× "
               f"ER → σ = {sigma_er:.0f}; il clustering resta {real[1] / cm[1]:.1f}× quello del configuration model, "
               f"quindi c'è chiusura triadica oltre l'effetto dei gradi. C(k) ∝ k^−{-slope:.2f}: clustering decrescente con il grado, "
               "segno di organizzazione gerarchica (più debole del β = 1 teorico).\n",
               f"- **Rich-club**: ρ(k) {'< 1 per k alti: gli hub NON formano un club, coerente con la disassortatività' if rho[-1] < 1 else '> 1 per k alti: gli hub tendono a connettersi tra loro'} "
               f"(ρ = {rho[-1]:.2f} al k più alto).\n",
               f"- **Community**: {best['Algoritmo']} ottiene la modularità più alta (Q = {best['Modularità Q']:.3f}, "
               f"{int(best['Community'])} community); Label Propagation degenera ({int(lp['Community'])} community, "
               f"Q = {lp['Modularità Q']:.3f}) perché gli hub propagano la stessa etichetta a quasi tutta la rete. "
               f"Le community catturano parte della classe (NMI Leiden vs classe = "
               f"{algo_tab.set_index('Algoritmo').loc['Leiden', 'NMI vs classe']:.3f}).\n"]

    REPORT_MD.write_text("".join(report), encoding="utf-8")
    print(f"Report scritto in {REPORT_MD}")


if __name__ == "__main__":
    main()
