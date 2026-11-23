"""Analisi delle misure di centralità – GitHub Social Network (MUSAE)
Nodi = sviluppatori GitHub, archi = mutual follow (grafo non orientato, connesso, 37.700 nodi, 289.003 archi).
Misure: grado, closeness, betweenness (esatte), eigenvector, PageRank, Katz.

uv run --python 3.12 --with python-igraph --with pandas --with pyarrow \\
        --with matplotlib --with scipy --with tabulate python centrality_analysis.py
Opzioni: --recompute  ricalcola le centralità anche se presenti in cache/
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import igraph as ig
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy import sparse, stats
from scipy.sparse.linalg import cg, eigsh

#########################
# CONFIGURAZIONE
#########################
ROOT = Path(__file__).resolve().parent.parent            # 2_CENTRALITY_ANALYSIS/
RAW = ROOT.parent / "data" / "raw"
CACHE = ROOT / "cache"
FIG_DIR = ROOT / "figures"
CENTR_PARQUET = CACHE / "centralities.parquet"
REPORT_MD = ROOT / "centrality_results.md"
SEED = 42
N_JOBS = max(1, (os.cpu_count() or 2) - 2)
TOP_K = 100                       # dimensione dei top-set per overlap e quota ML
PAGERANK_D = 0.85

MEASURES = {                      # colonna -> etichetta
    "degree": "Grado",
    "closeness": "Closeness",
    "betweenness": "Betweenness",
    "eigenvector": "Eigenvector",
    "pagerank": "PageRank",
    "katz": "Katz",
}

C_WEB, C_ML, C_GREY, C_INK = "#2a78d6", "#eb6834", "#898781", "#2b2a27"
CLASS_NAME = {0: "Web", 1: "ML"}
CLASS_COLOR = {0: C_WEB, 1: C_ML}
MEAS_COLOR = dict(zip(MEASURES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]))
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])

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
    df = df.copy()
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)
    return df.map(fmt).to_markdown(index=index, disable_numparse=True) + "\n"


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum())


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
    g.vs["orig_id"] = list(range(g.vcount()))   # induced_subgraph rinumera i vertici
    return g, target


#########################
# CALCOLO CENTRALITÀ (parallelo sulle sorgenti per le misure basate su cammini minimi)
#########################
_G = None


def _init_worker(n, edge_list):
    global _G
    _G = ig.Graph(n=n, edges=edge_list, directed=False)


def _betweenness_chunk(sources):
    # betweenness "subset": somma su una partizione delle sorgenti = betweenness esatta
    return np.array(_G.betweenness(sources=sources))


def _closeness_chunk(vertices):
    return np.array(_G.closeness(vertices=vertices))


def katz_params(g):
    """lambda_max di A e alpha = 0.85/lambda_max (Katz converge per alpha < 1/lambda_max)."""
    A = g.get_adjacency_sparse().astype(float)
    lam = float(eigsh(A, k=1, which="LA")[0][0])
    return lam, 0.85 / lam


def katz_centrality(g, alpha, beta=1.0):
    """Katz su matrice sparsa: risolve (I - alpha A) x = beta 1, normalizzato in norma L2."""
    A = g.get_adjacency_sparse().astype(float)
    # gradiente coniugato: I - alpha A è simmetrica definita positiva per alpha < 1/lambda_max
    x, info = cg(sparse.identity(A.shape[0], format="csr") - alpha * A, np.full(A.shape[0], beta), rtol=1e-10)
    assert info == 0, "Katz: gradiente coniugato non convergente"
    return x / np.linalg.norm(x)


def compute_centralities(g):
    n = g.vcount()
    chunks = [list(range(i, n, N_JOBS * 4)) for i in range(N_JOBS * 4)]
    with ProcessPoolExecutor(N_JOBS, initializer=_init_worker, initargs=(n, g.get_edgelist())) as ex:
        print(f"  betweenness esatta su {N_JOBS} processi...")
        btw = sum(ex.map(_betweenness_chunk, chunks))
        print("  closeness esatta...")
        clo = np.empty(n)
        for idx, vals in zip(chunks, ex.map(_closeness_chunk, chunks)):
            clo[idx] = vals
    deg = np.array(g.degree())
    print("  eigenvector, PageRank, Katz...")
    eig = np.array(g.eigenvector_centrality(scale=True))
    pr = np.array(g.pagerank(damping=PAGERANK_D))
    lam, alpha = katz_params(g)
    katz = katz_centrality(g, alpha)
    norm_btw = btw / ((n - 1) * (n - 2) / 2)
    df = pd.DataFrame({
        "id": range(n), "name": g.vs["name"], "ml": g.vs["ml"],
        "degree": deg / (n - 1), "degree_raw": deg,
        "closeness": clo, "betweenness": norm_btw, "eigenvector": eig, "pagerank": pr, "katz": katz,
    })
    return df


#########################
# MAIN
#########################
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    g, target = load_graph()
    n, m = g.vcount(), g.ecount()
    print(f"Grafo: {n} nodi, {m} archi")

    CACHE.mkdir(parents=True, exist_ok=True)
    if CENTR_PARQUET.exists() and not args.recompute:
        print("Centralità lette da cache")
        df = pd.read_parquet(CENTR_PARQUET)
    else:
        print("Calcolo centralità...")
        df = compute_centralities(g)
        df.to_parquet(CENTR_PARQUET)
    lam, alpha = katz_params(g)
    cols = list(MEASURES)
    N_ML = int(df["ml"].sum())
    base_ml = N_ML / n

    report = ["# Analisi delle misure di centralità – GitHub Social Network\n\n",
              f"Grafo non orientato e connesso: {fmt(n)} nodi, {fmt(m)} archi. "
              "Script: `codice/centrality_analysis.py` · Figure: `figures/` · Cache: `cache/centralities.parquet`.\n\n",
              "## 0. Misure calcolate\n\n",
              "| Misura | Definizione | Note |\n|:--|:--|:--|\n",
              "| Grado | $k_i/(n-1)$ | locale |\n",
              "| Closeness | $(n-1)/\\sum_j d(i,j)$ | esatta (grafo connesso) |\n",
              "| Betweenness | $\\sum_{s\\neq t\\neq i} \\sigma_{st}(i)/\\sigma_{st}$, normalizzata per $(n-1)(n-2)/2$ | esatta (Brandes), parallelizzata sulle sorgenti |\n",
              "| Eigenvector | autovettore dominante di $A$, scalato a max = 1 | |\n",
              f"| PageRank | damping $d = {PAGERANK_D}$ | |\n",
              f"| Katz | $x = \\alpha A x + \\beta$, $\\alpha = 0.85/\\lambda_{{max}} = {alpha:.3g}$ ($\\lambda_{{max}} = {lam:.2f}$), norm. L2 | |\n\n"]

    #########################
    # 1. STATISTICHE E CENTRALIZZAZIONE
    #########################
    print("1. Statistiche")
    rows = []
    for c in cols:
        x = df[c].values
        rows.append((MEASURES[c], x.min(), np.median(x), x.mean(), x.max(), x.max() / np.median(x), gini(x),
                     stats.skew(x)))
    st = pd.DataFrame(rows, columns=["Misura", "Min", "Mediana", "Media", "Max", "Max/Mediana", "Gini", "Skewness"])

    # Centralizzazione di Freeman: sum(max - c_i) / max teorico (grafo a stella)
    dmax = df["degree_raw"].max()
    cz_deg = (dmax - df["degree_raw"]).sum() / ((n - 1) * (n - 2))
    cz_btw = (df["betweenness"].max() - df["betweenness"]).sum() / (n - 1)
    cz_clo = (df["closeness"].max() - df["closeness"]).sum() / ((n - 1) * (n - 2) / (2 * n - 3))
    cz = pd.DataFrame([("Grado", cz_deg), ("Betweenness", cz_btw), ("Closeness", cz_clo)],
                      columns=["Misura", "Centralizzazione di Freeman"])
    report += ["## 1. Statistiche delle distribuzioni\n\n", md_table(st), "\n",
               "Il coefficiente di Gini misura la disuguaglianza della centralità tra i nodi (0 = tutti uguali, "
               "1 = tutto su un nodo).\n\n",
               "**Centralizzazione di Freeman** (1 = grafo a stella, 0 = tutti i nodi equivalenti):\n\n",
               md_table(cz), "\n"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, c in zip(axes.flat, cols):
        x = np.sort(df[c].values)
        y = 1 - np.arange(len(x)) / len(x)
        for cl in (0, 1):
            xc = np.sort(df.loc[df["ml"] == cl, c].values)
            ax.plot(xc, 1 - np.arange(len(xc)) / len(xc), color=CLASS_COLOR[cl], linewidth=1.6, label=CLASS_NAME[cl])
        ax.plot(x, y, color=C_INK, linewidth=1, linestyle="--", label="Tutti")
        ax.set_xscale("linear" if c == "closeness" else "log"); ax.set_yscale("log")
        ax.set_title(MEASURES[c], loc="left")
        ax.set_ylabel("P(X ≥ x)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("CCDF delle misure di centralità, per classe", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "01_ccdf_centralita.png")

    #########################
    # 2. TOP NODI
    #########################
    print("2. Top nodi")
    report += ["\n## 2. Nodi più centrali\n\n"]
    rank = df[cols].rank(ascending=False, method="min").astype(int)
    for c in cols:
        t = df.nlargest(10, c)[["id", "name", "ml", "degree_raw", c]].copy()
        t["ml"] = t["ml"].map(CLASS_NAME)
        for o in cols:
            if o != c:
                t[f"rank {MEASURES[o]}"] = rank.loc[t.index, o].values
        t = t.rename(columns={"ml": "classe", "degree_raw": "n. vicini", c: MEASURES[c]})
        report += [f"### Top 10 – {MEASURES[c]}\n\n", md_table(t), "\n"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, c in zip(axes.flat, cols):
        t = df.nlargest(10, c).iloc[::-1]
        ax.barh(t["name"], t[c], color=[CLASS_COLOR[v] for v in t["ml"]], height=0.65)
        ax.set_title(MEASURES[c], loc="left"); ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR[k]) for k in (0, 1)]
    fig.legend(handles, ["Web", "ML"], loc="upper right", ncol=2, frameon=False)
    fig.suptitle("Top 10 sviluppatori per misura di centralità", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "02_top10_per_misura.png")

    # Top ML developer (ranking tra i soli nodi ML)
    ml_rows = []
    for c in cols:
        t = df[df["ml"] == 1].nlargest(5, c)
        for pos, (_, r) in enumerate(t.iterrows(), 1):
            ml_rows.append((MEASURES[c], pos, r["name"], int(r["degree_raw"]), int(rank.loc[r.name, c])))
    top_ml = pd.DataFrame(ml_rows, columns=["Misura", "Pos. tra ML", "Nome", "Grado", "Rank globale"])
    report += ["### Sviluppatori ML più centrali (top 5 tra i nodi ML)\n\n", md_table(top_ml), "\n"]

    #########################
    # 3. CORRELAZIONI E OVERLAP
    #########################
    print("3. Correlazioni")
    sp = df[cols].corr(method="spearman")
    kt = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            kt.loc[a, b] = 1.0 if a == b else stats.kendalltau(df[a], df[b])[0] if a < b else np.nan
    kt = kt.combine_first(kt.T)
    tops = {c: set(df.nlargest(TOP_K, c)["id"]) for c in cols}
    jac = pd.DataFrame([[len(tops[a] & tops[b]) / len(tops[a] | tops[b]) for b in cols] for a in cols],
                       index=cols, columns=cols)
    lab = [MEASURES[c] for c in cols]
    report += ["\n## 3. Correlazione tra misure\n\n",
               "**Spearman** (sull'intero grafo):\n\n", md_table(sp.set_axis(lab).set_axis(lab, axis=1), index=True),
               "\n**Kendall τ**:\n\n", md_table(kt.set_axis(lab).set_axis(lab, axis=1), index=True),
               f"\n**Jaccard tra i top-{TOP_K}** (sovrapposizione dei nodi più centrali):\n\n",
               md_table(jac.set_axis(lab).set_axis(lab, axis=1), index=True), "\n"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, mat, title in [(axes[0], sp, "Correlazione di Spearman"),
                           (axes[1], jac, f"Jaccard tra i top-{TOP_K}")]:
        im = ax.imshow(mat.values, cmap=SEQ_BLUE, vmin=0, vmax=1)
        ax.set_xticks(range(len(cols)), lab, rotation=35, ha="right"); ax.set_yticks(range(len(cols)), lab)
        ax.grid(False)
        for i in range(len(cols)):
            for j in range(len(cols)):
                v = mat.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.6 else C_INK)
        ax.set_title(title, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save(fig, "03_correlazioni_overlap.png")

    pairs = [("degree", "betweenness"), ("degree", "closeness"), ("degree", "eigenvector"),
             ("eigenvector", "pagerank"), ("betweenness", "pagerank"), ("closeness", "eigenvector")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    order = df.sort_values("ml")  # ML disegnati sopra
    for ax, (a, b) in zip(axes.flat, pairs):
        for cl in (0, 1):
            s = order[order["ml"] == cl]
            ax.scatter(s[a], s[b].clip(lower=1e-9), s=3, alpha=0.25, color=CLASS_COLOR[cl], linewidths=0,
                       rasterized=True, label=CLASS_NAME[cl])
        ax.set_xscale("linear" if a == "closeness" else "log")
        ax.set_yscale("linear" if b == "closeness" else "log")
        ax.set_xlabel(MEASURES[a]); ax.set_ylabel(MEASURES[b])
        ax.set_title(f"ρ = {sp.loc[a, b]:.2f}", loc="left", fontsize=10)
    leg = axes[0, 0].legend(frameon=False, markerscale=4)
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.suptitle("Relazione tra coppie di misure (scala log tranne closeness; betweenness nulla portata a 1e-9)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "04_scatter_coppie.png")

    #########################
    # 4. BROKER: betweenness alta rispetto al grado
    #########################
    print("4. Broker")
    df["rank_deg"], df["rank_btw"] = rank["degree"], rank["betweenness"]
    cand = df[df["rank_btw"] <= 1000].copy()
    cand["guadagno_rank"] = cand["rank_deg"] - cand["rank_btw"]
    brokers = cand.nlargest(10, "guadagno_rank")[["id", "name", "ml", "degree_raw", "rank_deg", "rank_btw", "guadagno_rank"]]
    brokers["ml"] = brokers["ml"].map(CLASS_NAME)
    clust = np.array(g.transitivity_local_undirected(mode="zero"))
    brokers["clustering"] = clust[brokers["id"]]
    # frazione dei vicini ML per capire se fanno da ponte tra le classi
    brokers["quota vicini ML"] = [np.mean(np.array(g.vs[g.neighbors(i)]["ml"])) for i in brokers["id"]]
    zero_btw = (df["betweenness"] == 0).mean()
    report += ["\n## 4. Broker (betweenness alta rispetto al grado)\n\n",
               f"{zero_btw:.1%} dei nodi ha betweenness nulla (non si trovano su alcun cammino minimo tra altri nodi). "
               "Tra i primi 1000 per betweenness, i nodi che guadagnano più posizioni rispetto alla classifica per "
               "grado fanno da *ponte* tra zone diverse della rete:\n\n",
               md_table(brokers.rename(columns={"ml": "classe", "degree_raw": "grado"})), "\n"]

    #########################
    # 5. CENTRALITÀ E CLASSE (WEB vs ML)
    #########################
    print("5. Classe")
    rows = []
    for c in cols:
        w, ml = df.loc[df["ml"] == 0, c], df.loc[df["ml"] == 1, c]
        u, p = stats.mannwhitneyu(w, ml, alternative="two-sided")
        auc = u / (len(w) * len(ml))     # P(centralità Web > centralità ML)
        rows.append((MEASURES[c], w.median(), ml.median(), w.median() / ml.median(), auc, p))
    cls_tab = pd.DataFrame(rows, columns=["Misura", "Mediana Web", "Mediana ML", "Rapporto", "P(Web > ML)", "p-value MWU"])
    ks = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
    share = pd.DataFrame({MEASURES[c]: [df.nlargest(k, c)["ml"].mean() for k in ks] for c in cols}, index=ks)
    share.index.name = "k"
    report += ["\n## 5. Centralità e classe (Web vs ML)\n\n",
               "Test di Mann–Whitney U (bilaterale); P(Web > ML) = probabilità che un nodo Web scelto a caso "
               "sia più centrale di un nodo ML scelto a caso (0.5 = nessuna differenza).\n\n",
               md_table(cls_tab), "\n",
               f"**Quota di sviluppatori ML tra i top-k** (baseline = {base_ml:.3f}):\n\n",
               md_table(share, index=True), "\n"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    for c in cols:
        ax.plot(ks, share[MEASURES[c]], marker="o", markersize=4, linewidth=2, color=MEAS_COLOR[c], label=MEASURES[c])
    ax.axhline(base_ml, color=C_GREY, linestyle="--", linewidth=1)
    ax.text(ks[0], base_ml + 0.006, f"quota ML nel grafo = {base_ml:.1%}", ha="left", va="bottom", fontsize=8, color="#52514e")
    ax.set_xscale("log"); ax.set_xlabel("k (dimensione del top-k)"); ax.set_ylabel("Quota ML nei top-k")
    ax.set_title("Sviluppatori ML tra i nodi più centrali", loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    ax.set_ylim(0, base_ml + 0.04)
    ax = axes[1]
    y = np.arange(len(cols))
    ax.barh(y, cls_tab["P(Web > ML)"] - 0.5, left=0.5, color=C_WEB, height=0.55)
    ax.axvline(0.5, color=C_INK, linewidth=1)
    ax.set_yticks(y, cls_tab["Misura"]); ax.invert_yaxis(); ax.set_xlim(0.4, 0.75)
    for yi, v in zip(y, cls_tab["P(Web > ML)"]):
        ax.text(v + 0.005, yi, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("P(centralità Web > centralità ML)"); ax.grid(axis="y", visible=False)
    ax.set_title("Dominanza stocastica Web vs ML (Mann–Whitney)", loc="left")
    fig.tight_layout()
    save(fig, "05_centralita_per_classe.png")

    #########################
    # 6. ROBUSTEZZA: rimozione dei nodi più centrali
    #########################
    print("6. Robustezza")
    fracs = np.r_[0, np.linspace(0.0025, 0.05, 20)]
    rng = np.random.default_rng(SEED)
    orders = {c: df.sort_values(c, ascending=False)["id"].values for c in cols}
    orders["random"] = rng.permutation(n)
    res = {}
    for c, ordv in orders.items():
        gc, frag = [], []
        for f in fracs:
            k = int(round(f * n))
            h = g.copy()
            h.delete_vertices(ordv[:k].tolist())
            sizes = h.connected_components().sizes()
            gc.append(max(sizes) / n)
            frag.append(len(sizes))
        res[c] = (gc, frag)
    rob = pd.DataFrame({("Casuale" if c == "random" else MEASURES[c]): res[c][0] for c in orders},
                       index=[f"{f:.2%}" for f in fracs])
    rob.index.name = "Nodi rimossi"
    report += ["\n## 6. Robustezza: rimozione mirata dei nodi più centrali\n\n",
               "Frazione di nodi (sul totale iniziale) nella componente gigante dopo la rimozione dei top-x% "
               "per ciascuna misura (ordine statico, calcolato sul grafo intero), confrontata con la rimozione casuale.\n\n",
               md_table(rob.iloc[[0, 1, 2, 4, 8, 12, 16, 20]], index=True), "\n"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for c in orders:
        col = C_GREY if c == "random" else MEAS_COLOR[c]
        lbl = "Casuale" if c == "random" else MEASURES[c]
        ls = "--" if c == "random" else "-"
        axes[0].plot(fracs * 100, res[c][0], color=col, linewidth=2, linestyle=ls, label=lbl)
        axes[1].plot(fracs * 100, res[c][1], color=col, linewidth=2, linestyle=ls, label=lbl)
    axes[0].set_ylabel("Frazione di nodi nella componente gigante")
    axes[1].set_ylabel("Numero di componenti connesse")
    for ax in axes:
        ax.set_xlabel("% di nodi rimossi")
    axes[0].set_title("Dimensione della componente gigante", loc="left")
    axes[1].set_title("Frammentazione", loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "06_robustezza_rimozione.png")

    #########################
    # 7. VISUALIZZAZIONE: sottografo dei nodi più centrali
    #########################
    print("7. Visualizzazione")
    top_ids = df.nlargest(300, "pagerank")["id"].tolist()
    sub = g.induced_subgraph(top_ids)
    sub.delete_vertices([v.index for v in sub.vs if v.degree() == 0])
    keep = sub.vs["orig_id"]
    layout = np.array(sub.layout_fruchterman_reingold(niter=2000))
    btw_sub = df.set_index("id").loc[keep, "betweenness"].values
    size = 15 + 600 * (btw_sub / btw_sub.max()) ** 0.6
    fig, ax = plt.subplots(figsize=(9, 8))
    segs = layout[np.array(sub.get_edgelist())]
    ax.add_collection(LineCollection(segs, colors="#c9c8c2", linewidths=0.3, alpha=0.5, zorder=1))
    ax.scatter(layout[:, 0], layout[:, 1], s=size, c=[CLASS_COLOR[v] for v in sub.vs["ml"]],
               edgecolors="white", linewidths=0.6, zorder=2)
    for i in np.argsort(-btw_sub)[:8]:
        ax.annotate(sub.vs[i]["name"], layout[i], fontsize=7, color=C_INK, ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points", zorder=3)
    ax.set_axis_off()
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=CLASS_COLOR[k], markersize=7) for k in (0, 1)]
    ax.legend(handles, ["Web", "ML"], frameon=False, loc="lower left")
    ax.set_title(f"Sottografo indotto dai top-300 nodi per PageRank ({len(keep)} nodi non isolati, "
                 f"{sub.ecount()} archi)\nDimensione ∝ betweenness; etichette: top-8 per betweenness",
                 loc="left", fontsize=10)
    save(fig, "07_sottografo_top_pagerank.png")

    #########################
    # 8. SINTESI
    #########################
    best = cls_tab.set_index("Misura")
    r_db = sp.loc["degree", "betweenness"]; r_dc = sp.loc["degree", "closeness"]
    j_min = jac.where(~np.eye(len(cols), dtype=bool)).stack().idxmin()
    report += ["\n## 7. Sintesi\n\n",
               f"- **Forte concentrazione**: Gini della betweenness = {gini(df['betweenness']):.3f}, del grado = "
               f"{gini(df['degree']):.3f}; centralizzazione di Freeman del grado = {cz_deg:.3f}, della betweenness = "
               f"{cz_btw:.3f}. Pochi hub (es. `{df.loc[df['degree'].idxmax(), 'name']}`, "
               f"{fmt(int(df['degree_raw'].max()))} vicini) dominano tutte le classifiche.\n",
               f"- **Misure molto correlate nel complesso** (Spearman grado–betweenness = {r_db:.2f}, "
               f"grado–closeness = {r_dc:.2f}) ma con differenze ai vertici: il minimo Jaccard tra top-{TOP_K} "
               f"è tra {MEASURES[j_min[0]]} e {MEASURES[j_min[1]]} ({jac.loc[j_min]:.2f}).\n",
               f"- **Classe**: i nodi Web sono sistematicamente più centrali (P(Web > ML) da "
               f"{best['P(Web > ML)'].min():.2f} a {best['P(Web > ML)'].max():.2f}); la quota ML tra i top-{TOP_K} per "
               f"PageRank è {share.loc[TOP_K, 'PageRank']:.1%} contro una baseline del {base_ml:.1%}.\n",
               f"- **Robustezza**: rimuovendo il 5% dei nodi per grado la componente gigante scende a "
               f"{res['degree'][0][-1]:.1%} dei nodi, contro {res['random'][0][-1]:.1%} con rimozione casuale "
               "(rete scale-free: robusta ai guasti casuali, fragile agli attacchi mirati).\n"]

    REPORT_MD.write_text("".join(report), encoding="utf-8")
    print(f"Report scritto in {REPORT_MD}")


if __name__ == "__main__":
    main()
