# Progetto 2 – Social Network Analysis della rete di sviluppatori GitHub

Analisi descrittiva, analisi delle misure di centralità, analisi delle strutture
(k-core, clique, small-world, rich-club, community) e link prediction
sulla rete sociale degli sviluppatori GitHub.

## Dati
- **Fonte:** dataset "GitHub Social Network" (MUSAE), Rozemberczki, Allen, Sarkar,
  *Multi-scale Attributed Node Embedding* (https://github.com/benedekrozemberczki/MUSAE),
  disponibile anche su SNAP (https://snap.stanford.edu/data/github-social.html)
- **Raccolta:** API pubblica di GitHub, giugno 2019
- **Contenuto:**
  - nodi = 37.700 sviluppatori con almeno 10 repository con stella;
  - archi = 289.003 relazioni di *mutual follow* (grafo non orientato, non pesato, connesso);
  - etichetta binaria `ml_target`: sviluppatore web (0) o machine learning (1);
  - feature dei nodi: indici anonimizzati di località, repository con stella, datore di lavoro, e-mail.
- I dati non sono inclusi nel repository. I file vanno posizionati in `data/raw/`:
  `musae_git_edges.csv`, `musae_git_target.csv`, `musae_git_features.json`.

## Struttura
| Cartella | Contenuto |
|---|---|
| 1_DATASET_ANALYSIS/ | Analisi descrittiva del dataset (in preparazione) |
| 2_CENTRALITY_ANALYSIS/ | Misure di centralità: grado, closeness, betweenness, eigenvector, PageRank, Katz |
| 3_STRUCTURE_ANALYSIS/ | Connettività, k-core, clique, small-world, rich-club, community detection, visualizzazione |
| 4_LINK_PREDICTION/ | Link prediction (in preparazione) |
| tesina/ | Relazione in LaTeX |
| data/ | Dati grezzi (esclusi da Git) |

Ogni cartella di analisi contiene `codice/` (script), `figures/` (grafici) e un file `*_results.md`
con le tabelle dei risultati. I risultati intermedi sono salvati in `cache/` (esclusa da Git).

## Esecuzione
Gli script si lanciano dalla rispettiva cartella `codice/`; il comando completo con le dipendenze
è indicato nell'intestazione di ciascuno script, ad esempio:

```
uv run --python 3.12 --with python-igraph --with pandas --with pyarrow \
        --with matplotlib --with scipy --with tabulate python centrality_analysis.py
```

L'opzione `--recompute` ricalcola i risultati anche se già presenti in `cache/`.
