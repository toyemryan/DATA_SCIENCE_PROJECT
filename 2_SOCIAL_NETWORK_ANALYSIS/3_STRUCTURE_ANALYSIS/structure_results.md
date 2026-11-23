# Visualizzazione e analisi delle strutture – GitHub Social Network

Grafo non orientato: 37 700 nodi, 289 003 archi. Script: `codice/structure_analysis.py` · Figure: `figures/` · Cache: `cache/`.

## 1. Connettività e punti critici

| Misura                                      | Valore   |
|:--------------------------------------------|:---------|
| Componenti connesse                         | 1        |
| Punti di articolazione                      | 3 188    |
| Punti di articolazione (% nodi)             | 8.46%    |
| Bridge (archi ponte)                        | 5 241    |
| Nodi foglia (grado 1)                       | 5 045    |
| Bridge che non toccano una foglia           | 196      |
| Componenti biconnesse                       | 5 263    |
| Nodi nella componente biconnessa più grande | 32 416   |
| Seconda componente biconnessa (nodi)        | 4        |

**Punti di articolazione con grado più alto** (nodi che, rimossi, disconnettono il grafo):

| Nome              | Classe   | Grado   | Nuove componenti   | Nodi staccati   |
|:------------------|:---------|:--------|:-------------------|:----------------|
| dalinhuang99      | Web      | 9 458   | 368                | 379             |
| nfultz            | Web      | 7 085   | 121                | 123             |
| addyosmani        | Web      | 3 324   | 38                 | 41              |
| Bunlong           | Web      | 2 958   | 71                 | 73              |
| gabrielpconceicao | Web      | 2 468   | 47                 | 47              |
| rfthusn           | Web      | 2 343   | 46                 | 49              |
| nelsonic          | Web      | 1 924   | 52                 | 55              |
| getify            | Web      | 1 797   | 35                 | 37              |
| mdo               | Web      | 1 571   | 21                 | 22              |
| ronenhamias       | Web      | 1 568   | 54                 | 55              |


## 2. Decomposizione k-core

- Coreness massima (degenerazione del grafo): **k_max = 34**; il 34-core contiene 360 nodi e 10488 archi (densità 0.162), quota ML = 1.4%.
- Coreness mediana: Web = 6, ML = 4; Spearman(coreness, grado) = 0.991.

Dimensione del k-core (nodi con coreness ≥ k) e composizione per classe:

| k   | Nodi nel k-core   | Quota del grafo   | Quota ML nel k-core   |
|:----|:------------------|:------------------|:----------------------|
| 1   | 37 700            | 1                 | 0.2583                |
| 2   | 32 464            | 0.8611            | 0.2417                |
| 3   | 27 548            | 0.7307            | 0.2233                |
| 5   | 20 491            | 0.5435            | 0.1975                |
| 10  | 10 710            | 0.2841            | 0.1481                |
| 15  | 6 076             | 0.1612            | 0.1124                |
| 20  | 3 661             | 0.09711           | 0.07648               |
| 25  | 1 963             | 0.05207           | 0.03872               |
| 30  | 915               | 0.02427           | 0.01202               |


## 3. Clique

- Clique massimali con almeno 3 nodi: **223 672** (calcolo in 1.4s).
- Numero di clique ω(G) = **24** (4 clique di dimensione massima).
- Le 4 clique massime coinvolgono 26 nodi distinti e ne condividono 22: sono varianti dello stesso gruppo densissimo (quota ML = 3.8%; es. e911miri, favourch, forevertga, iamonuwa, perfectmak).

Composizione per classe delle clique massimali (quota di clique interamente Web, interamente ML, miste):

| Dimensione   | Clique   | Solo Web   | Solo ML   | Miste   |
|:-------------|:---------|:-----------|:----------|:--------|
| ≥ 3          | 223 672  | 0.8223     | 0.01341   | 0.1643  |
| ≥ 5          | 96 944   | 0.8789     | 0.001145  | 0.1199  |
| ≥ 10         | 1 496    | 0.6571     | 0         | 0.3429  |
| ≥ 15         | 106      | 0.3962     | 0         | 0.6038  |


## 4. Small-world e confronto con modelli nulli

Medie su 5 repliche (± dev. std). Distanza media stimata con BFS da 300 sorgenti (le stesse per tutti i grafi); per ER solo tra coppie connesse. Configuration model generato con l'algoritmo di Viger–Latapy (grafo semplice e connesso, stessa sequenza dei gradi).

| Grafo                                  | Transitività      | Clustering medio   | Distanza media   | Assortatività grado   | Quota comp. gigante   |
|:---------------------------------------|:------------------|:-------------------|:-----------------|:----------------------|:----------------------|
| GitHub (reale)                         | 0.01236           | 0.1675             | 3.271            | -0.07522              | 1                     |
| Erdős–Rényi G(n,m)                     | 0.0004168 ± 1e-05 | 0.0004133 ± 1e-05  | 4.141 ± 0.005    | 0.0007531 ± 0.002     | 1 ± 0                 |
| Configuration model (grado preservato) | 0.01225 ± 6e-05   | 0.09234 ± 0.0001   | 3.142 ± 0.009    | -0.07117 ± 5e-05      | 1 ± 0                 |

- Coefficiente small-world σ = (C/C_r)/(L/L_r): rispetto a ER = **513.3**, rispetto al configuration model = **1.74** (σ > 1 → small-world).
- Il clustering medio reale è 405× quello di ER e 1.8× quello del configuration model: il clustering **non** è spiegato dalla sola distribuzione dei gradi.
- Clustering gerarchico: C(k) ∝ k^(−β) con β = 0.65 (R² = 0.97, k ≥ 5); β ≈ 1 è la firma dei modelli gerarchici (Ravasz–Barabási).

## 5. Rich-club

φ(k) = densità del sottografo dei nodi con grado > k; ρ(k) normalizzato rispetto a 5 grafi ricablati che preservano i gradi (ρ > 1 → gli hub si connettono tra loro più del caso).

| k   | Nodi con grado > k   | φ(k)      | φ_rand(k)   | ρ(k) = φ/φ_rand   |
|:----|:---------------------|:----------|:------------|:------------------|
| 1   | 32 655               | 0.0005326 | 0.0005326   | 0.9999            |
| 5   | 19 052               | 0.001337  | 0.001332    | 1.004             |
| 12  | 9 923                | 0.003591  | 0.003567    | 1.007             |
| 24  | 4 801                | 0.009499  | 0.009616    | 0.9879            |
| 50  | 1 844                | 0.0285    | 0.0316      | 0.9018            |
| 103 | 611                  | 0.07608   | 0.1074      | 0.7087            |
| 211 | 200                  | 0.1769    | 0.3142      | 0.5631            |
| 432 | 70                   | 0.299     | 0.6316      | 0.4733            |
| 882 | 28                   | 0.5317    | 0.8524      | 0.6238            |

- ρ massimo = 1.01 a k = 10; ρ minimo = 0.46 a k = 516.

## 6. Community detection

### 6.1 Confronto tra algoritmi

| Algoritmo         | Community   | Modularità Q   | Quota community max   | Community ≥ 10 nodi   | NMI vs classe   | ARI vs classe   | Tempo (s)   |
|:------------------|:------------|:---------------|:----------------------|:----------------------|:----------------|:----------------|:------------|
| Louvain           | 30          | 0.453          | 0.2197                | 16                    | 0.1148          | 0.06575         | 0.2398      |
| Leiden            | 30          | 0.4591         | 0.1982                | 15                    | 0.1119          | 0.05864         | 2.377       |
| Label Propagation | 12          | 0.0001938      | 0.9991                | 1                     | 0.002596        | 0.001935        | 0.09737     |
| Infomap           | 1 470       | 0.3936         | 0.1515                | 739                   | 0.06765         | 0.02836         | 4.784       |
| Fast Greedy (CNM) | 407         | 0.3926         | 0.3101                | 26                    | 0.06583         | 0.04876         | 25.56       |

NMI tra le partizioni dei diversi algoritmi:

|                   | Louvain   | Leiden   | Label Propagation   | Infomap   | Fast Greedy (CNM)   |
|:------------------|:----------|:---------|:--------------------|:----------|:--------------------|
| Louvain           | 1         | 0.5584   | 0.004375            | 0.4011    | 0.3286              |
| Leiden            | 0.5584    | 1        | 0.005386            | 0.4132    | 0.3698              |
| Label Propagation | 0.004375  | 0.005386 | 1                   | 0.003278  | 0.01019             |
| Infomap           | 0.4011    | 0.4132   | 0.003278            | 1         | 0.3303              |
| Fast Greedy (CNM) | 0.3286    | 0.3698   | 0.01019             | 0.3303    | 1                   |

### 6.2 Leiden: effetto della risoluzione γ

Funzione di qualità RB-configuration: γ < 1 favorisce poche community grandi, γ > 1 molte piccole. La modularità è sempre riportata con γ = 1 per confrontabilità.

| γ    | Community   | Community ≥ 10 nodi   | Modularità Q (γ=1)   | Quota community max   | NMI vs classe   |
|:-----|:------------|:----------------------|:---------------------|:----------------------|:----------------|
| 0.1  | 1           | 1                     | 0                    | 1                     | 0               |
| 0.25 | 22          | 1                     | 0.0003598            | 0.9983                | 0.003994        |
| 0.5  | 17          | 4                     | 0.3788               | 0.5646                | 0.1988          |
| 1    | 30          | 15                    | 0.4591               | 0.1982                | 0.1119          |
| 2    | 46          | 42                    | 0.4198               | 0.08674               | 0.08029         |
| 4    | 133         | 129                   | 0.3694               | 0.05074               | 0.06224         |
| 8    | 320         | 320                   | 0.2964               | 0.01897               | 0.05064         |

### 6.3 Partizione Leiden: le 15 community più grandi

Conduttanza = archi uscenti / min(volume interno, volume esterno): più è bassa, più la community è separata dal resto della rete.

| Rank   | Nodi   | Quota nodi   | Quota ML   | Archi interni   | Densità interna   | Conduttanza   | Hub               | Grado hub   |
|:-------|:-------|:-------------|:-----------|:----------------|:------------------|:--------------|:------------------|:------------|
| 1      | 7 474  | 0.1982       | 0.7412     | 32 167          | 0.001152          | 0.2977        | nfultz            | 7 085       |
| 2      | 6 569  | 0.1742       | 0.05237    | 35 674          | 0.001654          | 0.4232        | addyosmani        | 3 324       |
| 3      | 6 246  | 0.1657       | 0.1214     | 27 386          | 0.001404          | 0.4521        | gabrielpconceicao | 2 468       |
| 4      | 5 143  | 0.1364       | 0.1806     | 35 197          | 0.002662          | 0.289         | ronenhamias       | 1 568       |
| 5      | 2 787  | 0.07393      | 0.1274     | 13 475          | 0.003471          | 0.4203        | rfthusn           | 2 343       |
| 6      | 2 650  | 0.07029      | 0.2668     | 8 001           | 0.00228           | 0.5246        | dalinhuang99      | 9 458       |
| 7      | 2 591  | 0.06873      | 0.05095    | 7 926           | 0.002362          | 0.4535        | philsturgeon      | 423         |
| 8      | 1 517  | 0.04024      | 0.2281     | 6 811           | 0.005923          | 0.4437        | onevcat           | 684         |
| 9      | 747    | 0.01981      | 0.2222     | 849             | 0.003047          | 0.7108        | nelsonic          | 1 924       |
| 10     | 567    | 0.01504      | 0.3527     | 2 000           | 0.01246           | 0.3747        | hunkim            | 253         |
| 11     | 508    | 0.01347      | 0.1575     | 2 262           | 0.01757           | 0.4124        | azer              | 313         |
| 12     | 443    | 0.01175      | 0.1129     | 1 802           | 0.01841           | 0.3898        | christiannwamba   | 207         |
| 13     | 277    | 0.007347     | 0.2671     | 696             | 0.01821           | 0.5464        | c9s               | 225         |
| 14     | 85     | 0.002255     | 0.1529     | 148             | 0.04146           | 0.4905        | neungkl           | 40          |
| 15     | 43     | 0.001141     | 0.4186     | 116             | 0.1285            | 0.4567        | iblancasa         | 47          |

- Le prime 15 community coprono il 99.9% dei nodi.
- Entropia della classe: globale = 0.824 bit, condizionata alla community = 0.600 bit → la community spiega il 27.2% dell'incertezza sulla classe (informazione mutua normalizzata sulla classe).
- Community con quota ML > 50%: 1 (tra quelle con ≥ 10 nodi), che contengono il 56.9% degli sviluppatori ML.

## 7. Sintesi

- **Connettività**: grafo connesso ma con 8.5% di punti di articolazione e 5 241 bridge, quasi tutti verso nodi foglia (5 045 nodi di grado 1): periferia ad albero appesa a un nucleo biconnesso di 86.0% dei nodi.
- **Nucleo**: degenerazione k_max = 34; il nucleo è quasi solo Web (quota ML 1.4% vs 25.8%), coerente con la centralità.
- **Small-world**: distanza media 3.27 (persino minore di ER, 4.14) con clustering 405× ER → σ = 513; il clustering resta 1.8× quello del configuration model, quindi c'è chiusura triadica oltre l'effetto dei gradi. C(k) ∝ k^−0.65: clustering decrescente con il grado, segno di organizzazione gerarchica (più debole del β = 1 teorico).
- **Rich-club**: ρ(k) < 1 per k alti: gli hub NON formano un club, coerente con la disassortatività (ρ = 0.66 al k più alto).
- **Community**: Leiden ottiene la modularità più alta (Q = 0.459, 30 community); Label Propagation degenera (12 community, Q = 0.000) perché gli hub propagano la stessa etichetta a quasi tutta la rete. Le community catturano parte della classe (NMI Leiden vs classe = 0.112).
