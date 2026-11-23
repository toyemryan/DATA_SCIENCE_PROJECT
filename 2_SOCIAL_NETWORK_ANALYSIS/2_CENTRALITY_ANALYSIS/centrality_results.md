# Analisi delle misure di centralità – GitHub Social Network

Grafo non orientato e connesso: 37 700 nodi, 289 003 archi. Script: `codice/centrality_analysis.py` · Figure: `figures/` · Cache: `cache/centralities.parquet`.

## 0. Misure calcolate

| Misura | Definizione | Note |
|:--|:--|:--|
| Grado | $k_i/(n-1)$ | locale |
| Closeness | $(n-1)/\sum_j d(i,j)$ | esatta (grafo connesso) |
| Betweenness | $\sum_{s\neq t\neq i} \sigma_{st}(i)/\sigma_{st}$, normalizzata per $(n-1)(n-2)/2$ | esatta (Brandes), parallelizzata sulle sorgenti |
| Eigenvector | autovettore dominante di $A$, scalato a max = 1 | |
| PageRank | damping $d = 0.85$ | |
| Katz | $x = \alpha A x + \beta$, $\alpha = 0.85/\lambda_{max} = 0.00571$ ($\lambda_{max} = 148.79$), norm. L2 | |

## 1. Statistiche delle distribuzioni

| Misura      | Min       | Mediana   | Media     | Max     | Max/Mediana   | Gini    | Skewness   |
|:------------|:----------|:----------|:----------|:--------|:--------------|:--------|:-----------|
| Grado       | 2.653e-05 | 0.0001592 | 0.0004067 | 0.2509  | 1576          | 0.6879  | 67.76      |
| Closeness   | 0.1414    | 0.3113    | 0.3137    | 0.5231  | 1.681         | 0.07382 | -0.2165    |
| Betweenness | 0         | 1.529e-06 | 5.959e-05 | 0.2696  | 1.763e+05     | 0.9255  | 121.3      |
| Eigenvector | 9.961e-12 | 0.003046  | 0.006669  | 1       | 328.3         | 0.6359  | 25.55      |
| PageRank    | 5.04e-06  | 1.296e-05 | 2.653e-05 | 0.01686 | 1301          | 0.5724  | 78.45      |
| Katz        | 0.001625  | 0.002466  | 0.003482  | 0.3247  | 131.7         | 0.3383  | 31.14      |

Il coefficiente di Gini misura la disuguaglianza della centralità tra i nodi (0 = tutti uguali, 1 = tutto su un nodo).

**Centralizzazione di Freeman** (1 = grafo a stella, 0 = tutti i nodi equivalenti):

| Misura      | Centralizzazione di Freeman   |
|:------------|:------------------------------|
| Grado       | 0.2505                        |
| Betweenness | 0.2695                        |
| Closeness   | 0.4188                        |


## 2. Nodi più centrali

### Top 10 – Grado

| id    | name              | classe   | n. vicini   | Grado   | rank Closeness   | rank Betweenness   | rank Eigenvector   | rank PageRank   | rank Katz   |
|:------|:------------------|:---------|:------------|:--------|:-----------------|:-------------------|:-------------------|:----------------|:------------|
| 31890 | dalinhuang99      | Web      | 9 458       | 0.2509  | 2                | 1                  | 1                  | 1               | 1           |
| 27803 | nfultz            | Web      | 7 085       | 0.1879  | 1                | 2                  | 2                  | 2               | 2           |
| 35773 | addyosmani        | Web      | 3 324       | 0.08817 | 4                | 4                  | 3                  | 3               | 3           |
| 19222 | Bunlong           | Web      | 2 958       | 0.07846 | 3                | 3                  | 4                  | 4               | 4           |
| 13638 | gabrielpconceicao | Web      | 2 468       | 0.06547 | 5                | 5                  | 5                  | 5               | 5           |
| 36652 | rfthusn           | Web      | 2 343       | 0.06215 | 9                | 6                  | 6                  | 6               | 6           |
| 18163 | nelsonic          | Web      | 1 924       | 0.05104 | 68               | 8                  | 22                 | 7               | 15          |
| 9051  | getify            | Web      | 1 797       | 0.04767 | 41               | 13                 | 9                  | 8               | 9           |
| 35008 | mdo               | Web      | 1 571       | 0.04167 | 13               | 14                 | 11                 | 10              | 10          |
| 10001 | ronenhamias       | Web      | 1 568       | 0.04159 | 8                | 7                  | 14                 | 9               | 13          |

### Top 10 – Closeness

| id    | name              | classe   | n. vicini   | Closeness   | rank Grado   | rank Betweenness   | rank Eigenvector   | rank PageRank   | rank Katz   |
|:------|:------------------|:---------|:------------|:------------|:-------------|:-------------------|:-------------------|:----------------|:------------|
| 27803 | nfultz            | Web      | 7 085       | 0.5231      | 2            | 2                  | 2                  | 2               | 2           |
| 31890 | dalinhuang99      | Web      | 9 458       | 0.5178      | 1            | 1                  | 1                  | 1               | 1           |
| 19222 | Bunlong           | Web      | 2 958       | 0.4663      | 4            | 3                  | 4                  | 4               | 4           |
| 35773 | addyosmani        | Web      | 3 324       | 0.4503      | 3            | 4                  | 3                  | 3               | 3           |
| 13638 | gabrielpconceicao | Web      | 2 468       | 0.4475      | 5            | 5                  | 5                  | 5               | 5           |
| 19253 | JonnyBanana       | Web      | 1 404       | 0.4473      | 13           | 10                 | 12                 | 14              | 11          |
| 36628 | SuriyaaKudoIsc    | Web      | 1 507       | 0.443       | 11           | 11                 | 8                  | 12              | 7           |
| 10001 | ronenhamias       | Web      | 1 568       | 0.4423      | 10           | 7                  | 14                 | 9               | 13          |
| 36652 | rfthusn           | Web      | 2 343       | 0.4396      | 6            | 6                  | 6                  | 6               | 6           |
| 22881 | ajsb85            | Web      | 905         | 0.4317      | 28           | 21                 | 16                 | 28              | 18          |

### Top 10 – Betweenness

| id    | name              | classe   | n. vicini   | Betweenness   | rank Grado   | rank Closeness   | rank Eigenvector   | rank PageRank   | rank Katz   |
|:------|:------------------|:---------|:------------|:--------------|:-------------|:-----------------|:-------------------|:----------------|:------------|
| 31890 | dalinhuang99      | Web      | 9 458       | 0.2696        | 1            | 2                | 1                  | 1               | 1           |
| 27803 | nfultz            | Web      | 7 085       | 0.2405        | 2            | 1                | 2                  | 2               | 2           |
| 19222 | Bunlong           | Web      | 2 958       | 0.05532       | 4            | 3                | 4                  | 4               | 4           |
| 35773 | addyosmani        | Web      | 3 324       | 0.04341       | 3            | 4                | 3                  | 3               | 3           |
| 13638 | gabrielpconceicao | Web      | 2 468       | 0.03534       | 5            | 5                | 5                  | 5               | 5           |
| 36652 | rfthusn           | Web      | 2 343       | 0.03084       | 6            | 9                | 6                  | 6               | 6           |
| 10001 | ronenhamias       | Web      | 1 568       | 0.02761       | 10           | 8                | 14                 | 9               | 13          |
| 18163 | nelsonic          | Web      | 1 924       | 0.0258        | 7            | 68               | 22                 | 7               | 15          |
| 33671 | shayan-taheri     | Web      | 1 347       | 0.02126       | 15           | 32               | 40                 | 13              | 27          |
| 19253 | JonnyBanana       | Web      | 1 404       | 0.02032       | 13           | 6                | 12                 | 14              | 11          |

### Top 10 – Eigenvector

| id    | name              | classe   | n. vicini   | Eigenvector   | rank Grado   | rank Closeness   | rank Betweenness   | rank PageRank   | rank Katz   |
|:------|:------------------|:---------|:------------|:--------------|:-------------|:-----------------|:-------------------|:----------------|:------------|
| 31890 | dalinhuang99      | Web      | 9 458       | 1             | 1            | 2                | 1                  | 1               | 1           |
| 27803 | nfultz            | Web      | 7 085       | 0.8475        | 2            | 1                | 2                  | 2               | 2           |
| 35773 | addyosmani        | Web      | 3 324       | 0.4542        | 3            | 4                | 4                  | 3               | 3           |
| 19222 | Bunlong           | Web      | 2 958       | 0.4182        | 4            | 3                | 3                  | 4               | 4           |
| 13638 | gabrielpconceicao | Web      | 2 468       | 0.3357        | 5            | 5                | 5                  | 5               | 5           |
| 36652 | rfthusn           | Web      | 2 343       | 0.2689        | 6            | 9                | 6                  | 6               | 6           |
| 7027  | jeresig           | Web      | 1 485       | 0.2494        | 12           | 15               | 18                 | 15              | 8           |
| 36628 | SuriyaaKudoIsc    | Web      | 1 507       | 0.2464        | 11           | 7                | 11                 | 12              | 7           |
| 9051  | getify            | Web      | 1 797       | 0.2326        | 8            | 41               | 13                 | 8               | 9           |
| 3712  | isaacs            | Web      | 1 264       | 0.2286        | 18           | 54               | 27                 | 19              | 12          |

### Top 10 – PageRank

| id    | name              | classe   | n. vicini   | PageRank   | rank Grado   | rank Closeness   | rank Betweenness   | rank Eigenvector   | rank Katz   |
|:------|:------------------|:---------|:------------|:-----------|:-------------|:-----------------|:-------------------|:-------------------|:------------|
| 31890 | dalinhuang99      | Web      | 9 458       | 0.01686    | 1            | 2                | 1                  | 1                  | 1           |
| 27803 | nfultz            | Web      | 7 085       | 0.01149    | 2            | 1                | 2                  | 2                  | 2           |
| 35773 | addyosmani        | Web      | 3 324       | 0.005057   | 3            | 4                | 4                  | 3                  | 3           |
| 19222 | Bunlong           | Web      | 2 958       | 0.004727   | 4            | 3                | 3                  | 4                  | 4           |
| 13638 | gabrielpconceicao | Web      | 2 468       | 0.003927   | 5            | 5                | 5                  | 5                  | 5           |
| 36652 | rfthusn           | Web      | 2 343       | 0.003868   | 6            | 9                | 6                  | 6                  | 6           |
| 18163 | nelsonic          | Web      | 1 924       | 0.003451   | 7            | 68               | 8                  | 22                 | 15          |
| 9051  | getify            | Web      | 1 797       | 0.002804   | 8            | 41               | 13                 | 9                  | 9           |
| 10001 | ronenhamias       | Web      | 1 568       | 0.002704   | 10           | 8                | 7                  | 14                 | 13          |
| 35008 | mdo               | Web      | 1 571       | 0.002401   | 9            | 13               | 14                 | 11                 | 10          |

### Top 10 – Katz

| id    | name              | classe   | n. vicini   | Katz    | rank Grado   | rank Closeness   | rank Betweenness   | rank Eigenvector   | rank PageRank   |
|:------|:------------------|:---------|:------------|:--------|:-------------|:-----------------|:-------------------|:-------------------|:----------------|
| 31890 | dalinhuang99      | Web      | 9 458       | 0.3247  | 1            | 2                | 1                  | 1                  | 1               |
| 27803 | nfultz            | Web      | 7 085       | 0.2687  | 2            | 1                | 2                  | 2                  | 2               |
| 35773 | addyosmani        | Web      | 3 324       | 0.1384  | 3            | 4                | 4                  | 3                  | 3               |
| 19222 | Bunlong           | Web      | 2 958       | 0.1282  | 4            | 3                | 3                  | 4                  | 4               |
| 13638 | gabrielpconceicao | Web      | 2 468       | 0.104   | 5            | 5                | 5                  | 5                  | 5               |
| 36652 | rfthusn           | Web      | 2 343       | 0.08753 | 6            | 9                | 6                  | 6                  | 6               |
| 36628 | SuriyaaKudoIsc    | Web      | 1 507       | 0.07466 | 11           | 7                | 11                 | 8                  | 12              |
| 7027  | jeresig           | Web      | 1 485       | 0.07383 | 12           | 15               | 18                 | 7                  | 15              |
| 9051  | getify            | Web      | 1 797       | 0.07278 | 8            | 41               | 13                 | 9                  | 8               |
| 35008 | mdo               | Web      | 1 571       | 0.07004 | 9            | 13               | 14                 | 11                 | 10              |

### Sviluppatori ML più centrali (top 5 tra i nodi ML)

| Misura      | Pos. tra ML   | Nome           | Grado   | Rank globale   |
|:------------|:--------------|:---------------|:--------|:---------------|
| Grado       | 1             | antirez        | 967     | 24             |
| Grado       | 2             | rasbt          | 729     | 34             |
| Grado       | 3             | bradfitz       | 618     | 40             |
| Grado       | 4             | amueller       | 478     | 65             |
| Grado       | 5             | sebastianruder | 398     | 78             |
| Closeness   | 1             | WillemJan      | 275     | 66             |
| Closeness   | 2             | bradfitz       | 618     | 97             |
| Closeness   | 3             | antirez        | 967     | 103            |
| Closeness   | 4             | surajit-techie | 82      | 114            |
| Closeness   | 5             | amiryeg        | 167     | 127            |
| Betweenness | 1             | rasbt          | 729     | 24             |
| Betweenness | 2             | antirez        | 967     | 26             |
| Betweenness | 3             | sebastianruder | 398     | 39             |
| Betweenness | 4             | amueller       | 478     | 46             |
| Betweenness | 5             | boxvc          | 187     | 47             |
| Eigenvector | 1             | antirez        | 967     | 23             |
| Eigenvector | 2             | bradfitz       | 618     | 48             |
| Eigenvector | 3             | cloudhead      | 312     | 142            |
| Eigenvector | 4             | WillemJan      | 275     | 162            |
| Eigenvector | 5             | rasbt          | 729     | 165            |
| PageRank    | 1             | antirez        | 967     | 21             |
| PageRank    | 2             | rasbt          | 729     | 27             |
| PageRank    | 3             | bradfitz       | 618     | 40             |
| PageRank    | 4             | amueller       | 478     | 50             |
| PageRank    | 5             | sebastianruder | 398     | 59             |
| Katz        | 1             | antirez        | 967     | 24             |
| Katz        | 2             | bradfitz       | 618     | 44             |
| Katz        | 3             | rasbt          | 729     | 95             |
| Katz        | 4             | cloudhead      | 312     | 142            |
| Katz        | 5             | WillemJan      | 275     | 153            |


## 3. Correlazione tra misure

**Spearman** (sull'intero grafo):

|             | Grado   | Closeness   | Betweenness   | Eigenvector   | PageRank   | Katz   |
|:------------|:--------|:------------|:--------------|:--------------|:-----------|:-------|
| Grado       | 1       | 0.7147      | 0.8611        | 0.7551        | 0.9726     | 0.7743 |
| Closeness   | 0.7147  | 1           | 0.6804        | 0.9838        | 0.6457     | 0.9827 |
| Betweenness | 0.8611  | 0.6804      | 1             | 0.6956        | 0.9023     | 0.7149 |
| Eigenvector | 0.7551  | 0.9838      | 0.6956        | 1             | 0.6815     | 0.9987 |
| PageRank    | 0.9726  | 0.6457      | 0.9023        | 0.6815        | 1          | 0.7052 |
| Katz        | 0.7743  | 0.9827      | 0.7149        | 0.9987        | 0.7052     | 1      |

**Kendall τ**:

|             | Grado   | Closeness   | Betweenness   | Eigenvector   | PageRank   | Katz   |
|:------------|:--------|:------------|:--------------|:--------------|:-----------|:-------|
| Grado       | 1       | 0.5421      | 0.7128        | 0.5827        | 0.8957     | 0.6061 |
| Closeness   | 0.5421  | 1           | 0.5134        | 0.897         | 0.4607     | 0.8902 |
| Betweenness | 0.7128  | 0.5134      | 1             | 0.5273        | 0.7328     | 0.5474 |
| Eigenvector | 0.5827  | 0.897       | 0.5273        | 1             | 0.4929     | 0.9723 |
| PageRank    | 0.8957  | 0.4607      | 0.7328        | 0.4929        | 1          | 0.5156 |
| Katz        | 0.6061  | 0.8902      | 0.5474        | 0.9723        | 0.5156     | 1      |

**Jaccard tra i top-100** (sovrapposizione dei nodi più centrali):

|             | Grado   | Closeness   | Betweenness   | Eigenvector   | PageRank   | Katz   |
|:------------|:--------|:------------|:--------------|:--------------|:-----------|:-------|
| Grado       | 1       | 0.3423      | 0.6667        | 0.6393        | 0.8868     | 0.6807 |
| Closeness   | 0.3423  | 1           | 0.3699        | 0.4599        | 0.3333     | 0.4493 |
| Betweenness | 0.6667  | 0.3699      | 1             | 0.5038        | 0.7391     | 0.5267 |
| Eigenvector | 0.6393  | 0.4599      | 0.5038        | 1             | 0.5748     | 0.9231 |
| PageRank    | 0.8868  | 0.3333      | 0.7391        | 0.5748        | 1          | 0.6129 |
| Katz        | 0.6807  | 0.4493      | 0.5267        | 0.9231        | 0.6129     | 1      |


## 4. Broker (betweenness alta rispetto al grado)

17.0% dei nodi ha betweenness nulla (non si trovano su alcun cammino minimo tra altri nodi). Tra i primi 1000 per betweenness, i nodi che guadagnano più posizioni rispetto alla classifica per grado fanno da *ponte* tra zone diverse della rete:

| id    | name           | classe   | grado   | rank_deg   | rank_btw   | guadagno_rank   | clustering   | quota vicini ML   |
|:------|:---------------|:---------|:--------|:-----------|:-----------|:----------------|:-------------|:------------------|
| 36859 | artificialsoph | ML       | 6       | 17 063     | 821        | 16 242          | 0            | 0.8333            |
| 3489  | pulver22       | ML       | 13      | 9 209      | 946        | 8 263           | 0.1154       | 0.7692            |
| 6666  | jpdias         | Web      | 15      | 8 059      | 963        | 7 096           | 0.09524      | 0.1333            |
| 26962 | SivilTaram     | Web      | 18      | 6 637      | 731        | 5 906           | 0.0915       | 0.5               |
| 13740 | n1try          | Web      | 19      | 6 243      | 838        | 5 405           | 0.05263      | 0.3158            |
| 34905 | GaoxingZheng   | ML       | 19      | 6 243      | 841        | 5 402           | 0.1111       | 0.5789            |
| 5479  | ananagame      | Web      | 26      | 4 338      | 622        | 3 716           | 0.1815       | 0.03846           |
| 28848 | smdsbz         | Web      | 26      | 4 338      | 831        | 3 507           | 0.1662       | 0.5769            |
| 2317  | FZambia        | Web      | 28      | 3 958      | 634        | 3 324           | 0.06085      | 0.1429            |
| 11260 | swordencao     | Web      | 27      | 4 135      | 976        | 3 159           | 0.05698      | 0.6296            |


## 5. Centralità e classe (Web vs ML)

Test di Mann–Whitney U (bilaterale); P(Web > ML) = probabilità che un nodo Web scelto a caso sia più centrale di un nodo ML scelto a caso (0.5 = nessuna differenza).

| Misura      | Mediana Web   | Mediana ML   | Rapporto   | P(Web > ML)   | p-value MWU   |
|:------------|:--------------|:-------------|:-----------|:--------------|:--------------|
| Grado       | 0.0001592     | 0.0001061    | 1.5        | 0.6128        | 2.062e-243    |
| Closeness   | 0.3187        | 0.2908       | 1.096      | 0.6631        | 0             |
| Betweenness | 2.037e-06     | 5.716e-07    | 3.563      | 0.574         | 7.59e-106     |
| Eigenvector | 0.004628      | 0.0008026    | 5.767      | 0.6822        | 0             |
| PageRank    | 1.375e-05     | 1.097e-05    | 1.254      | 0.5825        | 2.724e-130    |
| Katz        | 0.002897      | 0.001881     | 1.54       | 0.6732        | 0             |

**Quota di sviluppatori ML tra i top-k** (baseline = 0.258):

| k    | Grado   | Closeness   | Betweenness   | Eigenvector   | PageRank   | Katz   |
|:-----|:--------|:------------|:--------------|:--------------|:-----------|:-------|
| 10   | 0       | 0           | 0             | 0             | 0          | 0      |
| 25   | 0.04    | 0           | 0.04          | 0.04          | 0.04       | 0.04   |
| 50   | 0.06    | 0           | 0.12          | 0.04          | 0.08       | 0.04   |
| 100  | 0.05    | 0.02        | 0.13          | 0.02          | 0.1        | 0.03   |
| 250  | 0.096   | 0.032       | 0.164         | 0.028         | 0.116      | 0.032  |
| 500  | 0.086   | 0.02        | 0.156         | 0.02          | 0.114      | 0.026  |
| 1000 | 0.092   | 0.033       | 0.157         | 0.025         | 0.114      | 0.031  |
| 2500 | 0.104   | 0.084       | 0.1892        | 0.0404        | 0.1264     | 0.0464 |
| 5000 | 0.1286  | 0.0878      | 0.2054        | 0.0674        | 0.149      | 0.0744 |


## 6. Robustezza: rimozione mirata dei nodi più centrali

Frazione di nodi (sul totale iniziale) nella componente gigante dopo la rimozione dei top-x% per ciascuna misura (ordine statico, calcolato sul grafo intero), confrontata con la rimozione casuale.

| Nodi rimossi   | Grado   | Closeness   | Betweenness   | Eigenvector   | PageRank   | Katz   | Casuale   |
|:---------------|:--------|:------------|:--------------|:--------------|:-----------|:-------|:----------|
| 0.00%          | 1       | 1           | 1             | 1             | 1          | 1      | 1         |
| 0.25%          | 0.9397  | 0.9512      | 0.9374        | 0.9461        | 0.9381     | 0.9444 | 0.9972    |
| 0.50%          | 0.9237  | 0.9438      | 0.9213        | 0.9315        | 0.9223     | 0.9307 | 0.9945    |
| 1.00%          | 0.9     | 0.9326      | 0.8965        | 0.9167        | 0.8963     | 0.9139 | 0.9891    |
| 2.00%          | 0.8639  | 0.9112      | 0.8596        | 0.8903        | 0.8596     | 0.8852 | 0.9765    |
| 3.00%          | 0.834   | 0.8919      | 0.8273        | 0.8643        | 0.8301     | 0.8603 | 0.9657    |
| 4.00%          | 0.8075  | 0.8741      | 0.7961        | 0.842         | 0.8029     | 0.8382 | 0.9521    |
| 5.00%          | 0.7821  | 0.8543      | 0.7695        | 0.8206        | 0.7752     | 0.8153 | 0.941     |


## 7. Sintesi

- **Forte concentrazione**: Gini della betweenness = 0.925, del grado = 0.688; centralizzazione di Freeman del grado = 0.250, della betweenness = 0.270. Pochi hub (es. `dalinhuang99`, 9 458 vicini) dominano tutte le classifiche.
- **Misure molto correlate nel complesso** (Spearman grado–betweenness = 0.86, grado–closeness = 0.71) ma con differenze ai vertici: il minimo Jaccard tra top-100 è tra Closeness e PageRank (0.33).
- **Classe**: i nodi Web sono sistematicamente più centrali (P(Web > ML) da 0.57 a 0.68); la quota ML tra i top-100 per PageRank è 10.0% contro una baseline del 25.8%.
- **Robustezza**: rimuovendo il 5% dei nodi per grado la componente gigante scende a 78.2% dei nodi, contro 94.1% con rimozione casuale (rete scale-free: robusta ai guasti casuali, fragile agli attacchi mirati).
