# Forecasting degli inquinanti – confronto tra modelli

Previsione a un giorno (one-step-ahead). Training: fino al 2025-12-31 · Test: 2026-01-01 – 2026-10-26. Per ogni inquinante le metriche sono calcolate solo sui giorni-stazione in cui l'osservazione è valida e tutti i modelli hanno una previsione.

- **Skill %** = `1 − MAE_modello / MAE_persistenza`: miglioramento rispetto a "domani = oggi".
- **Precision / Recall / F1** riguardano i *superamenti* (giorni sopra la soglia dell'inquinante). Precision = quota dei superamenti previsti che si sono verificati; recall = quota dei superamenti osservati che sono stati previsti. "–" = non calcolabile (il modello non prevede mai un superamento, oppure nel test non ce ne sono).
- In fondo: errore per orizzonte da 1 a 7 giorni e previsione per i 7 giorni dopo l'ultimo dato.
- Spiegazione completa in [forecasting_explained.md](forecasting_explained.md).

## Sintesi: skill vs persistenza (%) per inquinante

Valori positivi = meglio della persistenza. Le 4 stazioni di ogni inquinante sono aggregate.

|       |   Persistenza |   Seasonal naive (t−7) |   ARIMA |   SARIMAX |   Prophet |   Prophet + lag |   XGBoost |
|:------|--------------:|-----------------------:|--------:|----------:|----------:|----------------:|----------:|
| PM10  |          +0,0 |                  −62,6 |    +6,4 |      +7,0 |     −21,1 |            +6,5 |      +7,7 |
| PM2.5 |          +0,0 |                  −63,6 |    +7,3 |      +6,6 |     −29,6 |            +3,8 |      +7,3 |
| NO2   |          +0,0 |                  −35,4 |    +5,0 |      +5,5 |     −32,1 |            −5,8 |      +9,1 |
| O3    |          +0,0 |                  −50,4 |    +5,4 |      +3,0 |     −64,4 |            −7,2 |      +4,4 |
| SO2   |          +0,0 |                  −49,4 |   −18,5 |      −2,2 |    −311,3 |           −48,9 |      −7,7 |
| CO    |          +0,0 |                  −57,2 |    +5,5 |      +6,9 |     −59,6 |            +0,2 |      +6,9 |
| C6H6  |          +0,0 |                  −37,8 |   +10,2 |      +6,6 |     −72,2 |            −7,1 |     +12,1 |

## Impostazione per inquinante

|   Inquinante |            Target |   Unità |   Soglia |                                  Riferimento soglia |   Giorni-stazione |   Superamenti |   Media osservata |
|-------------:|------------------:|--------:|---------:|----------------------------------------------------:|------------------:|--------------:|------------------:|
|         PM10 | media giornaliera |   µg/m³ |       50 |                 limite giornaliero, D.Lgs. 155/2010 |             1.038 |           105 |             27,88 |
|        PM2.5 | media giornaliera |   µg/m³ |       25 |      limite giornaliero dal 2030, Dir. UE 2024/2881 |             1.025 |            91 |             13,75 |
|          NO2 | media giornaliera |   µg/m³ |       50 |      limite giornaliero dal 2030, Dir. UE 2024/2881 |             1.087 |           162 |             34,15 |
|           O3 |     max media 8 h |   µg/m³ |      120 | valore obiettivo sul max media 8 h, D.Lgs. 155/2010 |             1.018 |           102 |             87,56 |
|          SO2 | media giornaliera |   µg/m³ |       50 |      limite giornaliero dal 2030, Dir. UE 2024/2881 |             1.045 |             0 |              2,73 |
|           CO | media giornaliera |   mg/m³ |        4 |      limite giornaliero dal 2030, Dir. UE 2024/2881 |             1.064 |             0 |              0,57 |
|         C6H6 | media giornaliera |   µg/m³ |        – |                       solo limite annuale (5 µg/m³) |               997 |             – |              0,83 |

## PM10

Unità: µg/m³. Soglia superamenti: 50 µg/m³ (limite giornaliero, D.Lgs. 155/2010).

### Complessivo (4 stazioni aggregate)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| XGBoost              |  7,91 |  13,25 |  −2,22 |        0,72 |     0,22 | 0,34 |      +7,7 |
| SARIMAX              |  7,97 |  13,61 |  −1,47 |        0,71 |     0,29 | 0,41 |      +7,0 |
| Prophet + lag        |  8,01 |  13,04 |  −2,14 |        0,62 |     0,25 | 0,35 |      +6,5 |
| ARIMA                |  8,02 |  13,73 |  −1,70 |        0,65 |     0,23 | 0,34 |      +6,4 |
| Persistenza          |  8,57 |  15,01 |   0,04 |        0,55 |     0,55 | 0,55 |      +0,0 |
| Prophet              | 10,37 |  15,43 |  −1,92 |        0,25 |     0,06 | 0,09 |     −21,1 |
| Seasonal naive (t−7) | 13,93 |  20,98 |   0,41 |        0,24 |     0,25 | 0,24 |     −62,6 |

### Per stazione

#### IT2211A – Acerra Zona Industriale  (giorni di test: 264, superamenti osservati: 62)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| Prophet + lag        | 11,38 |  15,80 |  −3,29 |        0,68 |     0,21 | 0,32 |      +6,2 |
| SARIMAX              | 11,43 |  15,68 |  −2,13 |        0,75 |     0,24 | 0,37 |      +5,8 |
| XGBoost              | 11,56 |  15,93 |  −4,67 |        0,75 |     0,24 | 0,37 |      +4,8 |
| ARIMA                | 11,77 |  16,14 |  −3,04 |        0,69 |     0,18 | 0,28 |      +3,0 |
| Persistenza          | 12,13 |  17,09 |   0,07 |        0,63 |     0,63 | 0,63 |      +0,0 |
| Prophet              | 13,91 |  19,01 |  −4,29 |        0,33 |     0,02 | 0,03 |     −14,6 |
| Seasonal naive (t−7) | 19,35 |  25,44 |  −0,44 |        0,27 |     0,27 | 0,27 |     −59,4 |

#### IT2212A – San Vitaliano Scuola Marconi  (giorni di test: 284, superamenti osservati: 31)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| XGBoost              |  9,43 |  16,07 |  −3,05 |        0,67 |     0,26 | 0,37 |      +6,5 |
| ARIMA                |  9,45 |  17,18 |  −1,48 |        0,62 |     0,42 | 0,50 |      +6,3 |
| SARIMAX              |  9,56 |  17,26 |  −1,70 |        0,68 |     0,48 | 0,57 |      +5,2 |
| Prophet + lag        |  9,62 |  15,50 |  −2,45 |        0,62 |     0,42 | 0,50 |      +4,6 |
| Persistenza          | 10,08 |  18,02 |   0,02 |        0,52 |     0,52 | 0,52 |      +0,0 |
| Prophet              | 13,37 |  18,86 |  −1,95 |        0,20 |     0,13 | 0,16 |     −32,5 |
| Seasonal naive (t−7) | 17,69 |  25,35 |   1,16 |        0,26 |     0,29 | 0,27 |     −75,4 |

#### IT1491A – Napoli NA07 Ferrovia  (giorni di test: 211, superamenti osservati: 10)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| XGBoost              |  6,73 |  12,38 |  −1,77 |           – |     0,00 |    – |     +20,1 |
| ARIMA                |  7,13 |  12,49 |  −1,26 |           – |     0,00 |    – |     +15,3 |
| SARIMAX              |  7,15 |  12,47 |  −1,02 |           – |     0,00 |    – |     +15,0 |
| Prophet + lag        |  7,17 |  12,26 |  −1,06 |        0,00 |     0,00 |    – |     +14,8 |
| Prophet              |  8,29 |  12,14 |  −0,57 |        1,00 |     0,10 | 0,18 |      +1,5 |
| Persistenza          |  8,42 |  16,04 |   0,07 |        0,20 |     0,20 | 0,20 |      +0,0 |
| Seasonal naive (t−7) | 10,92 |  18,18 |   0,63 |        0,00 |     0,00 |    – |     −29,8 |

#### IT2227A – Avellino Scuola Alighieri  (giorni di test: 279, superamenti osservati: 2)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| SARIMAX              |  3,68 |   6,00 |  −0,96 |           – |     0,00 |    – |      +2,2 |
| ARIMA                |  3,68 |   6,00 |  −0,98 |           – |     0,00 |    – |      +2,0 |
| Persistenza          |  3,76 |   6,08 |  −0,00 |        0,50 |     0,50 | 0,50 |      +0,0 |
| Prophet + lag        |  3,80 |   6,18 |  −1,55 |           – |     0,00 |    – |      −1,1 |
| XGBoost              |  3,81 |   5,88 |   0,60 |           – |     0,00 |    – |      −1,3 |
| Prophet              |  5,55 |   8,35 |  −0,69 |           – |     0,00 |    – |     −47,8 |
| Seasonal naive (t−7) |  7,24 |  10,99 |   0,29 |        0,00 |     0,00 |    – |     −92,7 |

### Modelli selezionati

- **IT2211A** (Acerra Zona Industriale): ARIMA(1, 0, 3) · SARIMAX(1, 0, 1)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **IT2212A** (San Vitaliano Scuola Marconi): ARIMA(2, 0, 2) · SARIMAX(1, 0, 1)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **IT1491A** (Napoli NA07 Ferrovia): ARIMA(1, 0, 1) · SARIMAX(1, 0, 1)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **IT2227A** (Avellino Scuola Alighieri): ARIMA(1, 1, 3) · SARIMAX(2, 1, 2)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **XGBoost**: un modello globale su 35 stazioni fisse (26.479 righe di training), 108 alberi (early stopping sugli ultimi 90 giorni di training)

### XGBoost – le 10 feature più importanti (gain)

| feature    |   importance |
|:-----------|-------------:|
| y_last     |        0,460 |
| y_lag1     |        0,193 |
| PM2.5_lag1 |        0,030 |
| reg_lag1   |        0,027 |
| y_roll30   |        0,026 |
| Stazione   |        0,026 |
| holiday    |        0,025 |
| NO2_lag1   |        0,024 |
| doy_sin    |        0,023 |
| reg_lag2   |        0,023 |

## PM2.5

Unità: µg/m³. Soglia superamenti: 25 µg/m³ (limite giornaliero dal 2030, Dir. UE 2024/2881).

### Complessivo (4 stazioni aggregate)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| XGBoost              |  4,22 |   8,41 |  −1,10 |        0,76 |     0,31 | 0,44 |      +7,3 |
| ARIMA                |  4,22 |   9,45 |  −0,68 |        0,78 |     0,43 | 0,55 |      +7,3 |
| SARIMAX              |  4,25 |   9,27 |  −0,56 |        0,71 |     0,46 | 0,56 |      +6,6 |
| Prophet + lag        |  4,38 |   8,16 |  −0,50 |        0,68 |     0,47 | 0,56 |      +3,8 |
| Persistenza          |  4,55 |  10,39 |   0,09 |        0,60 |     0,60 | 0,60 |      +0,0 |
| Prophet              |  5,90 |   9,12 |   0,29 |        0,57 |     0,47 | 0,51 |     −29,6 |
| Seasonal naive (t−7) |  7,45 |  12,96 |   0,39 |        0,40 |     0,43 | 0,41 |     −63,6 |

### Per stazione

#### IT2211A – Acerra Zona Industriale  (giorni di test: 252, superamenti osservati: 22)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| SARIMAX              |  4,44 |   6,09 |  −0,91 |        0,71 |     0,23 | 0,34 |      +7,7 |
| XGBoost              |  4,47 |   6,24 |  −1,16 |        0,80 |     0,18 | 0,30 |      +7,2 |
| ARIMA                |  4,54 |   6,31 |  −1,23 |        1,00 |     0,14 | 0,24 |      +5,7 |
| Persistenza          |  4,81 |   8,09 |   0,31 |        0,48 |     0,50 | 0,49 |      +0,0 |
| Prophet + lag        |  4,93 |   6,82 |  −1,20 |        0,33 |     0,05 | 0,08 |      −2,5 |
| Prophet              |  5,98 |   8,17 |  −1,59 |           – |     0,00 |    – |     −24,2 |
| Seasonal naive (t−7) |  7,84 |  11,43 |   0,18 |        0,29 |     0,27 | 0,28 |     −62,9 |

#### IT2212A – San Vitaliano Scuola Marconi  (giorni di test: 284, superamenti osservati: 52)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| XGBoost              |  5,92 |  11,74 |  −2,24 |        0,74 |     0,44 | 0,55 |      +5,4 |
| ARIMA                |  6,00 |  14,28 |  −0,61 |        0,77 |     0,63 | 0,69 |      +4,1 |
| Prophet + lag        |  6,14 |  10,96 |  −0,44 |        0,71 |     0,75 | 0,73 |      +1,9 |
| SARIMAX              |  6,20 |  13,96 |  −0,23 |        0,71 |     0,65 | 0,68 |      +0,9 |
| Persistenza          |  6,26 |  13,75 |   0,02 |        0,67 |     0,67 | 0,67 |      +0,0 |
| Prophet              |  8,73 |  12,66 |   1,50 |        0,56 |     0,81 | 0,66 |     −39,5 |
| Seasonal naive (t−7) | 11,02 |  17,49 |   0,91 |        0,47 |     0,54 | 0,50 |     −76,0 |

#### IT1491A – Napoli NA07 Ferrovia  (giorni di test: 210, superamenti osservati: 17)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| SARIMAX              |  4,30 |   9,95 |  −0,89 |        0,75 |     0,18 | 0,29 |     +18,0 |
| ARIMA                |  4,33 |  10,04 |  −0,94 |        0,75 |     0,18 | 0,29 |     +17,5 |
| Prophet + lag        |  4,38 |   9,72 |  −0,00 |        0,60 |     0,18 | 0,27 |     +16,4 |
| XGBoost              |  4,42 |  10,04 |  −1,64 |        1,00 |     0,06 | 0,11 |     +15,7 |
| Persistenza          |  5,24 |  13,45 |   0,02 |        0,53 |     0,53 | 0,53 |      +0,0 |
| Prophet              |  5,35 |   9,01 |   0,48 |        1,00 |     0,06 | 0,11 |      −2,1 |
| Seasonal naive (t−7) |  6,78 |  14,45 |   0,13 |        0,29 |     0,29 | 0,29 |     −29,3 |

#### IT2227A – Avellino Scuola Alighieri  (giorni di test: 279, superamenti osservati: 0)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| ARIMA                |  2,04 |   2,94 |  −0,04 |           – |        – |    – |      +0,6 |
| SARIMAX              |  2,05 |   3,01 |  −0,34 |           – |        – |    – |      +0,1 |
| Persistenza          |  2,06 |   3,01 |  −0,00 |           – |        – |    – |      +0,0 |
| Prophet + lag        |  2,08 |   3,02 |  −0,32 |           – |        – |    – |      −1,1 |
| XGBoost              |  2,11 |   2,92 |   0,51 |           – |        – |    – |      −2,6 |
| Prophet              |  3,35 |   4,57 |   0,62 |           – |        – |    – |     −63,0 |
| Seasonal naive (t−7) |  3,95 |   5,51 |   0,25 |           – |        – |    – |     −92,3 |

### Modelli selezionati

- **IT2211A** (Acerra Zona Industriale): ARIMA(1, 0, 3) · SARIMAX(1, 0, 1)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **IT2212A** (San Vitaliano Scuola Marconi): ARIMA(2, 0, 2) · SARIMAX(1, 0, 1)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **IT1491A** (Napoli NA07 Ferrovia): ARIMA(3, 0, 3) · SARIMAX(1, 0, 1)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **IT2227A** (Avellino Scuola Alighieri): ARIMA(2, 0, 2) · SARIMAX(2, 1, 2)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **XGBoost**: un modello globale su 34 stazioni fisse (26.573 righe di training), 100 alberi (early stopping sugli ultimi 90 giorni di training)

### XGBoost – le 10 feature più importanti (gain)

| feature   |   importance |
|:----------|-------------:|
| y_last    |        0,477 |
| y_lag1    |        0,196 |
| y_roll30  |        0,035 |
| Stazione  |        0,033 |
| reg_lag1  |        0,027 |
| month     |        0,024 |
| y_roll7   |        0,022 |
| reg_lag2  |        0,021 |
| NO2_lag1  |        0,021 |
| doy_cos   |        0,020 |

## NO2

Unità: µg/m³. Soglia superamenti: 50 µg/m³ (limite giornaliero dal 2030, Dir. UE 2024/2881).

### Complessivo (4 stazioni aggregate)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| XGBoost              |  7,00 |   9,08 |  −1,43 |        0,67 |     0,30 | 0,41 |      +9,1 |
| SARIMAX              |  7,27 |   9,45 |   0,35 |        0,53 |     0,49 | 0,51 |      +5,5 |
| ARIMA                |  7,32 |   9,49 |  −0,05 |        0,64 |     0,49 | 0,56 |      +5,0 |
| Persistenza          |  7,70 |  10,30 |  −0,05 |        0,60 |     0,60 | 0,60 |      +0,0 |
| Prophet + lag        |  8,15 |  10,81 |   0,08 |        0,45 |     0,41 | 0,43 |      −5,8 |
| Prophet              | 10,17 |  13,15 |   1,07 |        0,30 |     0,33 | 0,31 |     −32,1 |
| Seasonal naive (t−7) | 10,43 |  13,83 |  −0,00 |        0,43 |     0,43 | 0,43 |     −35,4 |

### Per stazione

#### IT2211A – Acerra Zona Industriale  (giorni di test: 275, superamenti osservati: 55)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| SARIMAX              |  6,83 |   8,69 |  −1,11 |        0,65 |     0,44 | 0,52 |      +2,1 |
| XGBoost              |  6,93 |   8,89 |  −3,11 |        0,75 |     0,11 | 0,19 |      +0,6 |
| ARIMA                |  6,95 |   8,86 |  −1,98 |        0,82 |     0,33 | 0,47 |      +0,3 |
| Persistenza          |  6,97 |   9,38 |  −0,10 |        0,64 |     0,64 | 0,64 |      +0,0 |
| Prophet + lag        |  9,32 |  11,69 |  −6,96 |        0,33 |     0,02 | 0,03 |     −33,7 |
| Seasonal naive (t−7) |  9,72 |  12,63 |  −0,48 |        0,36 |     0,33 | 0,34 |     −39,4 |
| Prophet              | 10,75 |  13,49 |  −7,78 |           – |     0,00 |    – |     −54,2 |

#### IT2212A – San Vitaliano Scuola Marconi  (giorni di test: 266, superamenti osservati: 28)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| XGBoost              |  7,67 |   9,74 |  −0,86 |        0,63 |     0,43 | 0,51 |     +10,7 |
| SARIMAX              |  7,92 |  10,15 |  −0,43 |        0,50 |     0,39 | 0,44 |      +7,7 |
| Prophet + lag        |  8,01 |  10,69 |   0,98 |        0,53 |     0,64 | 0,58 |      +6,7 |
| ARIMA                |  8,09 |  10,43 |  −0,82 |        0,59 |     0,36 | 0,44 |      +5,7 |
| Persistenza          |  8,58 |  11,40 |  −0,02 |        0,54 |     0,54 | 0,54 |      +0,0 |
| Prophet              | 10,27 |  13,17 |   1,70 |        0,39 |     0,68 | 0,49 |     −19,6 |
| Seasonal naive (t−7) | 11,93 |  15,55 |   0,48 |        0,41 |     0,46 | 0,43 |     −38,9 |

#### IT1491A – Napoli NA07 Ferrovia  (giorni di test: 269, superamenti osservati: 78)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| XGBoost              |  8,69 |  10,89 |  −0,75 |        0,67 |     0,38 | 0,49 |      +9,9 |
| ARIMA                |  9,13 |  11,47 |   3,17 |        0,60 |     0,67 | 0,63 |      +5,4 |
| Persistenza          |  9,65 |  12,42 |  −0,06 |        0,62 |     0,62 | 0,62 |      +0,0 |
| SARIMAX              |  9,67 |  11,95 |   3,25 |        0,49 |     0,58 | 0,53 |      −0,3 |
| Prophet + lag        | 10,36 |  13,31 |   4,97 |        0,43 |     0,62 | 0,51 |      −7,4 |
| Seasonal naive (t−7) | 12,35 |  16,37 |  −0,09 |        0,49 |     0,49 | 0,49 |     −28,0 |
| Prophet              | 13,36 |  16,54 |   7,49 |        0,26 |     0,45 | 0,33 |     −38,5 |

#### IT2227A – Avellino Scuola Alighieri  (giorni di test: 277, superamenti osservati: 1)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| SARIMAX              |  4,76 |   6,15 |  −0,28 |           – |     0,00 |    – |     +16,2 |
| XGBoost              |  4,79 |   6,21 |  −0,94 |           – |     0,00 |    – |     +15,7 |
| Prophet + lag        |  4,96 |   6,39 |   1,48 |           – |     0,00 |    – |     +12,6 |
| ARIMA                |  5,17 |   6,55 |  −0,51 |           – |     0,00 |    – |      +9,0 |
| Persistenza          |  5,68 |   7,35 |  −0,02 |        0,00 |     0,00 |    – |      +0,0 |
| Prophet              |  6,40 |   8,08 |   3,00 |           – |     0,00 |    – |     −12,7 |
| Seasonal naive (t−7) |  7,82 |   9,99 |   0,09 |        0,00 |     0,00 |    – |     −37,7 |

### Modelli selezionati

- **IT2211A** (Acerra Zona Industriale): ARIMA(2, 1, 3) · SARIMAX(2, 1, 2)x(0, 1, 1, 7) + Fourier(K=2) + festività
- **IT2212A** (San Vitaliano Scuola Marconi): ARIMA(3, 1, 3) · SARIMAX(2, 1, 2)x(0, 1, 1, 7) + Fourier(K=2) + festività
- **IT1491A** (Napoli NA07 Ferrovia): ARIMA(1, 0, 3) · SARIMAX(1, 0, 1)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **IT2227A** (Avellino Scuola Alighieri): ARIMA(2, 1, 2) · SARIMAX(2, 1, 2)x(0, 1, 1, 7) + Fourier(K=2) + festività
- **XGBoost**: un modello globale su 37 stazioni fisse (29.511 righe di training), 178 alberi (early stopping sugli ultimi 90 giorni di training)

### XGBoost – le 10 feature più importanti (gain)

| feature   |   importance |
|:----------|-------------:|
| y_last    |        0,566 |
| y_roll30  |        0,141 |
| y_roll7   |        0,096 |
| y_lag1    |        0,070 |
| Stazione  |        0,024 |
| dow       |        0,013 |
| holiday   |        0,012 |
| month     |        0,010 |
| reg_lag2  |        0,010 |
| doy_cos   |        0,009 |

## O3 (massimo giornaliero della media mobile su 8 ore)

Unità: µg/m³. Soglia superamenti: 120 µg/m³ (valore obiettivo sul max media 8 h, D.Lgs. 155/2010).

### Complessivo (4 stazioni aggregate)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| ARIMA                |  9,37 |  12,38 |  −1,43 |        0,67 |     0,48 | 0,56 |      +5,4 |
| XGBoost              |  9,46 |  12,45 |  −2,26 |        0,72 |     0,45 | 0,55 |      +4,4 |
| SARIMAX              |  9,60 |  12,61 |  −2,74 |        0,67 |     0,42 | 0,52 |      +3,0 |
| Persistenza          |  9,90 |  13,51 |   0,03 |        0,61 |     0,61 | 0,61 |      +0,0 |
| Prophet + lag        | 10,61 |  14,01 |  −2,51 |        0,49 |     0,39 | 0,43 |      −7,2 |
| Seasonal naive (t−7) | 14,89 |  19,52 |  −0,77 |        0,42 |     0,39 | 0,41 |     −50,4 |
| Prophet              | 16,27 |  21,27 |  −3,92 |        0,34 |     0,30 | 0,32 |     −64,4 |

### Per stazione

#### IT2276A – Ariano Irpino Villa Comunale  (giorni di test: 262, superamenti osservati: 33)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| SARIMAX              |  6,68 |   8,51 |   0,12 |        0,62 |     0,55 | 0,58 |      +8,2 |
| ARIMA                |  6,84 |   8,75 |  −0,20 |        0,64 |     0,55 | 0,59 |      +6,0 |
| XGBoost              |  6,89 |   8,69 |  −1,69 |        0,65 |     0,39 | 0,49 |      +5,3 |
| Persistenza          |  7,28 |   9,53 |   0,05 |        0,58 |     0,58 | 0,58 |      +0,0 |
| Prophet + lag        |  7,55 |  10,00 |   3,96 |        0,46 |     0,85 | 0,60 |      −3,8 |
| Seasonal naive (t−7) | 10,61 |  14,06 |   0,05 |        0,39 |     0,36 | 0,38 |     −45,8 |
| Prophet              | 11,56 |  15,41 |   8,75 |        0,33 |     0,85 | 0,48 |     −58,8 |

#### IT2277A – Napoli Parco Virgiliano  (giorni di test: 226, superamenti osservati: 20)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| ARIMA                |  8,50 |  11,55 |  −0,12 |        0,71 |     0,50 | 0,59 |      +5,3 |
| SARIMAX              |  8,56 |  11,81 |  −0,89 |        0,67 |     0,40 | 0,50 |      +4,7 |
| Persistenza          |  8,98 |  12,19 |   0,01 |        0,55 |     0,55 | 0,55 |      +0,0 |
| XGBoost              |  9,02 |  11,96 |  −0,51 |        0,60 |     0,45 | 0,51 |      −0,5 |
| Prophet + lag        |  9,31 |  12,41 |  −1,39 |        0,50 |     0,25 | 0,33 |      −3,7 |
| Seasonal naive (t−7) | 14,95 |  19,65 |  −1,24 |        0,29 |     0,25 | 0,27 |     −66,6 |
| Prophet              | 18,87 |  24,88 |  −1,51 |           – |     0,00 |    – |    −110,1 |

#### IT2212A – San Vitaliano Scuola Marconi  (giorni di test: 271, superamenti osservati: 47)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| ARIMA                | 11,36 |  14,79 |  −1,33 |        0,68 |     0,45 | 0,54 |      +6,4 |
| XGBoost              | 11,66 |  15,02 |  −4,14 |        0,83 |     0,51 | 0,63 |      +4,0 |
| SARIMAX              | 11,87 |  14,96 |  −4,85 |        0,74 |     0,36 | 0,49 |      +2,2 |
| Persistenza          | 12,14 |  16,37 |  −0,00 |        0,68 |     0,68 | 0,68 |      +0,0 |
| Prophet + lag        | 13,00 |  16,58 |  −6,48 |        0,64 |     0,15 | 0,24 |      −7,1 |
| Prophet              | 16,84 |  21,59 | −10,43 |        0,50 |     0,06 | 0,11 |     −38,7 |
| Seasonal naive (t−7) | 18,26 |  22,79 |  −0,84 |        0,49 |     0,49 | 0,49 |     −50,4 |

#### IT2223A – Volla via Filichito  (giorni di test: 259, superamenti osservati: 2)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| XGBoost              | 10,15 |  13,10 |  −2,41 |           – |     0,00 |    – |      +7,9 |
| ARIMA                | 10,61 |  13,38 |  −3,94 |           – |     0,00 |    – |      +3,7 |
| Persistenza          | 11,02 |  14,68 |   0,08 |        0,00 |     0,00 |    – |      +0,0 |
| SARIMAX              | 11,09 |  13,98 |  −5,03 |           – |     0,00 |    – |      −0,6 |
| Prophet + lag        | 12,36 |  15,77 |  −5,88 |           – |     0,00 |    – |     −12,1 |
| Seasonal naive (t−7) | 15,65 |  20,44 |  −1,11 |           – |     0,00 |    – |     −41,9 |
| Prophet              | 18,19 |  22,59 | −12,04 |           – |     0,00 |    – |     −65,0 |

### Modelli selezionati

- **IT2276A** (Ariano Irpino Villa Comunale): ARIMA(2, 1, 1) · SARIMAX(1, 0, 1)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **IT2277A** (Napoli Parco Virgiliano): ARIMA(2, 0, 1) · SARIMAX(1, 1, 1)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **IT2212A** (San Vitaliano Scuola Marconi): ARIMA(3, 1, 3) · SARIMAX(1, 0, 1)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **IT2223A** (Volla via Filichito): ARIMA(3, 0, 2) · SARIMAX(1, 0, 1)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **XGBoost**: un modello globale su 21 stazioni fisse (16.139 righe di training), 134 alberi (early stopping sugli ultimi 90 giorni di training)

### XGBoost – le 10 feature più importanti (gain)

| feature   |   importance |
|:----------|-------------:|
| y_last    |        0,509 |
| y_roll7   |        0,163 |
| y_lag1    |        0,075 |
| y_roll30  |        0,053 |
| doy_sin   |        0,016 |
| month     |        0,016 |
| y_lag7    |        0,015 |
| doy_cos   |        0,015 |
| Stazione  |        0,014 |
| y_lag2    |        0,014 |

## SO2

Unità: µg/m³. Soglia superamenti: 50 µg/m³ (limite giornaliero dal 2030, Dir. UE 2024/2881).

### Complessivo (4 stazioni aggregate)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| Persistenza          |  0,31 |   0,67 |  −0,00 |           – |        – |    – |      +0,0 |
| SARIMAX              |  0,31 |   0,62 |  −0,04 |           – |        – |    – |      −2,2 |
| XGBoost              |  0,33 |   0,61 |  −0,07 |           – |        – |    – |      −7,7 |
| ARIMA                |  0,36 |   0,65 |  −0,12 |           – |        – |    – |     −18,5 |
| Prophet + lag        |  0,46 |   0,69 |  −0,11 |           – |        – |    – |     −48,9 |
| Seasonal naive (t−7) |  0,46 |   0,85 |  −0,05 |           – |        – |    – |     −49,4 |
| Prophet              |  1,26 |   1,60 |  −0,30 |           – |        – |    – |    −311,3 |

### Per stazione

#### IT2211A – Acerra Zona Industriale  (giorni di test: 277, superamenti osservati: 0)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| Persistenza          |  0,17 |   0,24 |  −0,01 |           – |        – |    – |      +0,0 |
| SARIMAX              |  0,17 |   0,25 |  −0,02 |           – |        – |    – |      −0,3 |
| Prophet + lag        |  0,29 |   0,39 |  −0,19 |           – |        – |    – |     −73,2 |
| XGBoost              |  0,30 |   0,43 |  −0,20 |           – |        – |    – |     −78,0 |
| Seasonal naive (t−7) |  0,30 |   0,45 |  −0,05 |           – |        – |    – |     −81,6 |
| ARIMA                |  0,39 |   0,48 |  −0,37 |           – |        – |    – |    −133,2 |
| Prophet              |  0,81 |   1,03 |  −0,49 |           – |        – |    – |    −385,4 |

#### IT2212A – San Vitaliano Scuola Marconi  (giorni di test: 267, superamenti osservati: 0)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| XGBoost              |  0,36 |   0,82 |  −0,03 |           – |        – |    – |      +7,3 |
| Persistenza          |  0,39 |   0,99 |   0,00 |           – |        – |    – |      +0,0 |
| ARIMA                |  0,41 |   0,90 |  −0,04 |           – |        – |    – |      −4,3 |
| SARIMAX              |  0,43 |   0,90 |  −0,08 |           – |        – |    – |      −8,4 |
| Seasonal naive (t−7) |  0,56 |   1,10 |  −0,07 |           – |        – |    – |     −42,7 |
| Prophet + lag        |  0,60 |   0,96 |   0,27 |           – |        – |    – |     −52,2 |
| Prophet              |  1,60 |   1,96 |   1,21 |           – |        – |    – |    −306,7 |

#### IT1491A – Napoli NA07 Ferrovia  (giorni di test: 262, superamenti osservati: 0)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| XGBoost              |  0,53 |   0,75 |  −0,06 |           – |        – |    – |      +6,5 |
| ARIMA                |  0,54 |   0,76 |  −0,04 |           – |        – |    – |      +4,2 |
| SARIMAX              |  0,56 |   0,77 |  −0,03 |           – |        – |    – |      +1,6 |
| Persistenza          |  0,57 |   0,82 |  −0,00 |           – |        – |    – |      +0,0 |
| Prophet + lag        |  0,62 |   0,84 |  −0,24 |           – |        – |    – |     −10,1 |
| Seasonal naive (t−7) |  0,80 |   1,16 |  −0,02 |           – |        – |    – |     −41,3 |
| Prophet              |  1,72 |   2,06 |  −1,08 |           – |        – |    – |    −203,1 |

#### IT2273A – Polla Area tritovagliatore  (giorni di test: 239, superamenti osservati: 0)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| ARIMA                |  0,09 |   0,18 |  −0,02 |           – |        – |    – |      +1,1 |
| Persistenza          |  0,09 |   0,20 |  −0,00 |           – |        – |    – |      +0,0 |
| SARIMAX              |  0,09 |   0,19 |  −0,02 |           – |        – |    – |      −2,6 |
| XGBoost              |  0,11 |   0,19 |   0,01 |           – |        – |    – |     −28,3 |
| Seasonal naive (t−7) |  0,15 |   0,33 |  −0,05 |           – |        – |    – |     −69,6 |
| Prophet + lag        |  0,31 |   0,38 |  −0,31 |           – |        – |    – |    −250,3 |
| Prophet              |  0,92 |   0,98 |  −0,92 |           – |        – |    – |    −928,2 |

### Modelli selezionati

- **IT2211A** (Acerra Zona Industriale): ARIMA(2, 0, 3) · SARIMAX(2, 1, 2)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **IT2212A** (San Vitaliano Scuola Marconi): ARIMA(1, 1, 2) · SARIMAX(2, 1, 2)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **IT1491A** (Napoli NA07 Ferrovia): ARIMA(3, 0, 3) · SARIMAX(2, 0, 1)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **IT2273A** (Polla Area tritovagliatore): ARIMA(3, 0, 3) · SARIMAX(2, 1, 2)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **XGBoost**: un modello globale su 15 stazioni fisse (11.023 righe di training), 181 alberi (early stopping sugli ultimi 90 giorni di training)

### XGBoost – le 10 feature più importanti (gain)

| feature   |   importance |
|:----------|-------------:|
| y_last    |        0,539 |
| y_lag1    |        0,199 |
| y_roll7   |        0,050 |
| y_lag2    |        0,033 |
| y_roll30  |        0,019 |
| Stazione  |        0,016 |
| reg_lag1  |        0,013 |
| y_lag3    |        0,012 |
| month     |        0,012 |
| y_lag7    |        0,011 |

## CO

Unità: mg/m³. Soglia superamenti: 4 mg/m³ (limite giornaliero dal 2030, Dir. UE 2024/2881).

### Complessivo (4 stazioni aggregate)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| SARIMAX              | 0,114 |  0,163 | −0,004 |           – |        – |    – |      +6,9 |
| XGBoost              | 0,114 |  0,168 | −0,014 |           – |        – |    – |      +6,9 |
| ARIMA                | 0,116 |  0,168 | −0,001 |           – |        – |    – |      +5,5 |
| Prophet + lag        | 0,122 |  0,173 |  0,029 |           – |        – |    – |      +0,2 |
| Persistenza          | 0,122 |  0,186 | −0,001 |           – |        – |    – |      +0,0 |
| Seasonal naive (t−7) | 0,192 |  0,283 |  0,000 |           – |        – |    – |     −57,2 |
| Prophet              | 0,195 |  0,256 |  0,078 |           – |        – |    – |     −59,6 |

### Per stazione

#### IT2211A – Acerra Zona Industriale  (giorni di test: 232, superamenti osservati: 0)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| Persistenza          | 0,077 |  0,117 | −0,001 |           – |        – |    – |      +0,0 |
| XGBoost              | 0,078 |  0,110 | −0,009 |           – |        – |    – |      −1,5 |
| SARIMAX              | 0,080 |  0,113 | −0,002 |           – |        – |    – |      −3,9 |
| ARIMA                | 0,083 |  0,117 | −0,001 |           – |        – |    – |      −7,2 |
| Prophet + lag        | 0,090 |  0,119 |  0,017 |           – |        – |    – |     −16,3 |
| Seasonal naive (t−7) | 0,159 |  0,221 | −0,007 |           – |        – |    – |    −106,1 |
| Prophet              | 0,201 |  0,245 |  0,074 |           – |        – |    – |    −161,1 |

#### IT0898A – Napoli NA06 Museo Nazionale  (giorni di test: 287, superamenti osservati: 0)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| SARIMAX              | 0,130 |  0,184 |  0,001 |           – |        – |    – |     +15,5 |
| XGBoost              | 0,136 |  0,196 | −0,014 |           – |        – |    – |     +12,0 |
| Prophet + lag        | 0,136 |  0,200 |  0,048 |           – |        – |    – |     +11,5 |
| ARIMA                | 0,139 |  0,197 |  0,001 |           – |        – |    – |     +10,1 |
| Persistenza          | 0,154 |  0,228 | −0,001 |           – |        – |    – |      +0,0 |
| Prophet              | 0,184 |  0,259 |  0,103 |           – |        – |    – |     −19,6 |
| Seasonal naive (t−7) | 0,194 |  0,297 |  0,006 |           – |        – |    – |     −26,1 |

#### IT1491A – Napoli NA07 Ferrovia  (giorni di test: 273, superamenti osservati: 0)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| XGBoost              | 0,142 |  0,198 | −0,038 |           – |        – |    – |      +7,8 |
| ARIMA                | 0,144 |  0,196 | −0,009 |           – |        – |    – |      +6,5 |
| Prophet + lag        | 0,145 |  0,197 | −0,006 |           – |        – |    – |      +6,0 |
| SARIMAX              | 0,145 |  0,196 | −0,026 |           – |        – |    – |      +5,4 |
| Persistenza          | 0,154 |  0,217 | −0,001 |           – |        – |    – |      +0,0 |
| Prophet              | 0,197 |  0,262 |  0,007 |           – |        – |    – |     −27,9 |
| Seasonal naive (t−7) | 0,256 |  0,356 |  0,003 |           – |        – |    – |     −66,5 |

#### IT2227A – Avellino Scuola Alighieri  (giorni di test: 272, superamenti osservati: 0)

|                      |   MAE |   RMSE |   Bias |   Precision |   Recall |   F1 |   Skill % |
|:---------------------|------:|-------:|-------:|------------:|---------:|-----:|----------:|
| ARIMA                | 0,091 |  0,137 |  0,004 |           – |        – |    – |      +4,6 |
| XGBoost              | 0,094 |  0,142 |  0,006 |           – |        – |    – |      +2,3 |
| SARIMAX              | 0,094 |  0,138 |  0,010 |           – |        – |    – |      +1,9 |
| Persistenza          | 0,096 |  0,146 |  0,000 |           – |        – |    – |      +0,0 |
| Prophet + lag        | 0,112 |  0,156 |  0,055 |           – |        – |    – |     −16,7 |
| Seasonal naive (t−7) | 0,155 |  0,229 | −0,001 |           – |        – |    – |     −61,2 |
| Prophet              | 0,201 |  0,256 |  0,128 |           – |        – |    – |    −109,0 |

### Modelli selezionati

- **IT2211A** (Acerra Zona Industriale): ARIMA(3, 1, 3) · SARIMAX(2, 1, 2)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **IT0898A** (Napoli NA06 Museo Nazionale): ARIMA(2, 0, 2) · SARIMAX(1, 0, 1)x(0, 1, 1, 7) + Fourier(K=2) + festività
- **IT1491A** (Napoli NA07 Ferrovia): ARIMA(2, 0, 2) · SARIMAX(1, 0, 1)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **IT2227A** (Avellino Scuola Alighieri): ARIMA(2, 0, 2) · SARIMAX(1, 0, 1)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **XGBoost**: un modello globale su 21 stazioni fisse (16.555 righe di training), 127 alberi (early stopping sugli ultimi 90 giorni di training)

### XGBoost – le 10 feature più importanti (gain)

| feature   |   importance |
|:----------|-------------:|
| y_last    |        0,483 |
| y_lag1    |        0,226 |
| y_roll7   |        0,054 |
| y_roll30  |        0,026 |
| month     |        0,019 |
| reg_lag1  |        0,018 |
| reg_lag2  |        0,015 |
| doy_cos   |        0,015 |
| doy_sin   |        0,015 |
| holiday   |        0,014 |

## C6H6

Unità: µg/m³. Soglia superamenti: nessuna (solo limite annuale (5 µg/m³)).

### Complessivo (4 stazioni aggregate)

|                      |   MAE |   RMSE |   Bias |   Skill % |
|:---------------------|------:|-------:|-------:|----------:|
| XGBoost              |  0,29 |   0,48 |   0,00 |     +12,1 |
| ARIMA                |  0,29 |   0,48 |  −0,00 |     +10,2 |
| SARIMAX              |  0,30 |   0,49 |   0,01 |      +6,6 |
| Persistenza          |  0,33 |   0,55 |  −0,00 |      +0,0 |
| Prophet + lag        |  0,35 |   0,53 |   0,02 |      −7,1 |
| Seasonal naive (t−7) |  0,45 |   0,74 |   0,02 |     −37,8 |
| Prophet              |  0,56 |   0,80 |   0,20 |     −72,2 |

### Per stazione

#### IT2211A – Acerra Zona Industriale  (giorni di test: 259)

|                      |   MAE |   RMSE |   Bias |   Skill % |
|:---------------------|------:|-------:|-------:|----------:|
| XGBoost              |  0,23 |   0,39 |   0,00 |     +11,6 |
| SARIMAX              |  0,23 |   0,39 |  −0,01 |      +8,0 |
| ARIMA                |  0,24 |   0,40 |   0,00 |      +7,6 |
| Prophet + lag        |  0,25 |   0,43 |  −0,07 |      +3,6 |
| Persistenza          |  0,26 |   0,43 |  −0,00 |      +0,0 |
| Prophet              |  0,35 |   0,55 |   0,09 |     −35,3 |
| Seasonal naive (t−7) |  0,38 |   0,68 |   0,02 |     −47,6 |

#### IT2212A – San Vitaliano Scuola Marconi  (giorni di test: 260)

|                      |   MAE |   RMSE |   Bias |   Skill % |
|:---------------------|------:|-------:|-------:|----------:|
| ARIMA                |  0,35 |   0,63 |  −0,00 |      +8,9 |
| XGBoost              |  0,36 |   0,62 |   0,02 |      +7,7 |
| SARIMAX              |  0,37 |   0,63 |  −0,00 |      +4,5 |
| Persistenza          |  0,39 |   0,70 |  −0,00 |      +0,0 |
| Prophet + lag        |  0,46 |   0,69 |  −0,09 |     −18,5 |
| Seasonal naive (t−7) |  0,57 |   0,97 |   0,03 |     −47,0 |
| Prophet              |  0,72 |   1,06 |  −0,07 |     −86,5 |

#### IT1491A – Napoli NA07 Ferrovia  (giorni di test: 225)

|                      |   MAE |   RMSE |   Bias |   Skill % |
|:---------------------|------:|-------:|-------:|----------:|
| XGBoost              |  0,28 |   0,39 |  −0,04 |     +17,2 |
| ARIMA                |  0,29 |   0,41 |  −0,03 |     +14,2 |
| SARIMAX              |  0,30 |   0,41 |  −0,03 |     +11,5 |
| Persistenza          |  0,33 |   0,48 |  −0,00 |      +0,0 |
| Prophet + lag        |  0,38 |   0,49 |   0,19 |     −12,6 |
| Seasonal naive (t−7) |  0,38 |   0,51 |  −0,01 |     −12,7 |
| Prophet              |  0,70 |   0,85 |   0,58 |    −107,9 |

#### IT2227A – Avellino Scuola Alighieri  (giorni di test: 253)

|                      |   MAE |   RMSE |   Bias |   Skill % |
|:---------------------|------:|-------:|-------:|----------:|
| XGBoost              |  0,28 |   0,45 |   0,02 |     +13,2 |
| ARIMA                |  0,29 |   0,46 |   0,02 |     +10,3 |
| SARIMAX              |  0,31 |   0,47 |   0,08 |      +3,7 |
| Prophet + lag        |  0,31 |   0,49 |   0,05 |      +3,4 |
| Persistenza          |  0,33 |   0,53 |   0,00 |      +0,0 |
| Seasonal naive (t−7) |  0,46 |   0,70 |   0,03 |     −41,5 |
| Prophet              |  0,49 |   0,67 |   0,27 |     −51,6 |

### Modelli selezionati

- **IT2211A** (Acerra Zona Industriale): ARIMA(3, 1, 3) · SARIMAX(2, 1, 2)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **IT2212A** (San Vitaliano Scuola Marconi): ARIMA(2, 1, 3) · SARIMAX(2, 1, 2)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **IT1491A** (Napoli NA07 Ferrovia): ARIMA(1, 0, 3) · SARIMAX(1, 1, 1)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **IT2227A** (Avellino Scuola Alighieri): ARIMA(2, 0, 2) · SARIMAX(1, 0, 1)x(1, 0, 1, 7) + Fourier(K=2) + festività
- **XGBoost**: un modello globale su 25 stazioni fisse (18.570 righe di training), 142 alberi (early stopping sugli ultimi 90 giorni di training)

### XGBoost – le 10 feature più importanti (gain)

| feature    |   importance |
|:-----------|-------------:|
| y_last     |        0,526 |
| y_lag1     |        0,148 |
| y_roll7    |        0,076 |
| y_roll30   |        0,053 |
| month      |        0,017 |
| doy_cos    |        0,016 |
| reg_lag1   |        0,016 |
| y_lag7     |        0,015 |
| PM2.5_lag1 |        0,014 |
| holiday    |        0,014 |

## Orizzonte di previsione da 1 a 7 giorni

Il giorno t viene previsto con i soli dati fino a t−h. **h = 1** è la previsione a un giorno delle sezioni precedenti; **h = 7** è la previsione emessa una settimana prima.

- **Persistenza a h giorni**: ultimo valore noto al giorno t−h. **Seasonal naive**: valore di t−7 (noto per ogni h ≤ 7).
- **ARIMA / SARIMAX**: stessi parametri della previsione a un giorno; da ogni origine si prevedono ricorsivamente i 7 giorni successivi.
- **Prophet**: non usa i dati recenti, quindi la sua previsione è la stessa a ogni orizzonte.
- **Prophet + lag** e **XGBoost**: un modello *diretto* per ogni h, stimato con le feature spostate di h giorni (es. per h = 7 il valore più recente usato è quello di t−7).
- Campione comune: giorni-stazione con osservazione valida e previsione a tutti gli orizzonti. La skill è sempre rispetto alla persistenza **allo stesso orizzonte**.

### Sintesi: skill vs persistenza a 7 giorni (%)

|       |   Persistenza |   Seasonal naive (t−7) |   ARIMA |   SARIMAX |   Prophet |   Prophet + lag |   XGBoost |
|:------|--------------:|-----------------------:|--------:|----------:|----------:|----------------:|----------:|
| PM10  |          +0,0 |                   +0,0 |   +26,4 |     +26,7 |     +26,8 |           +25,1 |     +22,5 |
| PM2.5 |          +0,0 |                   +0,0 |   +26,1 |     +25,2 |     +23,5 |           +19,9 |     +22,5 |
| NO2   |          +0,0 |                   +0,0 |   +15,3 |     +14,4 |      +6,7 |            −1,7 |     +19,2 |
| O3    |          +0,0 |                   +0,0 |   +13,9 |     +14,8 |      −4,6 |            −1,8 |     +17,8 |
| SO2   |          +0,0 |                   +0,0 |   −39,0 |      −5,1 |    −159,2 |           −79,9 |     −63,7 |
| CO    |          +0,0 |                   +0,0 |   +15,3 |     +18,9 |      +2,2 |            −0,7 |     +18,1 |
| C6H6  |          +0,0 |                   +0,0 |   +23,1 |     +13,6 |     −20,0 |           −14,2 |     +15,2 |

### Sintesi: errore a 1 e a 7 giorni

|   Inquinante |   Unità |   MAE persistenza h=1 |   MAE persistenza h=7 |   Migliore a h=7 |   MAE migliore h=7 |   MAE Prophet |
|-------------:|--------:|----------------------:|----------------------:|-----------------:|-------------------:|--------------:|
|         PM10 |   µg/m³ |                  8,57 |                 14,18 |          Prophet |              10,37 |         10,37 |
|        PM2.5 |   µg/m³ |                  4,55 |                  7,71 |            ARIMA |               5,70 |          5,90 |
|          NO2 |   µg/m³ |                  7,70 |                 10,91 |          XGBoost |               8,82 |         10,17 |
|           O3 |   µg/m³ |                  9,90 |                 15,56 |          XGBoost |              12,80 |         16,27 |
|          SO2 |   µg/m³ |                  0,31 |                  0,49 |      Persistenza |               0,49 |          1,26 |
|           CO |   mg/m³ |                 0,122 |                 0,200 |          SARIMAX |              0,162 |         0,195 |
|         C6H6 |   µg/m³ |                  0,33 |                  0,47 |            ARIMA |               0,36 |          0,56 |

### PM10 – errore per orizzonte

MAE (µg/m³); 1.038 giorni-stazione per ogni orizzonte.

|                      |   h=1 |   h=2 |   h=3 |   h=4 |   h=5 |   h=6 |   h=7 |
|:---------------------|------:|------:|------:|------:|------:|------:|------:|
| Persistenza          |  8,57 | 10,98 | 12,46 | 13,28 | 14,57 | 14,81 | 14,18 |
| Seasonal naive (t−7) | 13,93 | 14,20 | 14,42 | 14,43 | 14,63 | 14,37 | 14,18 |
| ARIMA                |  8,02 |  9,52 | 10,07 | 10,29 | 10,40 | 10,47 | 10,43 |
| SARIMAX              |  7,97 |  9,51 | 10,06 | 10,26 | 10,36 | 10,39 | 10,40 |
| Prophet              | 10,37 | 10,37 | 10,37 | 10,37 | 10,37 | 10,37 | 10,37 |
| Prophet + lag        |  8,01 |  9,50 | 10,14 | 10,35 | 10,54 | 10,58 | 10,61 |
| XGBoost              |  7,91 |  9,96 | 10,29 | 10,61 | 11,01 | 10,81 | 10,99 |

Skill vs persistenza (%)

|                      |   h=1 |   h=2 |   h=3 |   h=4 |   h=5 |   h=6 |   h=7 |
|:---------------------|------:|------:|------:|------:|------:|------:|------:|
| Seasonal naive (t−7) | −62,6 | −29,3 | −15,7 |  −8,6 |  −0,4 |  +3,0 |  +0,0 |
| ARIMA                |  +6,4 | +13,3 | +19,2 | +22,6 | +28,6 | +29,3 | +26,4 |
| SARIMAX              |  +7,0 | +13,4 | +19,3 | +22,7 | +28,9 | +29,8 | +26,7 |
| Prophet              | −21,1 |  +5,5 | +16,8 | +21,9 | +28,8 | +30,0 | +26,8 |
| Prophet + lag        |  +6,5 | +13,5 | +18,6 | +22,1 | +27,7 | +28,6 | +25,1 |
| XGBoost              |  +7,7 |  +9,3 | +17,4 | +20,1 | +24,4 | +27,0 | +22,5 |

F1 sui superamenti (> 50 µg/m³)

|                      |   h=1 |   h=2 |   h=3 |   h=4 |   h=5 |   h=6 |   h=7 |
|:---------------------|------:|------:|------:|------:|------:|------:|------:|
| Persistenza          |  0,55 |  0,40 |  0,33 |  0,28 |  0,24 |  0,17 |  0,20 |
| Seasonal naive (t−7) |  0,24 |  0,22 |  0,22 |  0,21 |  0,21 |  0,20 |  0,20 |
| ARIMA                |  0,34 |  0,05 |  0,02 |     – |  0,02 |     – |     – |
| SARIMAX              |  0,41 |  0,11 |  0,08 |  0,08 |  0,11 |  0,09 |  0,11 |
| Prophet              |  0,09 |  0,09 |  0,09 |  0,09 |  0,09 |  0,09 |  0,09 |
| Prophet + lag        |  0,35 |  0,11 |  0,06 |  0,09 |  0,09 |  0,09 |  0,09 |
| XGBoost              |  0,34 |  0,02 |  0,04 |  0,04 |  0,04 |  0,04 |  0,04 |

XGBoost a 7 giorni – le 5 feature più importanti: `y_roll30` (0,184), `Stazione` (0,182), `month` (0,060), `holiday` (0,060), `doy_cos` (0,054)

### PM2.5 – errore per orizzonte

MAE (µg/m³); 1.025 giorni-stazione per ogni orizzonte.

|                      |   h=1 |   h=2 |   h=3 |   h=4 |   h=5 |   h=6 |   h=7 |
|:---------------------|------:|------:|------:|------:|------:|------:|------:|
| Persistenza          |  4,55 |  5,97 |  6,77 |  7,19 |  8,05 |  8,24 |  7,71 |
| Seasonal naive (t−7) |  7,45 |  7,72 |  7,82 |  7,80 |  7,95 |  7,91 |  7,71 |
| ARIMA                |  4,22 |  5,15 |  5,52 |  5,64 |  5,75 |  5,75 |  5,70 |
| SARIMAX              |  4,25 |  5,21 |  5,54 |  5,68 |  5,76 |  5,77 |  5,77 |
| Prophet              |  5,90 |  5,90 |  5,90 |  5,90 |  5,90 |  5,90 |  5,90 |
| Prophet + lag        |  4,38 |  5,26 |  5,66 |  5,81 |  5,97 |  6,06 |  6,18 |
| XGBoost              |  4,22 |  5,34 |  5,86 |  5,96 |  5,94 |  5,93 |  5,98 |

Skill vs persistenza (%)

|                      |   h=1 |   h=2 |   h=3 |   h=4 |   h=5 |   h=6 |   h=7 |
|:---------------------|------:|------:|------:|------:|------:|------:|------:|
| Seasonal naive (t−7) | −63,6 | −29,2 | −15,4 |  −8,5 |  +1,2 |  +4,0 |  +0,0 |
| ARIMA                |  +7,3 | +13,8 | +18,4 | +21,6 | +28,6 | +30,2 | +26,1 |
| SARIMAX              |  +6,6 | +12,8 | +18,3 | +21,0 | +28,4 | +29,9 | +25,2 |
| Prophet              | −29,6 |  +1,3 | +12,9 | +18,0 | +26,7 | +28,4 | +23,5 |
| Prophet + lag        |  +3,8 | +11,9 | +16,4 | +19,2 | +25,9 | +26,4 | +19,9 |
| XGBoost              |  +7,3 | +10,6 | +13,5 | +17,2 | +26,1 | +28,0 | +22,5 |

F1 sui superamenti (> 25 µg/m³)

|                      |   h=1 |   h=2 |   h=3 |   h=4 |   h=5 |   h=6 |   h=7 |
|:---------------------|------:|------:|------:|------:|------:|------:|------:|
| Persistenza          |  0,60 |  0,49 |  0,44 |  0,40 |  0,37 |  0,34 |  0,39 |
| Seasonal naive (t−7) |  0,41 |  0,42 |  0,40 |  0,40 |  0,39 |  0,38 |  0,39 |
| ARIMA                |  0,55 |  0,42 |  0,33 |  0,35 |  0,34 |  0,33 |  0,27 |
| SARIMAX              |  0,56 |  0,48 |  0,47 |  0,47 |  0,44 |  0,44 |  0,44 |
| Prophet              |  0,51 |  0,51 |  0,51 |  0,51 |  0,51 |  0,51 |  0,51 |
| Prophet + lag        |  0,56 |  0,52 |  0,53 |  0,52 |  0,52 |  0,49 |  0,49 |
| XGBoost              |  0,44 |  0,19 |  0,28 |  0,29 |  0,26 |  0,26 |  0,30 |

XGBoost a 7 giorni – le 5 feature più importanti: `y_roll30` (0,257), `Stazione` (0,123), `doy_cos` (0,079), `month` (0,063), `y_roll7` (0,062)

### NO2 – errore per orizzonte

MAE (µg/m³); 1.087 giorni-stazione per ogni orizzonte.

|                      |   h=1 |   h=2 |   h=3 |   h=4 |   h=5 |   h=6 |   h=7 |
|:---------------------|------:|------:|------:|------:|------:|------:|------:|
| Persistenza          |  7,70 |  9,81 | 10,67 | 11,04 | 11,43 | 11,72 | 10,91 |
| Seasonal naive (t−7) | 10,43 | 10,57 | 10,81 | 10,89 | 10,81 | 10,84 | 10,91 |
| ARIMA                |  7,32 |  8,75 |  9,12 |  9,18 |  9,20 |  9,24 |  9,24 |
| SARIMAX              |  7,27 |  8,65 |  9,10 |  9,24 |  9,32 |  9,35 |  9,34 |
| Prophet              | 10,17 | 10,17 | 10,17 | 10,17 | 10,17 | 10,17 | 10,17 |
| Prophet + lag        |  8,15 | 10,03 | 10,69 | 10,86 | 10,85 | 11,00 | 11,09 |
| XGBoost              |  7,00 |  8,23 |  8,64 |  8,82 |  8,77 |  8,88 |  8,82 |

Skill vs persistenza (%)

|                      |   h=1 |   h=2 |   h=3 |   h=4 |   h=5 |   h=6 |   h=7 |
|:---------------------|------:|------:|------:|------:|------:|------:|------:|
| Seasonal naive (t−7) | −35,4 |  −7,8 |  −1,3 |  +1,4 |  +5,5 |  +7,5 |  +0,0 |
| ARIMA                |  +5,0 | +10,8 | +14,5 | +16,8 | +19,5 | +21,1 | +15,3 |
| SARIMAX              |  +5,5 | +11,8 | +14,7 | +16,3 | +18,4 | +20,2 | +14,4 |
| Prophet              | −32,1 |  −3,7 |  +4,6 |  +7,9 | +11,0 | +13,2 |  +6,7 |
| Prophet + lag        |  −5,8 |  −2,3 |  −0,2 |  +1,6 |  +5,1 |  +6,1 |  −1,7 |
| XGBoost              |  +9,1 | +16,1 | +19,0 | +20,1 | +23,3 | +24,2 | +19,2 |

F1 sui superamenti (> 50 µg/m³)

|                      |   h=1 |   h=2 |   h=3 |   h=4 |   h=5 |   h=6 |   h=7 |
|:---------------------|------:|------:|------:|------:|------:|------:|------:|
| Persistenza          |  0,60 |  0,47 |  0,43 |  0,39 |  0,35 |  0,36 |  0,39 |
| Seasonal naive (t−7) |  0,43 |  0,41 |  0,40 |  0,40 |  0,40 |  0,41 |  0,39 |
| ARIMA                |  0,56 |  0,27 |  0,16 |  0,15 |  0,09 |  0,06 |  0,01 |
| SARIMAX              |  0,51 |  0,27 |  0,23 |  0,21 |  0,23 |  0,24 |  0,24 |
| Prophet              |  0,31 |  0,31 |  0,31 |  0,31 |  0,31 |  0,31 |  0,31 |
| Prophet + lag        |  0,43 |  0,33 |  0,32 |  0,32 |  0,31 |  0,33 |  0,32 |
| XGBoost              |  0,41 |  0,14 |  0,03 |     – |     – |  0,01 |  0,02 |

XGBoost a 7 giorni – le 5 feature più importanti: `Stazione` (0,382), `y_roll30` (0,347), `y_roll7` (0,039), `y_last` (0,036), `doy_cos` (0,023)

### O3 – errore per orizzonte

MAE (µg/m³); 1.018 giorni-stazione per ogni orizzonte.

|                      |   h=1 |   h=2 |   h=3 |   h=4 |   h=5 |   h=6 |   h=7 |
|:---------------------|------:|------:|------:|------:|------:|------:|------:|
| Persistenza          |  9,90 | 12,79 | 13,24 | 14,02 | 14,57 | 14,92 | 15,56 |
| Seasonal naive (t−7) | 14,89 | 15,23 | 15,56 | 15,64 | 15,64 | 15,56 | 15,56 |
| ARIMA                |  9,37 | 11,29 | 11,86 | 12,43 | 12,75 | 13,08 | 13,40 |
| SARIMAX              |  9,60 | 11,62 | 12,18 | 12,68 | 12,91 | 13,11 | 13,25 |
| Prophet              | 16,27 | 16,27 | 16,27 | 16,27 | 16,27 | 16,27 | 16,27 |
| Prophet + lag        | 10,61 | 13,31 | 14,62 | 15,04 | 15,38 | 15,78 | 15,84 |
| XGBoost              |  9,46 | 11,34 | 11,74 | 12,27 | 12,70 | 12,75 | 12,80 |

Skill vs persistenza (%)

|                      |   h=1 |   h=2 |   h=3 |   h=4 |   h=5 |   h=6 |   h=7 |
|:---------------------|------:|------:|------:|------:|------:|------:|------:|
| Seasonal naive (t−7) | −50,4 | −19,1 | −17,6 | −11,6 |  −7,4 |  −4,3 |  +0,0 |
| ARIMA                |  +5,4 | +11,7 | +10,4 | +11,3 | +12,5 | +12,3 | +13,9 |
| SARIMAX              |  +3,0 |  +9,1 |  +8,0 |  +9,5 | +11,4 | +12,1 | +14,8 |
| Prophet              | −64,4 | −27,3 | −23,0 | −16,1 | −11,7 |  −9,1 |  −4,6 |
| Prophet + lag        |  −7,2 |  −4,1 | −10,5 |  −7,3 |  −5,6 |  −5,8 |  −1,8 |
| XGBoost              |  +4,4 | +11,3 | +11,3 | +12,4 | +12,8 | +14,6 | +17,8 |

F1 sui superamenti (> 120 µg/m³)

|                      |   h=1 |   h=2 |   h=3 |   h=4 |   h=5 |   h=6 |   h=7 |
|:---------------------|------:|------:|------:|------:|------:|------:|------:|
| Persistenza          |  0,61 |  0,45 |  0,41 |  0,39 |  0,41 |  0,41 |  0,40 |
| Seasonal naive (t−7) |  0,41 |  0,39 |  0,38 |  0,37 |  0,37 |  0,40 |  0,40 |
| ARIMA                |  0,56 |  0,39 |  0,33 |  0,30 |  0,36 |  0,31 |  0,27 |
| SARIMAX              |  0,52 |  0,23 |  0,23 |  0,28 |  0,29 |  0,29 |  0,30 |
| Prophet              |  0,32 |  0,32 |  0,32 |  0,32 |  0,32 |  0,32 |  0,32 |
| Prophet + lag        |  0,43 |  0,33 |  0,29 |  0,31 |  0,31 |  0,31 |  0,30 |
| XGBoost              |  0,55 |  0,16 |  0,13 |  0,09 |  0,08 |  0,07 |  0,04 |

XGBoost a 7 giorni – le 5 feature più importanti: `y_roll30` (0,380), `y_roll7` (0,192), `y_last` (0,096), `month` (0,070), `Stazione` (0,029)

### SO2 – errore per orizzonte

MAE (µg/m³); 1.045 giorni-stazione per ogni orizzonte.

|                      |   h=1 |   h=2 |   h=3 |   h=4 |   h=5 |   h=6 |   h=7 |
|:---------------------|------:|------:|------:|------:|------:|------:|------:|
| Persistenza          |  0,31 |  0,40 |  0,44 |  0,47 |  0,49 |  0,50 |  0,49 |
| Seasonal naive (t−7) |  0,46 |  0,47 |  0,47 |  0,48 |  0,48 |  0,49 |  0,49 |
| ARIMA                |  0,36 |  0,49 |  0,54 |  0,58 |  0,62 |  0,66 |  0,68 |
| SARIMAX              |  0,31 |  0,40 |  0,43 |  0,46 |  0,49 |  0,50 |  0,51 |
| Prophet              |  1,26 |  1,26 |  1,26 |  1,26 |  1,26 |  1,26 |  1,26 |
| Prophet + lag        |  0,46 |  0,62 |  0,72 |  0,77 |  0,82 |  0,85 |  0,88 |
| XGBoost              |  0,33 |  0,45 |  0,60 |  0,64 |  0,67 |  0,70 |  0,80 |

Skill vs persistenza (%)

|                      |    h=1 |    h=2 |    h=3 |    h=4 |    h=5 |    h=6 |    h=7 |
|:---------------------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| Seasonal naive (t−7) |  −49,4 |  −16,0 |   −7,6 |   −1,4 |   +1,2 |   +1,6 |   +0,0 |
| ARIMA                |  −18,5 |  −22,9 |  −23,5 |  −23,1 |  −26,7 |  −32,2 |  −39,0 |
| SARIMAX              |   −2,2 |   +1,0 |   +1,3 |   +2,1 |   +1,0 |   −1,5 |   −5,1 |
| Prophet              | −311,3 | −213,6 | −187,2 | −166,1 | −157,1 | −154,7 | −159,2 |
| Prophet + lag        |  −48,9 |  −53,6 |  −62,7 |  −62,6 |  −67,1 |  −71,6 |  −79,9 |
| XGBoost              |   −7,7 |  −11,0 |  −37,0 |  −34,8 |  −36,6 |  −40,5 |  −63,7 |

XGBoost a 7 giorni – le 5 feature più importanti: `y_last` (0,412), `y_lag7` (0,091), `y_roll7` (0,091), `Stazione` (0,090), `y_roll30` (0,056)

### CO – errore per orizzonte

MAE (mg/m³); 1.064 giorni-stazione per ogni orizzonte.

|                      |   h=1 |   h=2 |   h=3 |   h=4 |   h=5 |   h=6 |   h=7 |
|:---------------------|------:|------:|------:|------:|------:|------:|------:|
| Persistenza          | 0,122 | 0,166 | 0,180 | 0,195 | 0,214 | 0,213 | 0,200 |
| Seasonal naive (t−7) | 0,192 | 0,195 | 0,197 | 0,200 | 0,200 | 0,201 | 0,200 |
| ARIMA                | 0,116 | 0,148 | 0,157 | 0,162 | 0,168 | 0,169 | 0,169 |
| SARIMAX              | 0,114 | 0,144 | 0,152 | 0,156 | 0,161 | 0,162 | 0,162 |
| Prophet              | 0,195 | 0,195 | 0,195 | 0,195 | 0,195 | 0,195 | 0,195 |
| Prophet + lag        | 0,122 | 0,163 | 0,179 | 0,189 | 0,195 | 0,197 | 0,201 |
| XGBoost              | 0,114 | 0,146 | 0,155 | 0,158 | 0,160 | 0,163 | 0,164 |

Skill vs persistenza (%)

|                      |   h=1 |   h=2 |   h=3 |   h=4 |   h=5 |   h=6 |   h=7 |
|:---------------------|------:|------:|------:|------:|------:|------:|------:|
| Seasonal naive (t−7) | −57,2 | −17,4 |  −9,0 |  −2,6 |  +6,8 |  +5,7 |  +0,0 |
| ARIMA                |  +5,5 | +11,0 | +12,8 | +16,6 | +21,5 | +20,4 | +15,3 |
| SARIMAX              |  +6,9 | +13,3 | +15,8 | +19,6 | +25,0 | +23,9 | +18,9 |
| Prophet              | −59,6 | −17,5 |  −8,4 |  −0,4 |  +8,8 |  +8,2 |  +2,2 |
| Prophet + lag        |  +0,2 |  +1,7 |  +0,7 |  +2,9 |  +9,1 |  +7,6 |  −0,7 |
| XGBoost              |  +6,9 | +12,1 | +14,2 | +18,8 | +25,4 | +23,5 | +18,1 |

XGBoost a 7 giorni – le 5 feature più importanti: `y_roll30` (0,308), `y_roll7` (0,167), `y_last` (0,145), `doy_cos` (0,048), `month` (0,047)

### C6H6 – errore per orizzonte

MAE (µg/m³); 997 giorni-stazione per ogni orizzonte.

|                      |   h=1 |   h=2 |   h=3 |   h=4 |   h=5 |   h=6 |   h=7 |
|:---------------------|------:|------:|------:|------:|------:|------:|------:|
| Persistenza          |  0,33 |  0,40 |  0,43 |  0,46 |  0,49 |  0,49 |  0,47 |
| Seasonal naive (t−7) |  0,45 |  0,46 |  0,47 |  0,47 |  0,48 |  0,47 |  0,47 |
| ARIMA                |  0,29 |  0,34 |  0,35 |  0,36 |  0,36 |  0,36 |  0,36 |
| SARIMAX              |  0,30 |  0,36 |  0,38 |  0,39 |  0,40 |  0,40 |  0,40 |
| Prophet              |  0,56 |  0,56 |  0,56 |  0,56 |  0,56 |  0,56 |  0,56 |
| Prophet + lag        |  0,35 |  0,44 |  0,49 |  0,51 |  0,52 |  0,53 |  0,53 |
| XGBoost              |  0,29 |  0,34 |  0,36 |  0,37 |  0,39 |  0,40 |  0,40 |

Skill vs persistenza (%)

|                      |   h=1 |   h=2 |   h=3 |   h=4 |   h=5 |   h=6 |   h=7 |
|:---------------------|------:|------:|------:|------:|------:|------:|------:|
| Seasonal naive (t−7) | −37,8 | −12,8 |  −8,7 |  −2,6 |  +3,1 |  +3,9 |  +0,0 |
| ARIMA                | +10,2 | +16,3 | +19,2 | +22,4 | +26,2 | +25,8 | +23,1 |
| SARIMAX              |  +6,6 | +11,4 | +12,2 | +14,5 | +19,0 | +17,8 | +13,6 |
| Prophet              | −72,2 | −38,8 | −30,4 | −22,4 | −13,5 | −14,3 | −20,0 |
| Prophet + lag        |  −7,1 | −10,1 | −13,9 | −10,8 |  −5,0 |  −8,3 | −14,2 |
| XGBoost              | +12,1 | +16,9 | +17,0 | +19,6 | +20,0 | +19,2 | +15,2 |

XGBoost a 7 giorni – le 5 feature più importanti: `y_roll30` (0,407), `y_roll7` (0,204), `y_last` (0,052), `doy_cos` (0,043), `month` (0,035)

## Previsione per i 7 giorni dopo l'ultimo dato (mar 27 ott – lun 2 nov 2026)

Emessa con i dati fino al 2026-10-26. Per ogni inquinante si usa il modello con il MAE medio (h = 1..7) più basso nel periodo di test. Tra parentesi l'intervallo empirico all'80%: 10° e 90° percentile degli errori di test a quell'orizzonte (in scala log, 4 stazioni insieme). L'intervallo è tarato sul 2026, quindi la copertura dell'80% vale in media su quel periodo, non per ogni giorno.

### PM10 – modello: SARIMAX (µg/m³, soglia 50)

|             |   Acerra Zona Industriale |   San Vitaliano Scuola Marconi |   Napoli NA07 Ferrovia |   Avellino Scuola Alighieri |
|:------------|--------------------------:|-------------------------------:|-----------------------:|----------------------------:|
| ultimo dato |                10 (26/10) |                     11 (26/10) |             17 (25/10) |                   6 (26/10) |
| mar 27 ott  |                18 (10–27) |                      16 (9–24) |             23 (13–34) |                    8 (4–12) |
| mer 28 ott  |                21 (11–35) |                     20 (10–32) |             24 (12–39) |                    8 (4–14) |
| gio 29 ott  |                24 (11–39) |                     23 (11–37) |             25 (12–40) |                    9 (4–15) |
| ven 30 ott  |                25 (12–41) |                     24 (11–40) |             25 (12–42) |                    9 (4–16) |
| sab 31 ott  |                25 (12–41) |                     25 (12–42) |             25 (12–42) |                   10 (4–16) |
| dom 1 nov   |                23 (11–38) |                     25 (12–42) |             24 (11–40) |                   10 (4–16) |
| lun 2 nov   |                24 (11–40) |                     26 (12–43) |             25 (12–42) |                   10 (4–16) |

### PM2.5 – modello: ARIMA (µg/m³, soglia 25)

|             |   Acerra Zona Industriale |   San Vitaliano Scuola Marconi |   Napoli NA07 Ferrovia |   Avellino Scuola Alighieri |
|:------------|--------------------------:|-------------------------------:|-----------------------:|----------------------------:|
| ultimo dato |                 4 (26/10) |                      6 (26/10) |             16 (20/10) |                   4 (26/10) |
| mar 27 ott  |                  6 (3–10) |                       7 (4–11) |              15 (9–23) |                     5 (2–7) |
| mer 28 ott  |                  8 (4–14) |                       9 (4–15) |              15 (7–25) |                    6 (2–10) |
| gio 29 ott  |                  9 (4–16) |                      10 (4–17) |              15 (7–26) |                    6 (2–11) |
| ven 30 ott  |                  9 (4–17) |                      10 (4–18) |              15 (7–27) |                    6 (3–11) |
| sab 31 ott  |                 10 (4–17) |                      10 (4–18) |              15 (7–27) |                    6 (3–12) |
| dom 1 nov   |                 10 (4–18) |                      11 (5–19) |              15 (7–27) |                    6 (3–12) |
| lun 2 nov   |                 10 (4–19) |                      11 (5–19) |              15 (7–27) |                    6 (3–12) |

### NO2 – modello: XGBoost (µg/m³, soglia 50)

|             |   Acerra Zona Industriale |   San Vitaliano Scuola Marconi |   Napoli NA07 Ferrovia |   Avellino Scuola Alighieri |
|:------------|--------------------------:|-------------------------------:|-----------------------:|----------------------------:|
| ultimo dato |                36 (26/10) |                     25 (26/10) |             35 (25/10) |                  21 (26/10) |
| mar 27 ott  |                38 (26–53) |                     26 (18–37) |             42 (29–59) |                  21 (15–30) |
| mer 28 ott  |                41 (26–60) |                     31 (20–45) |             49 (31–70) |                  24 (15–35) |
| gio 29 ott  |                40 (25–60) |                     32 (20–47) |             46 (29–68) |                  24 (15–35) |
| ven 30 ott  |                38 (24–59) |                     32 (21–50) |             44 (28–68) |                  22 (14–34) |
| sab 31 ott  |                42 (26–63) |                     33 (21–50) |             47 (29–71) |                  23 (14–35) |
| dom 1 nov   |                35 (21–52) |                     32 (20–48) |             42 (26–63) |                  21 (12–31) |
| lun 2 nov   |                40 (25–61) |                     33 (20–50) |             47 (29–71) |                  24 (14–36) |

### O3 – modello: XGBoost (µg/m³, soglia 120)

|             |   Ariano Irpino Villa Comunale |   Napoli Parco Virgiliano |   San Vitaliano Scuola Marconi |   Volla via Filichito |
|:------------|-------------------------------:|--------------------------:|-------------------------------:|----------------------:|
| ultimo dato |                     71 (26/10) |                54 (26/10) |                     72 (26/10) |            42 (26/10) |
| mar 27 ott  |                     73 (62–88) |                62 (53–74) |                     67 (57–81) |            38 (32–45) |
| mer 28 ott  |                     75 (62–94) |                69 (57–86) |                     66 (55–83) |            34 (28–43) |
| gio 29 ott  |                     77 (64–98) |                70 (58–89) |                     65 (54–83) |            36 (29–45) |
| ven 30 ott  |                     76 (63–98) |                74 (61–95) |                     64 (53–83) |            36 (30–47) |
| sab 31 ott  |                     76 (62–99) |                73 (60–96) |                     66 (54–86) |            39 (32–51) |
| dom 1 nov   |                    77 (63–102) |                73 (60–97) |                     67 (54–88) |            45 (36–59) |
| lun 2 nov   |                     75 (61–99) |                71 (58–94) |                     61 (49–80) |            41 (33–54) |

### SO2 – modello: Persistenza (µg/m³, soglia 50)

|             |   Acerra Zona Industriale |   San Vitaliano Scuola Marconi |   Napoli NA07 Ferrovia |   Polla Area tritovagliatore |
|:------------|--------------------------:|-------------------------------:|-----------------------:|-----------------------------:|
| ultimo dato |               5,3 (26/10) |                    3,1 (26/10) |            1,3 (25/10) |                  1,5 (26/10) |
| mar 27 ott  |             5,3 (4,6–6,1) |                  3,1 (2,6–3,6) |          1,3 (1,0–1,5) |                1,5 (1,2–1,8) |
| mer 28 ott  |             5,3 (4,4–6,4) |                  3,1 (2,5–3,7) |          1,3 (0,9–1,6) |                1,5 (1,1–1,9) |
| gio 29 ott  |             5,3 (4,3–6,6) |                  3,1 (2,4–3,9) |          1,3 (0,9–1,7) |                1,5 (1,1–2,0) |
| ven 30 ott  |             5,3 (4,2–6,8) |                  3,1 (2,4–4,0) |          1,3 (0,9–1,8) |                1,5 (1,0–2,0) |
| sab 31 ott  |             5,3 (4,2–6,9) |                  3,1 (2,3–4,1) |          1,3 (0,9–1,8) |                1,5 (1,0–2,1) |
| dom 1 nov   |             5,3 (4,2–7,0) |                  3,1 (2,4–4,1) |          1,3 (0,9–1,8) |                1,5 (1,0–2,1) |
| lun 2 nov   |             5,3 (4,3–6,8) |                  3,1 (2,4–4,1) |          1,3 (0,9–1,8) |                1,5 (1,1–2,1) |

### CO – modello: SARIMAX (mg/m³, soglia 4)

|             |   Acerra Zona Industriale |   Napoli NA06 Museo Nazionale |   Napoli NA07 Ferrovia |   Avellino Scuola Alighieri |
|:------------|--------------------------:|------------------------------:|-----------------------:|----------------------------:|
| ultimo dato |              0,43 (26/10) |                  0,46 (26/10) |           0,73 (25/10) |                0,56 (26/10) |
| mar 27 ott  |          0,40 (0,25–0,55) |              0,57 (0,39–0,74) |       0,76 (0,57–0,95) |            0,53 (0,36–0,69) |
| mer 28 ott  |          0,39 (0,19–0,58) |              0,68 (0,44–0,91) |       0,77 (0,52–1,01) |            0,48 (0,27–0,69) |
| gio 29 ott  |          0,39 (0,19–0,60) |              0,79 (0,53–1,06) |       0,77 (0,52–1,04) |            0,49 (0,27–0,71) |
| ven 30 ott  |          0,39 (0,18–0,61) |              0,77 (0,50–1,05) |       0,78 (0,51–1,06) |            0,50 (0,27–0,74) |
| sab 31 ott  |          0,39 (0,17–0,62) |              0,72 (0,45–1,00) |       0,78 (0,50–1,08) |            0,52 (0,28–0,77) |
| dom 1 nov   |          0,39 (0,16–0,62) |              0,50 (0,26–0,76) |       0,67 (0,41–0,96) |            0,49 (0,25–0,74) |
| lun 2 nov   |          0,40 (0,17–0,63) |              0,87 (0,56–1,18) |       0,79 (0,50–1,09) |            0,54 (0,29–0,79) |

### C6H6 – modello: ARIMA (µg/m³)

|             |   Acerra Zona Industriale |   San Vitaliano Scuola Marconi |   Napoli NA07 Ferrovia |   Avellino Scuola Alighieri |
|:------------|--------------------------:|-------------------------------:|-----------------------:|----------------------------:|
| ultimo dato |               0,3 (26/10) |                    0,8 (26/10) |            0,8 (25/10) |                 0,3 (21/10) |
| mar 27 ott  |             0,5 (0,2–0,8) |                  0,8 (0,4–1,2) |          1,1 (0,6–1,5) |               0,9 (0,5–1,3) |
| mer 28 ott  |             0,5 (0,1–0,8) |                  0,9 (0,4–1,4) |          1,1 (0,6–1,7) |               0,9 (0,4–1,4) |
| gio 29 ott  |             0,4 (0,1–0,8) |                  0,9 (0,4–1,4) |          1,1 (0,6–1,7) |               0,9 (0,4–1,4) |
| ven 30 ott  |             0,5 (0,1–0,9) |                  0,9 (0,4–1,4) |          1,1 (0,6–1,7) |               0,9 (0,4–1,4) |
| sab 31 ott  |             0,5 (0,1–1,0) |                  0,9 (0,4–1,4) |          1,1 (0,5–1,7) |               0,9 (0,4–1,4) |
| dom 1 nov   |             0,5 (0,1–0,9) |                  0,9 (0,4–1,5) |          1,1 (0,5–1,7) |               0,9 (0,4–1,4) |
| lun 2 nov   |             0,5 (0,1–0,9) |                  0,9 (0,4–1,4) |          1,1 (0,5–1,6) |               0,9 (0,4–1,4) |

## Note sui modelli

- **Prophet**: stagionalità annuale + settimanale + festività italiane (un modello per stazione).
- **Prophet + lag**: come sopra + regressori `ultimo valore osservato` e `media 7 giorni` (fino a t-1).
- **Feature XGBoost**: `y_*` = il target stesso (lag, medie mobili, ultimo valore osservato); `reg_*` = media regionale del target; `<inquinante>_lag1` = altri inquinanti il giorno prima.
