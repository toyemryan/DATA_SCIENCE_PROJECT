# Progetto 1 – Analisi della qualità dell'aria in Campania

Analisi descrittiva, classificazione, clustering e forecasting sui dati orari
di qualità dell'aria della rete di monitoraggio ARPA Campania.

## Dati
- **Fonte:** portale open data ARPAC, dataset "Dati RQA Orari Validati"
  (https://dati.arpacampania.it/dataset/dati-rqa-giornalieri-validati)
- **Licenza:** CC BY 4.0
- **Periodo:** marzo 2023 – ottobre 2026 (file mensili, scaricati il 29/10/2026)
- **Contenuto:** misure orarie di 7 inquinanti (PM10, PM2,5, NO2, O3, CO, SO2, C6H6)
- I dati non sono inclusi nel repository (dimensione > 100 MB).
  Il file concatenato va posizionato in data/raw/qualita_aria_campania_validati.csv.

## Struttura
| Cartella | Contenuto |
|---|---|
| 1_DATA_ANALYSIS/ | ETL e analisi descrittiva |
| 2_CLASSIFICATION/ | Classificazione |
| 3_CLUSTERING/ | Clustering delle stazioni |
| 4_FORECASTING/ | Previsione delle serie temporali |
| data/ | Dati grezzi e processati (esclusi da Git) |

Ogni cartella contiene codice/ (notebook) e igures/ (grafici).
