#  Data Science – Progetti

![anno](https://img.shields.io/badge/A.A.-2026--2027-lightgrey?style=flat)
![linguaggio](https://img.shields.io/badge/-Python%203.12-blue?style=flat&logo=python&logoColor=white)
![jupyter](https://img.shields.io/badge/-Jupyter-orange?style=flat&logo=jupyter&logoColor=white)
![pandas](https://img.shields.io/badge/-pandas-150458?style=flat&logo=pandas&logoColor=white)
![networkx](https://img.shields.io/badge/-NetworkX-2C5BB4?style=flat)
![latex](https://img.shields.io/badge/-LaTeX-008080?style=flat&logo=latex&logoColor=white)
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

Raccolta dei progetti del corso di **Data Science**, Laurea Magistrale in Ingegneria Informatica e dell'Automazione,
**Università Politecnica delle Marche**, Dipartimento di Ingegneria dell'Informazione – A.A. 2026-2027.

## 👥 Autori

| Studenti | |
|---|---|
| Ryan Parfait Toyem Tezem | Simone Murazzo |
| Andrea Scalzi | Maicol Lanni |

**Docente:** Prof. Domenico Ursino  
**Dottorando:** Christopher Buratti

##  Progetti

| # | Cartella | Argomento | Stato |
|---|---|---|---|
| 1️ | [`1_GLOBAL_ANALYSIS/`](1_GLOBAL_ANALYSIS/) | Qualità dell'aria in Campania: analisi descrittiva, classificazione, clustering e forecasting | ✅ completato |
| 2️ | [`2_SOCIAL_NETWORK_ANALYSIS/`](2_SOCIAL_NETWORK_ANALYSIS/) | Social Network Analysis della rete degli sviluppatori GitHub: centralità, comunità, link prediction | 🚧 in corso |
| 3️ | [`3_CHATBOT/`](3_CHATBOT/) | Chatbot | 🚧 in corso |

##  Organizzazione di un progetto

Tutti i progetti seguono la stessa struttura: i dati, una cartella numerata per ogni fase dell'analisi
e la relazione finale.

```text
N_NOME_PROGETTO/
├── data/
│   ├── raw/            # dati originali, mai modificati
│   └── processed/      # dati puliti e risultati intermedi prodotti dai notebook
├── 1_<FASE>/
│   ├── codice/         # notebook Jupyter della fase
│   ├── figures/        # figure generate, riprese dalla relazione
│   └── tables/         # tabelle LaTeX generate
├── 2_<FASE>/
│   └── ...
└── <relazione>/        # main.tex, bibliografia e immagini della relazione
```

- Le fasi sono numerate nell'ordine in cui vanno eseguite: ogni fase usa i file salvati dalle precedenti.
- Figure e tabelle sono generate dal codice e incluse direttamente nella relazione, così che i numeri
  del testo corrispondano sempre ai risultati.

## ⚙️ Ambiente

```bash
# 1. Creare e attivare un ambiente virtuale
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Installare le dipendenze
pip install -r requirements.txt

# 3. Avviare Jupyter
jupyter notebook
```

##  Licenza <a name="license"></a>

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

Quest'opera è distribuita con licenza
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International][cc-by-nc-sa].
I dataset utilizzati restano soggetti alle licenze delle rispettive fonti, indicate in ciascun progetto.

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
