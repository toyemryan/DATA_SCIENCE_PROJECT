"""Previsione della qualità dell'aria in Campania a 1–7 giorni: un unico script, dai dati grezzi ai grafici.

COSA FA
    Parte dal CSV orario delle centraline ARPAC e, per 7 inquinanti (PM10, PM2.5, NO2, O3, SO2, CO, C6H6),
    confronta sette modi di prevedere il valore giornaliero dei giorni successivi:

        Persistenza           "domani = oggi" (la baseline da battere)
        Seasonal naive        "domani = lo stesso giorno della settimana scorsa"
        ARIMA                 modello autoregressivo classico, un modello per stazione
        SARIMAX               ARIMA + stagionalità settimanale + stagionalità annuale + festività
        Prophet               trend + stagionalità + festività, senza usare i dati recenti
        Prophet + lag         Prophet + ultimo valore e media degli ultimi 7 giorni come regressori
        XGBoost               alberi di decisione, un modello unico per tutte le stazioni

LE TRE FASI (eseguibili insieme o separatamente)
    1. prep      CSV orario -> pulizia -> medie giornaliere per stazione   -> cache/daily_panel.parquet
    2. forecast  stima, previsione e valutazione di tutti i modelli        -> forecast_results.md + cache/*
    3. plots     i 18 grafici del report                                   -> figures/*.png

SCHEMA DI VALUTAZIONE
    - training fino al 31/12/2025, test dal 01/01/2026 al 26/10/2026 (ultimo giorno con dati)
    - orizzonte h = 1..7: il giorno t si prevede usando solo i dati fino al giorno t-h
    - i parametri dei modelli si stimano una volta sola sul training; nel test si aggiornano solo i dati
    - tutti i modelli lavorano su log(1 + valore); le previsioni si riconvertono con exp(x) - 1
    - in più: una previsione "operativa" per i 7 giorni dopo l'ultimo dato, con intervallo all'80%

COME SI ESEGUE (Prophet non supporta ancora Python 3.14, quindi si usa il 3.12)
    uv run --python 3.12 --with pandas --with pyarrow --with statsmodels --with prophet --with xgboost \\
           --with scikit-learn --with tabulate --with matplotlib python air_quality_forecast.py

    Solo alcune fasi:            ... python air_quality_forecast.py prep forecast
    Solo alcuni inquinanti:      ... python air_quality_forecast.py forecast --pollutants PM10 NO2

    Tempi indicativi su 20 core: prep ~1 min, forecast ~5 min, plots ~20 s.
    La spiegazione completa dei metodi e dei risultati è in forecasting_explained.md.
"""
import argparse
import itertools
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

# Un solo thread di calcolo per processo: il parallelismo è sulle serie (una per core), non dentro numpy.
# Va impostato PRIMA di importare numpy.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import holidays
import matplotlib

matplotlib.use("Agg")  # nessuna finestra: i grafici si salvano solo su file
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURAZIONE
#    Tutto ciò che si potrebbe voler cambiare è qui: file, periodi, stazioni, inquinanti e soglie.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

# ---- file di input e di output
ROOT = Path(__file__).resolve().parent.parent   # 4_FORECASTING/
CSV = ROOT.parent / "data" / "raw" / "qualita_aria_campania_validati.csv"   # dati orari grezzi (input)
CACHE = ROOT / "cache"
FIG_DIR = ROOT / "figures"
PANEL = CACHE / "daily_panel.parquet"                # fase 1: medie giornaliere stazione x giorno
OUT_PRED = CACHE / "forecast_predictions.parquet"    # fase 2: previsioni a 1 giorno nel test
OUT_PRED_H = CACHE / "forecast_predictions_h.parquet"  # fase 2: previsioni a 1..7 giorni nel test
OUT_NEXT7 = CACHE / "forecast_next7.parquet"         # fase 2: previsione dei 7 giorni dopo l'ultimo dato
OUT_IMP = CACHE / "xgb_feature_importance.csv"       # fase 2: importanza delle feature di XGBoost
OUT_COMP = CACHE / "prophet_components.parquet"      # fase 2: componenti stagionali stimate da Prophet
REPORT_MD = ROOT / "forecast_results.md"             # fase 2: tabelle complete dei risultati

# ---- periodi
TRAIN_END = pd.Timestamp("2025-12-31")               # ultimo giorno usato per stimare i modelli
TEST_START = TRAIN_END + pd.Timedelta(days=1)
TEST_END = pd.Timestamp("2026-10-26")                # ultimo giorno con dati
H = 7                                                # orizzonte massimo di previsione (giorni)
FC_END = TEST_END + pd.Timedelta(days=H)             # ultimo giorno della previsione operativa

# ---- regole di pulizia dei dati orari (vedi analisi_descrittiva_qualita_aria.md, §2)
PHYSICAL_LIMITS = {"PM10": 5000, "PM2.5": 5000, "NO2": 1000, "O3": 500, "SO2": 1000, "CO": 50, "C6H6": 100}
MIN_VALID_HOURS = 18   # un giorno è valido se ha almeno 18 ore valide su 24 (regola del 75%, D.Lgs. 155/2010)
STUCK_HOURS = 6        # sensore "bloccato": almeno 6 ore consecutive con lo stesso valore estremo

# ---- stazioni
STATION_NAMES = {
    "IT2211A": "Acerra Zona Industriale",
    "IT2212A": "San Vitaliano Scuola Marconi",
    "IT1491A": "Napoli NA07 Ferrovia",
    "IT2227A": "Avellino Scuola Alighieri",
    "IT0898A": "Napoli NA06 Museo Nazionale",
    "IT2273A": "Polla Area tritovagliatore",
    "IT2276A": "Ariano Irpino Villa Comunale",
    "IT2277A": "Napoli Parco Virgiliano",
    "IT2223A": "Volla via Filichito",
}
CORE = ["IT2211A", "IT2212A", "IT1491A", "IT2227A"]  # le 4 stazioni di valutazione del PM10

# ---- inquinanti
#   col       colonna del pannello giornaliero usata come target
#   unit      unità di misura
#   thr       soglia giornaliera per i "superamenti" (None = l'inquinante ha solo un limite annuale)
#   ref       da dove viene la soglia
#   stations  le 4 stazioni su cui si valuta (dove l'inquinante è misurato con continuità)
POLLUTANTS = {
    "PM10": dict(col="PM10", unit="µg/m³", thr=50, ref="limite giornaliero, D.Lgs. 155/2010", stations=CORE),
    "PM2.5": dict(col="PM2.5", unit="µg/m³", thr=25, ref="limite giornaliero dal 2030, Dir. UE 2024/2881",
                  stations=CORE),
    "NO2": dict(col="NO2", unit="µg/m³", thr=50, ref="limite giornaliero dal 2030, Dir. UE 2024/2881",
                stations=CORE),
    "O3": dict(col="O3_MDA8", unit="µg/m³", thr=120, ref="valore obiettivo sul max media 8 h, D.Lgs. 155/2010",
               stations=["IT2276A", "IT2277A", "IT2212A", "IT2223A"]),
    "SO2": dict(col="SO2", unit="µg/m³", thr=50, ref="limite giornaliero dal 2030, Dir. UE 2024/2881",
                stations=["IT2211A", "IT2212A", "IT1491A", "IT2273A"]),
    "CO": dict(col="CO", unit="mg/m³", thr=4, ref="limite giornaliero dal 2030, Dir. UE 2024/2881",
               stations=["IT2211A", "IT0898A", "IT1491A", "IT2227A"]),
    "C6H6": dict(col="C6H6", unit="µg/m³", thr=None, ref="solo limite annuale (5 µg/m³)", stations=CORE),
}
# inquinanti il cui valore di ieri entra come feature in XGBoost (escluso l'inquinante che si sta prevedendo)
XGB_OTHER_POLLUTANTS = ["PM10", "PM2.5", "NO2", "O3"]

# ---- modelli (nomi interni, usati come nomi di colonna) e nomi visualizzati nel report e nei grafici
MODELS = ["Persistence", "Seasonal naive (t-7)", "ARIMA", "SARIMAX", "Prophet", "Prophet + lags", "XGBoost"]
MODEL_LABELS = {"Persistence": "Persistenza", "Seasonal naive (t-7)": "Seasonal naive (t−7)",
                "Prophet + lags": "Prophet + lag"}

# ---- calendario italiano
IT_HOLIDAYS = holidays.Italy(years=range(2023, 2027))
IT_MONTHS = ["gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic"]
IT_WEEKDAYS = ["lun", "mar", "mer", "gio", "ven", "sab", "dom"]


def model_label(m: str) -> str:
    return MODEL_LABELS.get(m, m)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 1. PREPARAZIONE DEI DATI: CSV orario -> pannello giornaliero (una riga per stazione e giorno)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

def load_hourly() -> pd.DataFrame:
    """Legge il CSV orario. Le date compaiono in due formati diversi; tutto viene portato all'ora solare (+01:00)."""
    df = pd.read_csv(CSV, encoding="utf-8-sig", usecols=["Stazione", "Inquinante", "Data_ora", "Valore"])
    a = pd.to_datetime(df.Data_ora, format="%d-%m-%Y %H:%M:%S %z", errors="coerce")
    b = pd.to_datetime(df.Data_ora, format="%Y-%m-%d %H:%M:%S%z", errors="coerce")
    df["t"] = a.fillna(b).dt.tz_convert("Etc/GMT-1").dt.tz_localize(None).dt.floor("h")
    df["Inquinante"] = df.Inquinante.replace({"PM2,5": "PM2.5"})
    return df.drop(columns="Data_ora").sort_values(["Stazione", "Inquinante", "t"]).reset_index(drop=True)


def clean_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Scarta i valori non affidabili:
      - valori negativi (incluso il codice di mancante -9999)
      - valori oltre il limite fisico dell'inquinante
      - sensori bloccati: >= 6 ore consecutive identiche e sopra il 99,9° percentile dell'inquinante
      - serie stazione x inquinante x anno interamente a zero
    """
    p999 = df[df.Valore >= 0].groupby("Inquinante").Valore.quantile(0.999)

    # ogni "run" è una sequenza di ore consecutive con lo stesso valore, per stazione e inquinante
    run_id = (df.Valore != df.groupby(["Stazione", "Inquinante"]).Valore.shift()).cumsum()
    run_len = df.groupby(run_id).Valore.transform("size")
    stuck = (run_len >= STUCK_HOURS) & (df.Valore > df.Inquinante.map(p999))

    all_zero = (
        df.assign(v=df.Valore.where(df.Valore >= 0))
        .groupby(["Stazione", "Inquinante", df.t.dt.year]).v.transform("max").eq(0)
    )
    bad = (df.Valore < 0) | (df.Valore > df.Inquinante.map(PHYSICAL_LIMITS)) | stuck | all_zero
    return df[~bad]


def o3_mda8(df: pd.DataFrame) -> pd.DataFrame:
    """O3 MDA8: massimo giornaliero della media mobile su 8 ore (l'indicatore del valore obiettivo, 120 µg/m³).

    La media delle ore t-7..t è valida con almeno 6 ore su 8 ed è attribuita al giorno dell'ora t.
    Il massimo del giorno è valido con almeno 18 medie valide su 24.
    """
    out = []
    for st, g in df[df.Inquinante == "O3"].groupby("Stazione"):
        hourly = g.set_index("t").Valore
        hourly = hourly[~hourly.index.duplicated()]
        hourly = hourly.reindex(pd.date_range(hourly.index.min(), hourly.index.max(), freq="h"))
        mean8 = hourly.rolling(8, min_periods=6).mean()
        d = mean8.groupby(mean8.index.normalize()).agg(["max", "count"])
        d = d[d["count"] >= MIN_VALID_HOURS]["max"].rename("O3_MDA8").rename_axis("date").reset_index()
        out.append(d.assign(Stazione=st))
    return pd.concat(out, ignore_index=True)


def daily_means(df: pd.DataFrame) -> pd.DataFrame:
    """Media giornaliera per stazione e inquinante (valida con >= 18 ore), una colonna per inquinante, + O3_MDA8.
    I giorni non validi restano mancanti (NaN): non vengono riempiti."""
    g = df.groupby(["Stazione", "Inquinante", df.t.dt.normalize().rename("date")]).Valore.agg(["mean", "size"])
    g = g[g["size"] >= MIN_VALID_HOURS]["mean"]
    daily = g.unstack("Inquinante").reset_index()
    return daily.merge(o3_mda8(df), on=["Stazione", "date"], how="left")


def run_prep():
    """FASE 1."""
    print("Fase 1 – preparazione dei dati")
    daily = daily_means(clean_hourly(load_hourly()))
    CACHE.mkdir(exist_ok=True)
    daily.to_parquet(PANEL, index=False)
    print(f"{len(daily):,} righe stazione-giorno -> {PANEL}")


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 2. MODELLI
#    Tutte le funzioni lavorano su log(1 + valore).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

# ---- 2.1 utilità comuni ------------------------------------------------------------------------------------

def holiday_flag(dates: pd.DatetimeIndex) -> np.ndarray:
    """1 nei giorni festivi nazionali italiani e il 31 dicembre (fuochi d'artificio), 0 altrimenti."""
    return np.array([(d in IT_HOLIDAYS) or (d.month == 12 and d.day == 31) for d in dates.date], dtype=float)


def fourier_terms(dates: pd.DatetimeIndex, period: float = 365.25, k: int = 2) -> pd.DataFrame:
    """Coppie seno/coseno che descrivono un ciclo annuale "liscio" (k = numero di armoniche)."""
    t = (dates - pd.Timestamp("2023-01-01")).days.values
    cols = {}
    for i in range(1, k + 1):
        cols[f"sin{i}"] = np.sin(2 * np.pi * i * t / period)
        cols[f"cos{i}"] = np.cos(2 * np.pi * i * t / period)
    return pd.DataFrame(cols, index=dates)


def station_series(panel: pd.DataFrame, station: str, col: str) -> pd.Series:
    """log1p(target) giornaliero di una stazione, su calendario completo (giorni mancanti = NaN)."""
    s = panel[panel.Stazione == station].set_index("date")[col]
    idx = pd.date_range(s.index.min(), TEST_END, freq="D")
    return np.log1p(s.reindex(idx))


# ---- 2.2 ARIMA e SARIMAX ----------------------------------------------------------------------------------

def sarimax_exog(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Variabili esterne di SARIMAX: stagionalità annuale (Fourier, 2 armoniche) + festività."""
    x = fourier_terms(idx, k=2)
    x["holiday"] = holiday_flag(idx)
    return x


def fit_best_by_aic(y_train, orders, seasonal_orders, exog_train=None):
    """Stima tutte le combinazioni di ordini e tiene quella con l'AIC più basso."""
    best = None
    for order, sorder in itertools.product(orders, seasonal_orders):
        trend = "c" if order[1] == 0 and sorder[1] == 0 else "n"  # costante solo se la serie non è differenziata
        try:
            res = SARIMAX(y_train, exog=exog_train, order=order, seasonal_order=sorder, trend=trend,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
        except Exception:
            continue
        if np.isfinite(res.aic) and (best is None or res.aic < best[0]):
            best = (res.aic, order, sorder, res)
    if best is None:
        raise RuntimeError("nessun modello SARIMAX stimato")
    return best


def rolling_forecasts(res, y, exog_fn=None) -> pd.DataFrame:
    """Previsioni a 1..H giorni da ogni giorno di origine del test, con i parametri stimati sul training.

    Da ogni origine o si prevedono i giorni o+1..o+H usando solo i dati fino a o.
    Ritorna una tabella data x orizzonte: la colonna h contiene, per ogni data, la previsione emessa h giorni prima.
    L'ultima origine è TEST_END, quindi la tabella include anche i giorni dopo l'ultimo dato.
    """
    xe = exog_fn(pd.date_range(y.index[0], FC_END, freq="D")) if exog_fn else None
    out = {}
    for origin in pd.date_range(TRAIN_END - pd.Timedelta(days=H - 1), TEST_END, freq="D"):
        r = res.apply(y[:origin], exog=None if xe is None else xe[:origin])  # stessi parametri, dati aggiornati
        future = pd.date_range(origin + pd.Timedelta(days=1), periods=H, freq="D")
        fc = r.forecast(H, exog=None if xe is None else xe.loc[future])
        for h, (d, v) in enumerate(zip(future, fc.values), start=1):
            out[(d, h)] = v
    return pd.Series(out).unstack()


def run_arima(y):
    orders = [(p, d, q) for p in range(4) for d in (0, 1) for q in range(4)]
    _, order, _, res = fit_best_by_aic(y[:TRAIN_END], orders, [(0, 0, 0, 0)])
    return rolling_forecasts(res, y), f"ARIMA{order}"


def run_sarimax(y):
    x = sarimax_exog(y.index)
    orders = [(1, 0, 1), (2, 0, 1), (1, 1, 1), (2, 1, 2)]
    seasonal = [(1, 0, 1, 7), (0, 1, 1, 7), (1, 1, 1, 7)]  # stagionalità settimanale
    _, order, sorder, res = fit_best_by_aic(y[:TRAIN_END], orders, seasonal, x[:TRAIN_END])
    return rolling_forecasts(res, y, sarimax_exog), f"SARIMAX{order}x{sorder} + Fourier(K=2) + festività"


# ---- 2.3 Prophet ------------------------------------------------------------------------------------------

def new_prophet():
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.add_country_holidays(country_name="IT")
    return m


def run_prophet(y, dates):
    """Prophet standard: trend + stagionalità + festività, nessuna informazione sui giorni recenti.
    Per questo la previsione di un giorno è la stessa a qualunque orizzonte.
    Ritorna anche le componenti stimate su un anno intero, per i grafici."""
    train = y[:TRAIN_END].dropna().rename("y").rename_axis("ds").reset_index()
    m = new_prophet().fit(train)
    fc = m.predict(pd.DataFrame({"ds": dates}))
    comp = m.predict(pd.DataFrame({"ds": pd.date_range("2024-01-01", "2024-12-31", freq="D")}))
    comp = comp[["ds", "trend", "yearly", "weekly", "holidays"]]
    return pd.Series(fc.yhat.values, index=dates), comp


def run_prophet_lags(y, dates, h=1):
    """Prophet + due regressori: ultimo valore noto e media degli ultimi 7 giorni, entrambi fino a t-h.
    Modello "diretto": per ogni orizzonte h si stima un modello diverso, con i regressori spostati di h giorni."""
    y = y.reindex(pd.date_range(y.index[0], FC_END, freq="D"))
    df = pd.DataFrame({"y": y})
    df["lag1"] = y.ffill().shift(h)
    df["roll7"] = y.shift(h).rolling(7, min_periods=1).mean().ffill()
    df = df.dropna(subset=["lag1", "roll7"]).rename_axis("ds").reset_index()
    m = new_prophet()
    m.add_regressor("lag1")
    m.add_regressor("roll7")
    m.fit(df[(df.ds <= TRAIN_END) & df.y.notna()])
    fc = m.predict(df[df.ds.isin(dates)])
    return pd.Series(fc.yhat.values, index=fc.ds).reindex(dates)


# ---- 2.4 XGBoost ------------------------------------------------------------------------------------------

def build_xgb_features(panel: pd.DataFrame, col: str, h: int = 1) -> pd.DataFrame:
    """Tabella di training "globale": una riga per giorno e stazione (tutte le stazioni fisse, non solo le 4
    di valutazione), con le sole informazioni disponibili fino al giorno t-h.

    Feature del target stesso (y_*): valori ritardati, medie mobili a 7 e 30 giorni, ultimo valore osservato.
      Ritardi: per h = 1 sono 1, 2, 3, 7, 14; per h > 1 partono da h (h, h+1, h+2) + 7 e 14 se più lontani.
    Altri inquinanti (<inquinante>_lag<h>): il loro valore al giorno t-h.
    Media regionale del target (reg_*): a t-h e t-h-1.
    Calendario (giorno della settimana, giorno dell'anno, mese, festività): riferito al giorno previsto,
      che è noto in anticipo.
    """
    family = col.split("_")[0]  # "O3_MDA8" -> "O3"
    others = [p for p in XGB_OTHER_POLLUTANTS if p != family]
    lags = [h, h + 1, h + 2] + [lag for lag in (7, 14) if lag > h + 2]
    stations = panel[panel.Stazione.str.startswith("IT")].Stazione.unique()
    idx = pd.date_range(panel.date.min(), FC_END, freq="D")
    wide = (panel[panel.Stazione.isin(stations)]
            .set_index(["Stazione", "date"])[[col] + others]
            .pipe(np.log1p))
    full = wide.reindex(pd.MultiIndex.from_product([stations, idx], names=["Stazione", "date"]))
    regional = full[col].groupby(level="date").mean()

    frames = []
    for st, g in full.groupby(level="Stazione"):
        g = g.droplevel(0)
        f = pd.DataFrame(index=g.index)
        f["y"] = g[col]
        for lag in lags:
            f[f"y_lag{lag}"] = g[col].shift(lag)
        f["y_roll7"] = g[col].shift(h).rolling(7, min_periods=3).mean()
        f["y_roll30"] = g[col].shift(h).rolling(30, min_periods=10).mean()
        f["y_last"] = g[col].ffill().shift(h)
        for p in others:
            f[f"{p}_lag{h}"] = g[p].shift(h)
        f[f"reg_lag{h}"] = regional.shift(h).reindex(g.index)
        f[f"reg_lag{h + 1}"] = regional.shift(h + 1).reindex(g.index)
        f["Stazione"] = st
        frames.append(f)
    X = pd.concat(frames).rename_axis("date").reset_index()
    d = pd.DatetimeIndex(X.date)
    X["dow"] = d.dayofweek
    X["doy_sin"] = np.sin(2 * np.pi * d.dayofyear / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * d.dayofyear / 365.25)
    X["month"] = d.month
    X["holiday"] = holiday_flag(d)
    X["Stazione"] = X.Stazione.astype("category")
    return X


def run_xgboost(panel: pd.DataFrame, col: str, eval_stations, h: int = 1):
    """Un modello per (inquinante, orizzonte). Il numero di alberi si sceglie con early stopping sugli ultimi
    90 giorni del training, poi il modello si ristima su tutto il training con quel numero di alberi."""
    X = build_xgb_features(panel, col, h)
    feats = [c for c in X.columns if c not in ("y", "date")]
    train = X[(X.date <= TRAIN_END) & X.y.notna()]
    val_mask = train.date > TRAIN_END - pd.Timedelta(days=90)
    params = dict(n_estimators=2000, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5, tree_method="hist", enable_categorical=True, random_state=42)
    m = xgb.XGBRegressor(**params, early_stopping_rounds=100)
    m.fit(train.loc[~val_mask, feats], train.loc[~val_mask, "y"],
          eval_set=[(train.loc[val_mask, feats], train.loc[val_mask, "y"])], verbose=False)
    best_n = m.best_iteration + 1
    final = xgb.XGBRegressor(**{**params, "n_estimators": best_n}).fit(train[feats], train.y)

    test = X[(X.date > TRAIN_END) & X.Stazione.isin(eval_stations)].copy()  # test + 7 giorni dopo l'ultimo dato
    test["pred"] = final.predict(test[feats])
    imp = pd.Series(final.feature_importances_, index=feats).sort_values(ascending=False)
    info = dict(n_trees=best_n, n_stations=train.Stazione.nunique(), n_rows=len(train))
    return test.set_index(["Stazione", "date"])["pred"], info, imp


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 3. ESECUZIONE IN PARALLELO
#    Due tipi di lavoro indipendenti, distribuiti su tutti i core:
#      - una coppia (inquinante, stazione): baseline, ARIMA, SARIMAX, Prophet, Prophet + lag, h = 1..7
#      - una coppia (inquinante, orizzonte): un modello XGBoost
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

_PANEL = None  # pannello giornaliero, caricato una volta in ogni processo


def _init_worker():
    global _PANEL
    _PANEL = pd.read_parquet(PANEL)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")


def series_job(job):
    """Tutti i modelli per stazione di una coppia (inquinante, stazione), per gli orizzonti 1..H.

    Ritorna una tabella lunga (date, h, una colonna per modello, Observed) in scala log, dal primo giorno di
    test fino a FC_END (i giorni dopo TEST_END non hanno osservazione).
    """
    pol, st = job
    col = POLLUTANTS[pol]["col"]
    dates = pd.date_range(TEST_START, FC_END, freq="D")
    y = station_series(_PANEL, st, col)
    y_ext = y.reindex(pd.date_range(y.index[0], FC_END, freq="D"))
    specs = {}
    arima, specs["ARIMA"] = run_arima(y)
    sarimax, specs["SARIMAX"] = run_sarimax(y)
    prophet, comp = run_prophet(y, dates)
    frames = []
    for h in range(1, H + 1):
        p = pd.DataFrame(index=dates)
        p["Persistence"] = y_ext.ffill().shift(h).reindex(dates)                       # ultimo valore noto a t-h
        p["Seasonal naive (t-7)"] = y_ext.shift(7).fillna(y_ext.ffill().shift(h)).reindex(dates)  # valore a t-7
        p["ARIMA"] = arima[h].reindex(dates)
        p["SARIMAX"] = sarimax[h].reindex(dates)
        p["Prophet"] = prophet
        p["Prophet + lags"] = run_prophet_lags(y, dates, h)
        p["Observed"] = y_ext.reindex(dates)
        frames.append(p.assign(h=h))
    print(f"  {pol:6s} {st} {STATION_NAMES[st]}", flush=True)
    long = pd.concat(frames).rename_axis("date").reset_index().assign(Inquinante=pol, Stazione=st)
    return pol, st, long, specs, comp.assign(Inquinante=pol, Stazione=st)


def xgb_job(job):
    pol, h = job
    cfg = POLLUTANTS[pol]
    preds, info, imp = run_xgboost(_PANEL, cfg["col"], cfg["stations"], h)
    print(f"  XGBoost {pol:6s} h={h}", flush=True)
    return pol, h, preds, info, imp


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 4. VALUTAZIONE
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

def complete_cases(pred: pd.DataFrame, pol: str) -> pd.DataFrame:
    """Righe di un inquinante con osservazione valida e previsione di tutti i modelli (campione comune)."""
    return pred[pred.Inquinante == pol].dropna(subset=MODELS + ["Observed"])


def common_horizon_sample(pred_h: pd.DataFrame, pol: str) -> pd.DataFrame:
    """Come complete_cases, ma tiene solo i giorni-stazione con previsione valida a TUTTI gli orizzonti 1..H,
    così le metriche dei diversi orizzonti sono calcolate sugli stessi giorni e sono confrontabili."""
    e = complete_cases(pred_h, pol)
    return e[e.groupby(["Stazione", "date"]).h.transform("size") == H]


def metrics(y_true: pd.Series, y_pred: pd.Series, thr) -> dict:
    """MAE, RMSE, bias (previsto - osservato) e, se c'è una soglia, precision / recall / F1 sui superamenti.
    Precision e recall valgono NaN quando non sono calcolabili (nessun superamento previsto o osservato)."""
    err = y_pred - y_true
    out = {"MAE": err.abs().mean(), "RMSE": np.sqrt((err ** 2).mean()), "Bias": err.mean()}
    if thr is not None:
        obs_ex, pred_ex = y_true > thr, y_pred > thr
        tp = (obs_ex & pred_ex).sum()
        prec = tp / pred_ex.sum() if pred_ex.sum() else np.nan
        rec = tp / obs_ex.sum() if obs_ex.sum() else np.nan
        out |= {"Precision": prec, "Recall": rec,
                "F1": 2 * prec * rec / (prec + rec) if prec + rec > 0 else np.nan}
    return out


def evaluate(ev: pd.DataFrame, thr) -> pd.DataFrame:
    """Tabella modelli x metriche. Skill % = miglioramento del MAE rispetto alla persistenza."""
    t = pd.DataFrame({m: metrics(ev.Observed, ev[m], thr) for m in MODELS}).T
    t["Skill %"] = (1 - t.MAE / t.loc["Persistence", "MAE"]) * 100
    return t


def horizon_eval(pred_h: pd.DataFrame, pol: str):
    """Metriche per ciascun orizzonte h = 1..H sul campione comune."""
    e = common_horizon_sample(pred_h, pol)
    return e, {h: evaluate(e[e.h == h], POLLUTANTS[pol]["thr"]) for h in range(1, H + 1)}


def next7_forecast(fwd: pd.DataFrame, pred_h: pd.DataFrame, pols) -> pd.DataFrame:
    """Previsione per i 7 giorni dopo l'ultimo dato, con il modello che nel test ha il MAE medio (h = 1..7) più
    basso. Intervallo all'80%: 10° e 90° percentile degli errori di test (in scala log) a quell'orizzonte."""
    rows = []
    for pol in pols:
        e, tabs = horizon_eval(pred_h, pol)
        best = pd.DataFrame({h: t.MAE for h, t in tabs.items()}).mean(axis=1).idxmin()
        for h in range(1, H + 1):
            eh = e[e.h == h]
            q10, q90 = (np.log1p(eh.Observed) - np.log1p(eh[best])).quantile([0.1, 0.9])
            fh = fwd[(fwd.Inquinante == pol) & (fwd.h == h)].copy()
            fh["Modello"] = best
            fh["Previsione"] = fh[best]
            fh["Basso80"] = np.expm1(np.log1p(fh[best]) + q10).clip(lower=0)
            fh["Alto80"] = np.expm1(np.log1p(fh[best]) + q90)
            rows.append(fh)
    return pd.concat(rows, ignore_index=True)


def run_forecast(pols):
    """FASE 2: stima tutti i modelli in parallelo, salva le previsioni e scrive il report."""
    print("Fase 2 – modelli di previsione")
    panel = pd.read_parquet(PANEL)
    jobs = [(pol, st) for pol in pols for st in POLLUTANTS[pol]["stations"]]
    xjobs = [(pol, h) for pol in pols for h in range(1, H + 1)]
    print(f"{len(jobs)} serie (inquinante x stazione) + {len(xjobs)} modelli XGBoost (inquinante x orizzonte)...")
    with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=_init_worker) as ex:
        futures = [ex.submit(series_job, j) for j in jobs]
        xfutures = [ex.submit(xgb_job, j) for j in xjobs]
        results = [f.result() for f in futures]
        xresults = [f.result() for f in xfutures]

    # ---- raccolta dei risultati
    specs, comps, imps, xgb_info, xgb_long = {}, [], [], {}, []
    for pol, h, xp, info, imp in xresults:
        if h == 1:
            xgb_info[pol] = info
        imps.append(imp.rename("importance").rename_axis("feature").reset_index().assign(Inquinante=pol, h=h))
        xgb_long.append(xp.rename("XGBoost").reset_index().assign(Inquinante=pol, h=h))
    for pol, st, _, sp, comp in results:
        specs[(pol, st)] = sp
        comps.append(comp)
    long = pd.concat([r[2] for r in results], ignore_index=True)
    long = long.merge(pd.concat(xgb_long), on=["Inquinante", "Stazione", "date", "h"], how="left")
    long[MODELS + ["Observed"]] = np.expm1(long[MODELS + ["Observed"]])  # da scala log a unità originali

    # ---- tre viste: test a 1 giorno, test a 1..7 giorni, previsione oltre l'ultimo dato
    cols = ["date"] + MODELS + ["Observed", "Inquinante", "Stazione"]
    test = long[long.date <= TEST_END]
    pred_1 = test[test.h == 1][cols].reset_index(drop=True)
    pred_h = test[["h"] + cols].reset_index(drop=True)
    fwd = long[(long.date - TEST_END).dt.days == long.h].drop(columns="Observed")  # tutte emesse il TEST_END
    next7 = next7_forecast(fwd, pred_h, pols)
    imps = pd.concat(imps, ignore_index=True)

    pred_1.to_parquet(OUT_PRED, index=False)
    pred_h.to_parquet(OUT_PRED_H, index=False)
    next7.to_parquet(OUT_NEXT7, index=False)
    imps.to_csv(OUT_IMP, index=False)
    pd.concat(comps).to_parquet(OUT_COMP, index=False)
    write_report(pred_1, pred_h, next7, specs, xgb_info, imps, pols, panel)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 5. REPORT MARKDOWN (forecast_results.md)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

def it(v, digits: int = 2, sign: bool = False) -> str:
    """Numero in formato italiano: virgola decimale, punto per le migliaia, "–" se mancante."""
    if pd.isna(v):
        return "–"
    s = f"{v:+,.{digits}f}" if sign else f"{v:,.{digits}f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s.replace("-", "−")


def it_date(d) -> str:
    """Es. "mar 28 ott"."""
    d = pd.Timestamp(d)
    return f"{IT_WEEKDAYS[d.dayofweek]} {d.day} {IT_MONTHS[d.month - 1]}"


def md(df: pd.DataFrame, index: bool = True) -> str:
    """Tabella markdown con i numeri allineati a destra."""
    align = (["left"] if index else []) + ["right"] * df.shape[1]
    return df.to_markdown(index=index, colalign=align, disable_numparse=True)


def md_metrics(df: pd.DataFrame, digits: int) -> str:
    """Tabella markdown di una tabella di metriche (output di evaluate)."""
    df = df.rename(index=MODEL_LABELS).copy()
    for c in df.columns:
        if c == "Skill %":
            df[c] = df[c].map(lambda v: it(v, 1, sign=True))
        elif c in ("Precision", "Recall", "F1"):
            df[c] = df[c].map(lambda v: it(v, 2))
        else:
            df[c] = df[c].map(lambda v: it(v, digits))
    return md(df)


def write_report(pred_1, pred_h, next7, specs, xgb_info, imps, pols, panel):
    L = ["# Forecasting degli inquinanti – confronto tra modelli", "",
         f"Previsione a un giorno (one-step-ahead). Training: fino al {TRAIN_END.date()} · "
         f"Test: {TEST_START.date()} – {TEST_END.date()}. Per ogni inquinante le metriche sono calcolate solo sui "
         "giorni-stazione in cui l'osservazione è valida e tutti i modelli hanno una previsione.", "",
         "- **Skill %** = `1 − MAE_modello / MAE_persistenza`: miglioramento rispetto a \"domani = oggi\".",
         "- **Precision / Recall / F1** riguardano i *superamenti* (giorni sopra la soglia dell'inquinante). "
         "Precision = quota dei superamenti previsti che si sono verificati; recall = quota dei superamenti "
         "osservati che sono stati previsti. \"–\" = non calcolabile (il modello non prevede mai un superamento, "
         "oppure nel test non ce ne sono).",
         f"- In fondo: errore per orizzonte da 1 a {H} giorni e previsione per i {H} giorni dopo l'ultimo dato.",
         "- Spiegazione completa in [forecasting_explained.md](forecasting_explained.md).", ""]

    # ---- sintesi tra inquinanti
    L += ["## Sintesi: skill vs persistenza (%) per inquinante", "",
          "Valori positivi = meglio della persistenza. Le 4 stazioni di ogni inquinante sono aggregate.", ""]
    rows, setup = {}, []
    for pol in pols:
        cfg = POLLUTANTS[pol]
        ev = complete_cases(pred_1, pol)
        rows[pol] = evaluate(ev, cfg["thr"])["Skill %"]
        n_ex = int((ev.Observed > cfg["thr"]).sum()) if cfg["thr"] else None
        setup.append({"Inquinante": pol, "Target": "max media 8 h" if pol == "O3" else "media giornaliera",
                      "Unità": cfg["unit"], "Soglia": cfg["thr"] if cfg["thr"] else "–",
                      "Riferimento soglia": cfg["ref"], "Giorni-stazione": it(len(ev), 0),
                      "Superamenti": n_ex if n_ex is not None else "–",
                      "Media osservata": it(ev.Observed.mean(), 2)})
    summary = pd.DataFrame(rows).T[MODELS].rename(columns=MODEL_LABELS)
    L += [md(summary.map(lambda v: it(v, 1, sign=True))), "",
          "## Impostazione per inquinante", "", md(pd.DataFrame(setup).astype(str), index=False), ""]

    # ---- dettaglio per inquinante
    for pol in pols:
        cfg = POLLUTANTS[pol]
        digits = 3 if pol == "CO" else 2
        ev = complete_cases(pred_1, pol)
        L += [f"## {pol}" + (" (massimo giornaliero della media mobile su 8 ore)" if pol == "O3" else ""), "",
              f"Unità: {cfg['unit']}. Soglia superamenti: "
              + (f"{cfg['thr']} {cfg['unit']} ({cfg['ref']})." if cfg["thr"] else f"nessuna ({cfg['ref']})."), "",
              "### Complessivo (4 stazioni aggregate)", "",
              md_metrics(evaluate(ev, cfg["thr"]).sort_values("MAE"), digits), "", "### Per stazione", ""]
        for st in cfg["stations"]:
            g = ev[ev.Stazione == st]
            n_ex = f", superamenti osservati: {int((g.Observed > cfg['thr']).sum())}" if cfg["thr"] else ""
            L += [f"#### {st} – {STATION_NAMES[st]}  (giorni di test: {len(g)}{n_ex})", "",
                  md_metrics(evaluate(g, cfg["thr"]).sort_values("MAE"), digits), ""]
        L += ["### Modelli selezionati", ""]
        for st in cfg["stations"]:
            L.append(f"- **{st}** ({STATION_NAMES[st]}): {specs[(pol, st)]['ARIMA']} · {specs[(pol, st)]['SARIMAX']}")
        info = xgb_info[pol]
        L += [f"- **XGBoost**: un modello globale su {info['n_stations']} stazioni fisse "
              f"({it(info['n_rows'], 0)} righe di training), {info['n_trees']} alberi "
              "(early stopping sugli ultimi 90 giorni di training)", "",
              "### XGBoost – le 10 feature più importanti (gain)", "",
              md(imps[(imps.Inquinante == pol) & (imps.h == 1)].head(10).set_index("feature")[["importance"]]
                 .map(lambda v: it(v, 3))), ""]

    L += horizon_report(pred_h, next7, imps, pols, panel)
    L += ["## Note sui modelli", "",
          "- **Prophet**: stagionalità annuale + settimanale + festività italiane (un modello per stazione).",
          "- **Prophet + lag**: come sopra + regressori `ultimo valore osservato` e `media 7 giorni` (fino a t-1).",
          "- **Feature XGBoost**: `y_*` = il target stesso (lag, medie mobili, ultimo valore osservato); "
          "`reg_*` = media regionale del target; `<inquinante>_lag1` = altri inquinanti il giorno prima.", ""]
    REPORT_MD.write_text("\n".join(L))
    print(summary.round(1).to_string())
    print(f"\nReport -> {REPORT_MD}")


def horizon_report(pred_h, next7, imps, pols, panel) -> list:
    """Sezioni del report sull'orizzonte 1..7 giorni e sulla previsione dopo l'ultimo dato."""
    hcols = [f"h={h}" for h in range(1, H + 1)]
    L = ["## Orizzonte di previsione da 1 a 7 giorni", "",
         "Il giorno t viene previsto con i soli dati fino a t−h. **h = 1** è la previsione a un giorno delle "
         "sezioni precedenti; **h = 7** è la previsione emessa una settimana prima.", "",
         "- **Persistenza a h giorni**: ultimo valore noto al giorno t−h. **Seasonal naive**: valore di t−7 "
         "(noto per ogni h ≤ 7).",
         "- **ARIMA / SARIMAX**: stessi parametri della previsione a un giorno; da ogni origine si prevedono "
         "ricorsivamente i 7 giorni successivi.",
         "- **Prophet**: non usa i dati recenti, quindi la sua previsione è la stessa a ogni orizzonte.",
         "- **Prophet + lag** e **XGBoost**: un modello *diretto* per ogni h, stimato con le feature spostate "
         "di h giorni (es. per h = 7 il valore più recente usato è quello di t−7).",
         "- Campione comune: giorni-stazione con osservazione valida e previsione a tutti gli orizzonti. "
         "La skill è sempre rispetto alla persistenza **allo stesso orizzonte**.", ""]

    # ---- sintesi
    skill_h7, best_rows, tabs_all, n_all = {}, [], {}, {}
    for pol in pols:
        e, tabs = horizon_eval(pred_h, pol)
        tabs_all[pol], n_all[pol] = tabs, int((e.h == 1).sum())
        skill_h7[pol] = tabs[H]["Skill %"]
        mae = pd.DataFrame({h: t.MAE for h, t in tabs.items()})
        best_h7 = mae[H].idxmin()
        digits = 3 if pol == "CO" else 2
        best_rows.append({"Inquinante": pol, "Unità": POLLUTANTS[pol]["unit"],
                          "MAE persistenza h=1": it(mae.loc["Persistence", 1], digits),
                          f"MAE persistenza h={H}": it(mae.loc["Persistence", H], digits),
                          f"Migliore a h={H}": model_label(best_h7),
                          f"MAE migliore h={H}": it(mae.loc[best_h7, H], digits),
                          "MAE Prophet": it(mae.loc["Prophet", H], digits)})
    L += [f"### Sintesi: skill vs persistenza a {H} giorni (%)", "",
          md(pd.DataFrame(skill_h7).T[MODELS].rename(columns=MODEL_LABELS).map(lambda v: it(v, 1, sign=True))),
          "", "### Sintesi: errore a 1 e a 7 giorni", "",
          md(pd.DataFrame(best_rows).astype(str), index=False), ""]

    # ---- dettaglio per inquinante
    for pol in pols:
        tabs, cfg = tabs_all[pol], POLLUTANTS[pol]
        digits = 3 if pol == "CO" else 2
        mae = pd.DataFrame({f"h={h}": t.MAE for h, t in tabs.items()}).rename(index=MODEL_LABELS)
        sk = pd.DataFrame({f"h={h}": t["Skill %"] for h, t in tabs.items()}).rename(index=MODEL_LABELS)
        L += [f"### {pol} – errore per orizzonte", "",
              f"MAE ({cfg['unit']}); {it(n_all[pol], 0)} giorni-stazione per ogni orizzonte.", "",
              md(mae[hcols].map(lambda v: it(v, digits))), "", "Skill vs persistenza (%)", "",
              md(sk.drop(index=MODEL_LABELS["Persistence"])[hcols].map(lambda v: it(v, 1, sign=True))), ""]
        if cfg["thr"] is not None and tabs[1].F1.notna().any():
            f1 = pd.DataFrame({f"h={h}": t.F1 for h, t in tabs.items()}).rename(index=MODEL_LABELS)
            L += [f"F1 sui superamenti (> {cfg['thr']} {cfg['unit']})", "", md(f1[hcols].map(lambda v: it(v, 2))), ""]
        top = imps[(imps.Inquinante == pol) & (imps.h == H)].head(5)
        L += [f"XGBoost a {H} giorni – le 5 feature più importanti: "
              + ", ".join(f"`{f}` ({it(v, 3)})" for f, v in zip(top.feature, top.importance)), ""]

    # ---- previsione operativa
    L += [f"## Previsione per i 7 giorni dopo l'ultimo dato ({it_date(TEST_END + pd.Timedelta(days=1))} – "
          f"{it_date(FC_END)} {FC_END.year})", "",
          f"Emessa con i dati fino al {TEST_END.date()}. Per ogni inquinante si usa il modello con il MAE medio "
          "(h = 1..7) più basso nel periodo di test. Tra parentesi l'intervallo empirico all'80%: 10° e 90° "
          "percentile degli errori di test a quell'orizzonte (in scala log, 4 stazioni insieme). L'intervallo "
          "è tarato sul 2026, quindi la copertura dell'80% vale in media su quel periodo, non per ogni giorno.", ""]
    for pol in pols:
        cfg = POLLUTANTS[pol]
        digits = {"CO": 2, "SO2": 1, "C6H6": 1}.get(pol, 0)
        f = next7[next7.Inquinante == pol]
        best = f.Modello.iloc[0]
        tab = {}
        for st in cfg["stations"]:
            s = (panel[(panel.Stazione == st) & (panel.date <= TEST_END)].dropna(subset=[cfg["col"]])
                 .sort_values("date"))
            last = s.iloc[-1] if len(s) else None
            col = {"ultimo dato": (f"{it(last[cfg['col']], digits)} ({last.date.day}/{last.date.month})"
                                   if last is not None else "–")}
            for _, r in f[f.Stazione == st].sort_values("date").iterrows():
                col[it_date(r.date)] = f"{it(r.Previsione, digits)} ({it(r.Basso80, digits)}–{it(r.Alto80, digits)})"
            tab[STATION_NAMES[st]] = col
        thr = f", soglia {cfg['thr']}" if cfg["thr"] else ""
        L += [f"### {pol} – modello: {model_label(best)} ({cfg['unit']}{thr})", "", md(pd.DataFrame(tab)), ""]
    return L


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 6. GRAFICI (figures/*.png)
#    01–10: caso di studio PM10 · 11–14: confronto tra inquinanti · 15–18: orizzonte 1–7 giorni e previsione
#    Ogni figura è una funzione fig_XX(d); d contiene i dati letti da cache/ (vedi load_plot_data).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

# ---- palette e stile
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
YELLOW, MAGENTA, GREEN = "#eda100", "#e87ba4", "#008300"
NEUTRAL = "#b5b3ab"  # baseline e linee di riferimento
MODEL_COLOR = {"XGBoost": BLUE, "SARIMAX": ORANGE, "Prophet": AQUA, "Persistence": NEUTRAL}

PM10_STATIONS = {st: STATION_NAMES[st] for st in CORE}
PM10_LIMIT = POLLUTANTS["PM10"]["thr"]
THR = {p: cfg["thr"] for p, cfg in POLLUTANTS.items() if cfg["thr"] is not None}
POL_LABEL = {"O3": "O3 (max 8 h)"}
CASE_STATION = "IT2211A"  # Acerra ZI, la stazione usata per i grafici di dettaglio
ZOOM = ("2026-01-05", "2026-03-15")


def setup_matplotlib():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK2,
    })


# ---- piccoli aiuti per i grafici

def save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def title(fig, text, size=12):
    fig.suptitle(text, x=0.01, ha="left", fontsize=size, fontweight="bold")


def threshold_line(ax, horizontal=True, label=True):
    """Linea del limite giornaliero del PM10 (50 µg/m³)."""
    if horizontal:
        ax.axhline(PM10_LIMIT, color=MUTED, lw=0.9, zorder=1)
        if label:
            ax.annotate(f"limite {PM10_LIMIT} µg/m³", xy=(1, PM10_LIMIT), xycoords=("axes fraction", "data"),
                        xytext=(-2, 3), textcoords="offset points", ha="right", va="bottom",
                        fontsize=8, color=MUTED,
                        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5), zorder=5)
    else:
        ax.axvline(PM10_LIMIT, color=MUTED, lw=0.9, zorder=1)


def date_axis(ax, fmt="%b %Y"):
    """Etichette delle date con i mesi in italiano, indipendentemente dal locale del sistema."""
    def f(x, pos=None):
        d = mdates.num2date(x)
        return d.strftime(fmt.replace("%b", IT_MONTHS[d.month - 1]))
    ax.xaxis.set_major_formatter(FuncFormatter(f))


def mae(g, m):
    return (g[m] - g.Observed).abs().mean()


def hit_rates(obs, pred, thr):
    """Precision e recall sui superamenti, per i grafici: 0 (invece di NaN) se non si prevede mai un superamento."""
    o, f = obs > thr, pred > thr
    tp = (o & f).sum()
    return tp / max(f.sum(), 1), tp / o.sum()


def load_plot_data():
    pred_all = pd.read_parquet(OUT_PRED)
    imp_all = pd.read_csv(OUT_IMP)
    imp_all = imp_all[imp_all.h == 1]  # importanza del modello a un giorno
    pols = [p for p in POLLUTANTS if p in set(pred_all.Inquinante)]
    d = SimpleNamespace(
        panel=pd.read_parquet(PANEL),
        pred_all=pred_all,
        pred_h=pd.read_parquet(OUT_PRED_H),
        next7=pd.read_parquet(OUT_NEXT7),
        imp_all=imp_all,
        comp_all=pd.read_parquet(OUT_COMP),
        pols=pols,
        # inquinanti con almeno 10 superamenti nel test: gli unici per cui ha senso valutarli
        ex_pols=[p for p in pols if p in THR and (complete_cases(pred_all, p).Observed > THR[p]).sum() >= 10],
    )
    d.pred = pred_all[pred_all.Inquinante == "PM10"]   # caso di studio PM10
    d.ev = complete_cases(pred_all, "PM10")
    d.order = sorted(MODELS, key=lambda m: mae(d.ev, m), reverse=True)  # modelli dal peggiore al migliore
    return d


# ---- 01–10: caso di studio PM10 ---------------------------------------------------------------------------

def fig01_data_train_test(d):
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for ax, (st, name) in zip(axes, PM10_STATIONS.items()):
        s = d.panel[d.panel.Stazione == st].set_index("date").PM10
        s = s.reindex(pd.date_range(s.index.min(), s.index.max()))
        ax.plot(s.index, s.values, color=BLUE, lw=1.0)
        ax.axvspan(TEST_START, s.index.max(), color=GRID, alpha=0.45, lw=0, zorder=0)
        threshold_line(ax)
        ax.set_title(name)
        ax.set_ylim(0, min(160, np.nanmax(s.values) * 1.05))
        ax.set_ylabel("µg/m³")
    axes[0].annotate("periodo di test (2026)", xy=(TEST_START, 1), xycoords=("data", "axes fraction"),
                     xytext=(6, -4), textcoords="offset points", va="top", fontsize=9, color=INK2)
    axes[0].annotate("periodo di training (2023–2025)", xy=(d.panel.date.min(), 1),
                     xycoords=("data", "axes fraction"),
                     xytext=(4, -4), textcoords="offset points", va="top", fontsize=9, color=INK2)
    date_axis(axes[-1])
    title(fig, "PM10 medio giornaliero nelle quattro stazioni di valutazione", 13)
    fig.text(0.01, 0.945, "I vuoti sono giorni mancanti (es. apr–ott 2025). Asse y limitato a 160 µg/m³; "
             "alcuni picchi di Capodanno e di polveri lo superano.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "01_data_train_test.png")


def fig02_acf_pacf(d):
    y = np.log1p(d.panel[d.panel.Stazione == CASE_STATION].set_index("date").PM10)
    y = y.reindex(pd.date_range(y.index.min(), TRAIN_END))
    nlags = 28
    a = acf(y.values, nlags=nlags, missing="conservative")
    p = pacf(y.dropna().values, nlags=nlags)
    ci = 1.96 / np.sqrt(y.notna().sum())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, vals, name in [(axes[0], a, "Autocorrelazione (ACF)"), (axes[1], p, "Autocorrelazione parziale (PACF)")]:
        lags = np.arange(1, nlags + 1)
        ax.vlines(lags, 0, vals[1:], color=BLUE, lw=2)
        ax.plot(lags, vals[1:], "o", color=BLUE, ms=4.5, mec=SURFACE, mew=1.5)
        ax.axhspan(-ci, ci, color=GRID, alpha=0.7, lw=0)
        ax.axhline(0, color=AXIS, lw=0.9)
        for wk in (7, 14, 21, 28):
            ax.axvline(wk, color=GRID, lw=0.8, zorder=0)
        ax.set_title(name)
        ax.set_xlabel("lag (giorni)")
        ax.set_xticks([1, 7, 14, 21, 28])
    axes[0].set_ylim(-0.2, 1)
    title(fig, "Acerra ZI – quanto il log PM10 di oggi dipende dai giorni precedenti (dati di training)")
    fig.text(0.01, 0.9, "Banda ombreggiata = significatività al 95%. Decadimento lento dell'ACF + taglio della "
             "PACF dopo i lag 1–2 = tipico processo autoregressivo.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "02_acf_pacf_acerra.png")


def fig03_prophet_components(d):
    c = d.comp_all[(d.comp_all.Inquinante == "PM10") & (d.comp_all.Stazione == CASE_STATION)].copy()
    c["yearly_pct"] = (np.exp(c.yearly) - 1) * 100  # le componenti sono in scala log: effetto in %
    c["weekly_pct"] = (np.exp(c.weekly) - 1) * 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    ax.plot(c.ds, c.yearly_pct, color=AQUA, lw=2)
    ax.axhline(0, color=AXIS, lw=0.9)
    ax.set_title("Stagionalità annuale")
    ax.set_ylabel("effetto sul PM10 (%)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    date_axis(ax, "%b")
    ax = axes[1]
    wk = c.groupby(c.ds.dt.day_of_week).weekly_pct.first()
    ax.bar(range(7), wk.values, width=0.55, color=AQUA)
    ax.axhline(0, color=AXIS, lw=0.9)
    ax.set_xticks(range(7), [w.capitalize() for w in IT_WEEKDAYS])
    ax.set_title("Stagionalità settimanale")
    ax.grid(axis="x", visible=False)
    title(fig, "Acerra ZI – cosa ha imparato Prophet (effetto moltiplicativo rispetto al trend)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "03_prophet_components_acerra.png")


def fig04_forecast_test_period(d):
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for ax, (st, name) in zip(axes, PM10_STATIONS.items()):
        g = d.pred[d.pred.Stazione == st].set_index("date")
        ax.plot(g.index, g.Observed, color=INK, lw=1.2, label="Osservato")
        ax.plot(g.index, g.XGBoost, color=BLUE, lw=1.6, label="XGBoost (usa i valori recenti)")
        ax.plot(g.index, g.Prophet, color=AQUA, lw=1.6, label="Prophet (solo stagionalità)")
        threshold_line(ax)
        ax.set_title(name)
        ax.set_ylabel("µg/m³")
        ax.set_ylim(0, min(150, np.nanmax(g.Observed) * 1.05))
    axes[0].legend(loc="upper left", ncol=3, bbox_to_anchor=(0, 1.32))
    date_axis(axes[-1], "%b")
    title(fig, "Periodo di test 2026 – previsioni a un giorno vs PM10 osservato", 13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "04_forecast_test_period.png")


def fig05_zoom(d):
    g = d.pred[(d.pred.Stazione == CASE_STATION) & (d.pred.date.between(*ZOOM))].set_index("date")
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(g.index, g.Persistence, color=NEUTRAL, lw=1.6, label="Persistenza (valore di ieri)")
    ax.plot(g.index, g.SARIMAX, color=ORANGE, lw=2, label="SARIMAX")
    ax.plot(g.index, g.XGBoost, color=BLUE, lw=2, label="XGBoost")
    ax.plot(g.index, g.Observed, color=INK, lw=1.4, marker="o", ms=4.5, mec=SURFACE, mew=1.2, label="Osservato")
    threshold_line(ax)
    ax.set_ylabel("µg/m³")
    ax.legend(loc="upper left", ncol=4, bbox_to_anchor=(0, 1.12))
    date_axis(ax, "%d %b")
    title(fig, "Acerra ZI, gen–mar 2026 – le previsioni arrivano in ritardo rispetto alle osservazioni "
               "e appiattiscono i picchi")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "05_zoom_acerra.png")


def fig06_mae_by_station(d):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6))
    for ax, (st, name) in zip(axes.flat, PM10_STATIONS.items()):
        g = d.ev[d.ev.Stazione == st]
        vals = [mae(g, m) for m in d.order]
        colors = [NEUTRAL if m in ("Persistence", "Seasonal naive (t-7)") else BLUE for m in d.order]
        ax.barh(range(len(d.order)), vals, height=0.55, color=colors)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.1f}", (v, i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8,
                        color=INK2)
        ax.axvline(mae(g, "Persistence"), color=MUTED, lw=0.9)
        ax.set_yticks(range(len(d.order)), [model_label(m) for m in d.order])
        ax.set_title(f"{name}  (n={len(g)})")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(vals) * 1.15)
    for ax in axes[-1]:
        ax.set_xlabel("MAE (µg/m³) – più basso è meglio")
    title(fig, "Errore assoluto medio per stazione – grigio = baseline naive, linea verticale = persistenza")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "06_mae_by_station.png")


def fig07_scatter(d):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharex=True, sharey=True)
    lim = 140
    for ax, m in zip(axes, ["Persistence", "SARIMAX", "XGBoost"]):
        ax.scatter(d.ev.Observed, d.ev[m], s=12, color=BLUE, alpha=0.35, lw=0)
        ax.plot([0, lim], [0, lim], color=INK2, lw=1)
        threshold_line(ax, label=False)
        threshold_line(ax, horizontal=False)
        missed = ((d.ev.Observed > PM10_LIMIT) & (d.ev[m] <= PM10_LIMIT)).sum()
        ax.text(lim * 0.97, 4, f"superamenti mancati: {missed}", ha="right", fontsize=8.5, color=INK2)
        ax.set_title(model_label(m))
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("PM10 osservato (µg/m³)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("PM10 previsto (µg/m³)")
    axes[0].annotate("previsione perfetta", xy=(105, 105), xytext=(60, 125), fontsize=8.5, color=INK2,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    title(fig, "Previsto vs osservato (tutte e 4 le stazioni) – i modelli spingono i valori alti verso il centro")
    fig.text(0.01, 0.9, f"Riquadro in basso a destra = giorni sopra {PM10_LIMIT} µg/m³ che il modello ha previsto "
             f"sotto {PM10_LIMIT}.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "07_scatter_forecast_vs_observed.png")


def fig08_bias_by_level(d):
    bins = [0, 10, 20, 30, 40, 50, 70, 1000]
    labels = ["0–10", "10–20", "20–30", "30–40", "40–50", "50–70", ">70"]
    ev = d.ev.assign(level=pd.cut(d.ev.Observed, bins, labels=labels))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for m in ["Persistence", "SARIMAX", "Prophet", "XGBoost"]:
        b = (ev[m] - ev.Observed).groupby(ev.level, observed=False).mean()
        ax.plot(range(len(labels)), b.values, color=MODEL_COLOR[m], lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5,
                label=model_label(m))
        ax.annotate(model_label(m), (len(labels) - 1, b.values[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.axhline(0, color=INK2, lw=1)
    counts = ev.level.value_counts().reindex(labels)
    ax.set_xticks(range(len(labels)), [f"{lb}\n(n={n})" for lb, n in zip(labels, counts)])
    ax.set_xlim(-0.3, len(labels) - 0.3)
    ax.set_xlabel("fascia di PM10 osservato (µg/m³)")
    ax.set_ylabel("errore medio, previsto − osservato (µg/m³)")
    ax.legend(loc="lower left")
    title(fig, "Errore per livello di inquinamento – tutti i modelli sovrastimano i giorni puliti e sottostimano "
               "quelli inquinati")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "08_bias_by_level.png")


def fig09_exceedance_precision_recall(d):
    pr = pd.DataFrame([(m, *hit_rates(d.ev.Observed, d.ev[m], PM10_LIMIT)) for m in MODELS],
                      columns=["model", "precision", "recall"]).set_index("model").loc[d.order]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    yy = np.arange(len(pr))
    bar_h = 0.36
    ax.barh(yy + bar_h / 2 + 0.01, pr.precision, height=bar_h, color=BLUE,
            label="Precision – allarmi previsti che erano reali")
    ax.barh(yy - bar_h / 2 - 0.01, pr.recall, height=bar_h, color=ORANGE,
            label="Recall – superamenti reali che sono stati intercettati")
    for i, (pv, rv) in enumerate(zip(pr.precision, pr.recall)):
        ax.annotate(f"{pv:.2f}", (pv, i + bar_h / 2), xytext=(3, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK2)
        ax.annotate(f"{rv:.2f}", (rv, i - bar_h / 2), xytext=(3, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK2)
    ax.set_yticks(yy, [model_label(m) for m in pr.index])
    ax.set_xlim(0, 1.05)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.16), ncol=1)
    n_ex = int((d.ev.Observed > PM10_LIMIT).sum())
    title(fig, f"Rilevamento dei giorni con PM10 > {PM10_LIMIT} µg/m³ ({n_ex} osservati nel test set)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "09_exceedance_precision_recall.png")


def fig10_xgboost_importance(d):
    imp = d.imp_all[d.imp_all.Inquinante == "PM10"]
    top = imp.head(12).iloc[::-1]
    nice = {"y_last": "ultimo PM10 osservato", "y_lag1": "PM10 di ieri", "y_lag2": "PM10 di 2 giorni fa",
            "y_lag3": "PM10 di 3 giorni fa", "y_lag7": "PM10 di 7 giorni fa", "y_lag14": "PM10 di 14 giorni fa",
            "y_roll7": "PM10 media 7 giorni", "y_roll30": "PM10 media 30 giorni", "PM2.5_lag1": "PM2.5 di ieri",
            "NO2_lag1": "NO2 di ieri", "O3_lag1": "O3 di ieri", "reg_lag1": "PM10 regionale di ieri",
            "reg_lag2": "PM10 regionale di 2 giorni fa", "Stazione": "stazione", "holiday": "festività",
            "doy_sin": "giorno dell'anno (sin)", "doy_cos": "giorno dell'anno (cos)", "dow": "giorno della settimana",
            "month": "mese"}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(range(len(top)), top.importance, height=0.55, color=BLUE)
    for i, v in enumerate(top.importance):
        ax.annotate(f"{v:.3f}", (v, i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8,
                    color=INK2)
    ax.set_yticks(range(len(top)), [nice.get(f, f) for f in top.feature])
    ax.set_xlabel("importance (quota del gain totale)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, top.importance.max() * 1.15)
    title(fig, "XGBoost – quali input guidano la previsione")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "10_xgboost_feature_importance.png")


# ---- 11–14: confronto tra inquinanti ----------------------------------------------------------------------

def fig11_skill_heatmap(d):
    cols = [m for m in MODELS if m != "Persistence"]
    evs = {p: complete_cases(d.pred_all, p) for p in d.pols}
    skill = pd.DataFrame({m: [(1 - mae(evs[p], m) / mae(evs[p], "Persistence")) * 100 for p in d.pols]
                          for m in cols}, index=d.pols)
    diverging = LinearSegmentedColormap.from_list("div", ["#a32f2f", "#e34948", "#f0efec", "#2a78d6", "#184f95"])
    lim = 20  # colori saturati a ±20%
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(d.pols) + 1.6))
    ax.imshow(skill.clip(-lim, lim).values, cmap=diverging, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(d.pols)):
        for j in range(len(cols)):
            v = skill.iat[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 12 else INK, fontweight="bold" if v == skill.iloc[i].max() else "normal")
    ax.set_xticks(range(len(cols)), [model_label(m) for m in cols])
    ax.set_yticks(range(len(d.pols)), [POL_LABEL.get(p, p) for p in d.pols])
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(d.pols)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.xaxis.tick_top()
    title(fig, "Skill vs persistenza (%) per inquinante e modello – blu = meglio di \"domani = oggi\", rosso = peggio")
    fig.text(0.01, 0.005, f"In grassetto il modello migliore di ogni riga. Colori saturati a ±{lim}%; "
             "i numeri riportano il valore reale.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    save(fig, "11_skill_heatmap_pollutants.png")


def fig12_other_pollutants(d):
    example_station = {"PM2.5": "IT2212A", "NO2": "IT1491A", "O3": "IT2276A", "SO2": "IT2211A", "CO": "IT1491A",
                       "C6H6": "IT2212A"}
    show = [p for p in example_station if p in d.pols]
    fig, axes = plt.subplots(len(show), 1, figsize=(11, 2.35 * len(show) + 0.8), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, pol in zip(axes, show):
        st = example_station[pol]
        unit = POLLUTANTS[pol]["unit"]
        g = d.pred_all[(d.pred_all.Inquinante == pol) & (d.pred_all.Stazione == st)].set_index("date")
        ax.plot(g.index, g.Observed, color=INK, lw=1.1, label="Osservato")
        ax.plot(g.index, g.XGBoost, color=BLUE, lw=1.5, label="XGBoost (usa i valori recenti)")
        ax.plot(g.index, g.Prophet, color=AQUA, lw=1.5, label="Prophet (solo stagionalità)")
        if pol in THR and np.nanmax(g.Observed) > THR[pol] * 0.6:  # la soglia solo se è vicina ai dati
            ax.axhline(THR[pol], color=MUTED, lw=0.9, zorder=1)
            ax.annotate(f"soglia {THR[pol]} {unit}", xy=(1, THR[pol]),
                        xycoords=("axes fraction", "data"), xytext=(-2, 3), textcoords="offset points", ha="right",
                        va="bottom", fontsize=8, color=MUTED, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
        ax.set_title(f"{POL_LABEL.get(pol, pol)} – {STATION_NAMES[st]}")
        ax.set_ylabel(unit)
        ax.set_ylim(0, np.nanquantile(g.Observed, 0.995) * 1.15)
    axes[0].legend(loc="upper left", ncol=3, bbox_to_anchor=(0, 1.42))
    date_axis(axes[-1], "%b")
    title(fig, "Periodo di test 2026 – previsioni a un giorno per gli altri inquinanti (una stazione ciascuno)", 13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, "12_forecast_other_pollutants.png")


IMPORTANCE_GROUPS = [("target ieri / ultimo valore", BLUE), ("target giorni precedenti e medie", ORANGE),
                     ("altri inquinanti ieri", AQUA), ("media regionale del target", YELLOW),
                     ("calendario", MAGENTA), ("stazione", GREEN)]


def feature_group(f):
    """Gruppo di una feature XGBoost del modello a un giorno (vedi build_xgb_features)."""
    names = [g for g, _ in IMPORTANCE_GROUPS]
    if f in ("y_lag1", "y_last"):
        return names[0]
    if f.startswith("y_"):
        return names[1]
    if f.startswith("reg"):
        return names[3]
    if f.endswith("_lag1"):
        return names[2]
    if f == "Stazione":
        return names[5]
    return names[4]


def fig13_importance_groups(d):
    share = (d.imp_all.assign(group=d.imp_all.feature.map(feature_group))
             .pivot_table(index="Inquinante", columns="group", values="importance", aggfunc="sum")
             .reindex(index=d.pols[::-1], columns=[g for g, _ in IMPORTANCE_GROUPS]).fillna(0))
    share = share.div(share.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(d.pols) + 1.9))
    left = np.zeros(len(share))
    for gname, color in IMPORTANCE_GROUPS:
        v = share[gname].values
        ax.barh(range(len(share)), v, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2,
                label=gname)
        for i, (lft, w) in enumerate(zip(left, v)):
            if w >= 6:
                ax.text(lft + w / 2, i, f"{w:.0f}%", ha="center", va="center", fontsize=8.5,
                        color="white" if color in (BLUE, GREEN) else INK)
        left += v
    ax.set_yticks(range(len(share)), [POL_LABEL.get(p, p) for p in share.index])
    ax.set_xlim(0, 100)
    ax.set_xlabel("quota del gain totale (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, 1.2))
    title(fig, "XGBoost – da dove viene la previsione, per inquinante")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "13_xgb_importance_groups.png")


def fig14_exceedances(d):
    models = ["Persistence", "SARIMAX", "XGBoost"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.75 * len(d.ex_pols) + 1.9), sharey=True)
    bar_h = 0.24
    for ax, kind in zip(axes, ["Precision", "Recall"]):
        for k, m in enumerate(models):
            vals = []
            for p in d.ex_pols:
                e = complete_cases(d.pred_all, p)
                precision, recall = hit_rates(e.Observed, e[m], THR[p])
                vals.append(precision if kind == "Precision" else recall)
            yy = np.arange(len(d.ex_pols))[::-1] + (1 - k) * (bar_h + 0.03)
            ax.barh(yy, vals, height=bar_h, color=MODEL_COLOR[m], label=model_label(m))
            for y_, v in zip(yy, vals):
                ax.annotate(f"{v:.2f}", (v, y_), xytext=(3, 0), textcoords="offset points", va="center",
                            fontsize=8, color=INK2)
        ax.set_xlim(0, 1.1)
        ax.set_title(kind + (" – allarmi previsti che erano reali" if kind == "Precision"
                             else " – superamenti reali intercettati"))
        ax.grid(axis="y", visible=False)
    n_ex = {p: int((complete_cases(d.pred_all, p).Observed > THR[p]).sum()) for p in d.ex_pols}
    axes[0].set_yticks(np.arange(len(d.ex_pols))[::-1],
                       [f"{POL_LABEL.get(p, p)} > {THR[p]}\n(n={n_ex[p]})" for p in d.ex_pols])
    axes[0].legend(loc="upper left", ncol=3, bbox_to_anchor=(0, 1.22))
    title(fig, "Rilevamento dei superamenti per inquinante (4 stazioni aggregate)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "14_exceedances_pollutants.png")


# ---- 15–18: orizzonte da 1 a 7 giorni e previsione dopo l'ultimo dato ---------------------------------------

H_MODELS = ["Persistence", "SARIMAX", "Prophet", "XGBoost"]
HORIZON_COLOR = {1: "#104281", 3: "#2a78d6", 7: "#86b6ef"}  # rampa di blu: più chiaro = orizzonte più lungo


def fig15_mae_by_horizon(d):
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.4))
    hs = np.arange(1, H + 1)
    for ax, pol in zip(axes.flat, d.pols):
        e = common_horizon_sample(d.pred_h, pol)
        mae_h = {m: e.groupby("h").apply(lambda g: mae(g, m)).reindex(hs) for m in H_MODELS}
        for m in H_MODELS:
            ax.plot(hs, mae_h[m].values, color=MODEL_COLOR[m], lw=2, marker="o", ms=4.5, mec=SURFACE, mew=1.2,
                    ls="--" if m == "Prophet" else "-")
        ax.set_ylim(0, max(v.max() for v in mae_h.values()) * 1.12)
        ax.set_xticks(hs)
        ax.set_title(f"{POL_LABEL.get(pol, pol)} ({POLLUTANTS[pol]['unit']})")
        if pol == "SO2":  # Prophet è molto peggiore degli altri: lo si lascia fuori scala per leggere il resto
            ax.annotate(f"Prophet: MAE {it(mae_h['Prophet'].max(), 2)}\n(fuori scala)", xy=(0.97, 0.97),
                        xycoords="axes fraction", ha="right", va="top", fontsize=8, color=INK2)
            ax.set_ylim(0, max(mae_h[m].max() for m in H_MODELS if m != "Prophet") * 1.25)
    for ax in axes[-1]:
        ax.set_xlabel("orizzonte h (giorni)")
    for ax in axes[:, 0]:
        ax.set_ylabel("MAE")
    # l'ottavo riquadro (vuoto: gli inquinanti sono 7) ospita la legenda
    leg = axes.flat[-1]
    leg.axis("off")
    for i, m in enumerate(H_MODELS):
        y0 = 0.78 - i * 0.16
        leg.plot([0.05, 0.25], [y0, y0], color=MODEL_COLOR[m], lw=2.2, ls="--" if m == "Prophet" else "-",
                 transform=leg.transAxes)
        leg.text(0.3, y0, model_label(m) + (" (non usa i dati recenti)" if m == "Prophet" else ""), va="center",
                 fontsize=9, color=INK2, transform=leg.transAxes)
    title(fig, "Come cresce l'errore allungando l'orizzonte – sale nei primi 3–4 giorni, poi si stabilizza")
    fig.text(0.01, 0.935, "MAE sul periodo di test 2026, 4 stazioni aggregate. h = giorni tra l'ultimo dato usato e "
             "il giorno previsto. Asse y da zero; scale diverse per inquinante.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "15_mae_by_horizon.png")


def fig16_zoom_horizons(d):
    p = d.pred_h
    g = p[(p.Inquinante == "PM10") & (p.Stazione == CASE_STATION) & p.date.between(*ZOOM)]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for h, color in HORIZON_COLOR.items():
        gh = g[g.h == h].set_index("date")
        ax.plot(gh.index, gh.SARIMAX, color=color, lw=2,
                label=f"SARIMAX, previsione emessa {h} giorn{'o' if h == 1 else 'i'} prima")
    go = g[g.h == 1].set_index("date")
    ax.plot(go.index, go.Observed, color=INK, lw=1.4, marker="o", ms=4.5, mec=SURFACE, mew=1.2, label="Osservato")
    threshold_line(ax)
    ax.set_ylabel("µg/m³")
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0, 1.2))
    date_axis(ax, "%d %b")
    title(fig, "Acerra ZI, gen–mar 2026 – più l'orizzonte è lungo, più la previsione si appiattisce sulla media")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "16_zoom_horizons_acerra.png")


def fig17_exceedance_f1_by_horizon(d):
    hs = np.arange(1, H + 1)
    fig, axes = plt.subplots(1, len(d.ex_pols), figsize=(12, 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, pol in zip(axes, d.ex_pols):
        e = common_horizon_sample(d.pred_h, pol)
        for m in H_MODELS:
            f1 = []
            for h in hs:
                eh = e[e.h == h]
                precision, recall = hit_rates(eh.Observed, eh[m], THR[pol])
                f1.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0)
            ax.plot(hs, f1, color=MODEL_COLOR[m], lw=2, marker="o", ms=4.5, mec=SURFACE, mew=1.2,
                    ls="--" if m == "Prophet" else "-", label=model_label(m))
        n_ex = int((e[e.h == 1].Observed > THR[pol]).sum())
        ax.set_title(f"{POL_LABEL.get(pol, pol)} > {THR[pol]}  (n={n_ex})")
        ax.set_xticks(hs)
        ax.set_xlabel("orizzonte h (giorni)")
        ax.set_ylim(0, 0.75)
    axes[0].set_ylabel("F1 sui superamenti")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=4, bbox_to_anchor=(0.01, 0.9))
    title(fig, "Superamenti previsti con anticipo – dopo il primo giorno l'F1 crolla, soprattutto per XGBoost")
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    save(fig, "17_exceedance_f1_by_horizon.png")


def fig18_next7_pm10(d):
    f = d.next7[d.next7.Inquinante == "PM10"]
    best = f.Modello.iloc[0]
    last_day = d.pred_h.date.max()
    cut = last_day + pd.Timedelta(hours=12)  # linea verticale tra l'ultimo dato e il primo giorno previsto
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True, sharey=True)
    for ax, (st, name) in zip(axes.flat, PM10_STATIONS.items()):
        o = d.panel[(d.panel.Stazione == st) & (d.panel.date > last_day - pd.Timedelta(days=42))]
        o = o.set_index("date").PM10
        o = o.reindex(pd.date_range(o.index.min(), last_day))
        fs = f[f.Stazione == st].sort_values("date")
        ax.plot(o.index, o.values, color=INK, lw=1.3, marker="o", ms=3.5, mec=SURFACE, mew=1, label="Osservato")
        ax.fill_between(fs.date, fs.Basso80, fs.Alto80, color=BLUE, alpha=0.18, lw=0, label="intervallo 80%")
        ax.plot(fs.date, fs.Previsione, color=BLUE, lw=2, marker="o", ms=4, mec=SURFACE, mew=1,
                label=f"Previsione {model_label(best)}")
        ax.axvline(cut, color=MUTED, lw=0.9)
        threshold_line(ax, label=ax is axes.flat[1])
        ax.set_title(name)
        ax.set_ylim(0, 80)
    for ax in axes[:, 0]:
        ax.set_ylabel("µg/m³")
    for ax in axes[-1]:
        date_axis(ax, "%d %b")
    axes.flat[0].annotate("ultimo dato", xy=(cut, 1), xycoords=("data", "axes fraction"),
                          xytext=(-4, -4), textcoords="offset points", ha="right", va="top", fontsize=8.5, color=INK2)
    axes.flat[0].legend(loc="upper left", ncol=3, bbox_to_anchor=(0, 1.32))
    title(fig, f"PM10 – previsione per i 7 giorni dopo l'ultimo dato "
               f"({last_day.day} {IT_MONTHS[last_day.month - 1]} {last_day.year})", 13)
    fig.text(0.01, 0.925, "Intervallo all'80% empirico: percentili 10–90 degli errori di test 2026 a ciascun "
             "orizzonte. 1 novembre = festività.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "18_next7_pm10.png")


FIGURES = [
    fig01_data_train_test, fig02_acf_pacf, fig03_prophet_components, fig04_forecast_test_period, fig05_zoom,
    fig06_mae_by_station, fig07_scatter, fig08_bias_by_level, fig09_exceedance_precision_recall,
    fig10_xgboost_importance, fig11_skill_heatmap, fig12_other_pollutants, fig13_importance_groups,
    fig14_exceedances, fig15_mae_by_horizon, fig16_zoom_horizons, fig17_exceedance_f1_by_horizon, fig18_next7_pm10,
]


def run_plots():
    """FASE 3."""
    print("Fase 3 – grafici")
    setup_matplotlib()
    FIG_DIR.mkdir(exist_ok=True)
    d = load_plot_data()
    for make_figure in FIGURES:
        make_figure(d)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 7. AVVIO DA RIGA DI COMANDO
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

STEPS = ["prep", "forecast", "plots"]


def main():
    parser = argparse.ArgumentParser(description="Previsione della qualità dell'aria in Campania (vedi docstring).")
    parser.add_argument("steps", nargs="*", metavar="fase",
                        help=f"fasi da eseguire, in ordine: {', '.join(STEPS)} (default: tutte)")
    parser.add_argument("--pollutants", nargs="+", default=list(POLLUTANTS), metavar="INQ",
                        help=f"inquinanti per la fase forecast (default: tutti): {', '.join(POLLUTANTS)}")
    args = parser.parse_args()

    steps = args.steps or STEPS
    if unknown := set(steps) - set(STEPS):
        parser.error(f"fasi sconosciute: {sorted(unknown)}. Disponibili: {STEPS}")
    if unknown := set(args.pollutants) - set(POLLUTANTS):
        parser.error(f"inquinanti sconosciuti: {sorted(unknown)}. Disponibili: {list(POLLUTANTS)}")

    if "prep" in steps:
        run_prep()
    if "forecast" in steps:
        run_forecast(args.pollutants)
    if "plots" in steps:
        run_plots()


if __name__ == "__main__":
    main()
