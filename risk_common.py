#!/usr/bin/env python3
"""
Beursrally 2026 - risicomodel: de gedeelde basis.

Elk volatiliteitsmodel (risk_simple, risk_garch, risk_har, ...) gebruikt deze module, zodat alle modellen
op exact dezelfde datums, doelen en regels beoordeeld worden.

Afspraken
    Voorspellingsmoment  dag d na de slotkoers (beslismoment 08u00 op d+1), zoals in features.py.
    Doel                 gerealiseerde variantie over de volgende h EIGEN handelsdagen van het aandeel:
                         RV_h(d) = r_{d+1}^2 + ... + r_{d+h}^2   (log-rendementen in euro, uit data/clean)
    Voorspelling         v_h(d) = verwachte RV_h(d), in dezelfde eenheid (dagvariantie x h).
    Periodes             voorspellingen vanaf FC_START; keuzeperiode EVAL_START..SPLIT;
                         testperiode vanaf SPLIT (alleen bekijken met risk_compare.py --final).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CLEAN = DATA / "clean"
PROC = DATA / "processed"
FEAT = DATA / "features"
RISK = DATA / "risk"
FC_DIR = RISK / "forecasts"

HORIZONS = [1, 3, 5, 10, 20, 48]
FC_START = pd.Timestamp("2018-07-01")     # eerste voorspellingen (modellen hebben ~1,5 jaar opwarmtijd)
EVAL_START = pd.Timestamp("2019-01-01")   # begin keuzeperiode
SPLIT = pd.Timestamp("2022-01-01")        # begin testperiode
FLOOR = 1e-10                             # ondergrens voor RV = 0 (koers onveranderd) in QLIKE
LOGC = 1e-6                               # kleine constante in ln(variantie + c) voor HAR

VCOLS = [f"v_{h}" for h in HORIZONS]
RVCOLS = [f"rv_{h}" for h in HORIZONS]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_returns() -> pd.DataFrame:
    f = CLEAN / "logret.parquet"
    if not f.exists():
        sys.exit("data/clean/logret.parquet ontbreekt. Eerst:  python clean.py")
    return pd.read_parquet(f).astype("float64")


def own_series(lr: pd.DataFrame, c: str) -> pd.Series:
    """Rendementen van een effect op zijn eigen geldige dagen (zonder ontbrekende of verwijderde dagen)."""
    return lr[c].dropna()


# --------------------------------------------------------------------------- doelen
def realized_targets(lr: pd.DataFrame | None = None, rebuild: bool = False) -> pd.DataFrame:
    """Lange tabel (date, isin, rv_1 ... rv_48): gerealiseerde variantie over de volgende h eigen dagen.

    Wordt bewaard in data/risk/targets.parquet en opnieuw berekend als de rendementen nieuwer zijn.
    """
    f = RISK / "targets.parquet"
    src = CLEAN / "logret.parquet"
    given = lr is not None                       # eigen (bv. gewijzigde) rendementen: niet bewaren
    if not given and f.exists() and not rebuild and f.stat().st_mtime > src.stat().st_mtime:
        return pd.read_parquet(f)
    lr = load_returns() if lr is None else lr
    parts = []
    for c in lr.columns:
        s = own_series(lr, c)
        n = len(s)
        if n < 2:
            continue
        cs = np.concatenate([[0.0], np.cumsum(s.values ** 2)])
        out = {"date": s.index, "isin": c}
        for h in HORIZONS:
            v = np.full(n, np.nan)
            i = np.arange(n - h)
            v[i] = cs[i + 1 + h] - cs[i + 1]           # r_{i+1}^2 + ... + r_{i+h}^2
            out[f"rv_{h}"] = v
        parts.append(pd.DataFrame(out))
    t = pd.concat(parts, ignore_index=True)
    if not given:
        RISK.mkdir(parents=True, exist_ok=True)
        t.to_parquet(f, index=False)
    return t


# --------------------------------------------------------------------------- voorspellingen opslaan/lezen
def save_forecast(name: str, df: pd.DataFrame) -> None:
    """df: kolommen date, isin, v_1 ... v_48 (variantie over h dagen)."""
    FC_DIR.mkdir(parents=True, exist_ok=True)
    df = df[["date", "isin"] + VCOLS].copy()
    df = df[df["date"] >= FC_START]
    for c in VCOLS:
        df[c] = df[c].astype("float64").where(df[c] > 0)
    df.to_parquet(FC_DIR / f"{name}.parquet", index=False)
    log(f"Voorspellingen '{name}': {len(df):,} rijen, {df['isin'].nunique()} effecten -> data/risk/forecasts/{name}.parquet")


def load_forecast(name: str) -> pd.DataFrame:
    return pd.read_parquet(FC_DIR / f"{name}.parquet")


def available_forecasts() -> list[str]:
    return sorted(p.stem for p in FC_DIR.glob("*.parquet")) if FC_DIR.exists() else []


def from_daily_variance(var: pd.DataFrame) -> pd.DataFrame:
    """Brede tabel dagvariantie (datums x isin) -> lange tabel met v_h = h x dagvariantie."""
    s = var.stack().rename("var1").reset_index()
    s.columns = ["date", "isin", "var1"]
    s = s.dropna(subset=["var1"])                # dagen zonder handel of te weinig historiek: geen rij
    for h in HORIZONS:
        s[f"v_{h}"] = h * s["var1"]
    return s.drop(columns="var1")


# --------------------------------------------------------------------------- maatstaven
def qlike(rv: np.ndarray, f: np.ndarray) -> np.ndarray:
    """QLIKE-verlies: RV/F - ln(RV/F) - 1. Nul bij een perfecte voorspelling; onderschatten weegt zwaarder.

    De ondergrens voor RV verandert de VERSCHILLEN tussen modellen niet (de term ln RV is voor alle modellen gelijk).
    """
    x = np.maximum(rv, FLOOR) / f
    return x - np.log(x) - 1


def nw_tstat(d: np.ndarray, lag: int) -> float:
    """t-waarde van het gemiddelde van een tijdreeks met Newey-West-standaardfout (overlappende horizonnen)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 20:
        return np.nan
    e = d - d.mean()
    s = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        s += 2 * w * (e[k:] @ e[:-k]) / n
    return d.mean() / np.sqrt(s / n) if s > 0 else np.nan


def dm_test(loss_a: pd.Series, loss_b: pd.Series, dates: pd.Series, h: int) -> float:
    """Diebold-Mariano: t-waarde van gemiddeld(verlies A - verlies B), eerst per datum gemiddeld over aandelen.

    Negatief = A beter. Per datum middelen: aandelen op dezelfde dag hangen sterk samen.
    """
    d = (loss_a - loss_b).groupby(dates.values).mean().sort_index()
    return nw_tstat(d.values, lag=max(h, 5))


def period_mask(dates: pd.Series, period: str) -> pd.Series:
    if period == "keuze":
        return (dates >= EVAL_START) & (dates < SPLIT)
    if period == "test":
        return dates >= SPLIT
    raise ValueError(period)


def names() -> dict:
    t = pd.read_csv(DATA / "meta" / "tickers.csv", dtype=str)
    return dict(zip(t["isin"], t["name"]))


def ticker_to_isin() -> dict:
    t = pd.read_csv(DATA / "meta" / "tickers.csv", dtype=str)
    return dict(zip(t["ticker"], t["isin"]))


# --------------------------------------------------------------------------- kwartaalcijfers: exacte reactiedag
def earnings_reaction_k(rows: pd.DataFrame, known_ahead_days: int = 30) -> pd.Series:
    """Voor elke rij (date = dag d, isin): binnen hoeveel beursdagen na d valt de koersreactie op de volgende
    kwartaalcijfers? 1 = de eerstvolgende beursdag. NaN = geen cijfers bekend binnen het venster.

    Reactiedag: publicatie voor 15u Belgische tijd (Europese ochtend, VS voor de opening) -> dezelfde dag;
    later (VS na de slotbel) -> de volgende werkdag. Point-in-time: een datum telt alleen mee als hij op het
    beslismoment (08u00 op d+1) al bekend was, en dat nemen we aan tot 30 dagen vooraf.
    Ook cijfers die op dag d zelf na de slotbel kwamen (reactie op d+1) tellen mee: die zitten in het venster.
    """
    f = CLEAN / "earnings.parquet"
    out = pd.Series(np.nan, index=rows.index)
    if not f.exists():
        return out
    e = pd.read_parquet(f, columns=["isin", "known_at"]).dropna()
    if e.empty:
        return out
    kt = pd.to_datetime(e["known_at"])
    react = kt.dt.normalize().where(kt.dt.hour < 15, kt.dt.normalize() + pd.offsets.BDay(1))
    react = react.where(react.dt.dayofweek < 5, react + pd.offsets.BDay(1))
    e = e.assign(react=pd.to_datetime(react.values).astype("datetime64[ns]"), isin=e["isin"].astype(str))
    e = e.drop_duplicates(["isin", "react"]).sort_values("react")
    left = rows[["date", "isin"]].copy()
    left["isin"] = left["isin"].astype(str)
    left["date"] = pd.to_datetime(left["date"]).astype("datetime64[ns]")
    left["nxt"] = left["date"] + pd.Timedelta(days=1)
    left["_i"] = np.arange(len(left))
    left = left.sort_values("nxt")
    m = pd.merge_asof(left, e[["isin", "react", "known_at"]], left_on="nxt", right_on="react", by="isin",
                      direction="forward", tolerance=pd.Timedelta(days=known_ahead_days + 20))
    decision = m["date"] + pd.Timedelta(days=1, hours=8)
    ok = m["react"].notna() & (m["known_at"] - pd.Timedelta(days=known_ahead_days) <= decision)
    k = pd.Series(np.nan, index=m.index)
    d1 = m.loc[ok, "nxt"].values.astype("datetime64[D]")
    rd = m.loc[ok, "react"].values.astype("datetime64[D]")
    k[ok] = np.busday_count(d1, rd) + 1
    k.index = m["_i"].values
    out[:] = k.sort_index().values
    return out
