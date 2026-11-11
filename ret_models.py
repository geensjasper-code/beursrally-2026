#!/usr/bin/env python3
"""
Beursrally 2026 - rendementsmodel (stap 2): walk-forward voor vijf kandidaten en zes horizonnen.

Gebruik
    python ret_models.py                 # alle modellen
    python ret_models.py enet boost      # alleen deze modellen

Wat voorspeld wordt
    Het AANDEELSPECIFIEKE deel van het rendement: y_h - (gemiddelde y_h van alle aandelen op die dag), met
    y_h = ln(P(t+h) / Open(t+1)) uit features.py. Het marktdeel voorspellen we niet (vast 7% per jaar).

Kandidaten
    zero       alle aandelen gelijk (voorspelling 0)                                      niets geschat
    capm       (beta - gemiddelde beta) x 7% x h/252                                      niets geschat
    composite  gemiddelde van 4 signalen met vaste tekens uit de literatuur:              1 schaalfactor
               12-maandsmomentum (+), grootte (-), illiquiditeit (+), laatste cijferverrassing (+)
    enet       elastic net op alle kenmerken (ridge/lasso/mengsel gekozen per herschatting) ~45 coefficienten
    boost      gradient boosting (LightGBM) op dezelfde kenmerken                          bomen

Voorbereiding kenmerken
    Per dag omgezet naar rangen tussen -0,5 en +0,5 (alleen de doorsnede van die dag: geen lek).
    Ontbrekend: 0 plus een indicator 'ontbreekt' (boosting krijgt de lege waarden zelf).
    Marktbrede kenmerken (VIX, rente, maand, marktrendement) zijn op een dag voor elk aandeel gelijk en
    zeggen dus niets over de rangorde: niet gebruikt.

Zonder lek
    Walk-forward: elk kwartaal (boosting: elk halfjaar) opnieuw geschat op rijen tot herschattingsdatum
    - h - 5 beursdagen (embargo: hun doel is afgelopen). Instellingen (straf, ridge/lasso, bladeren) gekozen
    op de laatste 20% van de trainingsdatums (met een kloof van h dagen), nooit op de voorspelde periode.

Uitvoer: data/returns/pred_<model>.parquet (date, isin, p_1 ... p_48)
"""
from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FEAT = DATA / "features"
OUT = DATA / "returns"

HORIZONS = [1, 3, 5, 10, 20, 48]
FC_START = pd.Timestamp("2018-07-01")
MARKET_PA = 0.07
EMBARGO_EXTRA = 5
MAX_ROWS = int(os.environ.get("BR_RET_ROWS", "200000"))
BOOST_ROUNDS = int(os.environ.get("BR_RET_ROUNDS", "400"))

MARKET_FEATS = ["vix", "vix_chg_21", "rate_10y", "rate_10y_chg_21", "month", "days_to_year_end", "mkt_ret_21"]
SPARSE = ["last_surprise", "days_since_earnings", "days_to_earnings", "earnings_move_avg", "log_mcap",
          "short_pct_shares", "days_to_cover", "short_chg"]
COMPOSITE = {"mom_12_1": +1, "log_mcap": -1, "amihud": +1, "last_surprise": +1}
MODELS = ["zero", "capm", "composite", "enet", "boost"]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------- paneel
def load_panel(F: pd.DataFrame | None = None, T: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Kenmerken (gerangschikt per dag) + aandeelspecifieke doelen. F/T meegeven = gewijzigde data (lektest)."""
    F = pd.read_parquet(FEAT / "features.parquet") if F is None else F
    T = pd.read_parquet(FEAT / "targets.parquet") if T is None else T
    P = F.merge(T, on=["date", "isin"], how="left").sort_values(["date", "isin"]).reset_index(drop=True)
    feats = [c for c in F.columns if c not in ("date", "isin") and c not in MARKET_FEATS]
    g = P.groupby("date")
    R = g[feats].rank(pct=True) - 0.5
    R.columns = [f"r_{c}" for c in feats]
    for c in SPARSE:
        if c in feats:
            R[f"na_{c}"] = P[c].isna().astype("float32")
    for h in HORIZONS:
        y = P[f"y_{h}"].astype("float64")     # altijd float64: zelfde uitkomst ongeacht pandasversie of opslagtype
        P[f"yd_{h}"] = y - y.groupby(P["date"]).transform("mean")
    beta = P["beta_mkt"]
    P["capm_raw"] = beta - beta.groupby(P["date"]).transform("mean")
    comp = sum(sign * R[f"r_{c}"].fillna(0.0) for c, sign in COMPOSITE.items())
    P["composite_raw"] = comp / len(COMPOSITE)
    P = pd.concat([P[["date", "isin", "capm_raw", "composite_raw", "log_mcap"]
                     + [f"yd_{h}" for h in HORIZONS]], R.astype("float32")], axis=1)
    xcols = [c for c in R.columns]
    return P, xcols


class Calendar:
    def __init__(self, dates: pd.Series):
        self.dates = dates.to_numpy()
        self.cal = np.array(sorted(dates.unique()))
        day = np.searchsorted(self.cal, self.dates)
        self.bounds = np.searchsorted(day, np.arange(len(self.cal) + 1))

    def rows_upto(self, k: int) -> np.ndarray:
        return np.arange(0, self.bounds[k + 1]) if k >= 0 else np.array([], int)

    def rows_between(self, a: pd.Timestamp, b: pd.Timestamp) -> np.ndarray:
        lo = np.searchsorted(self.dates, np.datetime64(a))
        hi = np.searchsorted(self.dates, np.datetime64(b))
        return np.arange(lo, hi)

    def cutoff(self, refit: pd.Timestamp, h: int) -> int:
        """Laatste toegelaten trainingsdag: refit - h - EMBARGO_EXTRA beursdagen."""
        import ret_models
        return int(np.searchsorted(self.cal, np.datetime64(refit))) - h - ret_models.EMBARGO_EXTRA - 1


def subsample(idx: np.ndarray, rng) -> np.ndarray:
    if len(idx) <= MAX_ROWS:
        return idx
    half = MAX_ROWS // 2
    older = rng.choice(idx[:-half], half, replace=False)
    return np.sort(np.concatenate([older, idx[-half:]]))


def split_val(cal: Calendar, idx: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Laatste 20% van de trainingsdatums = validatie, met een kloof van h dagen ervoor."""
    d = cal.dates[idx]
    days = np.unique(d)
    n_val = max(20, int(0.2 * len(days)))
    v0 = days[-n_val]
    gap_end = days[max(0, len(days) - n_val - h)]
    return idx[d < gap_end], idx[d >= v0]


# --------------------------------------------------------------------------- modellen
def fit_predict(model: str, P: pd.DataFrame, xcols: list[str], cal: Calendar, h: int,
                train: np.ndarray, pred_rows: np.ndarray, rng) -> np.ndarray:
    y = P[f"yd_{h}"].to_numpy()
    if model == "zero":
        return np.zeros(len(pred_rows))
    if model == "capm":
        return P["capm_raw"].to_numpy()[pred_rows] * MARKET_PA * h / 252
    if model == "composite":
        x = P["composite_raw"].to_numpy()
        ok = np.isfinite(y[train])
        xt, yt = x[train][ok], y[train][ok]
        b = float(xt @ yt / (xt @ xt)) if (xt @ xt) > 0 else 0.0     # een schaalfactor, door de oorsprong
        return b * x[pred_rows]
    X = P[xcols]
    ok = np.isfinite(y[train])
    tr = subsample(train[ok], rng)
    fit_idx, val_idx = split_val(cal, tr, h)
    if len(fit_idx) < 5000 or len(val_idx) < 2000:
        return np.full(len(pred_rows), np.nan)
    if model == "enet":
        from sklearn.linear_model import ElasticNetCV
        Xa = X.iloc[np.concatenate([fit_idx, val_idx])].fillna(0.0).to_numpy(np.float64)
        ya = y[np.concatenate([fit_idx, val_idx])]
        cv = [(np.arange(len(fit_idx)), np.arange(len(fit_idx), len(fit_idx) + len(val_idx)))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ElasticNetCV(l1_ratio=[0.05, 0.5, 1.0], alphas=15, cv=cv, max_iter=3000, tol=1e-4,
                             selection="cyclic").fit(Xa, ya)
        fit_predict.last_info = {"alpha": float(m.alpha_), "l1_ratio": float(m.l1_ratio_),
                                 "nonzero": int((m.coef_ != 0).sum())}
        fit_predict.last_coef = pd.Series(m.coef_, index=xcols)
        return m.predict(X.iloc[pred_rows].fillna(0.0).to_numpy(np.float64))
    if model == "boost":
        import lightgbm as lgb
        best = None
        for leaves in (15, 63):
            params = {"objective": "regression", "learning_rate": 0.05, "num_leaves": leaves,
                      "min_data_in_leaf": 2000, "feature_fraction": 0.8, "bagging_fraction": 0.7,
                      "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "seed": 7, "deterministic": True,
                      "num_threads": int(os.environ.get("BR_RET_THREADS", "0"))}
            m = lgb.train(params, lgb.Dataset(X.iloc[fit_idx], y[fit_idx]), num_boost_round=BOOST_ROUNDS,
                          valid_sets=[lgb.Dataset(X.iloc[val_idx], y[val_idx])],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
            sc = m.best_score["valid_0"]["l2"]
            if best is None or sc < best[0]:
                best = (sc, m, leaves)
        _, m, leaves = best
        fit_predict.last_info = {"leaves": leaves, "trees": m.best_iteration}
        fit_predict.last_coef = pd.Series(m.feature_importance("gain"), index=xcols)
        return m.predict(X.iloc[pred_rows], num_iteration=m.best_iteration)
    raise ValueError(model)


def walk_forward(model: str, P: pd.DataFrame, xcols: list[str], start: pd.Timestamp, end: pd.Timestamp,
                 verbose: bool = True, horizons: list[int] | None = None) -> pd.DataFrame:
    cal = Calendar(P["date"])
    rng = np.random.default_rng(0)
    freq = "6MS" if model == "boost" else "QS"
    refits = pd.date_range(start.to_period("Q").start_time, end, freq=freq)
    out = P[["date", "isin"]].copy()
    info = []
    for h in (horizons or HORIZONS):
        pred = np.full(len(P), np.nan)
        for k, r0 in enumerate(refits):
            r1 = refits[k + 1] if k + 1 < len(refits) else end
            rows = cal.rows_between(max(r0, start), r1)
            if len(rows) == 0:
                continue
            k_cut = cal.cutoff(r0, h)                      # leest EMBARGO_EXTRA bij elke aanroep
            if k_cut < 120:
                continue
            train = cal.rows_upto(k_cut)
            pred[rows] = fit_predict(model, P, xcols, cal, h, train, rows, rng)
            if model in ("enet", "boost"):
                info.append({"model": model, "h": h, "herschatting": r0.date(), **getattr(fit_predict, "last_info", {})})
        out[f"p_{h}"] = pred
        if verbose:
            log(f"  {model} h = {h}: klaar")
    out = out[(out["date"] >= start) & (out["date"] < end)]
    if info:
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(info).to_csv(OUT / f"instellingen_{model}.csv", index=False)
    if model in ("enet", "boost") and hasattr(fit_predict, "last_coef"):
        fit_predict.last_coef.to_csv(OUT / f"laatste_{model}.csv")
    return out


def main():
    which = [m for m in sys.argv[1:] if m in MODELS] or MODELS
    log("Paneel laden (kenmerken per dag rangschikken) ...")
    P, xcols = load_panel()
    log(f"Paneel: {len(P):,} rijen, {P['isin'].nunique()} effecten, {len(xcols)} invoerkolommen")
    end = P["date"].max() + pd.Timedelta(days=1)
    OUT.mkdir(parents=True, exist_ok=True)
    for m in which:
        log(f"Model '{m}' ...")
        pred = walk_forward(m, P, xcols, FC_START, end)
        pred.to_parquet(OUT / f"pred_{m}.parquet", index=False)
        log(f"  -> data/returns/pred_{m}.parquet ({pred['p_5'].notna().sum():,} voorspellingen h=5)")
    log("Klaar. Volgende:  python ret_compare.py")


if __name__ == "__main__":
    main()
