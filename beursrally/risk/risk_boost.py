#!/usr/bin/env python3
"""
Beursrally 2026 - risicomodel 5: gradient boosting (LightGBM) als correctie op HAR.

Gebruik
    python br.py risk_boost            # correctie op HAR (eerst risk_har.py)          -> model 'boost'
    python br.py risk_boost --alleen   # boosting op zichzelf, zonder HAR               -> model 'boost_alleen'

Model (per horizon h)
    doel        y = ln(RV_h/h + c) - ln(HAR_h/h + c)     (hoeveel hoger of lager dan HAR?)
    invoer      alle HAR-kenmerken en alle blokken B1-B7 (ook shortposities), plus de HAR-voorspelling zelf
    voorspelling v_h = h * ((HAR_h/h + c) * exp(voorspeld) * gemiddelde(exp(residu)) - c)
    -> HAR doet het basiswerk en het extrapoleren; boosting zoekt wat HAR mist (niet-lineair, interacties).

Variant --alleen
    doel        y = ln(RV_h/h + c), zelfde kenmerken zonder de HAR-voorspelling
    voorspelling v_h = h * (exp(voorspeld) * gemiddelde(exp(residu)) - c)
    Voordeel: kan vanaf 2017 trainen (de correctievariant heeft HAR-voorspellingen nodig, die starten medio 2018).
    Nadeel: bomen kunnen niet voorbij het hoogste niveau uit de training voorspellen.

Zonder lek en zonder overfitting
    Elk half jaar (1 jan, 1 jul) opnieuw getraind, alleen op rijen waarvan het doel afgelopen is voor de
    herschattingsdatum (embargo h + 5 beursdagen). HAR-voorspellingen in de training zijn zelf walk-forward.
    Laatste 20% van de trainingsdatums = validatie (met een kloof van h dagen): vroeg stoppen en keuze
    tussen 15 en 63 bladeren. Er wordt NIETS afgesteld op de resultaten van de keuzeperiode.
    Voorzichtige instellingen: leersnelheid 0,05 (max. 600 bomen), minstens 1000 rijen per blad, L2-straf, steekproeven.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from beursrally.risk.risk_common import FC_START, HORIZONS, LOGC, RISK, load_forecast, log, save_forecast
from beursrally.risk.risk_har import BASE, BLOCKS, add_earnings_block, build_panel

MAX_TRAIN_ROWS = int(os.environ.get("BR_BOOST_ROWS", "300000"))
MAX_ROUNDS = int(os.environ.get("BR_BOOST_ROUNDS", "600"))
LEAVES_GRID = [15, 63]
EMBARGO_EXTRA = 5
FIRST_REFIT = pd.Timestamp("2019-01-01")


def params(leaves: int) -> dict:
    return {"objective": "regression", "learning_rate": 0.05, "num_leaves": leaves, "min_data_in_leaf": 1000,
            "feature_fraction": 0.8, "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 1.0,
            "verbose": -1, "seed": 7, "deterministic": True}


def main():
    import sys
    import lightgbm as lgb
    alone = "--alleen" in sys.argv
    name = "boost_alleen" if alone else "boost"
    log("Paneel opbouwen ...")
    P = build_panel()
    if not alone:
        har = load_forecast("har")
        P = P.merge(har, on=["date", "isin"], how="inner")
    cal = np.array(sorted(P["date"].unique()))
    log(f"Paneel ({'zonder HAR' if alone else 'met HAR-voorspellingen'}): {len(P):,} rijen vanaf {P['date'].min().date()}")
    block_cols = [c for cols in BLOCKS.values() for c in cols]
    out = P[["date", "isin"]].copy()
    importances = {}
    refits = pd.date_range(FIRST_REFIT, P["date"].max() + pd.Timedelta(days=1), freq="6MS")
    rng = np.random.default_rng(0)
    for h in HORIZONS:
        add_earnings_block(P, h)
        if alone:
            feats = BASE + block_cols
            y_all = np.log(P[f"rv_{h}"] / h + LOGC)
        else:
            P["ln_har"] = np.log(P[f"v_{h}"] / h + LOGC)
            feats = BASE + block_cols + ["ln_har"]
            y_all = np.log(P[f"rv_{h}"] / h + LOGC) - P["ln_har"]
        pred = np.full(len(P), np.nan)
        log(f"Horizon {h}: {len(refits)} trainingen ...")
        for k, r0 in enumerate(refits):
            r1 = refits[k + 1] if k + 1 < len(refits) else pd.Timestamp("2100-01-01")
            k_ref = np.searchsorted(cal, np.datetime64(r0))
            k_cut = k_ref - h - EMBARGO_EXTRA - 1
            if k_cut < 60:
                continue
            tr_days = cal[:k_cut + 1]
            n_val = max(20, int(0.2 * len(tr_days)))
            val_days = tr_days[-n_val:]
            fit_days = tr_days[:max(0, len(tr_days) - n_val - h)]         # kloof van h dagen voor de validatie
            ok = y_all.notna().to_numpy() & np.isfinite(y_all.to_numpy())
            fit_m = P["date"].isin(fit_days).to_numpy() & ok
            val_m = P["date"].isin(val_days).to_numpy() & ok
            if fit_m.sum() < 20000 or val_m.sum() < 5000:
                continue
            fit_idx = np.where(fit_m)[0]
            if len(fit_idx) > MAX_TRAIN_ROWS:                               # recentste rijen + steekproef ouder
                recent = fit_idx[-MAX_TRAIN_ROWS // 2:]
                older = rng.choice(fit_idx[:-MAX_TRAIN_ROWS // 2], MAX_TRAIN_ROWS // 2, replace=False)
                fit_idx = np.sort(np.concatenate([older, recent]))
            Xf, yf = P.iloc[fit_idx][feats], y_all.iloc[fit_idx]
            Xv, yv = P.loc[val_m, feats], y_all[val_m]
            best = None
            for leaves in LEAVES_GRID:
                m = lgb.train(params(leaves), lgb.Dataset(Xf, yf), num_boost_round=MAX_ROUNDS,
                              valid_sets=[lgb.Dataset(Xv, yv)],
                              callbacks=[lgb.early_stopping(30, verbose=False)])
                score = m.best_score["valid_0"]["l2"]
                if best is None or score < best[0]:
                    best = (score, m, leaves)
            _, model, leaves = best
            resid = yv.to_numpy() - model.predict(Xv, num_iteration=model.best_iteration)
            smear = float(np.mean(np.exp(resid)))
            rows = ((P["date"] >= r0) & (P["date"] < r1)).to_numpy()
            if rows.any():
                p = model.predict(P.loc[rows, feats], num_iteration=model.best_iteration)
                base = 1.0 if alone else P.loc[rows, f"v_{h}"].to_numpy() / h + LOGC
                pred[rows] = h * np.maximum(base * np.exp(p) * smear - LOGC, 1e-12)
            importances[h] = pd.Series(model.feature_importance("gain"), index=feats)
            log(f"  training {r0.date()}: {len(fit_idx):,} rijen, {leaves} bladeren, "
                f"{model.best_iteration} bomen")
        out[f"v_{h}"] = pred
    out = out[out["date"] >= FC_START]
    save_forecast(name, out.dropna(subset=[f"v_{h}" for h in HORIZONS], how="all"))
    imp = pd.DataFrame(importances)
    imp = imp / imp.sum()
    imp.to_csv(RISK / f"{name}_belang.csv")
    print(f"\n================ GRADIENT BOOSTING ({'op zichzelf' if alone else 'correctie op HAR'}): "
          "BELANGRIJKSTE KENMERKEN (laatste training) ================")
    print("Aandeel in de totale verbetering (gain), per horizon. Alleen ter interpretatie, niet voor selectie.")
    top = imp.mean(axis=1).sort_values(ascending=False).head(12).index
    print((imp.loc[top] * 100).round(1).to_string())
    print("==============================================================================================\n")
    log("Volgende:  python br.py risk_compare")


if __name__ == "__main__":
    main()
