#!/usr/bin/env python3
"""
Beursrally 2026 - risicomodel: alle volatiliteitsmodellen vergelijken.

Gebruik
    python risk_compare.py           # keuzeperiode 2019-2021: hiermee kiezen we
    python risk_compare.py --controle   # ook de controleperiode 2022 - midden 2025 (bescheiden bijsturen mag)
    python risk_compare.py --final      # ook de eindtoets midden 2025 - nu. Een keer, aan het einde!

Leest alle voorspellingen in data/risk/forecasts/ en beoordeelt ze op EXACT dezelfde rijen
(datum x aandeel waarvoor elk model een voorspelling heeft en het doel bekend is).

Maatstaven per model en horizon
    QLIKE      gemiddeld verlies RV/F - ln(RV/F) - 1 (lager = beter; onderschatten weegt zwaarder)
    t_DM       Diebold-Mariano tegenover EWMA 0,94 (negatief = beter dan EWMA; |t| > 2 = overtuigend)
    bias       gemiddelde RV / gemiddelde voorspelling (1 = gemiddeld juist; > 1 = onderschat)
    bias_ex    idem, zonder de coronacrash (15 feb - 15 mei 2020)
    helling    regressie ln RV op ln F (1 = goed gekalibreerd; < 1 = overreageert)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from risk_common import (EVAL_START, HORIZONS, LOGC, RISK, SPLIT, available_forecasts, dm_test, load_forecast, log,
                         names, period_mask, qlike, realized_targets, ticker_to_isin)

BASELINE = "ewma94"
SANITY = ["NVDA", "IONQ", "ASML.AS", "KBC.BR", "AGS.BR"]


def load_all(models: list[str]) -> pd.DataFrame:
    T = realized_targets()
    M = T
    for m in models:
        f = load_forecast(m).rename(columns={f"v_{h}": f"{m}|{h}" for h in HORIZONS})
        M = M.merge(f, on=["date", "isin"], how="inner")
    return M


def score(M: pd.DataFrame, models: list[str], h: int, period: str) -> tuple[pd.DataFrame, dict]:
    cols = [f"{m}|{h}" for m in models]
    rv_col = f"rv_{h}"
    sub = M[period_mask(M["date"], period)].dropna(subset=cols + [rv_col])
    sub = sub[np.isfinite(sub[cols]).all(axis=1) & (sub[cols] > 0).all(axis=1)]
    if sub.empty:
        return pd.DataFrame(), {}
    rv = sub[rv_col].to_numpy()
    losses = {m: pd.Series(qlike(rv, sub[f"{m}|{h}"].to_numpy()), index=sub.index) for m in models}
    rows = []
    for m in models:
        f = sub[f"{m}|{h}"].to_numpy()
        pos = rv > 0
        x, y = np.log(f[pos]), np.log(rv[pos])
        slope = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
        t = dm_test(losses[m], losses[BASELINE], sub["date"], h) if m != BASELINE and BASELINE in losses else np.nan
        d = sub["date"].to_numpy()
        ex = ~((d >= np.datetime64("2020-02-15")) & (d < np.datetime64("2020-05-15")))
        rows.append({"model": m, "QLIKE": losses[m].mean(), "t_DM": t, "bias": rv.mean() / f.mean(),
                     "bias_ex": rv[ex].mean() / f[ex].mean() if ex.any() else np.nan, "helling": slope})
    res = pd.DataFrame(rows).set_index("model")
    best = res["QLIKE"].idxmin()
    res["t_vs_best"] = [np.nan if m == best else dm_test(losses[m], losses[best], sub["date"], h) for m in models]
    meta = {"n": len(sub), "datums": sub["date"].nunique(), "best": best, "sub": sub}
    return res, meta


def winners(res: pd.DataFrame, meta: dict, close: float = 0.02) -> list[str]:
    """Beste model, plus modellen die er niet significant van verschillen EN er dicht bij liggen
    (QLIKE hoogstens 2% hoger). Alleen 'niet significant' volstaat niet: bij lange horizonnen heeft de test
    weinig onafhankelijke vensters, en dan zou een duidelijk slechter model meegemiddeld worden."""
    best = meta["best"]
    qb = res.at[best, "QLIKE"]
    tied = [m for m in res.index if m != best and res.at[m, "QLIKE"] <= qb * (1 + close)
            and (not np.isfinite(res.at[m, "t_vs_best"]) or res.at[m, "t_vs_best"] < 2)]
    return [best] + tied


def tail_check(meta: dict, models: list[str]) -> pd.Series:
    """Fractie dagen met |r_{d+1}| > 2 x voorspelde dagvolatiliteit (normaal ~4,6%; dikke staarten 5-7%)."""
    sub = meta["sub"]
    return pd.Series({m: float((sub["rv_1"] > 4 * sub[f"{m}|1"]).mean()) for m in models})


def sanity(models: list[str]) -> pd.DataFrame:
    t2i, nm = ticker_to_isin(), names()
    rows = []
    want = [t2i[tk] for tk in SANITY if tk in t2i]
    fcs = {m: load_forecast(m).query("isin in @want") for m in models}
    for tk in SANITY:
        isin = t2i.get(tk)
        if not isin:
            continue
        for m in models:
            f = fcs[m]
            f = f[f["isin"] == isin]
            if f.empty:
                continue
            last = f.sort_values("date").iloc[-1]
            rows.append({"aandeel": f"{nm.get(isin, tk)[:22]} ({tk})", "model": m, "datum": last["date"].date(),
                         "1d": np.sqrt(last["v_1"] * 252), "20d": np.sqrt(last["v_20"] / 20 * 252),
                         "48d": np.sqrt(last["v_48"] / 48 * 252)})
    return pd.DataFrame(rows)


def main():
    final = "--final" in sys.argv
    controle = "--controle" in sys.argv or final
    models = available_forecasts()
    if BASELINE not in models:
        sys.exit("Eerst:  python risk_simple.py  (EWMA 0,94 is de maatstaf)")
    log(f"Modellen: {', '.join(models)}. Laden ...")
    M = load_all(models)
    # gemiddelden van HAR met boosting (regel: bij gelijkwaardige modellen middelen; hier echt gemeten)
    for other in ("boost", "boost_alleen"):
        if "har" in models and other in models:
            name = f"gem_har_{other}"
            for h in HORIZONS:
                M[f"{name}|{h}"] = (M[f"har|{h}"] + M[f"{other}|{h}"]) / 2
            models.append(name)
    periods = ["keuze"] + (["controle"] if controle else []) + (["final"] if final else [])
    pd.set_option("display.width", 200)
    summary = []
    for period in periods:
        label = {"keuze": "KEUZEPERIODE 2019-2021", "controle": "CONTROLEPERIODE 2022 - MIDDEN 2025",
                 "final": "EINDTOETS MIDDEN 2025 - NU (eenmalig!)"}[period]
        print(f"\n================ VOLATILITEIT: {label} ================")
        print("Alle modellen op exact dezelfde rijen. QLIKE lager = beter. t_DM < -2: overtuigend beter dan EWMA 0,94.")
        for h in HORIZONS:
            res, meta = score(M, models, h, period)
            if res.empty:
                print(f"\nh = {h}: geen gegevens")
                continue
            w = winners(res, meta)
            print(f"\nh = {h} dagen  ({meta['n']:,} voorspellingen op {meta['datums']} datums)")
            show = res[["QLIKE", "t_DM", "bias", "bias_ex", "helling"]].copy()
            show["QLIKE"] = show["QLIKE"].round(4)
            show = show.round({"t_DM": 2, "bias": 2, "bias_ex": 2, "helling": 2})
            show["winnaar"] = ["<- beste" if m == w[0] else ("gelijkwaardig" if m in w else "") for m in show.index]
            show = show.reindex(res["QLIKE"].sort_values().index)
            print(show.sort_values("QLIKE").to_string(na_rep="-"))
            summary.append({"periode": period, "horizon": h, "winnaar": " + ".join(w)})
            if h == 1:
                tc = tail_check(meta, models)
                print("Dagen buiten 2 sigma (normaal 4,6%, met dikke staarten 5-7%): "
                      + ", ".join(f"{m} {v:.1%}" for m, v in tc.items()))
    sm = pd.DataFrame(summary)
    sm.to_csv(RISK / ("vergelijking_final.csv" if final else "vergelijking_controle.csv" if controle
                      else "vergelijking_keuze.csv"), index=False)
    print("\n================ WINNAAR PER HORIZON ================")
    print("Bij meerdere namen: niet significant verschillend van de beste en QLIKE binnen 2% -> we middelen ze.")
    print(sm.to_string(index=False))

    print("\n================ CONTROLE: BEKENDE AANDELEN (jaarvolatiliteit, laatste datum) ================")
    sn = sanity([m for m in models if not m.endswith("_cijfers") and not m.startswith("gem_")])
    if not sn.empty:
        for c in ("1d", "20d", "48d"):
            sn[c] = (sn[c] * 100).round(0).astype("Int64").astype(str) + "%"
        print(sn.to_string(index=False))
        print("Verwacht ruwweg: KBC/Ageas 15-30%, ASML 30-45%, Nvidia 40-60%, IonQ 70-120%.")
    print("=============================================================================================\n")
    if controle:
        print("Terugvalregel: zijn HAR/boosting in een latere periode NIET beter dan EWMA 0,94 (t_DM > -2) of ligt de\n"
              "bias ver van 1 (onder 0,7 of boven 1,4), dan valt die horizon terug op EWMA 0,94.\n")
    if not final:
        print("De eindtoets (midden 2025 - nu) is NIET bekeken. Dat doen we een keer, als alles vastligt.\n")


if __name__ == "__main__":
    main()
