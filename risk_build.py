#!/usr/bin/env python3
"""
Beursrally 2026 - het risicomodel voor VANDAAG (dagelijks na data.py update, clean.py, features.py).

Gebruik
    python risk_build.py

Wat het maakt (data/risk/)
    vol_vandaag.parquet     per aandeel de voorspelde variantie v_1 ... v_48 en de jaarvolatiliteit
        Basis (uit risk_compare.py, keuzeperiode 2019-2021):
            1-10 dagen: gemiddelde van HAR en boosting op zichzelf (boost_alleen); 20 en 48 dagen: HAR
        Europese aandelen en VS-aandelen zonder bruikbare opties: de basis
        VS-aandelen met bruikbare opties: lambda * (c * IV)^2 + (1 - lambda) * HAR   (c = 0,9, lambda = 0,7)
        Kwaliteitsfilter opties: minstens 4 noteringen rond de koers (90-110%) en een mediane
        bied-laatspreiding van hoogstens 30% van de middenprijs; anders HAR.
    corr_vandaag.parquet    correlatiematrix (het beste model uit risk_corr.py), op de laatste 500 beursdagen
    themas_vandaag.parquet  thema-indeling op dezelfde correlaties
    log/voorspellingen_JJJJMMDD.parquet   de voorspellingen van vandaag, ook met lambda 0 / 0,3 / 0,5 / 1,0,
                                          voor de voorwaartse test (python risk_forward.py)

De covariantiematrix over h dagen is Sigma_h = D_h R D_h, met D_h = wortel(v_h) per aandeel.
lambda aanpassen: zet het nieuwe getal in data/risk/lambda_aandelen.txt (zie risk_forward.py).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from risk_common import HORIZONS, PROC, RISK, load_forecast, load_returns, log, names
from risk_corr import (WINDOW, cluster, corr_factor, corr_factor_extended, corr_lw, corr_resid, corr_sample,
                       two_day_returns, window_matrix)
from risk_iv import atm_iv, fx_adjustment, iv_term

LAMBDA_DEFAULT = 0.7
COMBO_H = [1, 3, 5, 10, 20, 48]  # gemiddelde HAR + boost_alleen; 20/48 toegevoegd na de controleperiode 2022-2025
LAMBDA_VERSIONS = [0.0, 0.3, 0.5, 0.7, 1.0]
LOG_DIR = RISK / "log"


def stock_lambda() -> float:
    f = RISK / "lambda_aandelen.txt"
    if f.exists():
        try:
            return float(f.read_text().strip().replace(",", "."))
        except ValueError:
            pass
    return LAMBDA_DEFAULT


def option_quality(o: pd.DataFrame) -> pd.Series:
    """Per aandeel: genoeg en redelijk geprijsde opties rond de koers?"""
    o = o[o["spot"].notna() & (o["impliedVolatility"] > 0.05)].copy()
    o = o[(o["strike"] / o["spot"]).between(0.9, 1.1)]
    mid = (o["bid"] + o["ask"]) / 2
    o["spread"] = ((o["ask"] - o["bid"]) / mid).where(mid > 0)
    g = o.groupby("isin").agg(n=("strike", "size"), spread=("spread", "median"))
    return (g["n"] >= 4) & (g["spread"] <= 0.30)


def main():
    nm = names()
    t = pd.read_csv(RISK.parent / "meta" / "tickers.csv", dtype=str)
    tick = dict(zip(t["isin"], t["ticker"]))

    # 1. HAR: laatste voorspelling per aandeel
    har = load_forecast("har")
    last_date = har["date"].max()
    har = har.sort_values("date").groupby("isin").tail(1)
    stale = har["date"] < last_date - pd.Timedelta(days=7)
    if stale.any():
        log(f"{stale.sum()} aandelen zonder recente koers (laatste > 7 dagen oud): niet opgenomen.")
    har = har[~stale].set_index("isin")
    vol = pd.DataFrame(index=har.index)
    vol["datum"] = har["date"]
    try:
        bo = load_forecast("boost_alleen")
        bo = bo[bo["date"] == last_date].set_index("isin")
    except FileNotFoundError:
        bo = pd.DataFrame()
        log("Geen boost_alleen-voorspellingen: overal HAR (draai python risk_boost.py --alleen).")
    for h in HORIZONS:
        vol[f"har_{h}"] = har[f"v_{h}"]                 # 'har_' = de basis (HAR, of het gemiddelde met boosting)
        if h in COMBO_H and not bo.empty:
            b = bo[f"v_{h}"].reindex(vol.index)
            vol[f"har_{h}"] = np.where(b.notna(), (vol[f"har_{h}"] + b) / 2, vol[f"har_{h}"])
    vol["bron"] = "basis"

    # 2. opties voor VS-aandelen
    par_f = RISK / "iv_parameters.json"
    c = json.loads(par_f.read_text())["c"] if par_f.exists() else 0.9
    lam = stock_lambda()
    files = sorted(PROC.glob("options_*.parquet"))
    iv = pd.DataFrame()
    if files:
        o = pd.read_parquet(files[-1])
        asof = pd.to_datetime(o["downloaded"]).max().normalize()
        good = option_quality(o)
        term = iv_term(atm_iv(o[o["isin"].isin(good[good].index)]), asof)
        if not term.empty:
            adj = fx_adjustment(term["isin"].tolist())
            term = term[term["isin"].isin(adj.index) & term["isin"].isin(vol.index)].set_index("isin")
            for h in HORIZONS:
                term[f"iv_{h}"] = (term[f"ivv_{h}"] + h * adj.reindex(term.index)).clip(lower=0.25 * term[f"ivv_{h}"])
            iv = term
        log(f"Opties van {asof.date()}: {int(good.sum())} van {len(good)} aandelen door de kwaliteitsfilter.")
    for h in HORIZONS:
        vol[f"v_{h}"] = vol[f"har_{h}"]
        if not iv.empty:
            i = iv.index
            vol.loc[i, f"iv_{h}"] = iv[f"iv_{h}"]
            vol.loc[i, f"v_{h}"] = lam * c ** 2 * iv[f"iv_{h}"] + (1 - lam) * vol.loc[i, f"har_{h}"]
    if not iv.empty:
        vol.loc[iv.index, "bron"] = f"basis + opties (lambda {lam:.1f})"
    for h in (1, 20, 48):
        vol[f"jaarvol_{h}"] = np.sqrt(vol[f"v_{h}"] / h * 252)
    vol.insert(0, "naam", [nm.get(i, i) for i in vol.index])
    vol.insert(1, "ticker", [tick.get(i, "") for i in vol.index])
    vol.reset_index(names="isin").to_parquet(RISK / "vol_vandaag.parquet", index=False)

    # 3. log voor de voorwaartse test
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not iv.empty:
        rows = []
        for i in iv.index:
            for h in HORIZONS:
                rec = {"datum": vol.at[i, "datum"], "isin": i, "h": h, "har": vol.at[i, f"har_{h}"]}
                for lv in LAMBDA_VERSIONS:
                    rec[f"lam_{lv:.1f}"] = lv * c ** 2 * iv.at[i, f"iv_{h}"] + (1 - lv) * vol.at[i, f"har_{h}"]
                rows.append(rec)
        pd.DataFrame(rows).to_parquet(LOG_DIR / f"voorspellingen_{last_date:%Y%m%d}.parquet", index=False)

    # 4. correlaties en thema's van vandaag
    best = (RISK / "corr_keuze.txt").read_text().strip() if (RISK / "corr_keuze.txt").exists() else "lw"
    lr = load_returns()
    r2 = two_day_returns(lr)
    end = last_date + pd.Timedelta(days=1)
    W = window_matrix(r2, end)
    W = W.loc[:, W.columns.isin(vol.index)]
    X = W.to_numpy()
    if best.startswith("fac_"):
        # recente beursgangen (te kort voor het venster): ladingen schatten op de dagen dat ze noteren
        recent = [i for i in vol.index if i not in W.columns and i in r2.columns]
        raw = r2[r2.index < end].tail(len(W)).reindex(W.index)[recent]
        Rdf, est, short = corr_factor_extended(W, raw, int(best.split("_")[1]))
        if est:
            log(f"Correlaties geschat voor {len(est)} recente beursgang(en): " + ", ".join(nm.get(i, i) for i in est))
        if short:
            log(f"Te korte historiek (< 60 beursdagen), aanname in scen_core: " + ", ".join(nm.get(i, i) for i in short))
        Rdf.to_parquet(RISK / "corr_vandaag.parquet")
    else:
        R = corr_sample(X) if best == "sample" else corr_lw(X)[0]
        pd.DataFrame(R, index=W.columns, columns=W.columns).to_parquet(RISK / "corr_vandaag.parquet")
    st = RISK / "themas_stabiliteit.csv"
    nr, K = (1, 15)
    if st.exists():
        sd = pd.read_csv(st)
        if "factoren_weg" in sd:
            nr, K = (int(v) for v in sd.groupby(["factoren_weg", "K"])["ARI"].mean().idxmax())
    lab = cluster(corr_resid(X, nr), K)
    pd.DataFrame({"isin": W.columns, "naam": [nm.get(i, i) for i in W.columns], "thema": lab}) \
        .to_parquet(RISK / "themas_vandaag.parquet", index=False)

    # 5. rapport
    pd.set_option("display.width", 200)
    print("\n================ RISICO VANDAAG ================")
    print(f"Laatste beursdag in de data: {last_date.date()}. {len(vol)} aandelen.")
    print(vol["bron"].value_counts().to_string())
    print(f"Correlaties: model '{best}' op de laatste {WINDOW} beursdagen, {W.shape[1]} aandelen; {K} thema's.")
    show = vol[["naam", "ticker", "bron", "jaarvol_1", "jaarvol_20", "jaarvol_48"]].copy()
    for cc in ("jaarvol_1", "jaarvol_20", "jaarvol_48"):
        show[cc] = (show[cc] * 100).round(0).astype("Int64").astype(str) + "%"
    show = show.assign(_s=vol["jaarvol_48"]).sort_values("_s", ascending=False).drop(columns="_s")
    print("\nMeest beweeglijk (48 dagen):")
    print(show.head(10).to_string(index=False))
    print("\nMinst beweeglijk (48 dagen):")
    print(show.tail(5).to_string(index=False))
    if not iv.empty:
        r = np.sqrt(vol.loc[iv.index, "v_20"] / vol.loc[iv.index, "har_20"])
        print(f"\nVS-aandelen met opties: mediane volatiliteit mengvorm / basis = {r.median():.2f}")
    print("================================================\n")
    log("Klaar -> data/risk/vol_vandaag.parquet, corr_vandaag.parquet, themas_vandaag.parquet, log/")


if __name__ == "__main__":
    main()
