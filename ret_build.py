#!/usr/bin/env python3
"""
Beursrally 2026 - verwacht rendement voor VANDAAG (na features.py; keuzes uit ret_compare.py).

Gebruik
    python ret_build.py

Per horizon h (1, 3, 5, 10, 20, 48 beursdagen):
    1. Het gekozen model (data/returns/keuze.json) wordt opnieuw geschat op ALLE data tot vandaag,
       met hetzelfde embargo (de laatste h + 5 dagen hebben nog geen afgelopen doel).
    2. Voorspelling voor de laatste beursdag in de data.
    3. Gebruikt verwacht rendement  mu_h = 7% x h/252  +  w x b x voorspelling
       (bij een gemiddelde van twee modellen: z-scores van vandaag, gemiddeld, x de bewaarde schaal)
       (w = vertrouwen uit de keuzeperiode, b = kalibratie; w = 0 -> alle aandelen gelijk).
Uitvoer: data/returns/rendement_vandaag.parquet
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ret_models import DATA, HORIZONS, MARKET_PA, OUT, Calendar, fit_predict, load_panel, log


def main():
    keuze = json.loads((OUT / "keuze.json").read_text())
    log("Paneel laden ...")
    P, xcols = load_panel()
    cal = Calendar(P["date"])
    last = P["date"].max()
    rows = np.where(P["date"].to_numpy() == np.datetime64(last))[0]
    out = P.loc[rows, ["isin"]].copy()
    rng = np.random.default_rng(0)
    for h in HORIZONS:
        k = keuze[str(h)]
        k_cut = cal.cutoff(last + pd.Timedelta(days=1), h)
        train = cal.rows_upto(k_cut)
        if k["model"].startswith("gem:"):              # gemiddelde van twee modellen (z-scores van vandaag)
            parts = []
            for mname in k["model"][4:].split("+"):
                pm = np.nan_to_num(fit_predict(mname, P, xcols, cal, h, train, rows, rng), nan=0.0)
                sd = pm.std()
                parts.append((pm - pm.mean()) / sd if sd > 0 else np.zeros_like(pm))
            pred = (parts[0] + parts[1]) / 2 * k["schaal"]      # terug naar rendementseenheden
        else:
            pred = fit_predict(k["model"], P, xcols, cal, h, train, rows, rng)
            pred = np.nan_to_num(pred, nan=0.0)
        alpha = k["w"] * k["b"] * pred
        out[f"alpha_{h}"] = alpha
        out[f"mu_{h}"] = MARKET_PA * h / 252 + alpha
        log(f"h = {h}: model {k['model']}, w = {k['w']:.2f}, b = {k['b']:.2f}")
    t = pd.read_csv(DATA / "meta" / "tickers.csv", dtype=str)
    out.insert(1, "naam", out["isin"].map(dict(zip(t["isin"], t["name"]))))
    out.insert(2, "ticker", out["isin"].map(dict(zip(t["isin"], t["ticker"]))))
    out["datum"] = last
    out.to_parquet(OUT / "rendement_vandaag.parquet", index=False)

    pd.set_option("display.width", 200)
    show = out[["naam", "ticker", "mu_5", "mu_20", "mu_48", "alpha_48"]].copy()
    for c in ("mu_5", "mu_20", "mu_48", "alpha_48"):
        show[c] = (show[c] * 100).round(2).astype(str) + "%"
    srt = out["mu_48"].sort_values(ascending=False).index
    print(f"\n================ VERWACHT RENDEMENT VANDAAG ({last.date()}) ================")
    print(f"Marktdeel: 7% per jaar (h = 48: {MARKET_PA * 48 / 252:.2%}). alpha = aandeelspecifiek deel (w x b x voorspelling).")
    spread = out["alpha_48"].quantile(0.9) - out["alpha_48"].quantile(0.1)
    print(f"Spreiding alpha over 48 dagen (90e - 10e percentiel): {spread * 100:.2f}%")
    print("\nHoogste verwachting (48 dagen):")
    print(show.loc[srt[:12]].to_string(index=False))
    print("\nLaagste verwachting (48 dagen):")
    print(show.loc[srt[-5:]].to_string(index=False))
    print("==========================================================================\n")
    log("Klaar -> data/returns/rendement_vandaag.parquet")


if __name__ == "__main__":
    main()
