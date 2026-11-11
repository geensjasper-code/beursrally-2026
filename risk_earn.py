#!/usr/bin/env python3
"""
Beursrally 2026 - de cijfersprong als module bovenop de modellen zonder kenmerken (EWMA, GARCH, voortschrijdend).

Gebruik
    python risk_earn.py          # na risk_simple.py en risk_garch.py

Wat het doet
    Valt de publicatie van kwartaalcijfers binnen de horizon, dan:
        v_h(met cijfers) = v_h + max(sprong^2 - dagvariantie, 0)
    met sprong = de gemiddelde absolute koersreactie op de vorige 8 publicaties (earnings_move_avg uit
    features.py, alleen gebaseerd op wat bekend was). 'Minus dagvariantie': de cijferdag vervangt een gewone dag.
    Of de cijfers binnen de horizon vallen: exacte reactiedag (voor of na de opening), zie risk_common.
    HAR en boosting hebben de cijfers al als kenmerk; die krijgen geen sprong.

Bewaart per model een variant '<model>_cijfers' en toont het effect BINNEN en BUITEN cijfervensters
(keuzeperiode 2019-2021).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from risk_common import (FEAT, HORIZONS, available_forecasts, dm_test, earnings_reaction_k, load_forecast, log,
                         period_mask, qlike, realized_targets, save_forecast)

BASES = ["ewma94", "ewma97", "ewma99", "garch", "rolling60"]


def in_window(react_k: pd.Series, h: int) -> pd.Series:
    """De koersreactie op cijfers valt binnen de volgende h beursdagen (exacte reactiedag)."""
    return react_k.notna() & (react_k <= h)


def main():
    F = pd.read_parquet(FEAT / "features.parquet", columns=["date", "isin", "days_to_earnings", "earnings_move_avg"])
    T = realized_targets()
    have = [m for m in BASES if m in available_forecasts()]
    rows = []
    for m in have:
        f = load_forecast(m).merge(F, on=["date", "isin"], how="left")
        f["react_k"] = earnings_reaction_k(f).to_numpy()
        jump2 = f["earnings_move_avg"] ** 2
        daily = f["v_1"]
        g = f[["date", "isin"]].copy()
        for h in HORIZONS:
            bump = (jump2 - daily).clip(lower=0).where(in_window(f["react_k"], h), 0).fillna(0)
            g[f"v_{h}"] = f[f"v_{h}"] + bump
        save_forecast(f"{m}_cijfers", g)

        # effect binnen en buiten cijfervensters, keuzeperiode
        M = f[["date", "isin", "react_k"] + [f"v_{h}" for h in HORIZONS]].merge(
            g, on=["date", "isin"], suffixes=("", "_c")).merge(T, on=["date", "isin"])
        M = M[period_mask(M["date"], "keuze")]
        for h in (1, 3, 5, 10, 20):
            sub = M.dropna(subset=[f"v_{h}", f"v_{h}_c", f"rv_{h}"])
            win = in_window(sub["react_k"], h)
            for label, part in (("binnen", sub[win]), ("buiten", sub[~win])):
                if len(part) < 1000:
                    continue
                a = pd.Series(qlike(part[f"rv_{h}"].to_numpy(), part[f"v_{h}_c"].to_numpy()), index=part.index)
                b = pd.Series(qlike(part[f"rv_{h}"].to_numpy(), part[f"v_{h}"].to_numpy()), index=part.index)
                rows.append({"model": m, "h": h, "venster": label, "n": len(part),
                             "QLIKE zonder": b.mean(), "QLIKE met": a.mean(),
                             "t_DM": dm_test(a, b, part["date"], h) if label == "binnen" else np.nan})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n================ CIJFERSPRONG (keuzeperiode 2019-2021) ================")
    print("Binnen = cijfers vallen in de horizon. t_DM < -2: de sprong verbetert de voorspelling overtuigend.")
    print("Buiten de vensters verandert er niets (daar is de sprong 0); dat is een controle.")
    if not res.empty:
        show = res.copy()
        for c in ("QLIKE zonder", "QLIKE met"):
            show[c] = show[c].round(4)
        show["t_DM"] = show["t_DM"].round(2)
        print(show.to_string(index=False, na_rep="-"))
    print("=======================================================================\n")
    log("Volgende:  python risk_iv.py")


if __name__ == "__main__":
    main()
