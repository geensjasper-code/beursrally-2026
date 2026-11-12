#!/usr/bin/env python3
"""
Beursrally 2026 - risicomodel 1 en 2: voortschrijdend 60 dagen en EWMA.

Gebruik
    python br.py risk_simple

Model 1  rolling60   dagvariantie = gemiddelde van r^2 over de laatste 60 eigen dagen (min. 40)
Model 2  ewma94/97/99  s2_t = lam * s2_{t-1} + (1 - lam) * r_t^2   (min. 60 dagen historiek)

Beide nemen aan dat de huidige dagvariantie blijft gelden: v_h = h x dagvariantie.
Alleen terugkijkend: de voorspelling op dag d gebruikt rendementen tot en met dag d.
"""
from __future__ import annotations

import pandas as pd

from beursrally.risk.risk_common import load_returns, log, save_forecast, from_daily_variance, own_series

EWMA_LAMBDAS = {"ewma94": 0.94, "ewma97": 0.97, "ewma99": 0.99}


def daily_variance(lr: pd.DataFrame, kind: str, lam: float | None = None) -> pd.DataFrame:
    out = {}
    for c in lr.columns:
        r2 = own_series(lr, c) ** 2
        if kind == "rolling":
            out[c] = r2.rolling(60, min_periods=40).mean()
        else:
            out[c] = r2.ewm(alpha=1 - lam, adjust=False, min_periods=60).mean()
    return pd.DataFrame(out)


def main():
    log("Rendementen laden ...")
    lr = load_returns()
    log("Model 1: voortschrijdend 60 dagen ...")
    save_forecast("rolling60", from_daily_variance(daily_variance(lr, "rolling")))
    for name, lam in EWMA_LAMBDAS.items():
        log(f"Model 2: EWMA lambda = {lam} ...")
        save_forecast(name, from_daily_variance(daily_variance(lr, "ewma", lam)))
    log("Klaar. Volgende:  python br.py risk_garch")


if __name__ == "__main__":
    main()
