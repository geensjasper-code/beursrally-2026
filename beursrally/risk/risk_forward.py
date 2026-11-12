#!/usr/bin/env python3
"""
Beursrally 2026 - voorwaartse test van de optiemenging (lambda) voor VS-aandelen.

Gebruik
    python br.py risk_forward       # wekelijks, na data.py update + clean.py

Wat het doet
    risk_build.py bewaart elke dag de voorspellingen met lambda = 0 (alleen HAR), 0,3, 0,5, 0,7 en 1,0
    (alleen opties). Zodra de werkelijke koersen bekend zijn, scoort dit script ze met QLIKE.
    Beslissing over 1- en 5-daagse voorspellingen (veel onafhankelijke waarnemingen).

Beslisregel (vooraf vastgelegd)
    1. Pas oordelen na minstens 20 voorspellingsdagen met bekende uitkomst (~4 weken).
    2. Alleen wijzigen als een andere lambda SIGNIFICANT beter is dan de huidige (t <= -2, Diebold-Mariano,
       per dag gemiddeld over de aandelen) op zowel 1 als 5 dagen, of op een van beide zonder tegenspraak.
    3. Hoogstens een stap van 0,2 per keer, in de richting van de betere lambda.
    De nieuwe waarde zet je zelf in data/risk/lambda_aandelen.txt; risk_build.py gebruikt ze daarna.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from beursrally.risk.risk_build import LAMBDA_VERSIONS, LOG_DIR, stock_lambda
from beursrally.risk.risk_common import dm_test, load_returns, log, own_series, qlike

DECIDE_H = [1, 5]
MIN_DAYS = 20


def realized(lr: pd.DataFrame, isin: str, d: pd.Timestamp, h: int) -> float:
    s = own_series(lr, isin)
    fut = s[s.index > d]
    return float((fut.iloc[:h] ** 2).sum()) if len(fut) >= h else np.nan


def main():
    files = sorted(LOG_DIR.glob("voorspellingen_*.parquet")) if LOG_DIR.exists() else []
    if not files:
        print("Nog geen opgeslagen voorspellingen. Draai dagelijks python br.py risk_build.")
        return
    L = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    L = L[L["h"].isin(DECIDE_H)].drop_duplicates(["datum", "isin", "h"], keep="last")
    lr = load_returns()
    L["rv"] = [realized(lr, i, d, h) for i, d, h in zip(L["isin"], L["datum"], L["h"])]
    L = L.dropna(subset=["rv"])
    cur = stock_lambda()
    cur_col = f"lam_{cur:.1f}" if f"lam_{cur:.1f}" in L.columns else "lam_0.7"
    cols = [f"lam_{v:.1f}" for v in LAMBDA_VERSIONS]
    print("\n================ VOORWAARTSE TEST: OPTIES VS HAR (VS-aandelen) ================")
    print(f"Huidige lambda: {cur:.1f}. Logbestanden: {len(files)}.")
    verdict = {}
    for h in DECIDE_H:
        sub = L[L["h"] == h]
        n_days = sub["datum"].nunique()
        print(f"\nh = {h}: {len(sub):,} voorspellingen met bekende uitkomst op {n_days} dagen")
        if sub.empty:
            continue
        losses = {c: pd.Series(qlike(sub["rv"].to_numpy(), sub[c].to_numpy()), index=sub.index) for c in cols}
        rows = []
        for c in cols:
            t = np.nan if c == cur_col else dm_test(losses[c], losses[cur_col], sub["datum"], h)
            rows.append({"lambda": c.replace("lam_", ""), "QLIKE": round(losses[c].mean(), 4),
                         "t vs huidige": None if np.isnan(t) else round(t, 2)})
        print(pd.DataFrame(rows).to_string(index=False))
        better = [(float(r["lambda"]), r["t vs huidige"]) for r in rows
                  if r["t vs huidige"] is not None and r["t vs huidige"] <= -2]
        verdict[h] = (n_days, better)
    print("\nBESLISSING")
    enough = all(v[0] >= MIN_DAYS for v in verdict.values()) if verdict else False
    if not enough:
        print(f"  Nog te weinig gegevens (minstens {MIN_DAYS} dagen per horizon nodig). lambda blijft {cur:.1f}.")
    else:
        dirs = set()
        for h, (_, better) in verdict.items():
            for lv, _t in better:
                dirs.add(np.sign(lv - cur))
        if len(dirs) == 1:
            new = float(np.clip(cur + 0.2 * dirs.pop(), 0, 1))
            print(f"  Een andere lambda is significant beter: stap naar {new:.1f}.")
            print(f"  Zet {new:.1f} in data/risk/lambda_aandelen.txt.")
        elif len(dirs) > 1:
            print(f"  Tegenstrijdige signalen tussen de horizonnen: lambda blijft {cur:.1f}.")
        else:
            print(f"  Geen lambda significant beter dan {cur:.1f}: lambda blijft {cur:.1f}.")
    print("===============================================================================\n")
    log("Klaar.")


if __name__ == "__main__":
    main()
