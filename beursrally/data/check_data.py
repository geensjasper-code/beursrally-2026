#!/usr/bin/env python3
"""
Controle van de data in gewone taal, zonder de code te moeten lezen.

    python br.py check_data                 # standaard: ASML, Apple, NVIDIA, KBC, Coinbase
    python br.py check_data ASML.AS COIN    # of zelf tickers kiezen

Toont per aandeel de laatste koers (in de eigen munt en in euro), hoeveel historiek er is,
het rendement en de volatiliteit, en maakt een grafiek: data/controle.png.
Vergelijk de laatste koers met wat het Beursrally-platform toont.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from beursrally.paths import CONFIG, ROOT as HERE   # projectmap (data/ en config/ staan daar)
PROC = HERE / "data" / "processed"
DEFAULT = ["ASML.AS", "AAPL", "NVDA", "KBC.BR", "COIN"]


def main():
    if not (PROC / "close_eur.parquet").exists():
        sys.exit("Nog geen verwerkte data. Eerst:  python br.py data init")
    t = pd.read_csv(HERE / "data" / "meta" / "tickers.csv", dtype=str)
    eur = pd.read_parquet(PROC / "close_eur.parquet")
    loc = pd.read_parquet(PROC / "close_local.parquet")
    lr = pd.read_parquet(PROC / "logret_eur.parquet")
    wanted = sys.argv[1:] or DEFAULT

    print(f"\nData tot en met {eur.index.max():%d-%m-%Y}, {eur.shape[1]} effecten, "
          f"{len(eur)} beursdagen sinds {eur.index.min():%d-%m-%Y}\n")
    rows = []
    for tk in wanted:
        r = t[t["ticker"] == tk]
        if r.empty:
            print(f"{tk}: niet in de lijst (bestaat de ticker? zie data/meta/tickers.csv)")
            continue
        isin, name, ccy = r.iloc[0]["isin"], r.iloc[0]["name"], r.iloc[0]["currency"]
        if isin not in eur.columns or eur[isin].notna().sum() == 0:
            print(f"{name} ({tk}): GEEN KOERSEN")
            continue
        s_loc, s_eur, x = loc[isin].dropna(), eur[isin].dropna(), lr[isin].dropna()
        last = x.iloc[-252:]
        rows.append({
            "aandeel": name[:24],
            "laatste dag": f"{s_loc.index[-1]:%d-%m-%Y}",
            f"slot (eigen munt)": f"{s_loc.iloc[-1]:,.2f} {ccy}",
            "slot (EUR)": f"{s_eur.iloc[-1]:,.2f}",
            "historiek": f"{len(s_eur) / 252:.1f} jaar",
            "rendement 1 jaar": f"{np.exp(last.sum()) - 1:+.0%}",
            "volatiliteit (jaar)": f"{last.std() * np.sqrt(252):.0%}",
            "grootste dagbeweging": f"{np.exp(x.abs().max()) - 1:.0%}",
        })
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    print("\nControle: komt 'slot (eigen munt)' overeen met de koers op het Beursrally-platform op die dag?")
    print("(Amerikaanse aandelen toont het platform in dollar, de vertraging is 15 minuten.)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nVoor de grafiek:  pip install matplotlib")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for tk in wanted:
        r = t[t["ticker"] == tk]
        if r.empty or r.iloc[0]["isin"] not in eur.columns:
            continue
        s = eur[r.iloc[0]["isin"]].dropna()
        ax.plot(s.index, s / s.iloc[0] * 100, label=f"{r.iloc[0]['name'][:20]} ({tk})", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylabel("waarde in euro, start = 100 (log-schaal)")
    ax.set_title("Koersverloop in euro: zie je vreemde sprongen of gaten?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out = HERE / "data" / "controle.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nGrafiek: {out}")


if __name__ == "__main__":
    main()
