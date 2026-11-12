#!/usr/bin/env python3
"""
Beursrally 2026 - scenariogenerator (stap 4): bouwstenen van VANDAAG maken en controleren.

Gebruik
    python br.py scen_build                      # start = 16 november 2026 (of de eerstvolgende beursdag)
    python br.py scen_build --start 2026-11-23   # later in de competitie: opnieuw vanaf die dag

Invoer   data/risk/vol_vandaag.parquet, data/risk/corr_vandaag.parquet, data/returns/rendement_vandaag.parquet
Uitvoer  data/scenarios/params_vandaag.npz   (drift en standaardafwijking per dag en aandeel, correlatiewortel)
         Scenario's zelf worden NIET opgeslagen (20.000 x 48 x 370 = 1,4 GB): elk script maakt ze in een paar
         seconden opnieuw met generate(params, n, seed). Zelfde seed = zelfde scenario's.

Het rapport toont of de simulatie de voorspellingen terugvindt, en een eerste beeld van de uitkomsten.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from beursrally.scenarios.scen_core import HORIZONS, backend, build_params, generate, save_params

from beursrally.paths import CONFIG, ROOT as HERE   # projectmap (data/ en config/ staan daar)
DATA = HERE / "data"
OUT = DATA / "scenarios"
COMP_START = pd.Timestamp("2026-11-16")
N_CHECK = 20000


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def pct(x) -> str:
    return f"{x:+.1%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="eerste competitiedag (standaard 16 nov 2026)")
    a = ap.parse_args()
    f_vol, f_corr = DATA / "risk" / "vol_vandaag.parquet", DATA / "risk" / "corr_vandaag.parquet"
    f_mu = DATA / "returns" / "rendement_vandaag.parquet"
    for f in (f_vol, f_corr, f_mu):
        if not f.exists():
            sys.exit(f"{f.relative_to(HERE)} ontbreekt. Eerst risk_build.py en ret_build.py draaien.")
    vol = pd.read_parquet(f_vol).set_index("isin")
    ret = pd.read_parquet(f_mu).set_index("isin")
    R = pd.read_parquet(f_corr)
    data_date = pd.Timestamp(vol["datum"].max())
    start = pd.Timestamp(a.start) if a.start else max(COMP_START, data_date + pd.Timedelta(days=1))

    var = vol[[f"v_{h}" for h in HORIZONS]].set_axis(HORIZONS, axis=1)
    mu = pd.DataFrame(index=var.index, columns=HORIZONS, dtype=float)
    have = var.index.intersection(ret.index)
    for h in HORIZONS:
        mu[h] = 0.07 * h / 252                                    # zonder voorspelling: alleen het marktdeel
        mu.loc[have, h] = ret.loc[have, f"mu_{h}"]
    n_no_mu = len(var.index.difference(ret.index))

    from beursrally.data.data import US_EXCHANGES
    tk = pd.read_csv(DATA / "meta" / "tickers.csv", dtype=str)
    us = set(tk.loc[tk["exchange"].isin(US_EXCHANGES), "isin"])
    p = build_params(mu, var, R, data_date, start=start, us=us)
    f_shape = OUT / "vorm_keuze.json"
    if f_shape.exists():
        import json
        from beursrally.scenarios.scen_core import Shape
        d = json.loads(f_shape.read_text())
        p.shape = Shape(int(d["nu"]), float(d["vol_markt"]), float(d["vol_aandeel"]))
    OUT.mkdir(parents=True, exist_ok=True)
    save_params(p, OUT / "params_vandaag.npz")
    names = vol["naam"].to_dict()
    tick = vol["ticker"].to_dict()
    kind, dev = backend()
    log(f"Koersen tot {data_date.date()}, competitie {p.dates[0].date()} - {p.dates[-1].date()} "
        f"({len(p.dates)} beursdagen; {p.skip} dagen ervoor gesimuleerd en weggelaten)")
    log(f"{len(p.isins)} effecten | zonder correlaties (aangevuld): {len(p.info['zonder_correlaties'])} | "
        f"zonder rendementsvoorspelling (alleen marktdeel): {n_no_mu} | "
        f"VS-notering (dicht op 26 nov en 18 jan): {len(us & set(p.isins))}")
    if p.info["zonder_correlaties"]:
        log("  aangevuld: " + ", ".join(names.get(i, i) for i in p.info["zonder_correlaties"]))
    log(f"Rekenen op: {kind}" + (f" ({dev})" if dev is not None else "") + f" | vorm: {p.shape.label()}"
        + ("" if f_shape.exists() else " (nog geen vorm gekozen: python br.py scen_backtest --grid)"))

    t0 = time.time()
    final = np.concatenate([c[:, -1, :] for c in generate(p, N_CHECK, seed=0)])   # (paden, aandelen)
    log(f"{N_CHECK:,} scenario's in {time.time() - t0:.1f} s -> data/scenarios/params_vandaag.npz")

    # 1. controle: vindt de simulatie de voorspelling terug? (volledige periode vanaf dag 1 van de voorspelling)
    from beursrally.scenarios.scen_core import Params
    full = Params(p.isins, p.dates, 0, p.drift, p.sd, p.root, shape=p.shape)
    sims = np.concatenate([c[:, [h - 1 for h in HORIZONS if h <= p.T], :] for c in generate(full, N_CHECK, seed=1)])
    eu = np.array([i not in us for i in p.isins])                 # EU: kalenderdag h = eigen beursdag h
    print("\n================ CONTROLE: SIMULATIE TEGENOVER VOORSPELLING ================")
    print("Mediaan over de Europese aandelen van (gesimuleerd / voorspeld). Moet ~1,00 zijn.")
    rows = []
    for j, h in enumerate([h for h in HORIZONS if h <= p.T]):
        x = sims[:, j, eu].astype(np.float64)
        v, m = var[h].to_numpy()[eu], mu[h].to_numpy()[eu]
        rows.append({"h": h, "variantie": np.median(x.var(0) / v),
                     "gemiddelde - mu (in standaardfouten)": np.median((x.mean(0) - m) / np.sqrt(v / N_CHECK))})
    print(pd.DataFrame(rows).set_index("h").round(2).to_string())

    # 2. eerste beeld van de uitkomsten tot 22 januari
    simple = np.expm1(final.astype(np.float64))
    q = lambda a, s: np.quantile(a, s)
    print(f"\n================ UITKOMSTEN TOT {p.dates[-1].date()} (vorm: {p.shape.label()}) ================")
    vol48 = np.sqrt(simple.var(0))
    t = pd.DataFrame({"naam": [names.get(i, i) for i in p.isins], "ticker": [tick.get(i, "") for i in p.isins],
                      "mediaan": np.median(simple, 0), "p5": np.quantile(simple, 0.05, 0),
                      "p95": np.quantile(simple, 0.95, 0), "kans>+50%": (simple > 0.5).mean(0),
                      "kans<-30%": (simple < -0.3).mean(0), "sd": vol48}, index=p.isins)
    show = t.sort_values("sd", ascending=False).head(12).copy()
    for c in ("mediaan", "p5", "p95"):
        show[c] = show[c].map(pct)
    for c in ("kans>+50%", "kans<-30%"):
        show[c] = show[c].map(lambda v: f"{v:.1%}")
    print("Meest beweeglijke effecten:")
    print(show.drop(columns="sd").to_string(index=False))

    best = simple.max(1)
    top10 = t.sort_values("sd", ascending=False).index[:10]
    pos = [p.isins.index(i) for i in top10]
    port = simple[:, pos].mean(1)                                 # gelijk gewogen, zonder herbalanceren
    allw = simple.mean(1)
    print("\nPer scenario (verdeling over de scenario's):         p5      p25   mediaan     p75      p95")
    for lab, x in (("beste aandeel van het scenario", best),
                   ("10 meest beweeglijke, gelijk gewogen", port),
                   ("alle effecten, gelijk gewogen", allw)):
        print(f"  {lab:<45}" + "".join(f"{pct(v):>9}" for v in q(x, [0.05, 0.25, 0.5, 0.75, 0.95])))
    print("\nVorm (staarten, stressdagen, vol-onzekerheid) gekozen met scen_backtest.py --grid op 2019-2021.")
    print("==========================================================================================")
    log("Klaar.")


if __name__ == "__main__":
    main()
