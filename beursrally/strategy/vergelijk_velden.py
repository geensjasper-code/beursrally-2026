#!/usr/bin/env python3
"""
Beursrally 2026 - startportefeuilles vergelijken PER VELDTYPE.

Gebruik
    python br.py vergelijk_velden                                   # de twee laatste orderlijsten (70 / alle kandidaten)
    python br.py vergelijk_velden data/strategy/a.csv data/strategy/b.csv ...

Waarom
    strategy.py middelt het prijzengeld over zes velden (basis, minder actief, meer actief, veel gokkers, een hot
    thema, momentumjagers). Een portefeuille kan in dat gemiddelde winnen omdat ze heel sterk is in een of twee
    velden. Hier zie je per veld het verwachte prijzengeld en P(top 3 studenten), zodat je weet hoe sterk het
    resultaat afhangt van hoe het veld eruitziet.

Zelfde controleset als strategy.py (zelfde seeds), exacte spelregels (5 + 5 orders, hele aandelen, quizgeld bijkopen).
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from beursrally.game.field import FIELDS, draw_field, field_values, load_universe, our_prize
from beursrally.game.game import BuyHold, Rules, play
from beursrally.scenarios.scen_core import generate, load_params
from beursrally.strategy.strategy import CHECK_CHUNK, DATA, OUT, POS, WORKERS, _one_thread

N_B = 6000


def _chunk(job):
    p, n, ci, u, price0, cands = job
    rng = np.random.default_rng(202_000 + ci)                 # zelfde toeval als de controle in strategy.py
    cum = next(generate(p, n, seed=202 + 1000 * ci, chunk=n, device="numpy")).astype(np.float64)
    s = np.expm1(cum[:, -1, :])
    fields = []
    for cfg in FIELDS:
        fd = draw_field(cfg, u, rng)
        fields.append((fd, field_values(fd, s, rng)))
    out = {}
    for k, picks in cands.items():
        r = play(BuyHold(picks, weight=POS, quiz="bijkopen"), cum, price0, p.dates, rules=Rules(), seed=ci)
        per = []
        for fd, vals in fields:
            pr = our_prize(r.final, vals, fd.student)
            t3 = (np.where(fd.student[None], vals, -np.inf) > r.final[:, None]).sum(1) < 3
            per.append((pr, t3))
        out[k] = per
    return out


def main():
    files = [Path(f) for f in sys.argv[1:]]
    if not files:
        files = [OUT / "orders_start_eigen_uitsluiting.csv", OUT / "orders_start_eigen_uitsluiting_kalle.csv"]
    files = [f for f in files if f.exists()]
    if not files:
        raise SystemExit("Geen orderlijsten gevonden (data/strategy/orders_start*.csv)")
    p = load_params(DATA / "scenarios" / "params_vandaag.npz")
    close = pd.read_parquet(DATA / "processed" / "close_eur.parquet")
    price0 = close.ffill().iloc[-1].reindex(p.isins).to_numpy(float)
    u = load_universe(p.isins, price0)
    pos_of = {s: k for k, s in enumerate(p.isins)}
    cands = {}
    for f in files:
        o = pd.read_csv(f)
        cands[f.stem.replace("orders_start", "start")] = [pos_of[i] for i in o["isin"]]
        print(f"{f.name}: " + ", ".join(o["naam"]))
    _one_thread()
    jobs = [(p, min(CHECK_CHUNK, N_B - i), ci, u, price0, cands) for ci, i in enumerate(range(0, N_B, CHECK_CHUNK))]
    acc = {k: [([], []) for _ in FIELDS] for k in cands}
    print(f"\n{N_B:,} controlescenario's, {len(cands)} portefeuilles, {WORKERS} processen ...", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for done, out in enumerate(ex.map(_chunk, jobs), 1):
            for k, per in out.items():
                for f_i, (pr, t3) in enumerate(per):
                    acc[k][f_i][0].append(pr)
                    acc[k][f_i][1].append(t3)
            print(f"    blok {done}/{len(jobs)} klaar", flush=True)
    rows_p, rows_t = [], []
    for k in cands:
        rp, rt = {"portefeuille": k}, {"portefeuille": k}
        for f_i, cfg in enumerate(FIELDS):
            pr = np.concatenate(acc[k][f_i][0])
            t3 = np.concatenate(acc[k][f_i][1])
            rp[cfg.name] = f"{pr.mean():.0f} +/- {pr.std() / np.sqrt(len(pr)):.0f}"
            rt[cfg.name] = f"{t3.mean():.1%}"
        rp["gemiddeld"] = f"{np.mean([np.concatenate(acc[k][f][0]).mean() for f in range(len(FIELDS))]):.0f}"
        rows_p.append(rp)
        rows_t.append(rt)
    pd.set_option("display.width", 240)
    print("\n================ VERWACHT PRIJZENGELD PER VELDTYPE (euro) ================")
    print(pd.DataFrame(rows_p).set_index("portefeuille").to_string())
    print("\n================ P(TOP 3 STUDENTEN) PER VELDTYPE ================")
    print(pd.DataFrame(rows_t).set_index("portefeuille").to_string())
    print("\nZelfde scenario's voor elke portefeuille: verschillen binnen een kolom zijn gepaard (minder ruis dan de +/-).")


if __name__ == "__main__":
    main()
