#!/usr/bin/env python3
"""
Beursrally 2026 - handelsregel tijdens de wedstrijd (stap 6b): wanneer meer risico nemen, wanneer vastzetten?

Gebruik
    python br.py rules test               # test: doen de regels wat ze moeten doen?
    python br.py rules                    # zoeken op set A, vergelijken op set B (parallel)
    python br.py rules --snel             # minder scenario's (om te proberen)
    python br.py rules --start data/strategy/orders_start_eigen_uitsluiting.csv   # andere startportefeuille

Regels (bovenop kopen en houden met quizgeld bijkopen = R0)
    R1 inhalen     nog <= D beursdagen en meer dan X% achter op de 3e student:
                   de K rustigste posities verkopen, de K meest beweeglijke aandelen (niet in bezit) kopen. Eenmalig.
    R2 vastzetten  nog <= D beursdagen en minstens Y% voor op de 3e student:
                   de K wildste posities verkopen naar cash; geen aankopen meer. Eenmalig.
    R3             R1 en R2 samen (de beste instellingen van elk).
    Beslissing na de slotkoers van een controlepunt, uitvoering de volgende beursdag aan de opening,
    met alle spelregels (5 orders per dag: een ruil van 5 posities duurt 2 dagen; kosten; 10%-grens).

Controlepunten (wat je in het echt ziet: je eigen waarde en die van de 3e student)
    week   elke vrijdag (laatste beursdag van de week)
    dag    elke beursdag
    eind   elke vrijdag, en dagelijks in de laatste 10 beursdagen

Eerlijke vergelijking
    Elk scenario krijgt een van de 6 velden van field.py (om beurten); de regel reageert op de stand in DAT veld
    en het prijzengeld wordt tegen dat veld berekend. Set A kiest per regel en frequentie de beste instellingen,
    set B (nieuwe scenario's en velden) vergelijkt R0, R1, R2, R3 GEPAARD: zelfde scenario's, dus het verschil
    met R0 is nauwkeurig.

Uitvoer: data/strategy/regels_zoekset.csv, regels_controle.csv, regel_keuze.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from beursrally.game.field import FIELDS, POS, QUIZ_N, START, draw_field, field_values, load_universe, our_prize
from beursrally.game.game import BuyHold, Rules, play
from beursrally.scenarios.scen_core import generate, load_params

from beursrally.paths import CONFIG, ROOT as HERE   # projectmap (data/ en config/ staan daar)
DATA = HERE / "data"
OUT = DATA / "strategy"
CHUNK = 500
WORKERS = int(os.environ.get("BR_WORKERS", str(max(1, min(16, (os.cpu_count() or 2) - 1)))))
FREQS = ["week", "dag", "eind"]
R1_GRID = [(D, X, K) for D in (30, 20) for X in (0.05, 0.15) for K in (3, 5)]
R2_GRID = [(D, Y, K) for D in (15, 8) for Y in (0.0, 0.05) for K in (5, 10)]


def checkpoints(dates: pd.DatetimeIndex, freq: str) -> np.ndarray:
    T = len(dates)
    fri = pd.Series(np.arange(T), index=dates).groupby(dates.to_period("W-FRI").asi8).max().to_numpy()
    fri = fri[fri < T - 1]                                      # na de laatste dag valt niets meer te beslissen
    if freq == "week":
        return fri
    if freq == "dag":
        return np.arange(T - 1)
    return np.union1d(fri, np.arange(T - 11, T - 1))


# --------------------------------------------------------------------------- de regel
class RulePolicy:
    """Kopen en houden (BuyHold met bijkopen) + regels R1 (inhalen) en/of R2 (vastzetten)."""

    def __init__(self, targets, vol: np.ndarray, pool: np.ndarray, thr: np.ndarray, chk: np.ndarray, T: int,
                 r1=None, r2=None):
        self.base = BuyHold(targets, weight=POS, quiz="bijkopen")
        self.vol, self.pool, self.thr, self.T = vol, np.asarray(pool), thr, T
        self.chk = set(int(c) for c in chk)
        self.r1, self.r2 = r1, r2
        self.done1 = self.done2 = self.locked = None
        self.pending = []

    def orders(self, st):
        B = len(st.cash)
        if self.done1 is None:
            self.done1 = np.zeros(B, bool)
            self.done2 = np.zeros(B, bool)
            self.locked = np.zeros(B, bool)
            self.base.targets = np.broadcast_to(self.base.targets, (B, self.base.targets.shape[1])).copy()
        out = list(self.pending)
        self.pending = []
        for sec, side, amt in self.base.orders(st) or []:
            if side > 0:
                amt = np.where(self.locked, 0.0, amt)
            out.append((sec, side, amt))
        c = st.day - 1
        if c in self.chk:
            days_left = self.T - 1 - c
            ours = st.total
            th = self.thr[:, c]
            held = st.units > 0
            rows = np.arange(B)
            if self.r2 is not None:
                D, Y, K = self.r2
                go = ~self.done2 & (days_left <= D) & (ours >= th * (1 + Y)) & held.any(1)
                if go.any():
                    score = np.where(held, -self.vol[None], np.inf)          # wildste eerst
                    order = np.argsort(score, axis=1)[:, :K]
                    valid = np.take_along_axis(held, order, 1)
                    for k in range(order.shape[1]):
                        o = (order[:, k], -1, np.where(go & valid[:, k], 1.0, 0.0))
                        (out if k < 5 else self.pending).append(o)
                    self.done2 |= go
                    self.locked |= go
            if self.r1 is not None:
                D, X, K = self.r1
                go = ~self.done1 & ~self.locked & (days_left <= D) & (ours < th * (1 - X)) & held.any(1)
                if go.any():
                    score = np.where(held, self.vol[None], np.inf)           # rustigste eerst
                    sell = np.argsort(score, axis=1)[:, :K]
                    sv = np.take_along_axis(held, sell, 1) & go[:, None]
                    sold_val = (np.take_along_axis(st.units * st.price, sell, 1) * sv).sum(1)
                    pv = self.vol[self.pool]
                    pscore = np.where(held[:, self.pool], np.inf, -pv[None])  # meest beweeglijke, niet in bezit
                    buy = self.pool[np.argsort(pscore, axis=1)[:, :K]]
                    n_buy = sv.sum(1)
                    amt = np.where(n_buy > 0, (sold_val * (1 - 0.0035) - 5 * n_buy) / np.maximum(n_buy, 1), 0.0)
                    amt = np.minimum(amt, 0.099 * ours)
                    for k in range(sell.shape[1]):
                        out.append((sell[:, k], -1, np.where(sv[:, k], 1.0, 0.0)))
                    room = max(0, 5 - sell.shape[1])
                    for k in range(buy.shape[1]):
                        o = (buy[:, k], +1, np.where(sv[:, k], amt, 0.0))
                        (out if k < room else self.pending).append(o)
                        # het doel van BuyHold volgt de ruil (anders bouwt hij het oude aandeel opnieuw op)
                        tg = self.base.targets
                        hit = (tg == sell[:, [k]]) & sv[:, [k]]
                        tg[hit] = np.repeat(buy[:, k][:, None], tg.shape[1], 1)[hit]
                    self.done1 |= go
        return out


# --------------------------------------------------------------------------- een blok scenario's
def _block(job):
    ctx, n, bi, seed0, variants = job
    p, u, price0, vol, pool, starts = ctx["p"], ctx["u"], ctx["price0"], ctx["vol"], ctx["pool"], ctx["starts"]
    rng = np.random.default_rng(seed0 + bi)
    cum = next(generate(p, n, seed=seed0 * 7 + bi, chunk=n, device="numpy")).astype(np.float64)
    T = cum.shape[1]
    fidx = np.arange(n) % len(FIELDS)
    thr = np.zeros((n, T))
    fields = []
    quiz_cum = np.cumsum(~p.dates.isin(pd.to_datetime(list(Rules().no_quiz)))) * 100.0
    fa_from = int(np.searchsorted(p.dates, pd.Timestamp(Rules().fa_credit_date)))
    for f_i, cfg in enumerate(FIELDS):
        g = np.where(fidx == f_i)[0]
        fd = draw_field(cfg, u, rng)
        fin = field_values(fd, np.expm1(cum[g, -1, :]), rng)
        fields.append((g, fd, fin))
        # stand van de studenten op elke slotkoers (verwachte quiz tot dan, Finance Avenue vanaf 4 dec)
        st_idx = np.where(fd.student)[0]
        picks = fd.picks[st_idx]
        has = picks >= 0
        W = np.zeros((cum.shape[2], len(st_idx)), np.float32)
        r_, c_ = np.nonzero(has)
        np.add.at(W, (picks[r_, c_], r_), 1.0)
        k = has.sum(1)
        base = START - k * (5.0 + 0.0035 * POS * START)
        sim = np.expm1(cum[g].reshape(-1, cum.shape[2])).astype(np.float32) @ W   # (len(g)*T, Ps)
        sim = sim.reshape(len(g), T, -1).astype(np.float64) * POS * START
        val = sim + base[None, None] + fd.quiz_p[st_idx][None, None] * quiz_cum[None, :, None]
        val[:, fa_from:] += 500.0 * fd.fa_p[st_idx][None, None]
        thr[g] = np.partition(val, val.shape[2] - 3, axis=2)[:, :, -3]
    out = {}
    for sname, picks in starts.items():
        for vname, (freq, r1, r2) in variants.items():
            chk = checkpoints(p.dates, freq)
            pol = RulePolicy(picks, vol, pool, thr, chk, T, r1=r1, r2=r2)
            r = play(pol, cum, price0, p.dates, rules=Rules(), seed=bi)
            prize = np.zeros(n)
            t3 = np.zeros(n)
            for g, fd, fin in fields:
                prize[g] = our_prize(r.final[g], fin, fd.student)
                t3[g] = (np.where(fd.student[None], fin, -np.inf) > r.final[g][:, None]).sum(1) < 3
            out[(sname, vname)] = (prize, t3, r.executed.astype(float))
    return out


def run_set(ctx, n_total, seed0, variants, label):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    jobs = [(ctx, min(CHUNK, n_total - i), bi, seed0, variants) for bi, i in enumerate(range(0, n_total, CHUNK))]
    res = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for done, out in enumerate(ex.map(_block, jobs), 1):
            for key, arrs in out.items():
                res.setdefault(key, []).append(arrs)
            print(f"    {label}: blok {done}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
    return {k: tuple(np.concatenate([a[i] for a in v]) for i in range(3)) for k, v in res.items()}


def selftest() -> bool:
    from beursrally.scenarios.scen_core import trading_days
    dates = trading_days(pd.Timestamp("2026-11-13"))
    T, N, B = len(dates), 30, 2
    cum = np.zeros((B, T, N))
    price0 = np.full(N, 50.0)
    vol = np.linspace(0.1, 1.0, N)                              # effect 29 het wildst
    pool, start = np.arange(20, 30), np.arange(10)              # start: de 10 rustigste
    chk = checkpoints(dates, "week")
    ok_all = True

    def check(name, ok):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"  {'OK  ' if ok else 'FOUT'} {name}")

    print("Test handelsregels (vlakke koersen, vaste stand van de 3e student)")
    r = play(RulePolicy(start, vol, pool, np.full((B, T), 80_000.0), chk, T, r1=(30, 0.05, 3)), cum, price0, dates)
    held = r.units[0] > 0
    check("R1 (ver achter): de 3 rustigste verkocht, de 3 wildste uit de pool gekocht",
          not held[[0, 1, 2]].any() and list(np.where(held[20:])[0] + 20) == [27, 28, 29])
    r = play(RulePolicy(start, vol, pool, np.full((B, T), 30_000.0), chk, T, r2=(15, 0.0, 10)), cum, price0, dates)
    check(f"R2 K=10 (ver voor): alles naar cash, over 2 dagen (orders {r.executed[0]})",
          (r.units[0] > 0).sum() == 0 and r.executed[0] == 20)
    r = play(RulePolicy(start, vol, pool, np.full((B, T), 30_000.0), chk, T, r2=(15, 0.0, 5)), cum, price0, dates)
    check("R2 K=5: alleen de 5 wildste verkocht", list(np.where(r.units[0] > 0)[0]) == [0, 1, 2, 3, 4])
    r = play(RulePolicy(start, vol, pool, np.full((B, T), 55_000.0), chk, T, r1=(30, 0.05, 3), r2=(15, 0.05, 5)),
             cum, price0, dates)
    check("geen regel als we dicht bij de 3e student zitten (niets verkocht)", (r.units[0] > 0).sum() == 10)
    check(f"controlepunten: week {len(checkpoints(dates, 'week'))}, dag {len(checkpoints(dates, 'dag'))}, "
          f"eind {len(checkpoints(dates, 'eind'))}",
          len(checkpoints(dates, "dag")) == T - 1 and len(checkpoints(dates, "week")) == 9)
    print("ZELFTEST GESLAAGD" if ok_all else "ZELFTEST MISLUKT")
    return ok_all


# --------------------------------------------------------------------------- hoofdprogramma
def main():
    import sys
    if sys.argv[1:] == ["test"]:
        sys.exit(0 if selftest() else 1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--snel", action="store_true")
    ap.add_argument("--start", default=None, help="orderlijst (csv) met de startportefeuille")
    a = ap.parse_args()
    nA, nB = (1000, 1500) if a.snel else (4000, 6000)
    p = load_params(DATA / "scenarios" / "params_vandaag.npz")
    close = pd.read_parquet(DATA / "processed" / "close_eur.parquet")
    price0 = close.ffill().iloc[-1].reindex(p.isins).to_numpy(float)
    u = load_universe(p.isins, price0)
    vt = pd.read_parquet(DATA / "risk" / "vol_vandaag.parquet").set_index("isin").reindex(p.isins)
    vol = np.nan_to_num(vt["v_48"].to_numpy(), nan=0.0)
    names = vt["naam"].fillna("").to_numpy()
    ok = np.isfinite(price0) & (price0 > 0) & (price0 < POS * START) & ~u.is_etf
    by_vol = np.argsort(-np.where(ok, vol, -1))
    pool = by_vol[:20]
    f = Path(a.start) if a.start else max(OUT.glob("orders_start*.csv"), key=lambda x: x.stat().st_mtime)
    st = pd.read_csv(f)
    pos = {i: j for j, i in enumerate(p.isins)}
    chosen = np.array([pos[i] for i in st["isin"] if i in pos])
    starts = {"gekozen": chosen, "10 meest beweeglijk": by_vol[:10]}
    print(f"Startportefeuille uit {f.name}: " + ", ".join(names[chosen]))
    print(f"Inhaal-aandelen (R1): " + ", ".join(names[pool[:10]]) + ", ...")
    print(f"{WORKERS} processen | zoekset {nA:,}, controleset {nB:,} | vorm {p.shape.label()}")
    ctx = {"p": p, "u": u, "price0": price0, "vol": vol, "pool": pool, "starts": starts}

    # set A: alle instellingen
    variants = {"R0 kopen en houden": ("week", None, None)}
    for fq in FREQS:
        for r1 in R1_GRID:
            variants[f"R1 {fq} D={r1[0]} X={r1[1]:.0%} K={r1[2]}"] = (fq, r1, None)
        for r2 in R2_GRID:
            variants[f"R2 {fq} D={r2[0]} Y={r2[1]:.0%} K={r2[2]}"] = (fq, None, r2)
    print(f"Set A: {len(variants)} varianten x {len(starts)} startportefeuilles ...", flush=True)
    A = run_set(ctx, nA, 1001, variants, "zoekset")
    rowsA = [{"start": s, "variant": v, "prijzengeld": A[(s, v)][0].mean(), "P(top3)": A[(s, v)][1].mean(),
              "orders": A[(s, v)][2].mean()} for (s, v) in A]
    tA = pd.DataFrame(rowsA).sort_values(["start", "prijzengeld"], ascending=[True, False])
    OUT.mkdir(parents=True, exist_ok=True)
    tA.to_csv(OUT / "regels_zoekset.csv", index=False)
    g = tA[tA["start"] == "gekozen"].set_index("variant")["prijzengeld"]
    best = {}
    for fq in FREQS:
        for rr in ("R1", "R2"):
            cand = g[[v for v in g.index if v.startswith(f"{rr} {fq} ")]]
            best[(rr, fq)] = cand.idxmax()
    # set B: R0 + beste R1/R2 per frequentie + R3 = beide
    vB = {"R0 kopen en houden": variants["R0 kopen en houden"]}
    for fq in FREQS:
        b1, b2 = variants[best[("R1", fq)]], variants[best[("R2", fq)]]
        vB[best[("R1", fq)]] = b1
        vB[best[("R2", fq)]] = b2
        vB[f"R3 {fq}: {best[('R1', fq)][3:]} + {best[('R2', fq)][3:]}"] = (fq, b1[1], b2[2])
    print(f"Set B: {len(vB)} varianten (beste per regel en frequentie uit set A) ...", flush=True)
    Bres = run_set(ctx, nB, 2002, vB, "controleset")
    rows = []
    for s in starts:
        base = Bres[(s, "R0 kopen en houden")][0]
        for v in vB:
            pr, t3, ex = Bres[(s, v)]
            d = pr - base
            rows.append({"start": s, "variant": v, "prijzengeld": pr.mean(), "+/-": pr.std() / np.sqrt(len(pr)),
                         "verschil met R0": d.mean(), "+/- verschil": d.std() / np.sqrt(len(d)),
                         "P(top3 stud.)": t3.mean(), "orders": ex.mean()})
    tB = pd.DataFrame(rows)
    tB.to_csv(OUT / "regels_controle.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    for s in starts:
        sh = tB[tB["start"] == s].drop(columns="start").sort_values("prijzengeld", ascending=False).set_index("variant")
        for c in ("prijzengeld", "+/-", "verschil met R0", "+/- verschil"):
            sh[c] = sh[c].map(lambda v: f"{v:,.0f}")
        sh["P(top3 stud.)"] = sh["P(top3 stud.)"].map(lambda v: f"{v:.1%}")
        sh["orders"] = sh["orders"].map(lambda v: f"{v:.1f}")
        print(f"\n================ CONTROLESET: start = {s} ================")
        print(sh.to_string())
    sel = tB[tB["start"] == "gekozen"].set_index("variant")
    win = sel["verschil met R0"].idxmax()
    sig = sel.at[win, "verschil met R0"] > 2 * sel.at[win, "+/- verschil"]
    keuze = win if sig else "R0 kopen en houden"
    fq, r1, r2 = vB[keuze]
    (OUT / "regel_keuze.json").write_text(json.dumps({"variant": keuze, "frequentie": fq, "R1": r1, "R2": r2},
                                                    indent=2))
    print(f"\nGekozen regel: {keuze}" + ("" if sig else
          f"  (beste was {win}, maar niet duidelijk beter dan R0: verschil < 2 x de onzekerheid)"))
    print("-> data/strategy/regels_zoekset.csv, regels_controle.csv, regel_keuze.json")


if __name__ == "__main__":
    main()
