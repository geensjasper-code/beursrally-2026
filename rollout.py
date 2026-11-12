#!/usr/bin/env python3
"""
Beursrally 2026 - rollout (stap 6b-3): elke vrijdag opnieuw rekenen vanuit de huidige toestand.

Gebruik
    python rollout.py test       # test: kiest de rollout een duidelijk betere ruil, en laat hij ruis liggen?
    python rollout.py            # rollout tegenover kopen en houden, gepaard (parallel)
    python rollout.py --snel     # minder scenario's

Beslissing op elk controlepunt (vrijdag na de slot), uitvoering maandag aan de opening:
    1. Huidige toestand: onze posities en cash, de stand van alle concurrenten, de resterende dagen.
    2. 500 binnenste toekomsten voor de resterende dagen (zelfde vorm als de scenario's).
    3. Verwacht prijzengeld voor 'niets doen' en voor elke mogelijke actie:
         ruil   verkoop positie i (helemaal), koop j met de opbrengst + vrije cash, tot net onder 10%
         cash   verkoop positie i, niets terugkopen (gedeeltelijk vastzetten)
         nieuw  koop j met vrije cash (vanaf 1.500 euro), zonder te verkopen
       j uit: de 20 meest beweeglijke + het hot thema (betaalbaar, nog niet in bezit). Kosten: 5 euro + 0,35% per order.
    4. Kiezen op de helft van de binnenste toekomsten, BEVESTIGEN op de andere helft (tegen de winnaarsvloek: de
       beste van ~300 ruisende schattingen lijkt altijd goed). Alleen als de actie op beide helften duidelijk beter
       is (meer dan 2 x de onzekerheid). Dan opnieuw vanuit de nieuwe toestand: max. 2 acties (= max. 4 orders).
    Binnenste concurrenten: de 600 beste algemeen en de 600 beste studenten van dat moment (de rest kan de top 3
    in de resterende weken nauwelijks nog halen). Binnen een toekomst: na de actie kopen en houden.

Vergelijking: dezelfde buitenste scenario's en velden voor kopen en houden (R0) en de rollout.
Uitvoer: data/strategy/rollout.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from field import FIELDS, POS, PRIZES, START, draw_field, field_values, load_universe, our_prize
from game import BuyHold, Rules, play
from rules import checkpoints
from scen_core import Params, generate, load_params

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = DATA / "strategy"
WORKERS = int(os.environ.get("BR_WORKERS", str(max(1, min(16, (os.cpu_count() or 2) - 1)))))
M_INNER = 1000                   # helft om te kiezen, helft om te bevestigen
TOP_K = 600
MAX_ACT = 2
TAX, FIX = 0.0035, 5.0


# --------------------------------------------------------------------------- prijzengeld voor veel kandidaten tegelijk
def batch_prize(ours: np.ndarray, gv: np.ndarray, gid: np.ndarray, sv: np.ndarray, sid: np.ndarray) -> np.ndarray:
    """ours (C, M) onze eindwaarde per kandidaat en binnenste toekomst; gv/gid (M, 8) beste algemeen, sv/sid (M, 8)
    beste studenten. Geeft (C, M) prijzengeld (een prijs per persoon, de hoogste)."""
    C, M = ours.shape
    lists = {}
    for kind, v, i in (("G", gv, gid), ("S", sv, sid)):
        vv = np.concatenate([np.broadcast_to(v, (C, M, v.shape[1])), ours[:, :, None]], axis=2)
        ii = np.concatenate([np.broadcast_to(i, (C, M, i.shape[1])), np.full((C, M, 1), -1)], axis=2)
        o = np.argsort(-vv, axis=2, kind="stable")
        lists[kind] = np.take_along_axis(ii, o, 2)
    awarded = np.full((C, M, len(PRIZES)), -2)
    prize = np.zeros((C, M))
    for j, (kind, _, amount) in enumerate(PRIZES):
        lst = lists[kind]
        taken = (lst[..., :, None] == awarded[..., None, :]).any(-1)
        who = np.take_along_axis(lst, np.argmax(~taken, axis=2)[..., None], 2)[..., 0]
        awarded[..., j] = who
        prize += np.where(who == -1, amount, 0.0)
    return prize


def decide(units, cash, price, total, e_inner, tops, cand, vol, quiz_left, max_act=MAX_ACT):
    """Een rollout-beslissing voor een scenario. units/price (N,), e_inner (M, N) = exp(rendement rest),
    tops = (gv, gid, sv, sid). Geeft lijst acties (verkoop i of -1, koop j of -1, bedrag)."""
    acts = []
    units, cash = units.copy(), float(cash)
    for _ in range(max_act):
        held = np.where(units > 0)[0]
        val = units * price
        base = cash + quiz_left + (val[None, :] * e_inner).sum(1)              # (M,) niets doen
        opts = []                                                               # (verkoop, koop)
        buys = [j for j in cand if units[j] == 0 and price[j] <= 0.099 * total]
        for i in held:
            for j in buys:
                opts.append((i, j))
            opts.append((i, -1))
        if cash >= 1500:
            opts += [(-1, j) for j in buys]
        if not opts:
            break
        O = np.array(opts)
        si, bj = O[:, 0], O[:, 1]
        v_i = np.where(si >= 0, val[np.maximum(si, 0)], 0.0)
        proceeds = np.where(si >= 0, v_i * (1 - TAX) - FIX, 0.0)
        avail = proceeds + cash - 50.0                                          # kleine buffer
        amt = np.where(bj >= 0, np.clip(np.minimum(avail - FIX, 0.099 * total) / (1 + TAX), 0, None), 0.0)
        q = np.where(bj >= 0, np.floor(amt / price[np.maximum(bj, 0)]), 0.0)    # hele aandelen
        amt = q * np.where(bj >= 0, price[np.maximum(bj, 0)], 0.0)
        spend = np.where(q > 0, amt * (1 + TAX) + FIX, 0.0)
        e_i = np.where(si[:, None] >= 0, e_inner[:, np.maximum(si, 0)].T, 0.0)  # (C, M)
        e_j = np.where(bj[:, None] >= 0, e_inner[:, np.maximum(bj, 0)].T, 0.0)
        ours = base[None] - v_i[:, None] * e_i + (proceeds - spend)[:, None] + amt[:, None] * e_j
        allc = np.vstack([base[None], ours])
        pr = batch_prize(allc, *tops)
        d = pr[1:] - pr[0][None]
        # winnaarsvloek: de beste van ~300 ruisende schattingen ziet er altijd goed uit. Daarom kiezen op de ene
        # helft van de binnenste toekomsten en BEVESTIGEN op de andere helft (die bij het kiezen niet meetelde).
        h = d.shape[1] // 2
        dA, dB = d[:, :h], d[:, h:]
        gA = dA.mean(1) - 2 * dA.std(1) / np.sqrt(h)
        k = int(np.argmax(gA))
        gB = dB[k].mean() - 2 * dB[k].std() / np.sqrt(dB.shape[1])
        if gA[k] <= 0 or gB <= 0:
            break
        i, j = int(si[k]), int(bj[k])
        acts.append((i, j, float(amt[k])))
        if i >= 0:
            cash += proceeds[k]
            units[i] = 0
        if j >= 0:
            cash -= spend[k]
            units[j] += q[k]
    return acts


# --------------------------------------------------------------------------- de strategie in de spelmotor
class RolloutPolicy:
    def __init__(self, start, cand, vol, chk, T, info):
        self.build = BuyHold(start, weight=POS, quiz="bijkopen")
        self.cand, self.vol, self.chk, self.T, self.info = cand, vol, set(int(c) for c in chk), T, info
        self.n_act = None

    def orders(self, st):
        B = len(st.cash)
        if self.n_act is None:
            self.n_act = np.zeros(B, int)
        if st.day < 2:
            return self.build.orders(st)
        out = []
        c = st.day - 1
        if c in self.chk:
            sells, buys = [], []
            for b in range(B):
                ctx = self.info(b, c)
                if ctx is None:
                    continue
                e_inner, tops, quiz_left = ctx
                acts = decide(st.units[b], st.cash[b], st.price[b], st.total[b], e_inner, tops, self.cand,
                              self.vol, quiz_left)
                for i, j, amt in acts:
                    if i >= 0:
                        sells.append((b, i))
                    if j >= 0:
                        buys.append((b, j, amt))
                self.n_act[b] += len(acts)
            for k in range(MAX_ACT):
                for lst, side in ((sells, -1), (buys, +1)):
                    sec = np.zeros(B, int)
                    amt = np.zeros(B)
                    seen = {}
                    for item in lst:
                        b = item[0]
                        n = seen.get(b, 0)
                        seen[b] = n + 1
                        if n == k:
                            sec[b] = item[1]
                            amt[b] = 1.0 if side < 0 else item[2]
                    if amt.any():
                        out.append((sec, side, amt))
        # quizgeld bijkopen: PRECIES zoals kopen en houden (zelfde dagen, zelfde regel), naast eventuele acties.
        # Zo verschillen de twee strategieen alleen in de rollout-acties.
        if st.day % self.build.every == 0:
            held = st.units > 0
            val = st.units * st.price
            gap = np.where(held, POS * st.total[:, None] - val, -np.inf)
            order = np.argsort(-gap, axis=1)
            budget = st.cash.copy()
            for o in out:                                                   # wat de acties al uitgeven
                if o[1] > 0:
                    budget -= np.where(o[2] > 0, o[2] * (1 + TAX) + FIX, 0.0)
                else:
                    budget += np.where(o[2] > 0, val[np.arange(B), o[0]] * (1 - TAX) - FIX, 0.0)
            for k in range(5):
                j = order[:, k]
                g = gap[np.arange(B), j]
                a = np.minimum(g, (budget - 5.0) / (1 + TAX))
                a = np.where(a >= self.build.min_order, a, 0.0)
                budget -= np.where(a > 0, a * (1 + TAX) + 5.0, 0.0)
                out.append((j, +1, a))
        return out


# --------------------------------------------------------------------------- een blok buitenste scenario's
def _block(job):
    ctx, n, bi, seed0 = job
    p, u, price0, vol, cand, start = ctx["p"], ctx["u"], ctx["price0"], ctx["vol"], ctx["cand"], ctx["start"]
    rng = np.random.default_rng(seed0 + bi)
    cum = next(generate(p, n, seed=seed0 * 7 + bi, chunk=n, device="numpy")).astype(np.float64)
    T, N = cum.shape[1], cum.shape[2]
    chk = checkpoints(p.dates, "week")
    no_quiz = Rules().no_quiz
    qday = ~p.dates.isin(pd.to_datetime(list(no_quiz)))
    quiz_cum = np.cumsum(qday) * 100.0
    fa_idx = int(np.searchsorted(p.dates, pd.Timestamp(Rules().fa_credit_date)))
    fidx = np.arange(n) % len(FIELDS)
    fields = []
    for f_i, cfg in enumerate(FIELDS):
        g = np.where(fidx == f_i)[0]
        fd = draw_field(cfg, u, rng)
        fin = field_values(fd, np.expm1(cum[g, -1, :]), rng)
        has = fd.picks >= 0
        W = np.zeros((N, len(fd.picks)), np.float32)
        r_, c_ = np.nonzero(has)
        np.add.at(W, (fd.picks[r_, c_], r_), 1.0)
        base = START - has.sum(1) * (FIX + TAX * POS * START)
        fields.append({"g": g, "fd": fd, "fin": fin, "W": W, "base": base})
    # binnenste toekomsten per controlepunt (gedeeld: gemeenschappelijke toevalsgetallen)
    inner = {}
    for c in chk:
        rows = slice(p.skip + c + 1, None)
        q = Params(p.isins, p.dates[c + 1:], 0, p.drift[rows], p.sd[rows], p.root, shape=p.shape)
        inner[c] = np.exp(next(generate(q, M_INNER, seed=seed0 * 13 + bi * 100 + c, chunk=M_INNER,
                                        device="numpy"))[:, -1, :].astype(np.float64))
    row_field = {int(b): f for f in fields for b in f["g"]}

    def info(b, c):
        f = row_field[b]
        fd = f["fd"]
        e_c = np.exp(cum[b, c])                                                  # koersniveau op c t.o.v. start
        posval = POS * START * e_c                                               # waarde van een positie op c
        v_c = f["base"] + (np.expm1(cum[b, c]).astype(np.float32) @ f["W"]) * POS * START \
            + fd.quiz_p * quiz_cum[c] + (500.0 * fd.fa_p if c >= fa_idx else 0.0)
        top_all = np.argpartition(-v_c, TOP_K)[:TOP_K]
        sv_c = np.where(fd.student, v_c, -np.inf)
        top_st = np.argpartition(-sv_c, TOP_K)[:TOP_K]
        idx = np.union1d(top_all, top_st)
        Wk = f["W"][:, idx] * posval[:, None].astype(np.float32)
        e = inner[c]
        fut = v_c[idx][None] + ((e - 1.0).astype(np.float32) @ Wk) \
            + (fd.quiz_p[idx] * (quiz_cum[-1] - quiz_cum[c]))[None] \
            + (500.0 * fd.fa_p[idx] if c < fa_idx else 0.0)
        stud = fd.student[idx]
        o = np.argsort(-fut, axis=1)[:, :8]
        gv, gid = np.take_along_axis(fut, o, 1), idx[o]
        fs = np.where(stud[None], fut, -np.inf)
        o = np.argsort(-fs, axis=1)[:, :8]
        sv, sid = np.take_along_axis(fs, o, 1), idx[o]
        quiz_left = (quiz_cum[-1] - quiz_cum[c]) + (500.0 if c < fa_idx else 0.0)
        return e, (gv, gid, sv, sid), quiz_left

    out = {}
    for name, pol in (("R0 kopen en houden", BuyHold(start, weight=POS, quiz="bijkopen")),
                      ("rollout", RolloutPolicy(start, cand, vol, chk, T, info))):
        r = play(pol, cum, price0, p.dates, rules=Rules(), seed=bi)
        prize = np.zeros(n)
        t3 = np.zeros(n)
        for f in fields:
            g = f["g"]
            prize[g] = our_prize(r.final[g], f["fin"], f["fd"].student)
            t3[g] = (np.where(f["fd"].student[None], f["fin"], -np.inf) > r.final[g][:, None]).sum(1) < 3
        acts = pol.n_act if name == "rollout" else np.zeros(n)
        out[name] = (prize, t3, r.executed.astype(float), acts.astype(float))
    return out


def main():
    if sys.argv[1:] == ["test"]:
        sys.exit(0 if selftest() else 1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--snel", action="store_true")
    ap.add_argument("--start", default=None)
    a = ap.parse_args()
    n_out, blk = (120, 20) if a.snel else (900, 30)
    p = load_params(DATA / "scenarios" / "params_vandaag.npz")
    close = pd.read_parquet(DATA / "processed" / "close_eur.parquet")
    price0 = close.ffill().iloc[-1].reindex(p.isins).to_numpy(float)
    u = load_universe(p.isins, price0)
    vt = pd.read_parquet(DATA / "risk" / "vol_vandaag.parquet").set_index("isin").reindex(p.isins)
    vol = np.nan_to_num(vt["v_48"].to_numpy(), nan=0.0)
    names = vt["naam"].fillna("").to_numpy()
    ok = np.isfinite(price0) & (price0 > 0) & (price0 < POS * START) & ~u.is_etf
    by_vol = np.argsort(-np.where(ok, vol, -1))
    hot = [j for j in u.vol_top if u.theme[j] == u.hot and ok[j]]
    cand = np.array(list(dict.fromkeys(list(by_vol[:20]) + hot)))
    f = Path(a.start) if a.start else max(OUT.glob("orders_start*.csv"), key=lambda x: x.stat().st_mtime)
    pos = {i: j for j, i in enumerate(p.isins)}
    start = np.array([pos[i] for i in pd.read_csv(f)["isin"] if i in pos])
    print(f"Startportefeuille uit {f.name}: " + ", ".join(names[start]))
    print(f"Kandidaten om te kopen ({len(cand)}): " + ", ".join(names[cand]))
    print(f"{WORKERS} processen | {n_out} buitenste scenario's x {M_INNER} binnenste | controlepunten: elke vrijdag")
    ctx = {"p": p, "u": u, "price0": price0, "vol": vol, "cand": cand, "start": start}
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    jobs = [(ctx, min(blk, n_out - i), bi, 3003) for bi, i in enumerate(range(0, n_out, blk))]
    res = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for done, out in enumerate(ex.map(_block, jobs), 1):
            for k, v in out.items():
                res.setdefault(k, []).append(v)
            print(f"    blok {done}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
    res = {k: tuple(np.concatenate([x[i] for x in v]) for i in range(4)) for k, v in res.items()}
    base = res["R0 kopen en houden"][0]
    rows = []
    for k, (pr, t3, ex, acts) in res.items():
        d = pr - base
        rows.append({"strategie": k, "prijzengeld": pr.mean(), "+/-": pr.std() / np.sqrt(len(pr)),
                     "verschil met R0": d.mean(), "+/- verschil": d.std() / np.sqrt(len(d)),
                     "P(top3 stud.)": t3.mean(), "orders": ex.mean(), "acties": acts.mean()})
    t = pd.DataFrame(rows).set_index("strategie")
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "rollout.csv")
    sh = t.copy()
    for c in ("prijzengeld", "+/-", "verschil met R0", "+/- verschil"):
        sh[c] = sh[c].map(lambda v: f"{v:,.0f}")
    sh["P(top3 stud.)"] = sh["P(top3 stud.)"].map(lambda v: f"{v:.1%}")
    for c in ("orders", "acties"):
        sh[c] = sh[c].map(lambda v: f"{v:.1f}")
    print("\n================ ROLLOUT TEGENOVER KOPEN EN HOUDEN (zelfde scenario's) ================")
    print(sh.to_string())
    d, s = t.at["rollout", "verschil met R0"], t.at["rollout", "+/- verschil"]
    print(f"\nRollout {'duidelijk beter' if d > 2 * s else 'duidelijk slechter' if d < -2 * s else 'niet duidelijk anders'}"
          f" dan kopen en houden ({d:+,.0f} +/- {s:,.0f} euro). -> data/strategy/rollout.csv")


# --------------------------------------------------------------------------- test
def selftest() -> bool:
    ok_all = True

    def check(name, ok):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"  {'OK  ' if ok else 'FOUT'} {name}")

    print("Test rollout-beslissing (10 posities, eenvoudige binnenste toekomsten)")
    rng = np.random.default_rng(0)
    M, N = 2000, 12
    price = np.full(N, 50.0)
    units = np.zeros(N)
    units[:10] = 100.0                                                  # 10 posities van 5.000 euro
    total = 50_000.0 + 500
    # aandelen 0..9 rustig (vol 5%), 10 heel wild (vol 80%), 11 rustig
    sd = np.r_[np.full(10, 0.05), 0.8, 0.05]
    e = np.exp(rng.normal(-sd ** 2 / 2, sd, (M, N)))
    # drempel 3e plaats ~ +20%: met alleen rustige posities haal je dat bijna nooit
    gv = np.sort(rng.normal(60_000, 1500, (M, 8)), axis=1)[:, ::-1]
    gid = np.tile(np.arange(8), (M, 1))
    tops = (gv, gid, gv.copy(), gid.copy())
    acts = decide(units, 500.0, price, total, e, tops, np.array([10, 11]), sd, 0.0)
    check(f"ver achter met rustige posities: ruilt naar het wilde aandeel ({acts})",
          len(acts) >= 1 and acts[0][1] == 10)
    tops_easy = (np.full((M, 8), 40_000.0), gid, np.full((M, 8), 40_000.0), gid)
    acts = decide(units, 500.0, price, total, e, tops_easy, np.array([10, 11]), sd, 0.0)
    check(f"al zeker winnaar: geen actie (alleen kosten) ({acts})", acts == [])
    # winnaarsvloek: 20 kandidaten die precies hetzelfde zijn als wat we hebben (geen echte winst, wel kosten)
    sd2 = np.full(30, 0.4)
    e2 = np.exp(rng.normal(-0.08, 0.4, (M, 30)))
    units2 = np.zeros(30)
    units2[:10] = 100.0
    gv2 = np.sort(rng.normal(56_000, 3000, (M, 8)), axis=1)[:, ::-1]
    tops2 = (gv2, gid, gv2.copy(), gid.copy())
    n_act = sum(len(decide(units2, 500.0, np.full(30, 50.0), total, np.exp(rng.normal(-0.08, 0.4, (M, 30))),
                           tops2, np.arange(10, 30), sd2, 0.0)) for _ in range(10))
    check(f"geen echte winst (kandidaten gelijk aan bezit): bijna nooit een actie ({n_act} acties in 10 pogingen)",
          n_act <= 1)
    from scen_core import trading_days
    dates = trading_days(pd.Timestamp("2026-11-13"))
    cumx = np.cumsum(rng.normal(0, 0.03, (8, len(dates), 30)), axis=1)
    pol = RolloutPolicy(np.arange(10), np.arange(10, 20), np.ones(30), checkpoints(dates, "week"), len(dates),
                        lambda b, c: None)
    a = play(pol, cumx, np.full(30, 40.0), dates).final
    b = play(BuyHold(np.arange(10), weight=POS, quiz="bijkopen"), cumx, np.full(30, 40.0), dates).final
    check("rollout zonder acties = exact kopen en houden (zelfde bijkoopdagen)", np.allclose(a, b))
    pr = batch_prize(np.array([[70_000.0], [45_000.0]]), *(x[:1] for x in tops))
    check(f"prijzengeld per kandidaat: hoog = 10.000, laag = 0 ({pr.ravel()})", pr[0, 0] == 10_000 and pr[1, 0] == 0)
    print("ZELFTEST GESLAAGD" if ok_all else "ZELFTEST MISLUKT")
    return ok_all


if __name__ == "__main__":
    main()
