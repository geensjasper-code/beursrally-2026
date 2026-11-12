#!/usr/bin/env python3
"""
Beursrally 2026 - strategie (stap 6, deel 1): de startportefeuille met het hoogste verwachte prijzengeld.

Gebruik
    python strategy.py              # zoeken + controle + orderlijst voor maandag en dinsdag
    python strategy.py --snel       # minder scenario's (om te proberen)
    python strategy.py --zonder-recent   # zonder recente beursgangen (correlaties aangevuld, niet gemeten)
    python strategy.py --uitsluiten "Bpost,Ontex"   # deze namen of tickers niet kiezen (eigen oordeel)
    python strategy.py --kandidaten 120             # zoeken over de 120 meest beweeglijke (standaard 70)
    python strategy.py --kandidaten alle            # zoeken over alle betaalbare aandelen (traag)

Doel
    Verwacht prijzengeld voor ons (student), gemiddeld over de velden van field.py (basis,
    minder/meer actief, veel gokkers, een hot thema, momentumjagers; elk 32.000 deelnemers). Prijzen volgens
    het reglement (een prijs per persoon, doorschuiven).

Werkwijze
    1. Zoekscenario's (set A) en controlescenario's (set B, met nieuw getrokken velden): zo jagen we niet op
       toevalstreffers in de zoekset.
    2. Snel zoeken op set A met kopen-en-houden (10 posities van 9,5%, quiz en Finance Avenue):
       kandidaten = 70 (of --kandidaten N / alle) meest beweeglijke + 20 beste rendementsvoorspellingen + het
       hot thema (betaalbaar); willekeurige startpunten komen altijd uit de 70 meest beweeglijke (goede starts),
       wissels mogen naar elke kandidaat;
       startpunten = meest beweeglijk, beste voorspelling, hot thema, beste van 300 willekeurige
       + 15 willekeurige (het zoeken heeft meerdere lokale toppen); alle zoektochten lopen tegelijk in aparte
       processen; aandeel per aandeel wisselen zolang het verwachte prijzengeld met minstens 1% stijgt.
    3. Exacte controle op set B: de 6 beste verschillende eindportefeuilles en de vergelijkingen worden met de
       spelmotor gespeeld (5 + 5 orders in de volgorde van de orderlijst, hele aandelen, quizgeld bijkopen).
       De beste op set B wordt gekozen.
       Toets tegen toeval: 2000 willekeurige portefeuilles uit de kandidaten op dezelfde set B (snelle
       berekening); waar valt de gekozen portefeuille in die verdeling?
    4. Orderlijst: 5 aankopen voor maandag, 5 voor dinsdag (de meest beweeglijke eerst), elk 4.750 euro
       (9,5%: ruimte voor een hogere openingskoers zonder de 10%-grens te raken).

Uitvoer: data/strategy/orders_start.csv en data/strategy/vergelijking.csv
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from field import FIELDS, PRIZES, START, draw_field, field_values, load_universe, our_prize
from game import BuyHold, Rules, play
from scen_core import generate, load_params

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = DATA / "strategy"
POS = 0.095
N_POS = 10
TOP = 8
N_RAND_STARTS = 15
N_RAND_TEST = 2000
N_FINAL = 6


# --------------------------------------------------------------------------- snelle evaluatie
class FieldTops:
    """Per veld en scenario: de beste 8 algemeen en de beste 8 studenten (waarde en id). Genoeg om ons
    prijzengeld exact te bepalen (er zijn 6 prijzen)."""

    def __init__(self):
        self.g_val, self.g_id, self.s_val, self.s_id = [], [], [], []

    def add(self, vals: np.ndarray, student: np.ndarray, offset: int):
        for v, kv, ki in ((vals, self.g_val, self.g_id), (np.where(student[None], vals, -np.inf), self.s_val, self.s_id)):
            part = np.argpartition(-v, TOP, axis=1)[:, :TOP]
            o = np.take_along_axis(part, np.argsort(-np.take_along_axis(v, part, 1), axis=1), 1)
            kv.append(np.take_along_axis(v, o, 1))
            ki.append(o + offset)

    def done(self):
        for k in ("g_val", "g_id", "s_val", "s_id"):
            setattr(self, k, np.concatenate(getattr(self, k)))
        return self


def fast_prize(ours: np.ndarray, ft: FieldTops) -> np.ndarray:
    S = len(ours)
    rows = np.arange(S)
    lists = {}
    for kind, v, i in (("G", ft.g_val, ft.g_id), ("S", ft.s_val, ft.s_id)):
        vv = np.c_[v, ours]
        ii = np.c_[i, np.full(S, -1)]
        o = np.argsort(-vv, axis=1)
        lists[kind] = np.take_along_axis(ii, o, 1)
    awarded = np.full((S, len(PRIZES)), -2)
    prize = np.zeros(S)
    for j, (kind, _, amount) in enumerate(PRIZES):
        lst = lists[kind]
        taken = (lst[:, :, None] == awarded[:, None, :]).any(2)
        who = lst[rows, np.argmax(~taken, axis=1)]
        awarded[:, j] = who
        prize += np.where(who == -1, amount, 0.0)
    return prize


def our_bh(simple: np.ndarray, picks) -> np.ndarray:
    pos = POS * START
    return START + simple[:, picks].sum(1) * pos - len(picks) * (5 + 0.0035 * pos) + 46 * 100 + 500


def objective(simple, tops, picks) -> float:
    o = our_bh(simple, picks)
    return float(np.mean([fast_prize(o, ft).mean() for ft in tops]))


# --------------------------------------------------------------------------- controle (parallel)
CHECK_CHUNK = 500
WORKERS = int(os.environ.get("BR_WORKERS", str(max(1, min(16, (os.cpu_count() or 2) - 1)))))


def _check_chunk(job):
    """Een blok van de controleset in een eigen proces: scenario's maken (numpy), velden trekken,
    alle kandidaten exact spelen met de spelmotor. Zelfde code als zonder parallel; elk blok heeft eigen seeds."""
    p, n, ci, u, price0, cands, rand_sets = job
    rng = np.random.default_rng(202_000 + ci)
    cum = next(generate(p, n, seed=202 + 1000 * ci, chunk=n, device="numpy")).astype(np.float64)
    s = np.expm1(cum[:, -1, :])
    fields = []
    for cfg in FIELDS:
        fd = draw_field(cfg, u, rng)
        fields.append((fd, field_values(fd, s, rng)))
    out = {}
    for k, picks in cands.items():
        r = play(BuyHold(picks, weight=POS, quiz="bijkopen"), cum, price0, p.dates, rules=Rules(), seed=ci)
        pr = np.mean([our_prize(r.final, vals, fd.student) for fd, vals in fields], axis=0)
        t3 = np.mean([((np.where(fd.student[None], vals, -np.inf) > r.final[:, None]).sum(1) < 3)
                      for fd, vals in fields], axis=0)
        out[k] = (pr, t3, r.final)
    # toets tegen toeval: snelle berekening (zoals in het zoeken) voor willekeurige en gekozen portefeuilles
    tops = []
    for fd, vals in fields:
        ft = FieldTops()
        ft.add(vals, fd.student, offset=0)
        tops.append(ft.done())
    fast_r = np.array([objective(s, tops, r_) for r_ in rand_sets]) * n
    fast_c = {k: objective(s, tops, picks) * n for k, picks in cands.items()}
    out["__fast__"] = (fast_r, fast_c, n)
    return out


def _one_thread():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"                       # elk proces 1 draad: geen gedrang tussen processen


def parallel_check(p, nB, u, price0, cands, rand_sets=(), workers=WORKERS):
    _one_thread()
    jobs = [(p, min(CHECK_CHUNK, nB - i), ci, u, price0, cands, list(rand_sets))
            for ci, i in enumerate(range(0, nB, CHECK_CHUNK))]
    res = {k: {"prize": [], "t3": [], "final": []} for k in cands}
    fr, fc, ntot = np.zeros(len(rand_sets)), {k: 0.0 for k in cands}, 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for done, out in enumerate(ex.map(_check_chunk, jobs), 1):
            r_, c_, n_ = out.pop("__fast__")
            fr += r_
            ntot += n_
            for k in cands:
                fc[k] += c_[k]
            for k, (pr, t3, fin) in out.items():
                res[k]["prize"].append(pr)
                res[k]["t3"].append(t3)
                res[k]["final"].append(fin)
            print(f"    blok {done}/{len(jobs)} klaar", flush=True)
    return res, fr / ntot, {k: v / ntot for k, v in fc.items()}


# --------------------------------------------------------------------------- zoekset (parallel)
SET_A_CHUNK = 500


def _setA_chunk(job):
    """Een blok van de zoekset in een eigen proces: scenario's (numpy), velden trekken, top-8 per veld."""
    p, n, ci, u = job
    rng = np.random.default_rng(101_000 + ci)
    cum = next(generate(p, n, seed=101 + 1000 * ci, chunk=n, device="numpy"))
    s = np.expm1(cum[:, -1, :].astype(np.float64))
    parts = []
    for cfg in FIELDS:
        fd = draw_field(cfg, u, rng)
        ft = FieldTops()
        ft.add(field_values(fd, s, rng), fd.student, offset=ci * 10**6)
        parts.append((ft.g_val[0], ft.g_id[0], ft.s_val[0], ft.s_id[0]))
    return s, parts


def parallel_setA(p, nA, u, workers=WORKERS):
    _one_thread()
    jobs = [(p, min(SET_A_CHUNK, nA - i), ci, u) for ci, i in enumerate(range(0, nA, SET_A_CHUNK))]
    simple, tops = [], [FieldTops() for _ in FIELDS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for s, parts in ex.map(_setA_chunk, jobs):
            simple.append(s)
            for ft, (gv, gi, sv, si) in zip(tops, parts):
                ft.g_val.append(gv); ft.g_id.append(gi); ft.s_val.append(sv); ft.s_id.append(si)
    return np.concatenate(simple), [t.done() for t in tops]


# --------------------------------------------------------------------------- zoeken
def local_search(simple, tops, start, pool, log_name="", max_iter=12, quiet=False):
    cur = list(start)
    best = objective(simple, tops, cur)
    for it in range(max_iter):
        improved = False
        cand_best, swap = best, None
        for k in range(len(cur)):
            for c in pool:
                if c in cur:
                    continue
                trial = cur[:k] + [c] + cur[k + 1:]
                v = objective(simple, tops, trial)
                if v > cand_best:
                    cand_best, swap = v, (k, c)
        if swap and cand_best > best * 1.01:
            cur[swap[0]] = swap[1]
            best = cand_best
            improved = True
        if not improved:
            break
    if not quiet:
        print(f"  {log_name:<28} -> {best:7.1f} euro (zoekset)")
    return cur, best


_LS = {}


def _ls_init(simple, tops, pool, startpool):
    _one_thread()
    _LS.update(simple=simple, tops=tops, pool=pool, startpool=startpool)


def _ls_job(job):
    lab, start = job
    t = time.time()
    if start is None:                              # 'beste van 300 willekeurige': ook in een eigen proces
        rr = np.random.default_rng(7)
        rand = [list(rr.choice(_LS["startpool"], N_POS, replace=False)) for _ in range(300)]
        start = max(rand, key=lambda x: objective(_LS["simple"], _LS["tops"], x))
    res, val = local_search(_LS["simple"], _LS["tops"], start, _LS["pool"], quiet=True)
    return lab, [int(j) for j in res], val, time.time() - t


def parallel_search(simple, tops, pool, startpool, starts: dict, workers=WORKERS):
    """Lokaal zoeken vanaf elk startpunt tegelijk (elk proces krijgt set A een keer)."""
    out = {}
    jobs = [(k, None if v is None else [int(j) for j in v]) for k, v in starts.items()]
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs))), initializer=_ls_init,
                             initargs=(simple, tops, [int(j) for j in pool], [int(j) for j in startpool])) as ex:
        for lab, res, val, dt in ex.map(_ls_job, jobs):
            print(f"  {lab:<34} -> {val:7.1f} euro (zoekset, {dt:.0f} s)", flush=True)
            out[lab] = (res, val)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snel", action="store_true")
    ap.add_argument("--zonder-recent", action="store_true", help="recente beursgangen (aangevulde correlaties) uitsluiten")
    ap.add_argument("--kandidaten", default="70", help="aantal meest beweeglijke kandidaten, of 'alle'")
    ap.add_argument("--uitsluiten", default="", help="komma-gescheiden namen of tickers die niet gekozen mogen worden")
    a = ap.parse_args()
    nA, nB = (2000, 2000) if a.snel else (6000, 6000)
    t0 = time.time()
    p = load_params(DATA / "scenarios" / "params_vandaag.npz")
    close = pd.read_parquet(DATA / "processed" / "close_eur.parquet")
    price0 = close.ffill().iloc[-1].reindex(p.isins).to_numpy(float)
    u = load_universe(p.isins, price0)
    vol = pd.read_parquet(DATA / "risk" / "vol_vandaag.parquet").set_index("isin").reindex(p.isins)
    ret = pd.read_parquet(DATA / "returns" / "rendement_vandaag.parquet").set_index("isin").reindex(p.isins)
    names = vol["naam"].fillna("").to_numpy()
    tick = vol["ticker"].fillna("").to_numpy()
    ok = np.isfinite(price0) & (price0 > 0) & (price0 < POS * START) & ~u.is_etf
    if a.uitsluiten:
        want = [w.strip().lower() for w in a.uitsluiten.split(",") if w.strip()]
        nm = vol["naam"].fillna("").str.lower().to_numpy()
        tk_ = vol["ticker"].fillna("").str.lower().to_numpy()
        hit = np.array([any(w == t or w in n for w in want) for n, t in zip(nm, tk_)])
        ok &= ~hit
        print("Uitgesloten op eigen oordeel: " + ", ".join(vol["naam"].to_numpy()[hit].astype(str)))
    if a.zonder_recent:
        recent = set(p.info["zonder_correlaties"])
        ok &= ~np.array([i in recent for i in p.isins])
        print("Zonder recente beursgangen: " + ", ".join(n for n, i in zip(vol['naam'].fillna(''), p.isins) if i in recent))
    v48 = np.where(ok, vol["v_48"].to_numpy(), -np.inf)
    a48 = np.where(ok, ret["alpha_48"].fillna(-1).to_numpy(), -np.inf)
    n_ok = int(ok.sum())
    n_vol = n_ok if a.kandidaten.strip().lower() == "alle" else min(int(a.kandidaten), n_ok)
    order_v = [j for j in np.argsort(-v48) if ok[j]]
    by_vol = order_v[:n_vol]
    by_alpha = [j for j in np.argsort(-a48) if ok[j]][:20]
    hot = [j for j in u.vol_top if u.theme[j] == u.hot and ok[j]]
    pool = list(dict.fromkeys(by_vol + by_alpha + hot))
    startpool = list(dict.fromkeys(order_v[:70] + by_alpha + hot))   # waar willekeurige starts uit komen
    print(f"Scenario's: vorm {p.shape.label()} | kandidaten: {len(pool)} (van {n_ok} betaalbare) | zoekset {nA:,}, "
          f"controleset {nB:,}")

    # set A: alleen eindrendementen nodig; blokken van 500 tegelijk in aparte processen
    simpleA, topsA = parallel_setA(p, nA, u)
    print(f"Zoekset klaar ({time.time() - t0:.0f} s).")

    starts = {"meest beweeglijk": by_vol[:N_POS], "beste voorspelling": by_alpha[:N_POS]}
    if len(hot) >= 3:
        starts["hot thema + beweeglijk"] = (hot + [j for j in by_vol if j not in hot])[:N_POS]
    mix = [j for j in by_vol if a48[j] > np.median(a48[ok])]
    starts["beweeglijk met goede voorspelling"] = (mix + by_vol)[:N_POS]
    starts = {k: list(dict.fromkeys(v))[:N_POS] for k, v in starts.items()}
    starts["beste van 300 willekeurige"] = None            # wordt in een eigen proces gekozen
    r2 = np.random.default_rng(8)
    for i in range(N_RAND_STARTS):
        starts[f"willekeurig {i + 1}"] = list(r2.choice(startpool, N_POS, replace=False))
    print(f"Lokaal zoeken vanaf {len(starts)} startpunten, {WORKERS} processen tegelijk:", flush=True)
    found = parallel_search(simpleA, topsA, pool, startpool, starts)
    uniq = {}                                               # verschillende eindportefeuilles
    for lab, (res, val) in found.items():
        key = frozenset(res)
        if key not in uniq:
            uniq[key] = [lab, res, val, 0]
        uniq[key][3] += 1
    ranked = sorted(uniq.values(), key=lambda x: -x[2])
    print(f"\n{len(uniq)} verschillende eindportefeuilles; de beste {min(N_FINAL, len(ranked))} gaan naar de controle:")
    for lab, res, val, cnt in ranked[:N_FINAL]:
        print(f"  {val:7.1f} euro ({cnt}x gevonden, o.a. vanaf '{lab}'): " + ", ".join(names[res]))
    by_v = lambda picks: sorted(picks, key=lambda j: -v48[j])  # zelfde koopvolgorde als de orderlijst
    finalists = {f"gezocht vanaf: {lab} ({cnt}x)": by_v(res) for lab, res, val, cnt in ranked[:N_FINAL]}
    baselines = {"vergelijking: 10 meest beweeglijk": by_vol[:N_POS],
                 "vergelijking: 10 beste voorspelling": by_alpha[:N_POS]}

    # set B: exacte controle met de spelmotor
    cands = {**finalists, **baselines}
    print(f"Controle op een nieuwe set met de spelmotor ({time.time() - t0:.0f} s), {WORKERS} processen tegelijk ...",
          flush=True)
    r3 = np.random.default_rng(11)
    rand_test = [list(r3.choice(startpool, N_POS, replace=False)) for _ in range(N_RAND_TEST)]
    if len(pool) > len(startpool):                       # ook willekeurig uit de hele zoekpool
        rand_test += [list(r3.choice(pool, N_POS, replace=False)) for _ in range(N_RAND_TEST)]
    res, fast_rand, fast_cand = parallel_check(p, nB, u, price0, cands, rand_test)
    rows = []
    for k, d in res.items():
        pr, t3, fin = (np.concatenate(d[x]) for x in ("prize", "t3", "final"))
        se = pr.std() / np.sqrt(len(pr))
        rows.append({"strategie": k, "verwacht prijzengeld": pr.mean(), "+/-": se, "P(top 3 studenten)": t3.mean(),
                     "mediaan rendement": np.median(fin) / START - 1, "p95 rendement": np.quantile(fin, 0.95) / START - 1,
                     "aandelen": ", ".join(names[cands[k]])})
    t = pd.DataFrame(rows).sort_values("verwacht prijzengeld", ascending=False).set_index("strategie")
    OUT.mkdir(parents=True, exist_ok=True)
    tag = ("_zonder_recent" if a.zonder_recent else "") + ("_eigen_uitsluiting" if a.uitsluiten else "") + \
          ("" if a.kandidaten.strip() == "70" else f"_k{a.kandidaten.strip().lower()}")
    t.to_csv(OUT / f"vergelijking{tag}.csv")
    show = t.drop(columns="aandelen").copy()
    show["verwacht prijzengeld"] = show["verwacht prijzengeld"].map(lambda v: f"{v:,.0f} euro")
    show["+/-"] = show["+/-"].map(lambda v: f"{v:,.0f}")
    for c in ("P(top 3 studenten)",):
        show[c] = show[c].map(lambda v: f"{v:.1%}")
    for c in ("mediaan rendement", "p95 rendement"):
        show[c] = show[c].map(lambda v: f"{v:+.0%}")
    pd.set_option("display.width", 220)
    print("\n================ CONTROLESET (nieuwe scenario's en velden, exacte spelregels) ================")
    print(show.to_string())
    best = t.index[0]
    picks = cands[best]
    print(f"\nGekozen: {best}")
    fb = fast_cand[best]
    print(f"\nToets tegen toeval: {N_RAND_TEST} willekeurige portefeuilles op dezelfde controleset "
          f"(snelle berekening, kopen-en-houden); gekozen: {fb:.0f} euro")
    sets = [(f"uit de {len(startpool)} startkandidaten", fast_rand[:N_RAND_TEST])]
    if len(fast_rand) > N_RAND_TEST:
        sets.append((f"uit de hele zoekpool ({len(pool)})", fast_rand[N_RAND_TEST:]))
    for lab_, fr_ in sets:
        q = np.quantile(fr_, [0.5, 0.9, 0.99])
        print(f"  {lab_:<32} mediaan {q[0]:.0f} | p90 {q[1]:.0f} | p99 {q[2]:.0f} | beste {fr_.max():.0f} euro"
              f" -> gekozen beter dan {np.mean(fr_ < fb):.1%}")
    print("  (de beste willekeurige is op deze set uitgekozen en dus te rooskleurig; de gekozen niet)")
    order = sorted(picks, key=lambda j: -v48[j])
    lst = pd.DataFrame({"dag": ["maandag 16 nov"] * 5 + ["dinsdag 17 nov"] * 5,
                        "naam": names[order], "ticker": tick[order], "isin": np.array(p.isins)[order],
                        "bedrag_euro": 4750, "koers_nu_euro": np.round(price0[order], 2),
                        "ca_aantal": np.floor(4750 / price0[order]).astype(int),
                        "jaarvol": np.round(np.sqrt(vol["v_48"].to_numpy()[order] / 48 * 252), 2),
                        "alpha_48": np.round(ret["alpha_48"].to_numpy()[order], 4)})
    lst.to_csv(OUT / f"orders_start{tag}.csv", index=False)
    print("\n================ ORDERLIJST (voorlopig: opnieuw draaien met de data van vrijdag) ================")
    print(lst.to_string(index=False))
    print(f"-> data/strategy/orders_start{tag}.csv   ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
