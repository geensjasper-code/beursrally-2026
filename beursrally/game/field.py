#!/usr/bin/env python3
"""
Beursrally 2026 - concurrenten (stap 5): een gesimuleerd veld van spelers in dezelfde scenario's, en het
prijzengeld volgens het reglement.

Gebruik
    python br.py field test       # test van het prijzengeld
    python br.py field            # drempels per veld (wat heb je nodig voor de top 3?) + eenvoudige strategieen

Speltypes (elk met eigen portefeuillekeuze en quizscore; posities van 9,5%, kopen en houden)
    inactief     blijft in cash (of 1-2 posities); 40% vult de quiz trouw in (cash + quiz = ca. +9%)
    voorzichtig  3-8 ETF's en grote bekende aandelen
    casual       5-12 populaire namen (groot, Belgisch vaker gekozen)
    momentum     8-12 uit de 40 grootste stijgers van de laatste 6 maanden
    gokker       10 uit de 50 meest beweeglijke; vaak geconcentreerd in een thema
    Kosten: 5 euro + 0,35% per aankoop. Quiz: 100 euro per goed antwoord (46 vragen), Finance Avenue 500 euro.
    Per blok scenario's wordt een NIEUW veld getrokken: de strategie mag niet afgestemd raken op een toevallig veld.

Velden (32.000 deelnemers, 15% studenten; onzekerheid over wie meedoet: we kiezen een strategie die in alle
velden goed scoort)
    basis, minder actief, meer actief, veel gokkers, een hot thema, momentumjagers

Prijzengeld (reglement): algemeen 10.000 / 2.500 / 1.500, studenten 2.500 / 1.500 / 1.000. Een prijs per
persoon (de hoogste); wie in beide klassementen prijs heeft, laat de lagere aan de volgende. Wij zijn student.

Beperking: concurrenten handelen niet tijdens de wedstrijd (kopen en houden). Wie actief een winnaar najaagt,
zit gemiddeld niet hoger; de spreiding van hun uitkomst zit wel benaderd in de gokkers.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field as dfield
from pathlib import Path

import numpy as np
import pandas as pd

from beursrally.paths import CONFIG, ROOT as HERE   # projectmap (data/ en config/ staan daar)
DATA = HERE / "data"
START = 50_000.0
POS = 0.095
QUIZ_N = 46
PRIZES = [("G", 1, 10_000.0), ("G", 2, 2_500.0), ("S", 1, 2_500.0), ("G", 3, 1_500.0), ("S", 2, 1_500.0),
          ("S", 3, 1_000.0)]                        # van hoog naar laag: zo krijgt ieder zijn hoogste prijs
TYPES = ["inactief", "voorzichtig", "casual", "momentum", "gokker"]
QUIZ_P = {"inactief": 0.30, "voorzichtig": 0.70, "casual": 0.60, "momentum": 0.90, "gokker": 0.90}
QUIZ_SERIOUS = 0.40              # aandeel inactieve spelers dat de quiz wel trouw invult (cash + quiz: ~+9%)
FA_P = {"inactief": 0.02, "voorzichtig": 0.05, "casual": 0.05, "momentum": 0.15, "gokker": 0.15}


@dataclass
class FieldCfg:
    name: str
    n_total: int = 32_000
    student_share: float = 0.15
    mix_general: dict = dfield(default_factory=lambda: {"inactief": .20, "voorzichtig": .20, "casual": .30,
                                                        "momentum": .15, "gokker": .15})
    mix_students: dict = dfield(default_factory=lambda: {"inactief": .20, "voorzichtig": .10, "casual": .30,
                                                         "momentum": .15, "gokker": .25})
    theme_conc: float = 0.5          # kans dat een gokker zich op een thema concentreert
    hot_theme: bool = False          # alle thema-gokkers kiezen het thema met het sterkste momentum
    quiz_boost: float = 0.0          # extra kans op een juist quizantwoord (actiever veld)


FIELDS = [
    FieldCfg("basis"),
    FieldCfg("minder actief",
             mix_general={"inactief": .30, "voorzichtig": .25, "casual": .30, "momentum": .08, "gokker": .07},
             mix_students={"inactief": .30, "voorzichtig": .15, "casual": .35, "momentum": .08, "gokker": .12}),
    FieldCfg("meer actief", quiz_boost=0.05,
             mix_general={"inactief": .10, "voorzichtig": .15, "casual": .30, "momentum": .22, "gokker": .23},
             mix_students={"inactief": .08, "voorzichtig": .07, "casual": .30, "momentum": .22, "gokker": .33}),
    FieldCfg("veel gokkers",
             mix_general={"inactief": .15, "voorzichtig": .15, "casual": .25, "momentum": .15, "gokker": .30},
             mix_students={"inactief": .10, "voorzichtig": .05, "casual": .25, "momentum": .20, "gokker": .40}),
    FieldCfg("een hot thema", theme_conc=0.8, hot_theme=True),
    FieldCfg("momentumjagers",
             mix_general={"inactief": .15, "voorzichtig": .15, "casual": .25, "momentum": .30, "gokker": .15},
             mix_students={"inactief": .12, "voorzichtig": .08, "casual": .25, "momentum": .35, "gokker": .20}),
]


# --------------------------------------------------------------------------- het aanbod zoals spelers het zien
@dataclass
class Universe:
    isins: list
    is_etf: np.ndarray
    popularity: np.ndarray           # kans om gekozen te worden door een casual speler
    big: np.ndarray                  # indices: grote bekende aandelen
    mom_top: np.ndarray              # indices: grootste stijgers 6 maanden
    vol_top: np.ndarray              # indices: meest beweeglijk
    theme: np.ndarray                # themanummer per effect
    hot: int                         # thema met het sterkste momentum onder de beweeglijke aandelen
    price0: np.ndarray


def load_universe(isins: list[str], price0: np.ndarray) -> Universe:
    tk = pd.read_csv(DATA / "meta" / "tickers.csv", dtype=str).set_index("isin")
    vol = pd.read_parquet(DATA / "risk" / "vol_vandaag.parquet").set_index("isin").reindex(isins)
    F = pd.read_parquet(DATA / "features" / "features.parquet", columns=["date", "isin", "log_mcap", "mom_6_1"])
    F = F[F["date"] == F["date"].max()].set_index("isin").reindex(isins)
    th = pd.read_parquet(DATA / "risk" / "themas_vandaag.parquet").set_index("isin")["thema"].reindex(isins)
    ok = np.isfinite(price0) & (price0 > 0) & (price0 < POS * START)      # te duur voor een positie: niet kiesbaar
    is_etf = tk["type"].reindex(isins).eq("etf").to_numpy()
    mcap = F["log_mcap"].fillna(F["log_mcap"].median()).to_numpy()
    z = (mcap - mcap.mean()) / (mcap.std() or 1)
    belgian = np.array([i.startswith("BE") for i in isins])
    pop = np.exp(1.2 * z) * np.where(belgian, 2.5, 1.0) * ok
    pop = pop / pop.sum()
    stocks = ok & ~is_etf
    big = np.where(stocks & (z > np.quantile(z[stocks], 0.85)))[0]
    mom = F["mom_6_1"].to_numpy()
    mom_top = np.argsort(-np.where(stocks & np.isfinite(mom), mom, -np.inf))[:40]
    jv = vol["jaarvol_48"].to_numpy()
    vol_top = np.argsort(-np.where(stocks & np.isfinite(jv), jv, -np.inf))[:50]
    theme = th.fillna(-1).astype(int).to_numpy()
    vt = pd.DataFrame({"t": theme[vol_top], "m": np.nan_to_num(mom[vol_top])})
    vt = vt[vt["t"] >= 0].groupby("t")["m"].agg(["mean", "size"])
    vt = vt[vt["size"] >= 3]
    hot = int(vt["mean"].idxmax()) if len(vt) else -1
    return Universe(list(isins), is_etf, pop, big, mom_top, vol_top, theme, hot, price0)


# --------------------------------------------------------------------------- een veld trekken
@dataclass
class Field:
    picks: np.ndarray                # (P, 12) indices, -1 = geen
    ptype: np.ndarray                # (P,) typenummer
    student: np.ndarray              # (P,) bool
    quiz_p: np.ndarray               # (P,)
    fa_p: np.ndarray                 # (P,)


def _gumbel_topk(rng, n: int, n_items: int, logw: np.ndarray, k: np.ndarray, kmax: int) -> np.ndarray:
    """Per rij k[i] verschillende items trekken zonder teruglegging, kans ~ exp(logw) (Gumbel-truc).
    logw: (n_items,) of (n, n_items), -inf = niet kiesbaar. Geeft (n, kmax) met -1 waar niets."""
    keys = np.broadcast_to(logw, (n, n_items)) + rng.gumbel(size=(n, n_items))
    o = np.argsort(-keys, axis=1)[:, :kmax]
    ok = np.take_along_axis(keys, o, 1) > -np.inf
    o = np.where(ok & (np.arange(kmax)[None] < k[:, None]), o, -1)
    return o


def draw_field(cfg: FieldCfg, u: Universe, rng: np.random.Generator) -> Field:
    n_s = int(round(cfg.n_total * cfg.student_share))
    n_g = cfg.n_total - n_s
    types = np.concatenate([rng.choice(len(TYPES), n_g, p=[cfg.mix_general[t] for t in TYPES]),
                            rng.choice(len(TYPES), n_s, p=[cfg.mix_students[t] for t in TYPES])])
    student = np.r_[np.zeros(n_g, bool), np.ones(n_s, bool)]
    P, Nu = len(types), len(u.popularity)
    picks = np.full((P, 12), -1, int)
    log_pop = np.log(np.where(u.popularity > 0, u.popularity, np.nan))
    log_pop = np.where(np.isfinite(log_pop), log_pop, -np.inf)

    def logw_on(idx, w=None):
        lw = np.full(Nu, -np.inf)
        lw[idx] = 0.0 if w is None else np.log(w)
        return lw

    etfs = np.where(u.is_etf & (u.popularity > 0))[0]
    safe = np.r_[etfs, u.big]
    for t, name in enumerate(TYPES):
        rows = np.where(types == t)[0]
        n = len(rows)
        if n == 0:
            continue
        if name == "inactief":
            k = rng.choice([0, 0, 0, 1, 2], n)
            picks[rows, :2] = _gumbel_topk(rng, n, Nu, log_pop, k, 2)
        elif name == "voorzichtig":
            k = rng.integers(3, 9, n)
            picks[rows, :8] = _gumbel_topk(rng, n, Nu, logw_on(safe), k, 8)
        elif name == "casual":
            k = rng.integers(5, 13, n)
            picks[rows, :12] = _gumbel_topk(rng, n, Nu, log_pop, k, 12)
        elif name == "momentum":
            k = rng.integers(8, 13, n)
            picks[rows, :12] = _gumbel_topk(rng, n, Nu, logw_on(u.mom_top, np.linspace(2, 1, len(u.mom_top))), k, 12)
        else:                                                      # gokker: 10 posities
            vol_w = logw_on(u.vol_top, np.linspace(2, 1, len(u.vol_top)))
            themes = [th for th in np.unique(u.theme[u.vol_top]) if th >= 0
                      and (u.theme[u.vol_top] == th).sum() >= 3]
            in_theme = rng.random(n) < cfg.theme_conc if themes else np.zeros(n, bool)
            if cfg.hot_theme and u.hot in themes:
                th_choice = np.full(n, u.hot)
            else:
                th_choice = rng.choice(themes, n) if themes else np.full(n, -1)
            m = np.where(in_theme, rng.integers(4, 9, n), 0)
            # eerst m aandelen uit het eigen thema (onder de beweeglijke), dan aanvullen tot 10
            lw_theme = np.where((u.theme[None, :] == th_choice[:, None]) & np.isfinite(vol_w)[None], 0.0, -np.inf)
            first = _gumbel_topk(rng, n, Nu, lw_theme, m, 8)
            lw_rest = np.broadcast_to(vol_w, (n, Nu)).copy()
            r_i, c_i = np.nonzero(first >= 0)
            lw_rest[r_i, first[r_i, c_i]] = -np.inf
            got = (first >= 0).sum(1)
            rest = _gumbel_topk(rng, n, Nu, lw_rest, 10 - got, 10)
            comb = np.full((n, 12), -1, int)
            for i in range(10):                                    # samenvoegen zonder gaten
                take_first = i < got
                src = np.where(take_first, first[np.arange(n), np.minimum(i, 7)],
                               rest[np.arange(n), np.clip(i - got, 0, 9)])
                comb[:, i] = src
            picks[rows] = comb
    names = np.array(TYPES)[types]
    # quizscore per speler: spreiding rond het gemiddelde van zijn type (Beta-verdeling, sterkte 8);
    # bij inactieve spelers twee groepen: trouw (~97% juist, een derde foutloos) of nauwelijks (~10%)
    mean = np.clip(np.array([QUIZ_P[n] for n in names]) + cfg.quiz_boost, 0.02, 0.98)
    qp = rng.beta(mean * 8, (1 - mean) * 8)
    inact = names == "inactief"
    serious = rng.random(P) < QUIZ_SERIOUS
    qp[inact] = np.where(serious[inact], rng.beta(30, 1, inact.sum()), rng.beta(1, 9, inact.sum()))
    fa = np.array([FA_P[n] for n in names])
    fa[inact & serious] = 0.30                                 # trouwe quizspelers gaan vaker naar Finance Avenue
    return Field(picks, types, student, qp, fa)


def field_values(f: Field, simple: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """simple: (S, N) gewoon rendement tot 22 januari. Geeft (S, P) eindwaarde in euro."""
    S = simple.shape[0]
    has = f.picks >= 0
    k = has.sum(1)
    pos = POS * START
    costs = k * (5.0 + 0.0035 * pos)
    # winst = simple @ W met W (N, P) de posities per speler (ijl, als dichte matrix: snel en zuinig)
    W = np.zeros((simple.shape[1], len(k)), np.float32)
    rows, cols = np.nonzero(has)
    np.add.at(W, (f.picks[rows, cols], rows), 1.0)
    gain = (simple.astype(np.float32) @ W).astype(np.float64) * pos   # (S, P)
    quiz = rng.binomial(QUIZ_N, f.quiz_p, size=(S, len(k))) * 100.0
    fa = (rng.random((S, len(k))) < f.fa_p) * 500.0
    return START + gain - costs[None] + quiz + fa


# --------------------------------------------------------------------------- prijzengeld
def our_prize(ours: np.ndarray, vals: np.ndarray, student: np.ndarray, top: int = 6) -> np.ndarray:
    """Prijzengeld voor ons (student) per scenario, met de regel 'een prijs per persoon, de hoogste'.
    ours (S,), vals (S, P) concurrenten, student (P,)."""
    S = len(ours)
    allv = np.c_[vals, ours]                                        # wij = laatste kolom (id P)
    me = vals.shape[1]
    stud = np.r_[student, True]
    def top_ids(v):
        k = min(top, v.shape[1] - 1)
        part = np.argpartition(-v, k, axis=1)[:, :k + 1] if k < v.shape[1] - 1 else np.argsort(-v, axis=1)
        o = np.argsort(-np.take_along_axis(v, part, 1), axis=1)
        return np.take_along_axis(part, o, 1)
    g_ids = top_ids(allv)
    s_ids = top_ids(np.where(stud[None], allv, -np.inf))
    awarded = np.full((S, len(PRIZES)), -1)
    prize = np.zeros(S)
    rows = np.arange(S)
    for j, (kind, _, amount) in enumerate(PRIZES):
        lst = g_ids if kind == "G" else s_ids
        taken = (lst[:, :, None] == awarded[:, None, :]).any(2)     # (S, top) heeft al een prijs
        first = np.argmax(~taken, axis=1)
        who = lst[rows, first]
        awarded[:, j] = who
        prize += np.where(who == me, amount, 0.0)
    return prize


def thresholds(vals: np.ndarray, student: np.ndarray) -> dict:
    g = -np.sort(-np.partition(vals, vals.shape[1] - 3, axis=1)[:, -3:], axis=1)
    sv = vals[:, student]
    s = -np.sort(-np.partition(sv, sv.shape[1] - 3, axis=1)[:, -3:], axis=1)
    return {"alg_1": g[:, 0], "alg_3": g[:, 2], "stud_1": s[:, 0], "stud_3": s[:, 2]}


def selftest() -> bool:
    ok_all = True

    def check(name, ok):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"  {'OK  ' if ok else 'FOUT'} {name}")

    print("Test prijzengeld (een prijs per persoon, de hoogste; wij zijn student)")
    stud = np.array([True, False, True, False, True])          # spelers A..E
    def run(vals, ours):
        return our_prize(np.array([ours], float), np.array([vals], float), stud)[0]
    check("wij de hoogste van iedereen -> 10.000", run([90, 80, 70, 60, 50], 100) == 10_000)
    check("een student wint algemeen (A), wij 2e student -> studentenprijs 1 schuift door: 2.500",
          run([100, 50, 40, 30, 20], 60) == 2_500)
    check("algemeen: A(student) 1e, B 2e, C(student) 3e, wij 4e -> C neemt studenten-1 (2.500), algemeen-3e "
          "(1.500) schuift naar ons", run([100, 90, 80, 10, 5], 70) == 1_500)
    check("wij 4e student, maar student A neemt algemeen-1: studentenprijzen schuiven door -> 3e studentenprijs 1.000",
          run([100, 90, 80, 70, 60], 50) == 1_000)
    stud2 = np.array([False, False, False, True, True, True])
    check("3 niet-studenten bovenaan, 3 studenten voor ons -> niets",
          our_prize(np.array([10.0]), np.array([[100, 90, 80, 70, 60, 50]], float), stud2)[0] == 0)
    check("wij 2e algemeen achter een niet-student, 1e student -> 2.500 (een prijs, niet twee)",
          run([100, 50, 40, 30, 20], 90) == 2_500)
    print("ZELFTEST GESLAAGD" if ok_all else "ZELFTEST MISLUKT")
    return ok_all


# --------------------------------------------------------------------------- rapport
def main(n: int = 4000):
    from beursrally.game.game import BuyHold, Scripted, play, Rules
    from beursrally.scenarios.scen_core import generate, load_params
    f = DATA / "scenarios" / "params_vandaag.npz"
    if not f.exists():
        sys.exit("Eerst:  python br.py scen_build")
    p = load_params(f)
    close = pd.read_parquet(DATA / "processed" / "close_eur.parquet")
    price0 = close.ffill().iloc[-1].reindex(p.isins).to_numpy(float)
    u = load_universe(p.isins, price0)
    vol = pd.read_parquet(DATA / "risk" / "vol_vandaag.parquet").set_index("isin").reindex(p.isins)
    names = vol["naam"].fillna("").to_numpy()
    ok = np.isfinite(price0) & (price0 > 0) & (price0 < POS * START)
    top_vol = np.argsort(-np.where(ok, vol["v_48"].to_numpy(), -1))[:10]
    print(f"Concurrenten op {n:,} scenario's ({p.dates[0].date()} - {p.dates[-1].date()}), vorm: {p.shape.label()}")
    print(f"Hot thema (gokkers in het veld 'een hot thema'): " +
          ", ".join(names[u.vol_top[u.theme[u.vol_top] == u.hot]][:10]))
    ours = {"10 meest beweeglijke + bijkopen": BuyHold(top_vol, quiz="bijkopen"),
            "10 willekeurige": BuyHold(np.random.default_rng(3).choice(np.where(ok)[0], 10, replace=False),
                                       quiz="bijkopen"),
            "niets doen (quiz)": Scripted({})}
    out = {cfg.name: {"thr": [], **{k: [] for k in ours}} for cfg in FIELDS}
    t0 = time.time()
    rng = np.random.default_rng(11)
    for ci, cum in enumerate(generate(p, n, seed=5, chunk=1000)):
        cum = cum.astype(np.float64)
        simple = np.expm1(cum[:, -1, :])
        our_final = {k: play(pol, cum, price0, p.dates, rules=Rules(), seed=ci).final for k, pol in ours.items()}
        for cfg in FIELDS:
            fd = draw_field(cfg, u, rng)
            vals = field_values(fd, simple, rng)
            out[cfg.name]["thr"].append(thresholds(vals, fd.student))
            for k in ours:
                prize = our_prize(our_final[k], vals, fd.student)
                s_rank = (np.where(fd.student[None], vals, -np.inf) > our_final[k][:, None]).sum(1) + 1
                out[cfg.name][k].append((prize, s_rank <= 3))
    print(f"({time.time() - t0:.0f} s)\n")
    pct = lambda x: f"{x / START - 1:+.0%}"
    print("================ WAT HEB JE NODIG? (rendement incl. quiz, verdeling over scenario's) ================")
    rows = []
    for cfg in FIELDS:
        thr = {k: np.concatenate([d[k] for d in out[cfg.name]["thr"]]) for k in ("alg_1", "alg_3", "stud_1", "stud_3")}
        for k, lab in (("alg_1", "algemeen 1e"), ("alg_3", "algemeen 3e"), ("stud_1", "studenten 1e"),
                       ("stud_3", "studenten 3e")):
            q = np.quantile(thr[k], [0.1, 0.5, 0.9])
            rows.append({"veld": cfg.name, "plaats": lab, "p10": pct(q[0]), "mediaan": pct(q[1]), "p90": pct(q[2])})
    print(pd.DataFrame(rows).set_index(["veld", "plaats"]).to_string())
    print("Ter vergelijking: winnaars van eerdere edities en gelijkaardige wedstrijden haalden ruwweg +30% tot +66%.")
    print("\n================ EENVOUDIGE STRATEGIEEN (wij als student) ================")
    rows = []
    for cfg in FIELDS:
        for k in ours:
            pr = np.concatenate([a for a, _ in out[cfg.name][k]])
            t3 = np.concatenate([b for _, b in out[cfg.name][k]])
            rows.append({"veld": cfg.name, "strategie": k, "P(top 3 studenten)": f"{t3.mean():.1%}",
                         "P(prijs)": f"{(pr > 0).mean():.1%}", "verwacht prijzengeld": f"{pr.mean():,.0f} euro"})
    print(pd.DataFrame(rows).set_index(["veld", "strategie"]).to_string())
    print("\nStap 6 (strategy.py) zoekt de startportefeuille met het hoogste verwachte prijzengeld over alle velden.")


def cli():
    """Opdrachtregel: ook de zelftest/diagnose (br.py roept deze functie op)."""
    if sys.argv[1:] == ["test"]:
        sys.exit(0 if selftest() else 1)
    main()


if __name__ == "__main__":
    cli()
