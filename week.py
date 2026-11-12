#!/usr/bin/env python3
"""
Beursrally 2026 - wekelijkse beslissing vanuit de ECHTE stand (stap 6c).

Gebruik (zondagavond, na de data-pijplijn en stand_inlezen.py)
    python week.py --uitsluiten "Bpost,Sif Holding,Avantium,Alstom"
    python week.py --stand 2026-11-20       # een bepaalde stand (standaard de nieuwste in data/stand)
    python week.py --geen-fa                # niet ingeschreven voor Finance Avenue (geen 500 euro op 4 december)

Wat het doet
    1. Leest de stand: jouw posities en cash, de portefeuilles van de top 5 studenten en top 5 algemeen
       (data/stand/<datum>/portefeuilles.json) en beide klassementen (klassement.csv).
    2. Toekomsten: de scenario's van scen_build.py (van de eerstvolgende beursdag tot 22 januari).
    3. Concurrenten in elke toekomst:
         - de gekopieerde spelers met hun ECHTE posities en cash (kopen en houden, quiz beantwoord);
         - de rest van het veld gesimuleerd (zelfde speltypes als field.py), met huidige waarden die het echte
           klassement volgen: rang 1-20 exact, daaronder geinterpoleerd tot jouw eigen rang.
    4. De rollout-beslissing (zelfde regel als in rollout.py): niets doen, of max. 2 acties (ruil / verkoop /
       koop met cash), alleen als de winst op de ene helft van de toekomsten gekozen en op de andere helft
       bevestigd wordt. Kosten 5 euro + 0,35% per order, hele aandelen, 10%-grens bij aankoop.
    5. Daarnaast: quizgeld bijkopen in de posities die het verst onder 9,5% zitten (vanaf 500 euro).
    Uitvoer: de orders voor maandag (data/stand/<datum>/orders_maandag.csv) en het verwachte prijzengeld.

Niet automatisch: het nieuws. Kijk voor je de orders plaatst of er kapitaalverhogingen, overnamebiedingen of
schorsingen zijn bij de aandelen in de orderlijst (zie LEESMIJ.md).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from field import FIELDS, START, draw_field, load_universe
from game import Rules
from rollout import FIX, TAX, batch_prize, decide
from scen_core import generate, load_params

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STAND = DATA / "stand"
M_INNER = 2000           # toekomsten: helft om te kiezen, helft om te bevestigen
K_SIM = 600              # gesimuleerde concurrenten per groep (studenten / niet-studenten) die nog kunnen winnen
POS = 0.095


# --------------------------------------------------------------------------- stand
def load_stand(datum: str | None):
    dirs = sorted(d for d in STAND.iterdir() if d.is_dir() and (d / "portefeuilles.json").exists()) \
        if STAND.exists() else []
    if not dirs:
        raise SystemExit("Geen stand gevonden: eerst python stand_inlezen.py")
    d = STAND / datum if datum else dirs[-1]
    js = json.loads((d / "portefeuilles.json").read_text(encoding="utf-8"))
    kl = pd.read_csv(d / "klassement.csv") if (d / "klassement.csv").exists() else pd.DataFrame()
    return d, js, kl


def value_curve(top: pd.DataFrame, own_rank: int, own_value: float, n_total: int) -> np.ndarray:
    """Huidige waarde per rang 1..n: rang 1-20 uit het klassement, daaronder lineair in log(rang) tot jouw rang,
    daarna met dezelfde helling verder (ondergrens 60% van de start)."""
    pts = sorted({int(r): float(v) for r, v in zip(top["rang"], top["waarde"])}.items())
    if own_rank > pts[-1][0]:
        pts.append((own_rank, own_value))
    lr = np.log([r for r, _ in pts])
    vv = np.array([v for _, v in pts])
    ranks = np.arange(1, n_total + 1)
    out = np.interp(np.log(ranks), lr, vv)
    if len(pts) >= 2:
        slope = (vv[-1] - vv[-2]) / (lr[-1] - lr[-2])
        beyond = ranks > pts[-1][0]
        out[beyond] = vv[-1] + slope * (np.log(ranks[beyond]) - lr[-1])
    return np.maximum(out, 0.6 * START)


# --------------------------------------------------------------------------- hoofdprogramma
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stand", default=None)
    ap.add_argument("--uitsluiten", default="", help="zelfde lijst als bij strategy.py")
    ap.add_argument("--geen-fa", action="store_true")
    ap.add_argument("--kandidaten", type=int, default=70, help="meest beweeglijke aandelen om te kopen")
    a = ap.parse_args()
    rng = np.random.default_rng(2026)

    d, js, kl = load_stand(a.stand)
    ik = js.get("ik")
    spelers = {s["speler"]: s for s in js["spelers"]}
    if not ik or ik not in spelers:
        raise SystemExit("Je eigen portefeuille ontbreekt in de stand: kopieer 'Mijn Portefeuille' (stand_inlezen.py)")

    p = load_params(DATA / "scenarios" / "params_vandaag.npz")
    isins = list(p.isins)
    pos_of = {s: k for k, s in enumerate(isins)}
    close = pd.read_parquet(DATA / "processed" / "close_eur.parquet")
    price = close.ffill().iloc[-1].reindex(isins).to_numpy(float)
    data_date = close.index[-1].date()
    N = len(isins)
    u = load_universe(isins, price)
    vol = pd.read_parquet(DATA / "risk" / "vol_vandaag.parquet").set_index("isin").reindex(isins)
    names = vol["naam"].fillna(pd.Series(isins, index=vol.index)).to_numpy()

    # ---------------------------------------------------------------- resterende quiz en Finance Avenue
    rules = Rules()
    days = p.dates
    quiz_left = 100.0 * (~days.isin(pd.to_datetime(list(rules.no_quiz)))).sum()
    fa_left = 0.0 if a.geen_fa or days[0] > pd.Timestamp(rules.fa_credit_date) else rules.fa_bonus
    print(f"Stand {d.name} | koersen tot {data_date} | toekomsten {days[0].date()} - {days[-1].date()} "
          f"({len(days)} beursdagen) | nog quizgeld {quiz_left:,.0f} + Finance Avenue {fa_left:,.0f} EUR")

    # ---------------------------------------------------------------- toekomsten (rest van de competitie)
    e = np.exp(next(generate(p, M_INNER, seed=777, chunk=M_INNER))[:, -1, :].astype(np.float64))   # (M, N)

    def holdings(s):
        """units (N,), waarde buiten onze data (constant gehouden), lijst onbekende posities."""
        un = np.zeros(N)
        out_val, unknown = 0.0, []
        for pos in s["posities"]:
            k = pos_of.get(pos["isin"])
            if k is None or not np.isfinite(price[k]):
                out_val += pos["waarde"]
                unknown.append(pos["naam"])
            else:
                un[k] += pos["aantal"]
        return un, out_val, unknown

    # ---------------------------------------------------------------- wij
    me = spelers[ik]
    units, out_me, unk_me = holdings(me)
    cash = float(me["cash"])
    total = cash + float((units * np.nan_to_num(price)).sum()) + out_me
    if unk_me:
        print("LET OP: eigen posities niet in onze data (constant gehouden): " + ", ".join(unk_me))
    print(f"\nJIJ ({ik}): {total:,.2f} EUR (pagina: {me['totaal']:,.2f}) | cash {cash:,.2f} EUR")
    if abs(total / me["totaal"] - 1) > 0.05:
        print("LET OP: je waarde volgens onze koersen wijkt meer dan 5% af van de pagina: data bijwerken?")

    # ---------------------------------------------------------------- echte concurrenten
    stud_names = set(kl.loc[kl["soort"] == "student", "speler"]) if len(kl) else set()
    real_fut, real_stud, real_names = [], [], []
    for sp, s in spelers.items():
        if sp == ik:
            continue
        un, outv, unk = holdings(s)
        now_val = float(s["cash"]) + float((un * np.nan_to_num(price)).sum()) + outv
        fut = float(s["cash"]) + outv + (e * (un * np.nan_to_num(price))[None]).sum(1) + quiz_left + fa_left
        real_fut.append(fut)
        real_stud.append(sp in stud_names)
        real_names.append(sp)
        extra = f" | niet in data: {', '.join(unk)}" if unk else ""
        if abs(now_val / s["totaal"] - 1) > 0.05:
            extra += f" | LET OP: pagina zegt {s['totaal']:,.2f} EUR (koersen of aantallen kloppen niet?)"
        print(f"  concurrent {sp:<24} {now_val:>11,.2f} EUR {'student ' if sp in stud_names else 'algemeen'}{extra}")
    R = len(real_fut)
    real_fut = np.array(real_fut).T if R else np.zeros((M_INNER, 0))                 # (M, R)
    real_stud = np.array(real_stud, bool)

    # ---------------------------------------------------------------- gesimuleerd veld, geschaald op het klassement
    def pool(soort):
        sub = kl[kl["soort"] == soort] if len(kl) else pd.DataFrame()
        if sub.empty:
            return None
        n_tot = int(sub["deelnemers"].iloc[0])
        own_r = int(sub["eigen_rang"].iloc[0])
        return sub, n_tot, own_r

    pg, ps = pool("algemeen"), pool("student")
    if pg is None or ps is None:
        raise SystemExit("Beide klassementen nodig (studenten en algemeen): kopieer ze met stand_inlezen.py")
    sub_g, n_all, own_rg = pg
    sub_s, n_stu, own_rs = ps
    f_s = n_stu / n_all
    curve_s = value_curve(sub_s, own_rs, total, n_stu)
    curve_g = value_curve(sub_g, own_rg, total, n_all)
    # niet-studenten: rang k onder de niet-studenten ~ algemene rang k / (1 - f_s); de top 20 niet-studenten exact
    known_ns = sub_g[~sub_g["speler"].isin(stud_names)].sort_values("rang")
    n_ext = K_SIM + 20
    v_ns = curve_g[np.minimum((np.arange(1, n_ext + 1) / (1 - f_s)).astype(int), n_all - 1)]
    v_ns[:len(known_ns)] = known_ns["waarde"].to_numpy()[:n_ext]
    v_st = curve_s[:n_ext].copy()
    # spelers die we echt kennen (gekopieerd) uit de gesimuleerde rangen halen
    real_set = set(real_names)
    drop_s = sub_s[sub_s["speler"].isin(real_set)]["rang"].to_numpy() - 1
    drop_g = [k for k, sp in enumerate(known_ns["speler"]) if sp in real_set]
    v_st = np.delete(v_st, drop_s[drop_s < n_ext])[:K_SIM]
    v_ns = np.delete(v_ns, drop_g)[:K_SIM]
    print(f"\nVeld: {n_all:,} deelnemers ({n_stu:,} studenten). Jij: rang {own_rg:,} algemeen, {own_rs:,} studenten.")
    print(f"  3e plaats nu: studenten {sub_s.sort_values('rang')['waarde'].iloc[2]:,.0f} EUR, "
          f"algemeen {sub_g.sort_values('rang')['waarde'].iloc[2]:,.0f} EUR")

    # per veldtype een groep toekomsten; elke gesimuleerde speler krijgt een portefeuille uit dat veld
    fidx = np.arange(M_INNER) % len(FIELDS)
    gv = np.zeros((M_INNER, 8)); gid = np.zeros((M_INNER, 8), int)
    sv = np.zeros((M_INNER, 8)); sid = np.zeros((M_INNER, 8), int)
    for f_i, cfg in enumerate(FIELDS):
        g = np.where(fidx == f_i)[0]
        fd = draw_field(dataclasses.replace(cfg, n_total=max(n_all, 4 * K_SIM), student_share=max(f_s, 0.1)), u, rng)
        npk = (fd.picks >= 0).sum(1)
        sims = []
        for stud, vals in ((True, v_st), (False, v_ns)):
            cand = np.where((fd.student == stud) & (npk > 0))[0]          # hoge rangen: alleen wie belegd is
            pick = rng.choice(cand, len(vals), replace=len(cand) < len(vals))
            sims.append((pick, vals, stud))
        fut_all, stud_all, id_all = [real_fut[g]], [real_stud], [np.arange(R)]
        for pick, vals, stud in sims:
            W = np.zeros((N, len(pick)), np.float32)
            pk = fd.picks[pick]
            r_, c_ = np.nonzero(pk >= 0)
            per_pos = 0.95 * vals / np.maximum(npk[pick], 1)                # euro per positie, nu
            np.add.at(W, (pk[r_, c_], r_), per_pos[r_].astype(np.float32))
            fut = vals[None] + (e[g] - 1.0).astype(np.float32) @ W \
                + (fd.quiz_p[pick] * quiz_left)[None] + (fd.fa_p[pick] * fa_left)[None]
            fut_all.append(fut)
            stud_all.append(np.full(len(pick), stud))
            id_all.append(1000 + f_i * 10_000 + (0 if stud else 5000) + np.arange(len(pick)))
        F = np.concatenate(fut_all, 1)
        S = np.concatenate(stud_all)
        I = np.concatenate(id_all)
        o = np.argsort(-F, axis=1)[:, :8]
        gv[g], gid[g] = np.take_along_axis(F, o, 1), I[o]
        Fs = np.where(S[None], F, -np.inf)
        o = np.argsort(-Fs, axis=1)[:, :8]
        sv[g], sid[g] = np.take_along_axis(Fs, o, 1), I[o]
    tops = (gv, gid, sv, sid)

    # ---------------------------------------------------------------- beslissing
    ok = np.isfinite(price) & (price > 0) & ~u.is_etf
    if a.uitsluiten:
        want = [w.strip().lower() for w in a.uitsluiten.split(",") if w.strip()]
        nm = vol["naam"].fillna("").str.lower().to_numpy()
        tk = vol["ticker"].fillna("").str.lower().to_numpy()
        hit = np.array([any(w == t or w in n for w in want) for n, t in zip(nm, tk)])
        ok &= ~hit
    v48 = np.where(ok, vol["v_48"].to_numpy(), -np.inf)
    cand = [int(j) for j in np.argsort(-v48)[:a.kandidaten] if ok[j]]
    cand = list(dict.fromkeys(cand + [int(j) for j in u.vol_top if u.theme[j] == u.hot and ok[j]]))
    px = np.nan_to_num(price, nan=1e12)                                               # geen koers: niet kopen
    total_d = total                                                                   # 10%-grens: huidige waarde
    acts = decide(units, cash, px, total_d, e, tops, cand, None, quiz_left + fa_left + out_me)

    def our_final(un, ca):
        return ca + out_me + quiz_left + fa_left + (e * (un * np.nan_to_num(price))[None]).sum(1)

    base = our_final(units, cash)
    pr0 = batch_prize(base[None], *tops)[0]
    t3_0 = (np.where(sv > base[:, None], 1, 0).sum(1) < 3).mean()
    rows = []
    un2, ca2 = units.copy(), cash
    for i, j, amt in acts:
        if i >= 0:
            proceeds = un2[i] * price[i] * (1 - TAX) - FIX
            rows.append({"actie": "VERKOOP", "naam": names[i], "isin": isins[i], "aantal": int(un2[i]),
                         "ca_bedrag_euro": round(un2[i] * price[i], 2), "koers_nu": round(price[i], 2)})
            ca2 += proceeds
            un2[i] = 0
        if j >= 0:
            q = np.floor(amt / price[j])
            rows.append({"actie": "KOOP", "naam": names[j], "isin": isins[j], "aantal": int(q),
                         "ca_bedrag_euro": round(q * price[j], 2), "koers_nu": round(price[j], 2)})
            ca2 -= q * price[j] * (1 + TAX) + FIX
            un2[j] += q
    # quizgeld bijkopen: posities ver onder 9,5%, vanaf 500 euro, zolang er orders over zijn (max 5 per dag,
    # max 1 aankoop per effect per dag)
    n_orders = len(rows)
    bought = {r["isin"] for r in rows if r["actie"] == "KOOP"}
    val2 = un2 * np.nan_to_num(price)
    tot2 = ca2 + val2.sum() + out_me
    gap = np.where(un2 > 0, POS * tot2 - val2, -np.inf)
    budget = ca2 - 50.0
    for j in np.argsort(-gap):
        if n_orders >= 5 or not np.isfinite(gap[j]) or gap[j] < 500 or isins[j] in bought:
            break
        a_ = min(gap[j], (budget - FIX) / (1 + TAX))
        q = np.floor(a_ / price[j])
        if q * price[j] < 500:
            continue
        rows.append({"actie": "BIJKOPEN (quizgeld)", "naam": names[j], "isin": isins[j], "aantal": int(q),
                     "ca_bedrag_euro": round(q * price[j], 2), "koers_nu": round(price[j], 2)})
        budget -= q * price[j] * (1 + TAX) + FIX
        ca2 -= q * price[j] * (1 + TAX) + FIX
        un2[j] += q
        n_orders += 1

    fin = our_final(un2, ca2)
    pr1 = batch_prize(fin[None], *tops)[0]
    t3_1 = (np.where(sv > fin[:, None], 1, 0).sum(1) < 3).mean()
    se = lambda x: x.std() / np.sqrt(len(x))
    print("\n================ BESLISSING VOOR MAANDAG ================")
    print(f"  niets doen:      verwacht prijzengeld {pr0.mean():7.0f} EUR (+/- {se(pr0):.0f}) | "
          f"P(top 3 studenten) {t3_0:.1%}")
    if rows:
        print(f"  met deze orders: verwacht prijzengeld {pr1.mean():7.0f} EUR (+/- {se(pr1):.0f}) | "
              f"P(top 3 studenten) {t3_1:.1%}")
        print("  (verschil gepaard: dezelfde toekomsten; de rollout-acties zijn op twee helften bevestigd)\n")
        out = pd.DataFrame(rows)
        print(out.to_string(index=False))
        out.to_csv(d / "orders_maandag.csv", index=False)
        print(f"\n-> {d / 'orders_maandag.csv'}")
        print("Verkopen staan eerst: verkopen worden voor aankopen uitgevoerd. Geef aankopen in als BEDRAG als dat kan,")
        print("anders het aantal; kijk de koers maandagochtend na. Eerst het nieuws checken (LEESMIJ.md).")
    else:
        print("\n  -> NIETS DOEN deze week (geen actie is op beide helften van de toekomsten duidelijk beter).")


if __name__ == "__main__":
    main()
