#!/usr/bin/env python3
"""
Beursrally 2026 - lektest van het risicomodel.

Gebruik
    python risk_test.py

Idee (zoals bij clean.py en features.py)
    Verander alle rendementen NA een datum D willekeurig. Een voorspelling of thema-indeling voor een dag t <= D
    mag daardoor NIET veranderen: alles wat op t gebruikt wordt, lag toen al vast.
    Getest: voortschrijdend 60 / EWMA, GARCH, HAR (met embargo) en de thema's (bron van theme_mom_21).

Gevoeligheidscontrole (sabotage)
    HAR zonder embargo (trainingsrijen tot de dag voor de herschatting, hun doel loopt dus de toekomst in).
    Die versie MOET als lek gevonden worden; anders is de test niet scherp genoeg.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import risk_har
from risk_common import load_returns, log
from risk_corr import themes_over_time, two_day_returns
from risk_garch import garch_stock, quarter_starts
from risk_har import BASE, BLOCKS, WalkForward, add_earnings_block, build_panel
from risk_simple import daily_variance


def tamper(lr: pd.DataFrame, D: pd.Timestamp, rng) -> pd.DataFrame:
    x = lr.copy()
    fut = x.index > D
    x.loc[fut] = x.loc[fut].to_numpy() * rng.uniform(0.3, 3.0, size=x.loc[fut].shape)
    return x


def same(a: pd.DataFrame, b: pd.DataFrame, cols) -> tuple[bool, int]:
    """Gelijk op alle gemeenschappelijke rijen? Geeft ook het aantal vergeleken rijen terug: een lege
    vergelijking telt niet als geslaagd (dan is er niets getest)."""
    m = a.merge(b, on=["date", "isin"], suffixes=("_a", "_b"))
    if m.empty or len(a) != len(b):
        return False, len(m)
    return all(np.allclose(m[f"{c}_a"], m[f"{c}_b"], rtol=1e-8, atol=1e-14, equal_nan=True) for c in cols), len(m)


def check(label: str, ok: bool, expect_leak: bool = False) -> bool:
    status = ("LEK GEVONDEN (verwacht)" if not ok else "GEEN LEK GEVONDEN: test niet scherp!") if expect_leak \
        else ("OK" if ok else "LEK")
    print(f"  {label:<62} {status}")
    return (not ok) if expect_leak else ok


def main():
    rng = np.random.default_rng(3)
    lr = load_returns()
    cal = lr.index
    cols = list(lr.columns[rng.choice(lr.shape[1], 40, replace=False)])
    sub = lr[cols]
    # D = derde beursdag van een maand: dan zit een herschatting (1e van de maand) net voor D
    months = [m for m in pd.date_range("2020-03-01", "2025-06-01", freq="MS")]
    picks = [months[5], months[40]]
    Ds = [cal[cal >= m][2] for m in picks]
    allok = True
    print("\n================ LEKTEST RISICOMODEL ================")
    for D in Ds:
        print(f"\nDatum D = {D.date()} (alles na D willekeurig veranderd)")
        sub2 = tamper(sub, D, rng)

        # 1. voortschrijdend en EWMA
        for kind, lam in (("voortschrijdend 60", None), ("EWMA 0,94", 0.94)):
            k = "rolling" if lam is None else "ewma"
            a = daily_variance(sub, k, lam).stack().rename("v").reset_index()
            b = daily_variance(sub2, k, lam).stack().rename("v").reset_index()
            a.columns = b.columns = ["date", "isin", "v"]
            a, b = a.dropna(), b.dropna()
            ok, n = same(a[a["date"] <= D], b[b["date"] <= D], ["v"])
            allok &= check(f"{kind}: t <= D onveranderd ({n:,} vergeleken)", ok)

        # 2. GARCH: 5 aandelen met genoeg historiek voor D (anders is er niets te vergelijken)
        qs = quarter_starts(cal)
        okg, n_tot, changed = True, 0, False
        long_hist = [c for c in lr.columns if (lr[c].loc[:D].notna().sum() >= 700)]
        pick = list(rng.choice(long_hist, 5, replace=False))
        for c in pick:
            s1, s2 = lr[c].dropna(), tamper(lr[[c]], D, rng)[c].dropna()
            f1, _ = garch_stock(c, s1.index.values, s1.values, qs)
            f2, _ = garch_stock(c, s2.index.values, s2.values, qs)
            ok, n = same(f1[f1["date"] <= D], f2[f2["date"] <= D], ["v_1", "v_20"])
            okg &= ok
            n_tot += n
            after1, after2 = f1[f1["date"] > D], f2[f2["date"] > D]
            changed |= not np.allclose(after1["v_1"].to_numpy(), after2["v_1"].to_numpy())
        allok &= check(f"GARCH: t <= D onveranderd ({n_tot:,} vergeleken, 5 aandelen)", okg)
        allok &= check("Gevoeligheid: GARCH-voorspellingen NA D veranderen wel", changed)

        # 3. HAR (40 aandelen, h = 5 en 20, basis + B1 + B3)
        P1 = build_panel(sub)
        P2 = build_panel(sub2)
        use = BASE + BLOCKS["B1 prijsbereik"] + BLOCKS["B3 cijfers"]
        all_cols = BASE + [c for v in BLOCKS.values() for c in v]
        start = cal[cal >= D - pd.Timedelta(days=70)][0]
        for h in (5, 20):
            for P in (P1, P2):
                add_earnings_block(P, h)
            w1, w2 = WalkForward(P1, h, all_cols), WalkForward(P2, h, all_cols)
            end = D + pd.Timedelta(days=1)
            p1, p2 = w1.predict(use, start, end), w2.predict(use, start, end)
            m1 = (P1["date"] >= start) & (P1["date"] <= D)
            m2 = (P2["date"] >= start) & (P2["date"] <= D)
            ok = np.allclose(p1[m1.to_numpy()], p2[m2.to_numpy()], equal_nan=True, rtol=1e-8)
            allok &= check(f"HAR h = {h}: voorspellingen t <= D onveranderd", ok)

        # 3b. sabotage: HAR zonder embargo moet als lek gevonden worden
        old = risk_har.EMBARGO_EXTRA
        risk_har.EMBARGO_EXTRA = -5 - 5          # trainingsrijen tot de dag voor de herschatting
        try:
            h = 5
            for P in (P1, P2):
                add_earnings_block(P, h)
            w1, w2 = WalkForward(P1, h, all_cols), WalkForward(P2, h, all_cols)
            end = D + pd.Timedelta(days=1)
            p1, p2 = w1.predict(use, start, end), w2.predict(use, start, end)
            m1 = (P1["date"] >= start) & (P1["date"] <= D)
            ok = np.allclose(p1[m1.to_numpy()], p2[m1.to_numpy()], equal_nan=True, rtol=1e-8)
            allok &= check("SABOTAGE: HAR zonder embargo", ok, expect_leak=True)
        finally:
            risk_har.EMBARGO_EXTRA = old

        # 4. thema's (bron van theme_mom_21): kwartalen <= D onveranderd
        # (K = aantal thema's is een instelling gekozen op 2019-2021, zoals de HAR-blokken; hier vast)
        r2a = two_day_returns(lr)
        th1, K1, _ = themes_over_time(lr, r2a, K=15, n_remove=2)
        lr2 = tamper(lr, D, rng)
        th2, _, _ = themes_over_time(lr2, two_day_returns(lr2), K=15, n_remove=2)
        q = D.to_period("Q").start_time
        a = th1[th1["kwartaal"] <= q].sort_values(["kwartaal", "isin"])
        b = th2[th2["kwartaal"] <= q].sort_values(["kwartaal", "isin"])
        okt = len(a) > 0 and len(a) == len(b) and (a["thema"].to_numpy() == b["thema"].to_numpy()).all()
        allok &= check(f"Thema's van kwartalen <= {q.date()} onveranderd", okt)
        c1 = th1[th1["kwartaal"] > q + pd.offsets.QuarterBegin()]
        c2 = th2[th2["kwartaal"] > q + pd.offsets.QuarterBegin()]
        changed = len(c1) != len(c2) or (c1.sort_values(["kwartaal", "isin"])["thema"].to_numpy()
                                         != c2.sort_values(["kwartaal", "isin"])["thema"].to_numpy()).any()
        allok &= check("Gevoeligheid: thema's NA D veranderen wel", changed)
    print(f"\nLektest risicomodel: {'GESLAAGD' if allok else 'MISLUKT'}")
    print("=====================================================\n")
    log("Klaar.")


if __name__ == "__main__":
    main()
