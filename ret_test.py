#!/usr/bin/env python3
"""
Beursrally 2026 - lektest van het rendementsmodel.

Gebruik
    python ret_test.py               # lektest
    python ret_test.py diagnose      # bij een LEK: stap voor stap zoeken waar het verschil ontstaat

Idee
    Verander de toekomst na een datum D: alle kenmerken van dagen na D, en elk doel y_h waarvan het venster
    (dag t+1 tot t+h) voorbij D loopt. Voorspellingen voor dagen t <= D mogen daardoor NIET veranderen.
    Getest: composite (schaal geschat op het verleden), elastic net en boosting, h = 5 en 20.

Eerst een herhaalbaarheidscontrole: hetzelfde model twee keer op IDENTIEKE data moet exact hetzelfde geven.
    Alles draait op een rekendraad, zodat afrondingsverschillen tussen draden geen rol spelen.

Gevoeligheidscontrole (sabotage)
    Zonder embargo (trainen tot de dag voor de herschatting): het doel van de laatste trainingsrijen loopt
    dan voorbij D. Dat MOET als lek gevonden worden.
"""
from __future__ import annotations

import os

# Een enkele rekendraad: met veel draden kan de volgorde van optellingen per run verschillen (afronding in de
# 15e decimaal), en dan kan een bijna-gelijke keuze (straf, aantal bomen) omslaan. Dat is geen lek, maar maakt
# de lektest onbetrouwbaar. Moet gezet worden VOOR numpy/sklearn/lightgbm geladen worden.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BR_RET_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("BR_RET_ROWS", "60000")
os.environ.setdefault("BR_RET_ROUNDS", "60")

import numpy as np
import pandas as pd

import ret_models
from ret_models import FEAT, load_panel, log, walk_forward

HS = [5, 20]


def tamper(F: pd.DataFrame, T: pd.DataFrame, D: pd.Timestamp, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    F2, T2 = F.copy(), T.copy()
    num = [c for c in F.columns if c not in ("date", "isin")]
    fut = (F2["date"] > D).to_numpy()
    # zelfde datatype houden: anders veranderen ook de ONgewijzigde waarden een heel klein beetje (afronding)
    for c in num:
        v = F2[c].to_numpy().copy()
        v[fut] = (v[fut] * rng.uniform(-2, 3, fut.sum())).astype(v.dtype)
        F2[c] = v
    cal = np.array(sorted(T2["date"].unique()))
    kD = np.searchsorted(cal, np.datetime64(D), side="right") - 1
    for c in [c for c in T2.columns if c.startswith("y_")]:
        h = int(c.split("_")[1])
        first_bad = cal[max(0, kD - h + 1)]          # vanaf deze dag loopt het venster voorbij D
        bad = (T2["date"] >= first_bad).to_numpy()
        v = T2[c].to_numpy().copy()
        v[bad] = rng.normal(0, 0.2, bad.sum()).astype(v.dtype)
        T2[c] = v
    # zelfcontrole: wat NIET gewijzigd mocht worden, moet bit voor bit gelijk zijn
    for src, dst, mask_of in ((F, F2, lambda c: fut), (T, T2, None)):
        for c in [c for c in src.columns if c not in ("date", "isin")]:
            if mask_of is not None:
                keep = ~fut
            else:
                h = int(c.split("_")[1])
                keep = (T2["date"] < cal[max(0, kD - h + 1)]).to_numpy()
            a = src[c].to_numpy()[keep].astype("float64")
            b = dst[c].to_numpy()[keep].astype("float64")
            same = (a == b) | (np.isnan(a) & np.isnan(b))
            if not same.all():
                raise RuntimeError(f"Testfout: kolom {c} veranderde ook op ongewijzigde rijen "
                                   f"({int((~same).sum())} rijen; type {src[c].dtype} -> {dst[c].dtype}).")
    return F2, T2


def compare(a: pd.DataFrame, b: pd.DataFrame, D, h) -> tuple[bool, int]:
    m = a.merge(b, on=["date", "isin"], suffixes=("_a", "_b"))
    m = m[m["date"] <= D].dropna(subset=[f"p_{h}_a", f"p_{h}_b"])
    if m.empty:
        return False, 0
    return bool(np.allclose(m[f"p_{h}_a"], m[f"p_{h}_b"], rtol=1e-6, atol=1e-10)), len(m)


def check(label, ok, expect_leak=False):
    status = ("LEK GEVONDEN (verwacht)" if not ok else "GEEN LEK GEVONDEN: test niet scherp!") if expect_leak \
        else ("OK" if ok else "LEK")
    print(f"  {label:<64} {status}")
    return (not ok) if expect_leak else ok


def main():
    rng = np.random.default_rng(11)
    F = pd.read_parquet(FEAT / "features.parquet")
    T = pd.read_parquet(FEAT / "targets.parquet")
    cal = np.array(sorted(F["date"].unique()))
    allok = True
    print("\n================ LEKTEST RENDEMENTSMODEL ================")
    for q in (pd.Timestamp("2021-04-01"), pd.Timestamp("2024-07-01")):
        D = pd.Timestamp(cal[np.searchsorted(cal, np.datetime64(q)) + 2])   # 3 beursdagen na een herschatting
        start = q - pd.Timedelta(days=95)
        end = D + pd.Timedelta(days=1)
        print(f"\nDatum D = {D.date()} (kenmerken na D en doelen die voorbij D lopen: willekeurig veranderd)")
        P1, x1 = load_panel(F, T)
        F2, T2 = tamper(F, T, D, rng)
        P2, x2 = load_panel(F2, T2)
        P1b, _ = load_panel(F.copy(), T.copy())
        for model in ("enet", "boost"):
            a = walk_forward(model, P1, x1, start, end, verbose=False, horizons=HS)
            b = walk_forward(model, P1b, x1, start, end, verbose=False, horizons=HS)
            for h in HS:
                ok, n = compare(a, b, pd.Timestamp("2100-01-01"), h)
                allok &= check(f"herhaalbaar: {model} h = {h} twee keer identiek ({n:,} vergeleken)", ok and n > 0)
        for model in ("composite", "enet", "boost"):
            a = walk_forward(model, P1, x1, start, end, verbose=False, horizons=HS)
            b = walk_forward(model, P2, x2, start, end, verbose=False, horizons=HS)
            for h in HS:
                ok, n = compare(a, b, D, h)
                allok &= check(f"{model} h = {h}: voorspellingen t <= D onveranderd ({n:,} vergeleken)", ok and n > 0)
        old = ret_models.EMBARGO_EXTRA
        ret_models.EMBARGO_EXTRA = -20 - 5        # trainen tot de dag voor de herschatting (h = 20)
        try:
            a = walk_forward("composite", P1, x1, start, end, verbose=False, horizons=[20])
            b = walk_forward("composite", P2, x2, start, end, verbose=False, horizons=[20])
            ok, n = compare(a, b, D, 20)
            allok &= check("SABOTAGE: composite zonder embargo (h = 20)", ok, expect_leak=True)
        finally:
            ret_models.EMBARGO_EXTRA = old
    print(f"\nLektest rendementsmodel: {'GESLAAGD' if allok else 'MISLUKT'}")
    print("=========================================================\n")
    log("Klaar.")


def diagnose():
    """Bouw een falend geval stap voor stap na en meld de EERSTE plek waar origineel en gewijzigd verschillen."""
    rng = np.random.default_rng(11)
    F = pd.read_parquet(FEAT / "features.parquet")
    T = pd.read_parquet(FEAT / "targets.parquet")
    cal_all = np.array(sorted(F["date"].unique()))
    print("\n================ DIAGNOSE LEKTEST ================")
    print(f"features: {F.shape}, dubbele (datum, isin): {int(F.duplicated(['date', 'isin']).sum())}; "
          f"targets: {T.shape}, dubbel: {int(T.duplicated(['date', 'isin']).sum())}")
    print("datatypes features:", dict(F.dtypes.astype(str).value_counts()))
    print("datatypes targets:", dict(T.dtypes.astype(str).value_counts()))
    for q in (pd.Timestamp("2021-04-01"), pd.Timestamp("2024-07-01")):
        D = pd.Timestamp(cal_all[np.searchsorted(cal_all, np.datetime64(q)) + 2])
        F2, T2 = tamper(F, T, D, rng)
        P1, x1 = load_panel(F, T)
        P2, x2 = load_panel(F2, T2)
        print(f"\nD = {D.date()}")
        same_rows = (P1["date"].to_numpy() == P2["date"].to_numpy()).all() and (P1["isin"].to_numpy() == P2["isin"].to_numpy()).all()
        print(f"  zelfde rijvolgorde in het paneel: {same_rows}")
        print(f"  type doelen: origineel {T['y_5'].dtype} / gewijzigd {T2['y_5'].dtype}; "
              f"in het paneel {P1['yd_5'].dtype} / {P2['yd_5'].dtype}")
        cal = ret_models.Calendar(P1["date"])
        start = q - pd.Timedelta(days=95)
        for h in HS:
            y1, y2 = P1[f"yd_{h}"].to_numpy(), P2[f"yd_{h}"].to_numpy()
            refits = pd.date_range(start.to_period("Q").start_time, D + pd.Timedelta(days=1), freq="QS")
            for r0 in refits:
                k_cut = cal.cutoff(r0, h)
                tr = cal.rows_upto(k_cut)
                pr = cal.rows_between(max(r0, start), min(D + pd.Timedelta(days=1), r0 + pd.offsets.QuarterBegin()))
                exact = lambda a, b: ~((a == b) | (np.isnan(a) & np.isnan(b)))     # bit-voor-bit, geen marge
                dy = exact(y1[tr], y2[tr])
                X1, X2 = P1[x1].to_numpy()[tr], P2[x2].to_numpy()[tr]
                dX = exact(X1, X2)
                Xp1, Xp2 = P1[x1].to_numpy()[pr], P2[x2].to_numpy()[pr]
                dXp = exact(Xp1, Xp2)
                last_train = pd.Timestamp(cal.cal[k_cut]).date() if k_cut >= 0 else None
                print(f"  h = {h}, herschatting {r0.date()} (training t/m {last_train}): "
                      f"verschil y {int(dy.sum())}, X-training {int(dX.sum())}, X-voorspeld {int(dXp.sum())}")
                for label, mask, rows, cols in (("y", dy, tr, None), ("X-training", dX, tr, x1), ("X-voorspeld", dXp, pr, x1)):
                    if mask.any():
                        idx = np.argwhere(mask)[0]
                        r = rows[idx[0]]
                        col = cols[idx[1]] if cols is not None and mask.ndim == 2 else f"yd_{h}"
                        print(f"     eerste verschil in {label}: rij {r}, datum {P1.at[r, 'date'].date()}, "
                              f"isin {P1.at[r, 'isin']}, kolom {col}")
                        dates = np.unique(P1["date"].to_numpy()[rows[np.unique(np.argwhere(mask)[:, 0])]])
                        print(f"     datums met verschillen: {len(dates)}, van {pd.Timestamp(dates.min()).date()} "
                              f"tot {pd.Timestamp(dates.max()).date()}")
    print("==================================================\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "diagnose":
        diagnose()
    else:
        main()
