#!/usr/bin/env python3
"""
Beursrally 2026 - risicomodel 3: GARCH(1,1) per aandeel.

Gebruik
    python risk_garch.py            # alle aandelen (eerste keer 15-40 min; hervat automatisch na onderbreking)
    python risk_garch.py --opnieuw  # alles opnieuw schatten
    python risk_garch.py test       # zelftest: vindt de schatter bekende parameters terug?

Model
    s2_{t+1} = omega + alpha * r_t^2 + beta * s2_t
    Langetermijnniveau  V = omega / (1 - alpha - beta);  persistentie  phi = alpha + beta.
    Over h dagen:  v_h = h*V + (s2_{t+1} - V) * (1 - phi^h) / (1 - phi)
    -> voor lange horizonnen trekt de voorspelling terug naar V (dat kan EWMA niet).

Eigen implementatie met scipy (zelfde model en schatter als het pakket arch, dat op
    Python 3.15 onder Windows niet installeert); gecontroleerd tegen arch.

Schatten zonder lek
    Elk kwartaal opnieuw geschat op de laatste 1000 eigen dagen VOOR de eerste dag van dat kwartaal (min. 500).
    Binnen het kwartaal blijven de parameters vast; de variantie wordt dagelijks bijgewerkt met de rendementen
    tot en met dag d. Mislukt de schatting, of is phi >= 0,999 (bijna een random walk), dan EWMA(0,94) voor
    dat kwartaal (gemarkeerd als terugval).
"""
from __future__ import annotations

import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from risk_common import FC_START, HORIZONS, RISK, VCOLS, load_returns, log, own_series, save_forecast

WINDOW, MIN_OBS = 1000, 500
TMP = RISK / "garch_tmp"


def quarter_starts(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    q = pd.date_range(FC_START.to_period("Q").start_time, index.max() + pd.offsets.QuarterBegin(), freq="QS")
    return [d for d in q if d <= index.max()]


def _h_sum(s2_next: np.ndarray, V: float, phi: float) -> dict:
    out = {}
    for h in HORIZONS:
        if abs(1 - phi) < 1e-9:
            out[f"v_{h}"] = h * s2_next
        else:
            out[f"v_{h}"] = h * V + (s2_next - V) * (1 - phi ** h) / (1 - phi)
    return out


def _backcast(x2: np.ndarray) -> float:
    """Startwaarde van de variantie: gewogen gemiddelde van de eerste 75 kwadraten (zoals het pakket arch)."""
    n = min(75, len(x2))
    w = 0.94 ** np.arange(n)
    return float(w @ x2[:n] / w.sum())


def _neg_loglik(theta: np.ndarray, x2: np.ndarray, s0: float) -> float:
    """-log-likelihood (normaal) van GARCH(1,1); de recursie loopt in C via scipy.signal.lfilter."""
    from scipy.signal import lfilter
    om, al, be = theta
    if om <= 0 or al < 0 or be < 0 or al + be >= 0.99999:
        return 1e12
    u = om + al * np.concatenate([[s0], x2[:-1]])          # omega + alpha * r_{t-1}^2
    s2 = lfilter([1.0], [1.0, -be], u, zi=[be * s0])[0]    # s2_t = u_t + beta * s2_{t-1}
    if not np.all(s2 > 0):
        return 1e12
    return 0.5 * float(np.sum(np.log(s2) + x2 / s2))


def fit_garch(r: np.ndarray) -> tuple[float, float, float, float]:
    """Schat GARCH(1,1) met maximale likelihood. Geeft omega, alpha, beta (rendementen als fractie)
    en de variantie van de laatste dag van het venster. Rekent intern in procenten (stabieler)."""
    from scipy.optimize import minimize
    x = 100 * np.asarray(r, float)
    x2 = x ** 2
    s0 = _backcast(x2)
    v = x2.mean()
    starts = [(a, b) for a in (0.03, 0.06, 0.10) for b in (0.80, 0.90, 0.95) if a + b < 0.995]
    best = min(starts, key=lambda ab: _neg_loglik(np.array([v * (1 - sum(ab)), *ab]), x2, s0))
    th0 = np.array([v * (1 - sum(best)), *best])
    res = minimize(_neg_loglik, th0, args=(x2, s0), method="L-BFGS-B",
                   bounds=[(1e-8 * v, 10 * v), (0.0, 1.0), (0.0, 1.0)])
    om, al, be = res.x
    if not res.success and _neg_loglik(res.x, x2, s0) >= 1e12:
        raise RuntimeError("optimalisatie mislukt")
    from scipy.signal import lfilter
    u = om + al * np.concatenate([[s0], x2[:-1]])
    s2 = lfilter([1.0], [1.0, -be], u, zi=[be * s0])[0]
    return om / 1e4, al, be, float(s2[-1]) / 1e4


def garch_stock(isin: str, dates: np.ndarray, r: np.ndarray, qstarts: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Voorspellingen en parameters voor een aandeel. Draait in een apart proces."""
    warnings.filterwarnings("ignore")
    dates = pd.DatetimeIndex(dates)
    ewma = pd.Series(r ** 2).ewm(alpha=0.06, adjust=False, min_periods=60).mean().to_numpy()
    rows, params = [], []
    for k, q0 in enumerate(qstarts):
        q1 = qstarts[k + 1] if k + 1 < len(qstarts) else pd.Timestamp("2100-01-01")
        in_q = np.where((dates >= q0) & (dates < q1))[0]
        if len(in_q) == 0:
            continue
        past = np.where(dates < q0)[0]
        win = past[-WINDOW:]
        fallback, why = False, ""
        if len(win) < MIN_OBS:
            fallback, why = True, "te weinig historiek"
        else:
            try:
                om, al, be, s2_last = fit_garch(r[win])
                phi = al + be
                if not (np.isfinite([om, al, be]).all() and om > 0 and al >= 0 and be >= 0 and phi < 0.999):
                    fallback, why = True, "persistentie >= 0,999 of ongeldige parameters"
            except Exception as e:  # noqa: BLE001
                fallback, why = True, f"schatting mislukt ({type(e).__name__})"
        if fallback:
            v1 = ewma[in_q]
            ok = np.isfinite(v1)
            out = {"date": dates[in_q][ok], "isin": isin}
            out.update({f"v_{h}": h * v1[ok] for h in HORIZONS})
            rows.append(pd.DataFrame(out))
            params.append({"isin": isin, "kwartaal": q0, "omega": np.nan, "alpha": np.nan, "beta": np.nan,
                           "terugval": True, "reden": why, "n": len(win)})
            continue
        # variantie voor de eerste dag van het kwartaal: recursie vanaf de laatste geschatte waarde
        s2 = om + al * r[win[-1]] ** 2 + be * s2_last          # variantie van dag in_q[0]
        V = om / (1 - phi)
        nxt = np.empty(len(in_q))
        for j, i in enumerate(in_q):                            # na de slotkoers van dag i: voorspel dag i+1
            s2 = om + al * r[i] ** 2 + be * s2
            nxt[j] = s2
        out = {"date": dates[in_q], "isin": isin}
        out.update(_h_sum(nxt, V, phi))
        rows.append(pd.DataFrame(out))
        params.append({"isin": isin, "kwartaal": q0, "omega": om, "alpha": al, "beta": be,
                       "terugval": False, "reden": "", "n": len(win)})
    fc = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "isin"] + VCOLS)
    return fc, pd.DataFrame(params)


def main():
    redo = "--opnieuw" in sys.argv
    log("Rendementen laden ...")
    lr = load_returns()
    qstarts = quarter_starts(lr.index)
    TMP.mkdir(parents=True, exist_ok=True)
    if redo:
        for f in TMP.glob("*.parquet"):
            f.unlink()
    done = {f.stem.rsplit("_", 1)[0] for f in TMP.glob("*_fc.parquet")}
    todo = [c for c in lr.columns if c not in done and len(own_series(lr, c)) > 60]
    workers = max(1, (os.cpu_count() or 2) - 1)
    log(f"GARCH: {len(todo)} van {lr.shape[1]} aandelen te schatten, {len(qstarts)} kwartalen, "
        f"{workers} processen tegelijk ...")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for c in todo:
            s = own_series(lr, c)
            futs[ex.submit(garch_stock, c, s.index.values, s.values, qstarts)] = c
        for n, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            fc, pr = fut.result()
            fc.to_parquet(TMP / f"{c}_fc.parquet", index=False)
            pr.to_parquet(TMP / f"{c}_par.parquet", index=False)
            if n % 25 == 0 or n == len(todo):
                log(f"  {n}/{len(todo)}")
    fc = pd.concat([pd.read_parquet(f) for f in TMP.glob("*_fc.parquet")], ignore_index=True)
    pr = pd.concat([pd.read_parquet(f) for f in TMP.glob("*_par.parquet")], ignore_index=True)
    save_forecast("garch", fc)
    pr.to_parquet(RISK / "garch_parameters.parquet", index=False)
    ok = pr[~pr["terugval"]]
    print("\n================ GARCH ================")
    print(f"Schattingen: {len(pr)} (aandeel x kwartaal); terugval naar EWMA: {pr['terugval'].mean():.1%}")
    if len(pr):
        reasons = pr.loc[pr["terugval"], "reden"].str.replace(r" \(.*", "", regex=True).value_counts()
        for r_, n_ in reasons.items():
            print(f"  {r_}: {n_}")
    if len(ok):
        print(f"\nMediaan alpha {ok['alpha'].median():.3f}, beta {ok['beta'].median():.3f}, "
              f"persistentie {(ok['alpha'] + ok['beta']).median():.3f}")
        phi = (ok["alpha"] + ok["beta"]).clip(1e-6, 0.999)
        hl = np.log(0.5) / np.log(phi)
        print(f"Halfwaardetijd van een schok (mediaan): {hl.median():.0f} beursdagen")
    print("=======================================\n")
    log("Volgende:  python risk_har.py")


# --------------------------------------------------------------------------- zelftest
def _simulate(om, al, be, n, rng, s2=None):
    s2 = om / (1 - al - be) if s2 is None else s2
    r = np.empty(n)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = om + al * r[t] ** 2 + be * s2
    return r, s2


def self_test() -> bool:
    """Drie onafhankelijke controles, zonder het pakket arch en zonder echte data."""
    from scipy.signal import lfilter
    rng = np.random.default_rng(2026)
    ok = True
    print("\n================ ZELFTEST GARCH ================")

    # 1. de snelle recursie (lfilter) tegenover een gewone lus
    om, al, be = 0.02, 0.08, 0.90
    x2 = (rng.standard_normal(1000) * 1.5) ** 2
    s0 = _backcast(x2)
    fast = lfilter([1.0], [1.0, -be], om + al * np.concatenate([[s0], x2[:-1]]), zi=[be * s0])[0]
    slow, s2 = np.empty(1000), s0
    for t in range(1000):
        s2 = om + al * (x2[t - 1] if t else s0) + be * s2
        slow[t] = s2
    dev = np.max(np.abs(fast / slow - 1))
    good = dev < 1e-10
    ok &= good
    print(f"1. Recursie snel = lus (max. relatief verschil {dev:.1e})" + " " * 8 + ("OK" if good else "FOUT"))

    # 2. bekende parameters terugvinden (2000 dagen per reeks, 10 reeksen per parameterset)
    print("2. Bekende parameters terugvinden (mediaan over 10 gesimuleerde reeksen van 2000 dagen):")
    print("   waar: alpha  beta   persist. jaarvol  |  geschat: alpha  beta   persist. jaarvol")
    for om, al, be in [(2e-6, 0.08, 0.90), (1e-5, 0.05, 0.93), (8e-6, 0.12, 0.80), (3e-6, 0.04, 0.95)]:
        est = []
        for _ in range(10):
            r, _ = _simulate(om, al, be, 2000, rng)
            o, a, b, _ = fit_garch(r)
            est.append((a, b, a + b, np.sqrt(o / (1 - a - b) * 252) if a + b < 1 else np.nan))
        e = np.nanmedian(np.array(est), axis=0)
        vol = np.sqrt(om / (1 - al - be) * 252)
        good = abs(e[2] - (al + be)) < 0.02 and abs(e[3] / vol - 1) < 0.10 and abs(e[0] - al) < 0.03
        ok &= good
        print(f"         {al:.3f}  {be:.3f}  {al + be:.3f}   {vol:5.1%}   |           "
              f"{e[0]:.3f}  {e[1]:.3f}  {e[2]:.3f}   {e[3]:5.1%}   {'OK' if good else 'FOUT'}")

    # 3. de formule voor h dagen vooruit tegenover simulatie
    om, al, be = 5e-6, 0.08, 0.90
    s2_next = 3 * om / (1 - al - be)                          # start in een onrustige periode
    form = _h_sum(np.array([s2_next]), om / (1 - al - be), al + be)
    sims = np.zeros(len(HORIZONS))
    n_paths = 20000
    for _ in range(n_paths):
        r, _ = _simulate(om, al, be, max(HORIZONS), rng, s2=s2_next)
        c = np.cumsum(r ** 2)
        sims += np.array([c[h - 1] for h in HORIZONS])
    sims /= n_paths
    print("3. Verwachte variantie over h dagen: formule tegenover 20.000 gesimuleerde paden:")
    for h, s_ in zip(HORIZONS, sims):
        f = float(form[f"v_{h}"][0])
        good = abs(f / s_ - 1) < 0.04
        ok &= good
        print(f"   h = {h:>2}: formule {f:.6f}  simulatie {s_:.6f}  verschil {f / s_ - 1:+.1%}   {'OK' if good else 'FOUT'}")
    print(f"\nZelftest GARCH: {'GESLAAGD' if ok else 'MISLUKT'}")
    print("================================================\n")
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        sys.exit(0 if self_test() else 1)
    main()
