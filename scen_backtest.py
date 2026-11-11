#!/usr/bin/env python3
"""
Beursrally 2026 - scenariogenerator (stap 4c): klopt de simulatie met wat er echt gebeurde?

Gebruik
    python scen_backtest.py            # keuzeperiode 2019-2021 met de gekozen vorm (of normaal)
    python scen_backtest.py --grid     # keuzeperiode: alle vormen van het rooster, beste bewaren
    python scen_backtest.py --final    # testperiode 2022-2026: EEN keer, als alles af is

Vormen (scen_core.Shape): stressdagen (nu), vol-onzekerheid voor de hele markt en per aandeel.
Keuze: laagste afwijking van de staartpercentages (gewogen naar hoeveel onafhankelijke waarnemingen een
toets heeft); daarna wordt elke uitbreiding weggelaten die de afwijking met minder dan 5% verbetert.
Bewaard in data/scenarios/vorm_keuze.json; scen_build.py gebruikt die.

Hoe
    Op elke startdatum D bouwen we de scenario's met ALLEEN wat op D bekend was:
        volatiliteit   walk-forward-voorspellingen van het risicomodel (HAR, gemiddeld met boosting tot h = 10)
        rendement      walk-forward-voorspellingen van het rendementsmodel (de gekozen modellen, w en b)
        samenhang      factormodel (fac_10) op de 500 beursdagen voor D
    (Opties zitten er niet in: van opties hebben we geen historiek.)
    Dan vergelijken we met de echte rendementen vanaf de opening na D (zelfde definitie als de modellen).

Toetsen (h = 5, 20 en 48 beursdagen)
    1. PIT per aandeel: in welk percentiel van de scenario's valt het echte rendement? Bij een goede simulatie
       valt 1% onder het 1e percentiel, 5% onder het 5e, ... ook in de staarten. Twee versies:
         absoluut   inclusief de marktbeweging (een gemeenschappelijke trekking per datum: veel ruis)
         relatief   ten opzichte van het gemiddelde aandeel van die dag (aandeelspecifiek deel)
    2. Gemiddelde naar volatiliteit: verwacht de simulatie voor de meest beweeglijke aandelen een hoger gewoon
       rendement dan er echt was? (Convexiteit: E[gewoon rendement] = exp(mu + s^2/2) - 1.)
    3. Portefeuilles en uitschieters na 48 dagen: 10 meest beweeglijke, willekeurige 10-aandelenportefeuilles,
       het beste aandeel, de spreiding tussen aandelen. Precies wat de top van het klassement bepaalt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from risk_common import HORIZONS, RISK, load_forecast, load_returns, log, nw_tstat
from risk_corr import corr_factor, corr_lw, corr_sample, two_day_returns, window_matrix
from scen_core import Shape, build_params, generate, trading_days

HERE = Path(__file__).resolve().parent
RET = HERE / "data" / "returns"
FEAT = HERE / "data" / "features"
COMBO_H = [1, 3, 5, 10]                      # zoals risk_build.py: HAR + boost_alleen gemiddeld
TEST_H = [5, 20, 48]
N_SCEN = int(os.environ.get("BR_BT_SCEN", "2000"))
N_RANDOM_PORT = 300
MARKET_PA = 0.07
KEUZE = (pd.Timestamp("2019-01-01"), pd.Timestamp("2022-01-01"))
TEST = (pd.Timestamp("2022-01-01"), pd.Timestamp("2100-01-01"))
GRID = [Shape(nu, vm, va) for nu in (0, 5) for vm in (0.0, 0.25, 0.5, 0.75) for va in (0.0, 0.25, 0.5, 0.75, 1.0)]
SHAPE_FILE = HERE / "data" / "scenarios" / "vorm_keuze.json"
BANDS = [("<1%", 0.01), ("<5%", 0.05), ("<10%", 0.10), (">90%", 0.10), (">95%", 0.05), (">99%", 0.01)]


# --------------------------------------------------------------------------- historische invoer
class History:
    def __init__(self):
        har = load_forecast("har")
        try:
            bo = load_forecast("boost_alleen")
        except FileNotFoundError:
            bo = None
        v = har.set_index(["date", "isin"])
        if bo is not None:
            b = bo.set_index(["date", "isin"]).reindex(v.index)
            for h in COMBO_H:
                v[f"v_{h}"] = np.where(b[f"v_{h}"].notna(), (v[f"v_{h}"] + b[f"v_{h}"]) / 2, v[f"v_{h}"])
        self.var = v
        self.targets = pd.read_parquet(FEAT / "targets.parquet").set_index(["date", "isin"])
        self.keuze = json.loads((RET / "keuze.json").read_text())
        self.preds = {}
        for h in HORIZONS:
            for m in self._parts(self.keuze[str(h)]["model"]):
                if m not in self.preds:
                    self.preds[m] = pd.read_parquet(RET / f"pred_{m}.parquet").set_index(["date", "isin"])
        best = (RISK / "corr_keuze.txt").read_text().strip() if (RISK / "corr_keuze.txt").exists() else "fac_10"
        self.corr_kind = best
        self.r2 = two_day_returns(load_returns())
        self.dates = np.array(sorted(har["date"].unique()))

    @staticmethod
    def _parts(model: str) -> list[str]:
        return model[4:].split("+") if model.startswith("gem:") else [model]

    def alpha(self, D: pd.Timestamp, isins: pd.Index, h: int) -> pd.Series:
        k = self.keuze[str(h)]
        parts = []
        for m in self._parts(k["model"]):
            p = self.preds[m]
            try:
                s = p.xs(D, level="date")[f"p_{h}"].reindex(isins)
            except KeyError:
                return pd.Series(0.0, index=isins)
            parts.append(s)
        if k["model"].startswith("gem:"):
            z = [(s - s.mean()) / s.std() if s.std() > 0 else s * 0 for s in parts]
            pred = (z[0] + z[1]) / 2 * k["schaal"]
        else:
            pred = parts[0]
        return (k["w"] * k["b"] * pred).fillna(0.0)

    def inputs(self, D: pd.Timestamp):
        v = self.var.xs(D, level="date")[[f"v_{h}" for h in HORIZONS]].dropna()
        v.columns = HORIZONS
        y = self.targets.xs(D, level="date")[[f"y_{h}" for h in TEST_H]].reindex(v.index)
        keep = y[f"y_{TEST_H[-1]}"].notna()
        v, y = v[keep], y[keep]
        mu = pd.DataFrame({h: MARKET_PA * h / 252 + self.alpha(D, v.index, h) for h in HORIZONS})
        W = window_matrix(self.r2, D + pd.Timedelta(days=1))
        W = W.loc[:, W.columns.isin(v.index)]
        X = W.to_numpy()
        if self.corr_kind.startswith("fac_"):
            R = corr_factor(X, int(self.corr_kind.split("_")[1]))
        elif self.corr_kind == "sample":
            R = corr_sample(X)
        else:
            R = corr_lw(X)[0]
        R = pd.DataFrame(R, index=W.columns, columns=W.columns)
        return mu, v, R, y


# --------------------------------------------------------------------------- per datum
def pit(sim: np.ndarray, real: np.ndarray) -> np.ndarray:
    """sim (S, K), real (K,): aandeel scenario's onder de werkelijkheid (gelijke waarden half)."""
    return (sim < real).mean(0) + 0.5 * (sim == real).mean(0)


def prepare(hist: History, D: pd.Timestamp):
    mu, var, R, y = hist.inputs(D)
    p = build_params(mu, var, R, D, end=D + pd.Timedelta(days=110))
    return p, var, y


def one_date(prep, D: pd.Timestamp, shape: Shape, n_scen: int, picks_seed: int) -> dict:
    p, var, y = prep
    rng = np.random.default_rng(picks_seed)
    idx = [h - 1 for h in TEST_H]
    sim = np.concatenate([c[:, idx, :] for c in generate(p, n_scen, seed=int(D.strftime("%Y%m%d")), shape=shape)])
    sim = sim.astype(np.float64)                                   # (S, 3, N) log-rendement
    real = y.to_numpy(float).T                                     # (3, N)
    out = {"datum": D, "n": len(var)}
    for j, h in enumerate(TEST_H):
        ok = np.isfinite(real[j])
        s, r = sim[:, j, ok], real[j, ok]
        out[f"pit_abs_{h}"] = pit(s, r)
        out[f"pit_rel_{h}"] = pit(s - s.mean(1, keepdims=True), r - r.mean())
    # gemiddelde naar volatiliteit (gewoon rendement, relatief tot het gemiddelde aandeel), h = 48
    s48, r48 = np.expm1(sim[:, -1, :]), np.expm1(real[-1])
    sim_rel = (s48 - s48.mean(1, keepdims=True)).mean(0)
    real_rel = r48 - r48.mean()
    q = pd.qcut(var[48].rank(method="first"), 5, labels=False).to_numpy()
    out["vol_bucket"] = pd.DataFrame({"q": q, "sim": sim_rel, "echt": real_rel,
                                      "jaarvol": np.sqrt(var[48].to_numpy() / 48 * 252)})
    # portefeuilles en uitschieters na 48 dagen (gewone rendementen, kopen en houden, gelijk gewogen)
    top = np.argsort(-var[48].to_numpy())[:10]
    out["pit_top10"] = pit(s48[:, top].mean(1, keepdims=True), np.array([r48[top].mean()]))[0]
    picks = np.array([rng.choice(len(r48), 10, replace=False) for _ in range(N_RANDOM_PORT)])
    sim_p = s48[:, picks].mean(2)                                  # (S, P)
    out["pit_rand10"] = pit(sim_p, r48[picks].mean(1))
    out["pit_best"] = pit(s48.max(1, keepdims=True), np.array([r48.max()]))[0]
    out["pit_spread"] = pit(s48.std(1, keepdims=True), np.array([r48.std()]))[0]
    out["best_echt"] = r48.max()
    out["best_sim_med"] = float(np.median(s48.max(1)))
    out["best_sim"] = s48.max(1)[:500]                             # verdeling van het beste aandeel (steekproef)
    return out


# --------------------------------------------------------------------------- samenvatting
def tails(u: np.ndarray) -> dict:
    u = u[np.isfinite(u)]
    return {"n": len(u), "<1%": (u < 0.01).mean(), "<5%": (u < 0.05).mean(), "<10%": (u < 0.10).mean(),
            ">90%": (u > 0.90).mean(), ">95%": (u > 0.95).mean(), ">99%": (u > 0.99).mean(),
            "gem.": u.mean()}


def tail_table(res: list[dict]) -> pd.DataFrame:
    rows = []
    for kind in ("abs", "rel"):
        for h in TEST_H:
            u = np.concatenate([r[f"pit_{kind}_{h}"] for r in res])
            rows.append({"toets": f"aandeel {'absoluut' if kind == 'abs' else 'relatief'} h={h}", **tails(u)})
    for key, lab in (("pit_top10", "10 meest beweeglijke (h=48)"), ("pit_rand10", "willekeurige 10 (h=48)"),
                     ("pit_best", "beste aandeel (h=48)"), ("pit_spread", "spreiding tussen aandelen (h=48)")):
        u = np.concatenate([np.atleast_1d(r[key]) for r in res])
        rows.append({"toets": lab, **tails(u)})
    return pd.DataFrame(rows).set_index("toets")


# per toets: hoeveel ONAFHANKELIJKE waarnemingen ongeveer (aandelen op dezelfde datum hangen samen)
N_EFF = {"aandeel relatief h=20": 20, "aandeel relatief h=48": 20, "aandeel absoluut h=48": 3,
         "willekeurige 10 (h=48)": 3, "10 meest beweeglijke (h=48)": 1, "beste aandeel (h=48)": 1,
         "spreiding tussen aandelen (h=48)": 1}


def loss(t: pd.DataFrame, n_dates: int) -> float:
    """Som van gekwadrateerde, gestandaardiseerde afwijkingen van de staartpercentages (zoals een chi-kwadraat)."""
    tot = 0.0
    for test, m in N_EFF.items():
        n = m * n_dates
        for band, e in BANDS:
            tot += (t.at[test, band] - e) ** 2 / (e * (1 - e) / n)
    return tot


def report(res: list[dict], label: str, n_scen: int = N_SCEN) -> None:
    pd.set_option("display.width", 200)
    fmt = lambda v: f"{v:.1%}" if isinstance(v, float) else v
    print(f"\n================ SCENARIO-BACKTEST: {label} ({len(res)} startdatums, {n_scen:,} scenario's elk) "
          "================")
    print("PIT: in welk percentiel van de scenario's viel de werkelijkheid? Verwacht: <1% = 1%, <5% = 5%, "
          "<10% = 10%, gem. = 50%.")
    t = tail_table(res)
    show = t.copy()
    for c in show.columns:
        if c != "n":
            show[c] = show[c].map(fmt)
    print(show.to_string())
    print("  Te veel in de staarten (bv. <1% = 3%): de simulatie onderschat uitschieters (dikkere staarten nodig).")
    print("  Te weinig (bv. <10% = 4%): de simulatie overschat de beweeglijkheid.")
    print("  Absoluut: per datum een marktbeweging; met ~36 overlappende datums is dat ruw.")

    print("\nGewoon rendement na 48 dagen, relatief tot het gemiddelde aandeel, per volatiliteitsgroep:")
    vb = pd.concat([r["vol_bucket"].assign(d=r["datum"]) for r in res])
    rows = []
    for q, g in vb.groupby("q"):
        per_d = g.groupby("d")[["sim", "echt"]].mean()
        diff = per_d["sim"] - per_d["echt"]
        rows.append({"groep": {0: "1 rustigst", 4: "5 beweeglijkst"}.get(q, str(q + 1)),
                     "jaarvol": f"{g['jaarvol'].median():.0%}", "simulatie": f"{per_d['sim'].mean():+.2%}",
                     "echt": f"{per_d['echt'].mean():+.2%}", "sim - echt": f"{diff.mean():+.2%}",
                     "t": f"{nw_tstat(diff.to_numpy(), 3):+.2f}"})
    print(pd.DataFrame(rows).set_index("groep").to_string())
    print("  'sim - echt' > 0 met |t| > 2 in de beweeglijkste groep: de simulatie is te optimistisch over")
    print("  beweeglijke aandelen (mu bevat de volatiliteitsrem niet volledig).")
    real_best = np.array([r["best_echt"] for r in res])
    sim_best = np.concatenate([r["best_sim"] for r in res])
    qs = [0.5, 0.75, 0.9, 0.95]
    print("\nBeste aandeel na 48 dagen (over alle startdatums):   p50     p75     p90     p95     max")
    print("  echt (36 waarden)                            " + "".join(f"{v:+8.0%}" for v in np.quantile(real_best, qs))
          + f"{real_best.max():+8.0%}")
    print("  simulatie                                    " + "".join(f"{v:+8.0%}" for v in np.quantile(sim_best, qs))
          + f"{np.quantile(sim_best, 0.999):+8.0%} (p99,9)")
    print("  Simulatie p90 duidelijk boven echt p90: de extremen zijn te wild -> voorzichtiger vorm kiezen.")
    print(f"Afwijking (lager = beter): {loss(t, len(res)):.0f}")
    print("=" * 100)


def load_shape() -> Shape:
    if SHAPE_FILE.exists():
        d = json.loads(SHAPE_FILE.read_text())
        return Shape(int(d["nu"]), float(d["vol_markt"]), float(d["vol_aandeel"]))
    return Shape()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="testperiode 2022-2026 (een keer, aan het einde)")
    ap.add_argument("--grid", action="store_true", help="alle vormen van het rooster vergelijken (keuzeperiode)")
    a = ap.parse_args()
    if a.grid and a.final:
        sys.exit("--grid alleen op de keuzeperiode.")
    lo, hi = TEST if a.final else KEUZE
    log("Historiek laden (voorspellingen risico en rendement, rendementen) ...")
    hist = History()
    d = pd.Series(hist.dates)
    d = d[(d >= lo) & (d < hi)]
    starts = [pd.Timestamp(x) for x in d.groupby(d.dt.to_period("M")).min().tolist()]
    preps = {}
    for D in starts:
        try:
            pr = prepare(hist, D)
        except KeyError:
            continue
        if len(pr[1]) >= 50:
            preps[D] = pr
    if not preps:
        sys.exit("Geen bruikbare startdatums.")
    log(f"{len(preps)} startdatums voorbereid")
    period = "TESTPERIODE 2022-2026" if a.final else "KEUZEPERIODE 2019-2021"
    if not a.grid:
        shape = load_shape()
        t0 = time.time()
        res = [one_date(pr, D, shape, N_SCEN, k) for k, (D, pr) in enumerate(preps.items())]
        log(f"Gesimuleerd in {time.time() - t0:.0f} s")
        report(res, f"{period}, vorm: {shape.label()}")
        return
    n_grid = int(os.environ.get("BR_BT_GRID_SCEN", "1000"))
    scores, results = {}, {}
    t0 = time.time()
    for sh in GRID:
        res = [one_date(pr, D, sh, n_grid, k) for k, (D, pr) in enumerate(preps.items())]
        scores[sh] = loss(tail_table(res), len(res))
        results[sh] = res
        log(f"  {sh.label():<48} afwijking {scores[sh]:8.0f}")
    best = min(scores, key=scores.get)
    # eenvoud: elke uitbreiding weglaten die minder dan 5% verbetert
    for field_, off in (("vol_aandeel", 0.0), ("vol_markt", 0.0), ("nu", 0)):
        simpler = Shape(**{**best.__dict__, field_: off})
        if simpler in scores and scores[simpler] <= scores[best] * 1.05:
            best = simpler
    log(f"Rooster in {time.time() - t0:.0f} s. Gekozen: {best.label()} "
        f"(afwijking {scores[best]:.0f}; normaal {scores[Shape()]:.0f})")
    SHAPE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SHAPE_FILE.write_text(json.dumps({"nu": best.nu, "vol_markt": best.vol_markt, "vol_aandeel": best.vol_aandeel,
                                      "afwijking": scores[best], "afwijking_normaal": scores[Shape()]}, indent=2))
    log(f"-> {SHAPE_FILE.relative_to(HERE)}")
    report(results[Shape()], f"{period}, NORMAAL", n_grid)
    if best != Shape():
        report(results[best], f"{period}, GEKOZEN: {best.label()}", n_grid)


if __name__ == "__main__":
    main()
