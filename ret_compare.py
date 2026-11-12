#!/usr/bin/env python3
"""
Beursrally 2026 - rendementsmodel: alle kandidaten vergelijken en kiezen.

Gebruik
    python ret_compare.py           # keuzeperiode 2019-2021: hiermee kiezen we
    python ret_compare.py --controle   # ook de controleperiode 2022 - midden 2025 (bescheiden bijsturen mag)
    python ret_compare.py --final      # ook de eindtoets midden 2025 - nu (een keer, aan het einde)

Maatstaven (per model en horizon, op exact dezelfde rijen)
    IC        rang-IC: correlatie tussen voorspelde en gerealiseerde rangorde per dag, gemiddeld
    t_IC      t-waarde van de IC (Newey-West: overlappende horizonnen)
    IC_groot  idem, alleen de grootste helft van de aandelen (controle op survivorship bias)
    top-bodem gemiddeld rendement van de beste 10% voorspellingen min de slechtste 10% (over h dagen)
    RMSE      wortel van de gemiddelde kwadratische fout (in procent)
    R2_OS     1 - (fout model / fout 'alles gelijk'); t_DM: Diebold-Mariano (negatief = beter dan alles gelijk)
    b         kalibratie: helling van het gerealiseerde op het voorspelde (1 = juiste grootte)

Kosten (reglement): 5 euro + 0,35% per order, dus ~0,9% per heen-en-terug bij een positie van ~5.000 euro.
    IC_min = kosten / (spreiding tussen aandelen x z), met z = gemiddelde score van de 10 beste uit ~370
    (~2,2 standaardafwijkingen). Onder IC_min levert handelen op dit signaal gemiddeld niets op na kosten.

Keuze per horizon (vooraf vastgelegd)
    1. Neem de twee modellen met de hoogste IC.
    2. Toets of hun IC-verschil significant is (t van het verschil per dag, Newey-West).
    3. Niet significant (|t| < 2): gebruik het GEMIDDELDE van beide (per dag gestandaardiseerde voorspellingen,
       z-scores, gemiddeld). Wel significant: de beste.
    Vertrouwen w = max(0, 1 - 1/t_IC^2) van het gekozen (gemiddelde) model.
Bewaart data/returns/keuze.json (model, IC, t, w, b, IC_min per horizon).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from ret_models import FEAT, HORIZONS, MODELS, OUT, log

EVAL_START, SPLIT = pd.Timestamp("2019-01-01"), pd.Timestamp("2022-01-01")
CONTROLE_END = pd.Timestamp("2025-07-01")


def _mask(d, period):
    if period == "keuze":
        return (d >= EVAL_START) & (d < SPLIT)
    if period == "controle":
        return (d >= SPLIT) & (d < CONTROLE_END)
    if period == "final":
        return d >= CONTROLE_END
    if period == "samen":
        return (d >= EVAL_START) & (d < CONTROLE_END)
    if period == "alles":
        return d >= EVAL_START
    return d >= SPLIT
COST_ROUND_TRIP = 2 * (5.0 / 5000 + 0.0035)          # 0,9%
Z_TOP10 = 2.2


def nw_t(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return np.nan
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else np.nan


def per_date_ic(df: pd.DataFrame, pcol: str, ycol: str) -> pd.Series:
    r = df.groupby("date")[[pcol, ycol]].rank()
    r["date"] = df["date"].to_numpy()
    def corr(g):
        if g[pcol].nunique() < 5:
            return np.nan
        return np.corrcoef(g[pcol], g[ycol])[0, 1]
    return r.groupby("date").apply(corr, include_groups=False)


def zscore_by_date(x: pd.Series, dates: pd.Series) -> pd.Series:
    g = x.groupby(dates)
    sd = g.transform("std")
    return ((x - g.transform("mean")) / sd).where(sd > 0, 0.0)


def combo_scale(M: pd.DataFrame, a: str, b: str, h: int) -> float:
    """Typische spreiding (per dag) van de twee eigen voorspellingen, in rendement: om het gemiddelde van de
    z-scores terug in rendementseenheden te zetten (zodat RMSE, R2_OS en b vergelijkbaar zijn)."""
    sd = M.groupby("date")[[f"{a}|{h}", f"{b}|{h}"]].std().mean()
    return float(sd.mean())


def add_combo(M: pd.DataFrame, a: str, b: str, h: int, scale: float | None = None) -> tuple[str, float]:
    name = f"gem:{a}+{b}"
    scale = combo_scale(M, a, b, h) if scale is None else scale
    z = (zscore_by_date(M[f"{a}|{h}"], M["date"]) + zscore_by_date(M[f"{b}|{h}"], M["date"])) / 2
    M[f"{name}|{h}"] = z * scale
    return name, scale


def ic_diff_t(M: pd.DataFrame, a: str, b: str, h: int, period: str) -> float:
    d = M["date"]
    sub = M[_mask(d, period)]
    sub = sub.dropna(subset=[f"{a}|{h}", f"{b}|{h}", f"yd_{h}"])
    ia = per_date_ic(sub.assign(_p=sub[f"{a}|{h}"]), "_p", f"yd_{h}")
    ib = per_date_ic(sub.assign(_p=sub[f"{b}|{h}"]), "_p", f"yd_{h}")
    return nw_t((ia - ib).to_numpy(), h)


def load(models):
    T = pd.read_parquet(FEAT / "targets.parquet")
    F = pd.read_parquet(FEAT / "features.parquet", columns=["date", "isin", "log_mcap"])
    M = T.merge(F, on=["date", "isin"], how="left")
    for h in HORIZONS:
        y = M[f"y_{h}"]
        M[f"yd_{h}"] = y - y.groupby(M["date"]).transform("mean")
    M["groot"] = M.groupby("date")["log_mcap"].rank(pct=True) >= 0.5
    for m in models:
        p = pd.read_parquet(OUT / f"pred_{m}.parquet").rename(columns={f"p_{h}": f"{m}|{h}" for h in HORIZONS})
        M = M.merge(p, on=["date", "isin"], how="inner")
    return M


def evaluate(M: pd.DataFrame, models: list[str], h: int, period: str) -> pd.DataFrame:
    d = M["date"]
    sub = M[_mask(d, period)]
    cols = [f"{m}|{h}" for m in models]
    sub = sub.dropna(subset=cols + [f"yd_{h}"])
    y = sub[f"yd_{h}"]
    sig_cs = y.groupby(sub["date"]).std().mean()
    rows = []
    e0 = y ** 2
    for m in models:
        p = sub[f"{m}|{h}"]
        rec = {"model": m, "n": len(sub)}
        if p.abs().sum() > 0:
            ic = per_date_ic(sub.assign(_p=p), "_p", f"yd_{h}")
            rec["IC"], rec["t_IC"] = float(ic.mean()), nw_t(ic.to_numpy(), h)
            g = sub[sub["groot"]]
            icg = per_date_ic(g.assign(_p=g[f"{m}|{h}"]), "_p", f"yd_{h}")
            rec["IC_groot"] = icg.mean()
            q = p.groupby(sub["date"]).rank(pct=True)
            tb = (y[q > 0.9].groupby(sub["date"][q > 0.9]).mean() - y[q <= 0.1].groupby(sub["date"][q <= 0.1]).mean())
            rec["top-bodem"] = tb.mean()
            rec["b"] = float((p * y).sum() / (p * p).sum())
        e = (y - p) ** 2
        rec["RMSE%"] = float(np.sqrt(e.mean()) * 100)
        rec["R2_OS%"] = float((1 - e.sum() / e0.sum()) * 100)
        dd = (e - e0).groupby(sub["date"]).mean()
        rec["t_DM"] = np.nan if m == "zero" else nw_t(dd.to_numpy(), h)
        rows.append(rec)
    res = pd.DataFrame(rows).set_index("model")
    res.attrs = {"sigma_cs": float(sig_cs), "ic_min": float(COST_ROUND_TRIP / (sig_cs * Z_TOP10)),
                 "dates": int(sub["date"].nunique())}
    return res


def main():
    alles = "--alles" in sys.argv                  # na de eindtoets: w en b over 2019 - nu (zelfde regel, alle data)
    final = "--final" in sys.argv or alles
    controle = "--controle" in sys.argv or final
    write_from = "alles" if alles else "samen"
    models = [m for m in MODELS if (OUT / f"pred_{m}.parquet").exists()]
    if not models:
        sys.exit("Geen voorspellingen gevonden. Eerst:  python ret_models.py")
    log(f"Modellen: {', '.join(models)}. Laden ...")
    M = load(models)
    pd.set_option("display.width", 220)
    keuze = {}
    later = {}
    for period in ["keuze"] + (["controle", "samen"] if controle else []) + (["final"] if final else []) \
            + (["alles"] if alles else []):
        title = {"keuze": "KEUZEPERIODE 2019-2021", "controle": "CONTROLEPERIODE 2022 - MIDDEN 2025",
                 "samen": "KEUZE + CONTROLE SAMEN: 2019 - MIDDEN 2025"
                          + ("" if alles else " (hieruit komen w en b)"),
                 "final": "EINDTOETS MIDDEN 2025 - NU (eenmalig!)",
                 "alles": "ALLES: 2019 - NU (na de eindtoets; hieruit komen w en b)"}[period]
        print(f"\n================ RENDEMENTSMODEL: {title} ================")
        print("IC = rang-IC (correlatie voorspelde en echte rangorde). t_IC > 2: overtuigend. "
              "t_DM < -2: overtuigend beter dan 'alles gelijk'.")
        for h in HORIZONS:
            res = evaluate(M, models, h, period)
            cand0 = res.drop(index="zero", errors="ignore").dropna(subset=["IC"]).sort_values("IC", ascending=False)
            note = ""
            combo, scale = None, 1.0
            if period == "keuze" and len(cand0) >= 2:
                m1, m2 = cand0.index[0], cand0.index[1]
                td = ic_diff_t(M, m1, m2, h, period)
                if np.isfinite(td) and abs(td) < 2:
                    combo, scale = add_combo(M, m1, m2, h)
                    note = f"  IC-verschil {m1} - {m2}: t = {td:.2f} (niet significant) -> gemiddelde van beide"
                else:
                    note = f"  IC-verschil {m1} - {m2}: t = {td:.2f} (significant) -> {m1}"
            elif period != "keuze" and str(h) in keuze and keuze[str(h)]["model"].startswith("gem:"):
                a_, b_ = keuze[str(h)]["model"][4:].split("+")
                combo, scale = add_combo(M, a_, b_, h, keuze[str(h)]["schaal"])
            if combo:
                res = evaluate(M, models + [combo], h, period)
            a = res.attrs
            show = res.drop(columns="n").copy()
            for c in ("IC", "IC_groot"):
                if c in show:
                    show[c] = show[c].round(4)
            if "top-bodem" in show:
                show["top-bodem"] = (show["top-bodem"] * 100).round(2).astype(str) + "%"
            show = show.round({"t_IC": 2, "RMSE%": 3, "R2_OS%": 3, "t_DM": 2, "b": 3})
            cand = res.drop(index="zero", errors="ignore").dropna(subset=["IC"])
            if combo:
                best = combo
            elif period != "keuze" and str(h) in keuze:
                best = keuze[str(h)]["model"]
            else:
                best = cand["IC"].idxmax() if not cand.empty else "zero"
            t = float(cand.at[best, "t_IC"]) if best != "zero" else 0.0
            w = max(0.0, 1 - 1 / t ** 2) if t > 1 else 0.0
            print(f"\nh = {h} dagen  ({res['n'].iloc[0]:,} voorspellingen op {a['dates']} datums; "
                  f"spreiding tussen aandelen {a['sigma_cs'] * 100:.2f}%; IC_min na kosten {a['ic_min']:.3f})")
            print(show.to_string(na_rep="-"))
            if note:
                print(note)
            ic_best = float(cand.at[best, "IC"]) if best != "zero" else 0.0
            verdict = "boven IC_min" if ic_best > a["ic_min"] else "ONDER IC_min (handelen loont gemiddeld niet)"
            print(f"  -> beste: {best} (IC {ic_best:.4f}, t {t:.2f}); vertrouwen w = {w:.2f}; {verdict}")
            if period != "keuze":
                later.setdefault(period, {})[h] = {"model": best, "IC": ic_best, "t_IC": t, "w": w,
                                                   "b": float(cand.at[best, "b"]) if best != "zero" else 0.0}
            if period == "keuze":
                keuze[str(h)] = {"model": best, "schaal": float(scale) if combo else 1.0, "IC": ic_best, "t_IC": t, "w": w,
                                 "b": float(cand.at[best, "b"]) if best != "zero" else 0.0,
                                 "IC_min": float(a["ic_min"]), "boven_IC_min": bool(ic_best > a["ic_min"])}
    (OUT / "keuze.json").write_text(json.dumps(keuze, indent=2))
    print("\n================ KEUZE PER HORIZON (keuzeperiode) ================")
    kt = pd.DataFrame(keuze).T[["model", "IC", "t_IC", "w", "b", "IC_min", "boven_IC_min"]]
    for c in ("IC", "t_IC", "w", "b", "IC_min"):
        kt[c] = kt[c].astype(float)
    kt.index.name = "h"
    print(kt.round({"IC": 4, "t_IC": 2, "w": 2, "b": 2, "IC_min": 3}).to_string())
    print("Gebruikt rendement = 7%/jaar x h/252  +  w x b x voorspelling.")
    print("==================================================================\n")
    for period, d in later.items():
        print(f"================ {period.upper()}: GEKOZEN MODELLEN BUITEN DE KEUZEPERIODE ================")
        rows = []
        adv = {}
        for h, v in d.items():
            k = keuze[str(h)]
            if v["t_IC"] < 2 or v["IC"] <= 0:
                adv[str(h)] = {"w": 0.0, "b": float(k["b"])}
                advies = "w = 0 (alleen marktdeel)"
            else:
                adv[str(h)] = {"w": float(v["w"]), "b": float(v["b"])}
                advies = f"w = {v['w']:.2f}, b = {v['b']:.2f}"
            rows.append({"h": h, "model": v["model"], "IC keuze": round(k["IC"], 4), "IC hier": round(v["IC"], 4),
                         "t_IC hier": round(v["t_IC"], 2), "b hier": round(v["b"], 2),
                         "w keuze": round(k["w"], 2), "gevolg": advies})
        print(pd.DataFrame(rows).set_index("h").to_string())
        if period == write_from:
            (OUT / "w_controle.json").write_text(json.dumps(adv, indent=2))
            span = "2019 - nu" if alles else "2019 - midden 2025"
            print(f"-> data/returns/w_controle.json: ret_build.py gebruikt w en b uit {span} "
                  "(bestand weg = keuzeperiode)")
            print("Regel (optie C): vertrouwen gemeten over de hele periode voor de eindtoets; niet significant "
                  "(t < 2) of IC <= 0 -> w = 0.\n")
        else:
            print("(Alleen ter informatie; w en b komen uit de periode "
                  f"{'2019 - nu' if alles else '2019 - midden 2025 samen'}.)\n")
    if not final:
        print("De eindtoets (midden 2025 - nu) is NIET bekeken. Dat doen we een keer, als alles vastligt.\n")


if __name__ == "__main__":
    main()
