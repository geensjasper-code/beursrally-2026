#!/usr/bin/env python3
"""
Beursrally 2026 - risicomodel 4: HAR (Heterogeneous Autoregressive), een regressie voor alle aandelen samen.

Gebruik
    python br.py risk_har

Model (een regressie per horizon h, voor alle aandelen samen)
    ln(RV_h/h + c) = b0 + b1 ln RV_1d + b2 ln RV_5d + b3 ln RV_22d + b4 ln RV_66d + (gekozen blokken) + fout
    voorspelling  v_h = h * exp(voorspeld) * gemiddelde(exp(residu))   (Duan: van log naar gemiddelde)

Blokken (vaste volgorde, vooraf vastgelegd)
    B1 prijsbereik      Parkinson-variantie (hoog/laag) over 5 en 22 dagen
    B2 markt            ln VIX, verandering VIX over 21 dagen
    B3 cijfers          koersreactie op cijfers binnen de horizon (0/1, exacte reactiedag: voor of na de opening),
                        en dat maal ln(gemiddelde koersreactie^2)
    B4 asymmetrie       som van de negatieve rendementen over 5 en 22 dagen
    B5 grootte          log marktkapitalisatie, ln Amihud (illiquiditeit)
    B6 shortposities    short_pct_shares, days_to_cover (alleen VS vanaf 2021)
    B7 distress         distress_score, drawdown_all
    Ontbrekende waarden: 0 plus een indicator 'ontbreekt' (geen invulling met toekomstige gemiddelden).

Selectie (alleen keuzeperiode 2019-2021)
    Start met de basis; voeg blokken een voor een toe; een blok blijft alleen als QLIKE significant daalt
    (Diebold-Mariano t <= -2). Per horizon apart.

Zonder lek
    Maandelijks opnieuw geschat op alle rijen tot (herschattingsdatum - h - 5 beursdagen): het doel van een
    trainingsrij mag niet voorbij de herschattingsdatum lopen (embargo).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from beursrally.risk.risk_common import (EVAL_START, FC_START, FEAT, HORIZONS, LOGC, PROC, RISK, SPLIT, dm_test, earnings_reaction_k,
                         load_returns, log, own_series, qlike, realized_targets, save_forecast)

EMBARGO_EXTRA = 5
BASE = ["lnrv1", "lnrv5", "lnrv22", "lnrv66"]
BLOCKS = {
    "B1 prijsbereik": ["lnpk5", "lnpk22", "pk_na"],
    "B2 markt": ["lnvix", "vix_chg_21", "vix_na"],
    "B3 cijfers": ["earn_in", "earn_x_move", "move_na"],
    "B4 asymmetrie": ["neg5", "neg22"],
    "B5 grootte": ["log_mcap", "ln_amihud", "mcap_na", "amihud_na"],
    "B6 shortposities": ["short_pct_shares", "days_to_cover", "short_na"],
    "B7 distress": ["distress_score", "drawdown_all", "distress_na"],
}
FEATS_FROM_FILE = ["vix", "vix_chg_21", "days_to_earnings", "earnings_move_avg", "log_mcap", "amihud",
                   "short_pct_shares", "days_to_cover", "distress_score", "drawdown_all"]


# --------------------------------------------------------------------------- paneel
def build_panel(lr: pd.DataFrame | None = None) -> pd.DataFrame:
    """Paneel met alle kenmerken en doelen. lr meegeven = andere rendementen (gebruikt door de lektest)."""
    injected = lr is not None
    lr = load_returns() if lr is None else lr
    hi = pd.read_parquet(PROC / "high_local.parquet")
    lo = pd.read_parquet(PROC / "low_local.parquet")
    parts = []
    for c in lr.columns:
        s = own_series(lr, c)
        if len(s) < 70:
            continue
        r2 = s ** 2
        d = pd.DataFrame(index=s.index)
        d["lnrv1"] = np.log(r2 + LOGC)
        d["lnrv5"] = np.log(r2.rolling(5, min_periods=4).mean() + LOGC)
        d["lnrv22"] = np.log(r2.rolling(22, min_periods=18).mean() + LOGC)
        d["lnrv66"] = np.log(r2.rolling(66, min_periods=50).mean() + LOGC)
        neg = s.clip(upper=0)
        d["neg5"] = neg.rolling(5, min_periods=4).sum()
        d["neg22"] = neg.rolling(22, min_periods=18).sum()
        if c in hi.columns and c in lo.columns:
            H, L = hi[c].reindex(s.index), lo[c].reindex(s.index)
            pk = (np.log(H / L) ** 2 / (4 * math.log(2))).where((H > 0) & (L > 0) & (H >= L))
            d["lnpk5"] = np.log(pk.rolling(5, min_periods=4).mean() + LOGC)
            d["lnpk22"] = np.log(pk.rolling(22, min_periods=18).mean() + LOGC)
        else:
            d["lnpk5"] = d["lnpk22"] = np.nan
        d["isin"] = c
        parts.append(d.reset_index(names="date"))
    P = pd.concat(parts, ignore_index=True).dropna(subset=BASE)
    F = pd.read_parquet(FEAT / "features.parquet", columns=["date", "isin"] + FEATS_FROM_FILE)
    P = P.merge(F, on=["date", "isin"], how="left")
    P = P.merge(realized_targets(lr if injected else None), on=["date", "isin"], how="left")
    P["react_k"] = earnings_reaction_k(P).to_numpy()

    # blokken met indicator 'ontbreekt' (0 invullen; de indicator vangt het niveauverschil op)
    P["pk_na"] = P["lnpk5"].isna() | P["lnpk22"].isna()
    P["lnvix"] = np.log(P["vix"])
    P["vix_na"] = P["lnvix"].isna() | P["vix_chg_21"].isna()
    mv = P["earnings_move_avg"]
    P["lnmove2"] = np.log(mv ** 2 + LOGC)
    P["move_na"] = mv.isna()
    P["ln_amihud"] = np.log(P["amihud"] + 1e-6)
    P["mcap_na"] = P["log_mcap"].isna()
    P["amihud_na"] = P["ln_amihud"].isna()
    P["short_na"] = P["short_pct_shares"].isna() | P["days_to_cover"].isna()
    P["distress_na"] = P["distress_score"].isna() | P["drawdown_all"].isna()
    for c in ["lnpk5", "lnpk22", "lnvix", "vix_chg_21", "log_mcap", "ln_amihud", "short_pct_shares",
              "days_to_cover", "distress_score", "drawdown_all", "neg5", "neg22", "lnmove2"]:
        P[c] = P[c].fillna(0.0)
    for c in ["pk_na", "vix_na", "move_na", "mcap_na", "amihud_na", "short_na", "distress_na"]:
        P[c] = P[c].astype(float)
    return P.sort_values(["date", "isin"]).reset_index(drop=True)


def add_earnings_block(P: pd.DataFrame, h: int) -> None:
    """Cijfers binnen de horizon: de koersreactie valt binnen de volgende h beursdagen (react_k <= h)."""
    ein = (P["react_k"].notna() & (P["react_k"] <= h)).astype(float)
    P["earn_in"] = ein
    P["earn_x_move"] = ein * P["lnmove2"] * (1 - P["move_na"])


# --------------------------------------------------------------------------- walk-forward met cumulatieve sommen
class WalkForward:
    """Uitbreidend venster, maandelijkse herschatting, embargo. Snel: X'X per dag, daarna cumulatief."""

    def __init__(self, P: pd.DataFrame, h: int, cols: list[str]):
        self.h, self.cols = h, cols
        self.dates = P["date"].to_numpy()
        self.cal = np.array(sorted(P["date"].unique()))
        X = np.column_stack([np.ones(len(P))] + [P[c].to_numpy(float) for c in cols])
        y = np.log(P[f"rv_{h}"].to_numpy(float) / h + LOGC)
        ok = np.isfinite(y)
        self.X, self.y, self.ok = X, y, ok
        day = np.searchsorted(self.cal, self.dates)
        nP = X.shape[1]
        G = np.zeros((len(self.cal), nP, nP))
        b = np.zeros((len(self.cal), nP))
        yy = np.zeros(len(self.cal))
        n = np.zeros(len(self.cal))
        bounds = np.searchsorted(day, np.arange(len(self.cal) + 1))
        for k in range(len(self.cal)):
            sl = slice(bounds[k], bounds[k + 1])
            m = ok[sl]
            Xd, yd = X[sl][m], y[sl][m]
            G[k], b[k], yy[k], n[k] = Xd.T @ Xd, Xd.T @ yd, yd @ yd, m.sum()
        self.G, self.b, self.yy, self.n = G.cumsum(0), b.cumsum(0), yy.cumsum(), n.cumsum()
        self.bounds = bounds

    def predict(self, use: list[str], start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        """Voorspelde variantie v_h voor alle rijen met datum in [start, end); NaN elders."""
        idx = np.array([0] + [1 + self.cols.index(c) for c in use])
        out = np.full(len(self.dates), np.nan)
        months = pd.date_range(start.to_period("M").start_time, end, freq="MS")
        for m0 in months:
            m1 = m0 + pd.offsets.MonthBegin()
            k_ref = np.searchsorted(self.cal, np.datetime64(m0))          # eerste beursdag >= m0
            k_cut = k_ref - self.h - EMBARGO_EXTRA - 1                    # laatste toegelaten trainingsdag
            if k_cut < 250:
                continue
            A = self.G[k_cut][np.ix_(idx, idx)]
            bb = self.b[k_cut][idx]
            nn = self.n[k_cut]
            beta = np.linalg.solve(A + 1e-8 * np.eye(len(idx)), bb)
            lo = np.searchsorted(self.dates, np.datetime64(max(m0, start)))
            hi = np.searchsorted(self.dates, np.datetime64(min(m1, end)))
            if hi <= lo:
                continue
            smear = self._smear(idx, beta, k_cut)
            pred = self.X[lo:hi][:, idx] @ beta
            out[lo:hi] = self.h * np.maximum(np.exp(pred) * smear - LOGC, 1e-12)
        return out

    def _smear(self, idx, beta, k_cut, n_max: int = 200_000) -> float:
        """Duan-correctie van log naar gemiddelde: gemiddelde van exp(residu) op de laatste trainingsrijen.

        Beter dan exp(s2/2), omdat de residuen (zeker bij h = 1: ln van een enkel kwadraat) niet normaal zijn.
        """
        end_row = self.bounds[k_cut + 1]
        rows = np.arange(max(0, end_row - 3 * n_max), end_row)
        rows = rows[self.ok[rows]][-n_max:]
        e = self.y[rows] - self.X[rows][:, idx] @ beta
        return float(np.mean(np.exp(e)))

    def coefficients(self, use: list[str], at: pd.Timestamp) -> pd.Series:
        idx = np.array([0] + [1 + self.cols.index(c) for c in use])
        k_cut = np.searchsorted(self.cal, np.datetime64(at)) - self.h - EMBARGO_EXTRA - 1
        A, bb = self.G[k_cut][np.ix_(idx, idx)], self.b[k_cut][idx]
        return pd.Series(np.linalg.solve(A + 1e-8 * np.eye(len(idx)), bb), index=["constante"] + use)


# --------------------------------------------------------------------------- selectie en voorspellingen
def select_blocks(P: pd.DataFrame, wf: WalkForward, h: int) -> tuple[list[str], list[dict]]:
    in_eval = ((P["date"] >= EVAL_START) & (P["date"] < SPLIT)).to_numpy()
    rv = P[f"rv_{h}"].to_numpy(float)
    ev = in_eval & np.isfinite(rv)
    dates = P.loc[ev, "date"].reset_index(drop=True)

    def loss(use):
        f = wf.predict(use, FC_START, SPLIT)[ev]
        return pd.Series(qlike(rv[ev], f))

    use = list(BASE)
    cur = loss(use)
    log_rows = [{"horizon": h, "blok": "basis", "QLIKE": cur.mean(), "t_DM": np.nan, "gehouden": True}]
    for name, cols in BLOCKS.items():
        new = loss(use + cols)
        t = dm_test(new, cur, dates, h)
        keep = bool(np.isfinite(t) and t <= -2)
        log_rows.append({"horizon": h, "blok": name, "QLIKE": new.mean(), "t_DM": t, "gehouden": keep})
        if keep:
            use, cur = use + cols, new
    return use, log_rows


def main():
    log("Paneel opbouwen (kenmerken per aandeel per dag) ...")
    P = build_panel()
    log(f"Paneel: {len(P):,} rijen, {P['isin'].nunique()} aandelen, {P['date'].min().date()} tot {P['date'].max().date()}")
    all_cols = BASE + [c for cols in BLOCKS.values() for c in cols]
    out = P[["date", "isin"]].copy()
    sel_log, coefs = [], []
    end = P["date"].max() + pd.Timedelta(days=1)
    for h in HORIZONS:
        add_earnings_block(P, h)
        log(f"Horizon {h}: walk-forward opbouwen en blokken selecteren ...")
        wf = WalkForward(P, h, all_cols)
        use, rows = select_blocks(P, wf, h)
        sel_log += rows
        kept = [r["blok"] for r in rows[1:] if r["gehouden"]]
        log(f"  gekozen: basis{' + ' + ', '.join(kept) if kept else ''}")
        out[f"v_{h}"] = wf.predict(use, FC_START, end)
        cf = wf.coefficients(use, end)
        coefs.append(cf.rename(f"h={h}"))
    save_forecast("har", out)
    sel = pd.DataFrame(sel_log)
    sel.to_csv(RISK / "har_selectie.csv", index=False)
    cf = pd.concat(coefs, axis=1)
    cf.to_csv(RISK / "har_coefficienten.csv")

    pd.set_option("display.width", 200)
    print("\n================ HAR: BLOKSELECTIE (keuzeperiode 2019-2021) ================")
    print("Per horizon: QLIKE na toevoegen van het blok, t-waarde DM tegenover het model ervoor (<= -2 = gehouden).")
    show = sel.copy()
    show["QLIKE"] = show["QLIKE"].round(4)
    show["t_DM"] = show["t_DM"].round(2)
    print(show.pivot(index="blok", columns="horizon", values="t_DM").reindex(["basis"] + list(BLOCKS))
          .to_string(na_rep="-"))
    print("\nGehouden blokken:")
    print(show.pivot(index="blok", columns="horizon", values="gehouden").reindex(list(BLOCKS))
          .replace({True: "ja", False: "."}).to_string())
    print("\nCoefficienten basis (laatste schatting):")
    print(cf.loc[["constante"] + BASE].round(3).to_string())
    lvl = [c for c in BASE + ["lnpk5", "lnpk22"] if c in cf.index]
    s = cf.loc[lvl].fillna(0).sum()
    print(f"\nSom van de volatiliteitscoefficienten incl. Parkinson (persistentie, ~0,8-1,0 verwacht): "
          + ", ".join(f"h={h}: {v:.2f}" for h, v in zip(HORIZONS, s.values)))
    print("=============================================================================\n")
    log("Volgende:  python br.py risk_compare")


if __name__ == "__main__":
    main()
