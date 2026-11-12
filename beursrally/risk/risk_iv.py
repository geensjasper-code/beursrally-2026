#!/usr/bin/env python3
"""
Beursrally 2026 - risicomodel 6: implied volatility (opties) gecombineerd met het basismodel.

Gebruik
    python br.py risk_iv              # backtest op indexniveau (2002-2021) + vandaag per aandeel
    python br.py risk_iv --final      # ook de testperiode 2022-2026 (een keer, aan het einde)
    python br.py risk_iv --synthetic  # test met nepdata (fake_yf.py)

Deel 1 - backtest op indexniveau (de enige plek waar we historische implied volatility hebben)
    VIX -> S&P 500 en VXN -> Nasdaq 100, vanaf 2000 (eenmalig opgehaald via Yahoo, bewaard in
    data/risk/iv_historiek.parquet). Per index: GARCH(1,1) walk-forward (elk kwartaal opnieuw), en de mengvorm
        v_h = lambda * (c * VIX/100)^2 * h/252 + (1 - lambda) * v_h(GARCH)
    Waarom zo lang? Op 2019-2021 alleen waren c en lambda niet te bepalen (een crash meer of minder keerde
    de schatting om). 2002-2021 bevat de dotcomcrash, 2008, 2011, 2015-16, 2018 en corona.

    Keuzeregel (vooraf vastgelegd)
        Schat c en lambda op 2002-2021, en ter controle apart op 2002-2010 en 2011-2021.
        Liggen beide deelschattingen dicht bij de volledige (c binnen 0,15 en lambda binnen 0,2): gebruik de
        volledige schatting. Anders: literatuurwaarden c = 0,9 en lambda = 0,7.

Deel 2 - vandaag, per Amerikaans aandeel met opties
    1. Implied volatility rond de koers per vervaldag (calls en puts, uitoefenprijs 90-110% van de koers).
    2. Interpolatie naar elke horizon h met totale variantie w(T) = IV^2 x T (lineair in w tussen vervaldagen).
    3. Van dollar naar euro: var_EUR = var_USD - var_FX - 2 cov(r_EUR, r_FX)
       (r_EUR = r_USD - r_FX; geschat op de laatste 250 dagen).
    4. Geen cijfersprong op het optiedeel: opties prijzen de cijferdatum al in.
    5. Menging met het basismodel (HAR): v_h = lambda (c IV)^2 + (1 - lambda) v_h(HAR).
    -> data/risk/iv_vandaag.parquet; het rapport vergelijkt de mengvorm met HAR per aandeel.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from beursrally.risk.risk_common import CLEAN, HORIZONS, PROC, RISK, dm_test, load_forecast, log, names, qlike
from beursrally.risk.risk_garch import _h_sum, fit_garch

PAIRS = {"sp500": "vix", "nasdaq100": "vxn"}
YAHOO = {"^GSPC": "sp500", "^NDX": "nasdaq100", "^VIX": "vix", "^VXN": "vxn"}
HIST_START = "2000-01-01"
C_GRID = np.round(np.arange(0.60, 1.501, 0.05), 2)
L_GRID = np.round(np.arange(0.0, 1.001, 0.1), 2)
FIT_H = [5, 10, 20, 48]
PERIODS = {"2002-2021": ("2002-07-01", "2022-01-01"), "2002-2010": ("2002-07-01", "2011-01-01"),
           "2011-2021": ("2011-01-01", "2022-01-01"), "test 2022-2026": ("2022-01-01", "2100-01-01")}
LITERATURE = {"c": 0.9, "lambda": 0.7}
TOL_C, TOL_L = 0.15, 0.2


# --------------------------------------------------------------------------- deel 1: indices
def long_history(synthetic: bool) -> pd.DataFrame:
    """Slotkoersen van S&P 500, Nasdaq 100, VIX en VXN vanaf 2000 (eenmalig ophalen, daarna uit het bestand)."""
    f = RISK / "iv_historiek.parquet"
    if f.exists() and not synthetic:
        h = pd.read_parquet(f)
        if h.index.max() >= pd.Timestamp.now().normalize() - pd.Timedelta(days=10):
            return h
    if synthetic:
        import beursrally.data.fake_yf as yf
    else:
        import yfinance as yf
    import time
    last_err = None
    for attempt in range(4):
        try:
            df = yf.download(list(YAHOO), start=HIST_START, auto_adjust=False, progress=False)
            close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df
            close = close.rename(columns=YAHOO)
            close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
            if close.reindex(columns=["sp500", "vix"]).notna().sum().min() > 1000:   # VXN mag ontbreken
                RISK.mkdir(parents=True, exist_ok=True)
                if not synthetic:
                    close.to_parquet(f)
                return close
            last_err = "te weinig gegevens"
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(5 * (attempt + 1))
    log(f"Lange historiek niet op te halen ({last_err}); terugval op data/clean (vanaf 2016).")
    return pd.read_parquet(CLEAN / "market_level.parquet")[list(YAHOO.values())]


def index_garch(r: pd.Series) -> pd.DataFrame:
    """Walk-forward GARCH op een indexreeks: kwartaal per kwartaal opnieuw geschat."""
    r = r.dropna()
    dates = r.index
    qs = pd.date_range(dates[0] + pd.Timedelta(days=760), dates.max() + pd.offsets.QuarterBegin(), freq="QS")
    rows = []
    for k, q0 in enumerate(qs):
        q1 = qs[k + 1] if k + 1 < len(qs) else pd.Timestamp("2100-01-01")
        win = np.where(dates < q0)[0][-1000:]
        in_q = np.where((dates >= q0) & (dates < q1))[0]
        if len(win) < 500 or len(in_q) == 0:
            continue
        try:
            om, al, be, s2_last = fit_garch(r.values[win])
        except Exception:  # noqa: BLE001
            continue
        phi = min(al + be, 0.9989)
        s2 = om + al * r.values[win[-1]] ** 2 + be * s2_last
        nxt = []
        for i in in_q:
            s2 = om + al * r.values[i] ** 2 + be * s2
            nxt.append(s2)
        d = pd.DataFrame(_h_sum(np.array(nxt), om / (1 - phi), phi), index=dates[in_q])
        rows.append(d)
    return pd.concat(rows)


def index_targets(r: pd.Series) -> pd.DataFrame:
    r = r.dropna()
    cs = np.concatenate([[0.0], np.cumsum(r.values ** 2)])
    n = len(r)
    out = pd.DataFrame(index=r.index)
    for h in HORIZONS:
        v = np.full(n, np.nan)
        i = np.arange(n - h)
        v[i] = cs[i + 1 + h] - cs[i + 1]
        out[f"rv_{h}"] = v
    return out


def index_backtest(final: bool, synthetic: bool) -> dict:
    hist = long_history(synthetic)
    data = {}
    for idx, ivn in PAIRS.items():
        if idx not in hist or ivn not in hist:
            continue
        px = hist[idx].dropna()
        r = np.log(px).diff().dropna()
        g = index_garch(r)
        t = index_targets(r)
        iv = (hist[ivn].reindex(g.index) / 100) ** 2 / 252          # dagvariantie volgens de opties
        d = g.join(t, how="inner")
        d["iv_day"] = iv
        data[idx] = d.dropna(subset=["iv_day"])
    if not data:
        print("Geen VIX/VXN of indexreeksen gevonden: literatuurwaarden.")
        return {**LITERATURE, "bron": "literatuur (geen data)"}

    def losses(d, h, c, lam):
        f = lam * (c ** 2) * d["iv_day"] * h + (1 - lam) * d[f"v_{h}"]
        return pd.Series(qlike(d[f"rv_{h}"].to_numpy(), f.to_numpy()), index=d.index)

    def window(d, period):
        a, b = PERIODS[period]
        return d[(d.index >= a) & (d.index < b)]

    def mean_loss(period, c, lam):
        tot = []
        for d in data.values():
            dd = window(d, period)
            for h in FIT_H:
                dd2 = dd.dropna(subset=[f"rv_{h}"])
                if len(dd2) > 50:
                    tot.append(losses(dd2, h, c, lam).mean())
        return float(np.mean(tot)) if tot else np.inf

    est = {}
    for period in ("2002-2021", "2002-2010", "2011-2021"):
        grid = [(c, lam, mean_loss(period, c, lam)) for c in C_GRID for lam in L_GRID]
        c_, l_, _ = min(grid, key=lambda x: x[2])
        est[period] = (float(c_), float(l_))
    c_f, l_f = est["2002-2021"]
    stable = all(abs(est[p][0] - c_f) <= TOL_C + 1e-9 and abs(est[p][1] - l_f) <= TOL_L + 1e-9
                 for p in ("2002-2010", "2011-2021"))
    if stable:
        par = {"c": c_f, "lambda": l_f, "bron": "indexbacktest 2002-2021 (stabiel)"}
    else:
        par = {**LITERATURE, "bron": "literatuur (schattingen niet stabiel)"}
    par["schattingen"] = {k: {"c": v[0], "lambda": v[1]} for k, v in est.items()}
    (RISK / "iv_parameters.json").write_text(json.dumps(par, indent=2))

    first = {k: v.index.min().date() for k, v in data.items()}
    print("\n================ IMPLIED VOLATILITY: BACKTEST OP INDEXNIVEAU ================")
    print("Eerste voorspelling: " + ", ".join(f"{k} {v}" for k, v in first.items()))
    print(f"Rooster: c van {C_GRID[0]:.2f} tot {C_GRID[-1]:.2f}, lambda van 0 tot 1; horizonnen {FIT_H}, beide indices samen.")
    for k, (c_, l_) in est.items():
        edge = "   (rand van het rooster)" if c_ in (C_GRID[0], C_GRID[-1]) and l_ > 0 else ""
        print(f"  {k:<10}  c = {c_:.2f}, lambda = {l_:.1f}{edge}")
    print(f"  Literatuur  c = {LITERATURE['c']:.2f}, lambda = {LITERATURE['lambda']:.1f}")
    print(f"  Stabiel (deelperiodes binnen c +-{TOL_C}, lambda +-{TOL_L}): {'JA' if stable else 'NEE'}")
    print(f"  GEBRUIKT: c = {par['c']:.2f}, lambda = {par['lambda']:.1f}  ({par['bron']})")

    periods = ["2002-2010", "2011-2021"] + (["test 2022-2026"] if final else [])
    for period in periods:
        print(f"\n{period}: QLIKE (lager = beter), met de GEBRUIKTE c en lambda")
        rows = []
        for name, d in data.items():
            dd = window(d, period)
            for h in HORIZONS:
                dd2 = dd.dropna(subset=[f"rv_{h}"])
                if len(dd2) < 50:
                    continue
                lg = losses(dd2, h, 1.0, 0.0)
                li = losses(dd2, h, par["c"], 1.0)
                lb = losses(dd2, h, par["c"], par["lambda"])
                dates = pd.Series(dd2.index, index=dd2.index)
                rows.append({"index": name, "h": h, "GARCH": lg.mean(), "alleen opties": li.mean(),
                             "mengvorm": lb.mean(), "t_DM meng vs GARCH": dm_test(lb, lg, dates, h)})
        t = pd.DataFrame(rows)
        if t.empty:
            continue
        for c in ("GARCH", "alleen opties", "mengvorm"):
            t[c] = t[c].round(4)
        t["t_DM meng vs GARCH"] = t["t_DM meng vs GARCH"].round(2)
        print(t.to_string(index=False))
    if not final:
        print("\nDe testperiode 2022-2026 is NIET bekeken (python br.py risk_iv --final, een keer aan het einde).")
    print("=============================================================================\n")
    return par


# --------------------------------------------------------------------------- deel 2: vandaag per aandeel
def atm_iv(o: pd.DataFrame) -> pd.DataFrame:
    """Implied volatility rond de koers per (aandeel, vervaldag): gewogen naar nabijheid van de koers."""
    o = o[(o["impliedVolatility"] > 0.05) & (o["impliedVolatility"] < 5) & o["spot"].notna()].copy()
    o["m"] = o["strike"] / o["spot"]
    o = o[o["m"].between(0.9, 1.1)]
    if "bid" in o:
        o = o[(o["bid"].fillna(0) > 0) | (o["openInterest"].fillna(0) > 0)]
    o["w"] = 1 / (np.abs(o["m"] - 1) + 0.02)
    g = o.groupby(["isin", "expiry"]).apply(
        lambda x: pd.Series({"iv": np.average(x["impliedVolatility"], weights=x["w"]), "n": len(x)}),
        include_groups=False).reset_index()
    return g[g["n"] >= 2]


def iv_term(atm: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Per aandeel en horizon de optievariantie in dollars (totale variantie, lineair geinterpoleerd)."""
    rows = []
    for isin, g in atm.groupby("isin"):
        g = g.sort_values("expiry")
        T = ((pd.to_datetime(g["expiry"]) - asof).dt.days / 365.25).to_numpy()
        keep = T > 2 / 365
        if not keep.any():
            continue
        T, iv = T[keep], g["iv"].to_numpy()[keep]
        w = iv ** 2 * T
        rec = {"isin": isin}
        for h in HORIZONS:
            tau = h / 252
            if tau <= T[0]:
                wt = iv[0] ** 2 * tau
            elif tau >= T[-1]:
                wt = iv[-1] ** 2 * tau
            else:
                wt = np.interp(tau, T, w)
            rec[f"ivv_{h}"] = wt
        rec["iv_kort"] = iv[0]
        rows.append(rec)
    return pd.DataFrame(rows)


def fx_adjustment(isins: list[str]) -> pd.Series:
    """Dagelijkse correctie van dollar- naar eurovariantie: -var_FX - 2 cov(r_EUR, r_FX), laatste 250 dagen."""
    lr = pd.read_parquet(CLEAN / "logret.parquet")
    fx = pd.read_parquet(CLEAN / "market_ret.parquet")["eurusd"]
    out = {}
    for i in isins:
        if i not in lr:
            continue
        d = pd.concat([lr[i], fx], axis=1, keys=["e", "f"]).dropna().tail(250)
        if len(d) < 100:
            continue
        out[i] = -d["f"].var() - 2 * d["e"].cov(d["f"])
    return pd.Series(out)


def live_blend(par: dict) -> None:
    files = sorted(PROC.glob("options_*.parquet"))
    if not files:
        print("Nog geen optiedata (data/processed/options_*.parquet): deel 2 overgeslagen.")
        return
    o = pd.read_parquet(files[-1])
    asof = pd.to_datetime(o["downloaded"]).max().normalize()
    atm = atm_iv(o)
    term = iv_term(atm, asof)
    if term.empty:
        print("Geen bruikbare optieketens: deel 2 overgeslagen.")
        return
    adj = fx_adjustment(term["isin"].tolist())
    term = term[term["isin"].isin(adj.index)].copy()
    har = load_forecast("har")
    last = har.sort_values("date").groupby("isin").tail(1).set_index("isin")
    term = term[term["isin"].isin(last.index)]
    c, lam = par["c"], par["lambda"]
    for h in HORIZONS:
        eur = (term[f"ivv_{h}"] + h * term["isin"].map(adj)).clip(lower=0.25 * term[f"ivv_{h}"])
        term[f"iv_eur_{h}"] = eur
        base = term["isin"].map(last[f"v_{h}"])
        term[f"har_{h}"] = base
        term[f"v_{h}"] = lam * c ** 2 * eur + (1 - lam) * base
    term["optiedatum"] = asof
    term.to_parquet(RISK / "iv_vandaag.parquet", index=False)

    nm = names()
    h = 20
    r = pd.DataFrame({"aandeel": term["isin"].map(lambda i: str(nm.get(i, i))[:26]),
                      "HAR": np.sqrt(term[f"har_{h}"] / h * 252),
                      "opties (EUR)": np.sqrt(term[f"iv_eur_{h}"] / h * 252),
                      "mengvorm": np.sqrt(term[f"v_{h}"] / h * 252)})
    r["opties/HAR"] = r["opties (EUR)"] / r["HAR"]
    print("================ IMPLIED VOLATILITY VANDAAG (VS-aandelen, horizon 20 dagen, jaarvolatiliteit) ================")
    print(f"Optiedata van {asof.date()}; {len(term)} aandelen. c = {c:.2f}, lambda = {lam:.1f}.")
    print(f"Mediaan opties/HAR: {r['opties/HAR'].median():.2f}  (rond 1,0-1,2 verwacht: opties iets duurder)")
    print(f"Mediaan dollar->euro-correctie op jaarvariantie: {adj.median() * 252:+.4f}")
    fmt = lambda df: df.assign(**{k: (df[k] * 100).round(0).astype(int).astype(str) + "%"
                                  for k in ("HAR", "opties (EUR)", "mengvorm")}).round({"opties/HAR": 2})
    print("\nOpties zien VEEL MEER risico dan de historiek (gebeurtenis op komst?):")
    print(fmt(r.sort_values("opties/HAR", ascending=False).head(10)).to_string(index=False))
    print("\nOpties zien VEEL MINDER risico dan de historiek:")
    print(fmt(r.sort_values("opties/HAR").head(5)).to_string(index=False))
    print("===============================================================================================================\n")


def main():
    final = "--final" in sys.argv
    log("Deel 1: backtest op indexniveau (VIX, VXN, vanaf 2000) ...")
    par = index_backtest(final, "--synthetic" in sys.argv)
    log("Deel 2: implied volatility per aandeel vandaag ...")
    live_blend(par)
    log("Volgende:  python br.py risk_compare")


if __name__ == "__main__":
    main()
