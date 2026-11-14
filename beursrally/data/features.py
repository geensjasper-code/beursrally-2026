#!/usr/bin/env python3
"""
Beursrally 2026 - stap 2a: kenmerken (features) en doelvariabelen (targets).

Gebruik
    python br.py features          # alle kenmerken + doelvariabelen -> data/features/ + rapport
    python br.py features test     # lektest: veranderen de kenmerken op dag d als je de toekomst wijzigt?

Tijdsafspraak (zie 'Voorkomen van datalekken' in de specificatie)
    Kenmerken met datum d gebruiken alleen data tot en met de slotkoers van dag d.
    Beslismoment = 08u00 op de ochtend na d: dan zijn alle slotkoersen van d bekend (ook VS en EUR/USD).
    Doelvariabelen starten bij de OPENING van de volgende handelsdag: y_h = ln(slot na h dagen / opening d+1).

Alles wordt per effect op zijn EIGEN handelsdagen berekend (geen kunstmatige 0%-dagen) en alleen
terugkijkend (rolling windows), daarna op de gemeenschappelijke kalender gezet.
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from beursrally.paths import CONFIG, ROOT as HERE   # projectmap (data/ en config/ staan daar)
DATA = HERE / "data"
CLEAN = DATA / "clean"
PROC = DATA / "processed"
OUT = DATA / "features"

HORIZONS = [1, 3, 5, 10, 20, 48]
DECISION_HOUR = 8                      # beslismoment: 08u00 op de dag na d
LN = np.log


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------- invoer
def load_inputs() -> dict:
    need = ["close_eur", "open_eur", "logret", "market_level"]
    for n in need:
        if not (CLEAN / f"{n}.parquet").exists():
            sys.exit(f"data/clean/{n}.parquet ontbreekt. Eerst:  python br.py clean")
    inp = {n: pd.read_parquet(CLEAN / f"{n}.parquet") for n in need}
    inp["zones"] = pd.read_csv(CLEAN / "zones.csv", index_col=0)["zone"].to_dict()
    # ruwe tabellen uit data.py (niet opgeschoond): volume, hoog/laag, ongefilterde slotkoers voor 'geen handel'
    for n in ("volume", "high_local", "low_local", "close_local", "close_eur"):
        inp["raw_" + n] = pd.read_parquet(PROC / f"{n}.parquet")
    for k in ("earnings", "shares", "short_interest"):
        f = CLEAN / f"{k}.parquet"
        inp[k] = pd.read_parquet(f) if f.exists() else pd.DataFrame()
    return inp


# --------------------------------------------------------------------------- per effect
def _own(s: pd.Series) -> pd.Series:
    return s.dropna()


def stock_features(isin: str, inp: dict) -> pd.DataFrame:
    """Alle prijs- en volumekenmerken van een effect, op zijn eigen handelsdagen."""
    P = _own(inp["close_eur"][isin])
    if len(P) < 30:
        return pd.DataFrame()
    days = P.index
    r = inp["logret"][isin].reindex(days)                      # opgeschoonde rendementen (artefacten = NaN)
    zone = inp["zones"].get(isin, "US")
    mk = inp["market_level"]
    own_mkt = "stoxx600" if zone == "EU" else "sp500"
    M = mk[own_mkt].dropna().reindex(days, method="ffill") if own_mkt in mk else pd.Series(np.nan, index=days)
    B = mk["bitcoin"].dropna().reindex(days, method="ffill") if "bitcoin" in mk else pd.Series(np.nan, index=days)
    m, b = LN(M).diff(), LN(B).diff()

    f = pd.DataFrame(index=days)
    # A. momentum en trend
    for k in (1, 5, 21):
        f[f"r_{k}"] = r.rolling(k, min_periods=max(1, int(0.8 * k))).sum()
    f["mom_6_1"] = LN(P.shift(21) / P.shift(126))
    f["mom_12_1"] = LN(P.shift(21) / P.shift(252))
    f["dist_52w_high"] = LN(P / P.rolling(252, min_periods=60).max())
    f["dist_ma50"] = LN(P / P.rolling(50, min_periods=40).mean())
    f["dist_ma200"] = LN(P / P.rolling(200, min_periods=150).mean())
    f["drawdown_all"] = LN(P / P.cummax())

    # B. volatiliteit en risico
    f["vol_20"] = r.rolling(20, min_periods=15).std() * np.sqrt(252)
    f["vol_60"] = r.rolling(60, min_periods=40).std() * np.sqrt(252)
    H = inp["raw_high_local"][isin].reindex(days)
    L = inp["raw_low_local"][isin].reindex(days)
    hl2 = LN(H / L).where((H > 0) & (L > 0) & r.notna()) ** 2
    f["vol_parkinson_20"] = np.sqrt(hl2.rolling(20, min_periods=15).mean() / (4 * np.log(2)) * 252)
    f["max_ret_21"] = r.rolling(21, min_periods=15).max()
    f["skew_60"] = r.rolling(60, min_periods=40).skew()
    cov_m = r.rolling(252, min_periods=120).cov(m)
    beta = cov_m / m.rolling(252, min_periods=120).var()
    f["beta_mkt"] = beta
    f["beta_btc"] = r.rolling(252, min_periods=120).cov(b) / b.rolling(252, min_periods=120).var()
    f["idio_vol_60"] = (r - beta * m).rolling(60, min_periods=40).std() * np.sqrt(252)
    f["mkt_ret_21"] = m.rolling(21, min_periods=15).sum()

    # C. volume en liquiditeit (ruw volume; dagen zonder handel tellen hier juist mee)
    rawP = _own(inp["raw_close_eur"][isin])
    V = inp["raw_volume"][isin].reindex(rawP.index)
    v20, v252 = V.rolling(20, min_periods=15).mean(), V.rolling(252, min_periods=120).mean()
    f["abn_volume"] = LN(v20 / v252).replace([np.inf, -np.inf], np.nan).reindex(days)
    dv = (V * rawP).where(V > 0)
    rr = LN(rawP).diff().abs()
    f["amihud"] = (rr / dv * 1e6).rolling(60, min_periods=30).mean().reindex(days)
    f["zero_vol_share"] = (V == 0).astype(float).rolling(60, min_periods=30).mean().reindex(days)

    # E. distress (0 tot 3 signalen)
    f["distress_score"] = ((f["dist_52w_high"] < LN(0.3)).astype(float)
                           + (f["vol_60"] > 1.0).astype(float)
                           + (f["mom_12_1"] < LN(0.4)).astype(float)).where(f["vol_60"].notna())

    # doelvariabelen: vanaf de opening van de volgende eigen handelsdag
    O = inp["open_eur"][isin].reindex(days)
    tg = pd.DataFrame(index=days)
    o_next = O.shift(-1)
    for h in HORIZONS:
        tg[f"y_{h}"] = LN(P.shift(-h) / o_next)
    f = f.astype("float32")
    f.insert(0, "isin", isin)
    tg = tg.astype("float32")
    tg.insert(0, "isin", isin)
    return f.reset_index(names="date"), tg.reset_index(names="date")


# --------------------------------------------------------------------------- kwartaalcijfers
def earnings_features(grid: pd.DataFrame, inp: dict) -> pd.DataFrame:
    """Per (datum, effect): laatste gepubliceerde verrassing, dagen sinds/tot cijfers, gemiddelde koerssprong."""
    e = inp["earnings"]
    out = grid[["date", "isin"]].copy()
    cols = ["last_surprise", "days_since_earnings", "days_to_earnings", "earn_within_5", "earn_within_20",
            "earnings_move_avg"]
    if e.empty:
        for c in cols:
            out[c] = np.nan
        return out
    out["decision"] = out["date"] + pd.Timedelta(days=1, hours=DECISION_HOUR)
    e = e[e["isin"].isin(out["isin"].unique())].copy()
    e = e.dropna(subset=["known_at"]).sort_values("known_at")
    g = out.sort_values("decision")

    last = pd.merge_asof(g, e[["isin", "known_at", "surprisepct"]], left_on="decision", right_on="known_at",
                         by="isin", direction="backward")
    since = (last["decision"] - last["known_at"]).dt.days
    last["last_surprise"] = last["surprisepct"].where(since <= 120)
    last["days_since_earnings"] = since.where(since <= 365)

    nxt = pd.merge_asof(g, e[["isin", "known_at"]].rename(columns={"known_at": "next_at"}),
                        left_on="decision", right_on="next_at", by="isin", direction="forward",
                        tolerance=pd.Timedelta(days=30))          # datum ~30 dagen vooraf bekend verondersteld
    to = (nxt["next_at"] - nxt["decision"]).dt.days
    last["days_to_earnings"] = to.values
    last["earn_within_5"] = (to <= 7).astype(float).where(to.notna(), 0.0).values
    last["earn_within_20"] = (to <= 28).astype(float).where(to.notna(), 0.0).values

    # gemiddelde absolute koersreactie op de vorige 8 publicaties (pas bekend na de reactiedag)
    lr = inp["logret"]
    rows = []
    for isin, grp in e.groupby("isin"):
        if isin not in lr.columns:
            continue
        s = lr[isin].dropna()
        if s.empty:
            continue
        for _, row in grp.iterrows():
            day = row["known_at"].normalize()
            # publicatie na 15u Belgische tijd (na Euronext-opening/VS-voorbeurs): reactie de volgende handelsdag
            later = s.index[s.index > day] if row["known_at"].hour >= 15 else s.index[s.index >= day]
            if len(later):
                rows.append((isin, later[0], abs(s[later[0]])))
    if rows:
        mv = pd.DataFrame(rows, columns=["isin", "react_day", "move"]).sort_values(["isin", "react_day"])
        mv["earnings_move_avg"] = mv.groupby("isin")["move"].transform(lambda x: x.rolling(8, min_periods=2).mean())
        mv = mv.sort_values("react_day")
        avg = pd.merge_asof(g.sort_values("date"), mv[["isin", "react_day", "earnings_move_avg"]],
                            left_on="date", right_on="react_day", by="isin", direction="backward")
        avg = avg.set_index(["date", "isin"])["earnings_move_avg"]
    else:
        avg = pd.Series(dtype=float)
    res = last.set_index(["date", "isin"])[cols[:-1]]
    res["earnings_move_avg"] = avg.reindex(res.index)
    return res.reset_index()


# --------------------------------------------------------------------------- grootte en shortposities
FUND_COLS = ["log_mcap", "short_pct_shares", "days_to_cover", "short_chg"]


def size_short_features(grid: pd.DataFrame, inp: dict) -> pd.DataFrame:
    """Marktkapitalisatie en shortpositie zoals bekend op het beslismoment (08u00 op d+1).

    log_mcap          ln(aantal aandelen x slotkoers in euro). Aantal = laatste gepubliceerde (45 dagen vertraging).
    short_pct_shares  shortpositie / aantal aandelen (FINRA, alleen VS-aandelen, vanaf juni 2021).
    days_to_cover     shortpositie / gemiddeld dagvolume (FINRA).
    short_chg         ln(shortpositie / vorige shortpositie): wordt er bij- of afgebouwd?
    Alles op de basis van de koersen (gecorrigeerd voor splitsingen), dus een splitsing geeft geen sprong.
    """
    out = grid[["date", "isin"]].copy()
    for c in FUND_COLS:
        out[c] = np.nan
    out["decision"] = out["date"] + pd.Timedelta(days=1, hours=DECISION_HOUR)
    g = out.sort_values("decision").reset_index(drop=True)

    sh = inp.get("shares", pd.DataFrame())
    if not sh.empty:
        sh = sh[sh["isin"].isin(g["isin"].unique())].sort_values("known_at")
        m = pd.merge_asof(g[["date", "isin", "decision"]], sh[["isin", "known_at", "shares_adj"]],
                          left_on="decision", right_on="known_at", by="isin", direction="backward",
                          tolerance=pd.Timedelta(days=400))
        px = inp["close_eur"].stack().rename("px").reset_index()
        px.columns = ["date", "isin", "px"]
        m = m.merge(px, on=["date", "isin"], how="left")
        g["shares_adj"] = m["shares_adj"].to_numpy()
        g["log_mcap"] = np.log(m["shares_adj"] * m["px"]).to_numpy()

    si = inp.get("short_interest", pd.DataFrame())
    if not si.empty:
        si = si[si["isin"].isin(g["isin"].unique())].sort_values(["isin", "settlement_date"]).copy()
        si["short_chg"] = np.log(si["short_adj"] / si.groupby("isin")["short_adj"].shift(1)) \
            .replace([np.inf, -np.inf], np.nan)
        si = si.sort_values("known_at")
        m = pd.merge_asof(g[["decision", "isin"]], si[["isin", "known_at", "short_adj", "days_to_cover", "short_chg"]],
                          left_on="decision", right_on="known_at", by="isin", direction="backward",
                          tolerance=pd.Timedelta(days=40))       # ouder dan 40 dagen: niet meer actueel
        sh_now = g["shares_adj"] if "shares_adj" in g else np.nan
        g["short_pct_shares"] = (m["short_adj"] / sh_now).clip(0, 2).to_numpy()
        g["days_to_cover"] = m["days_to_cover"].clip(0, 100).to_numpy()
        g["short_chg"] = m["short_chg"].clip(-3, 3).to_numpy()
    return g[["date", "isin"] + FUND_COLS].astype({c: "float32" for c in FUND_COLS})


# --------------------------------------------------------------------------- markt en kalender
def market_features(dates: pd.DatetimeIndex, inp: dict) -> pd.DataFrame:
    mk = inp["market_level"].reindex(dates, method="ffill")
    f = pd.DataFrame(index=dates)
    if "vix" in mk:
        f["vix"] = mk["vix"]
        f["vix_chg_21"] = mk["vix"] - mk["vix"].shift(21)
    if "us_rate_10y" in mk:
        f["rate_10y"] = mk["us_rate_10y"]
        f["rate_10y_chg_21"] = mk["us_rate_10y"] - mk["us_rate_10y"].shift(21)
    f["month"] = dates.month
    f["days_to_year_end"] = (pd.to_datetime(dates.year.astype(str) + "-12-31") - dates).days
    return f.astype("float32").reset_index(names="date")


# --------------------------------------------------------------------------- alles samen
def build(inp: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats, tgts = [], []
    cols = inp["close_eur"].columns
    for n, isin in enumerate(cols, 1):
        res = stock_features(isin, inp)
        if isinstance(res, tuple):
            feats.append(res[0])
            tgts.append(res[1])
        if n % 100 == 0:
            log(f"  {n}/{len(cols)} effecten")
    F = pd.concat(feats, ignore_index=True)
    T = pd.concat(tgts, ignore_index=True)
    F = F.merge(earnings_features(F[["date", "isin"]], inp), on=["date", "isin"], how="left")
    F = F.merge(size_short_features(F[["date", "isin"]], inp), on=["date", "isin"], how="left")
    F = F.merge(market_features(pd.DatetimeIndex(sorted(F["date"].unique())), inp), on="date", how="left")
    return F, T


def live_features() -> pd.DataFrame:
    """Alleen live te gebruiken (momentopname van vandaag): shortpositie en marktkapitalisatie."""
    f = PROC / "info.parquet"
    if not f.exists():
        return pd.DataFrame()
    i = pd.read_parquet(f)
    out = pd.DataFrame({"isin": i["isin"],
                        "short_pct": pd.to_numeric(i.get("shortPercentOfFloat"), errors="coerce"),
                        "log_mcap": np.log(pd.to_numeric(i.get("marketCap"), errors="coerce").where(lambda x: x > 0))})
    return out


# --------------------------------------------------------------------------- rapport
def rank_ic(F: pd.DataFrame, T: pd.DataFrame, feat: str, y: str, step: int) -> tuple[float, float, int]:
    """Gemiddelde rang-IC over niet-overlappende datums (elke 'step' dagen) en een eenvoudige t-waarde."""
    d = F[["date", "isin", feat]].merge(T[["date", "isin", y]], on=["date", "isin"]).dropna()
    dates = np.array(sorted(d["date"].unique()))[::step]
    d = d[d["date"].isin(dates)]
    ics = d.groupby("date").apply(lambda x: x[feat].rank().corr(x[y].rank()) if len(x) > 30 else np.nan,
                                  include_groups=False).dropna()
    if len(ics) < 10:
        return np.nan, np.nan, len(ics)
    return ics.mean(), ics.mean() / (ics.std() / np.sqrt(len(ics))), len(ics)


def report(F: pd.DataFrame, T: pd.DataFrame) -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # constante kolommen per dag: IC bestaat niet
    feats = [c for c in F.columns if c not in ("date", "isin")]
    rows = []
    for c in feats:
        x = F[c]
        ic5, t5, n5 = rank_ic(F, T, c, "y_5", 5) if c not in ("month", "days_to_year_end") else (np.nan,) * 3
        ic20, t20, _ = rank_ic(F, T, c, "y_20", 20) if c not in ("month", "days_to_year_end") else (np.nan,) * 3
        rows.append({"kenmerk": c, "dekking": f"{x.notna().mean():.0%}", "mediaan": x.median(),
                     "p5": x.quantile(0.05), "p95": x.quantile(0.95),
                     "IC_5d": ic5, "t_5d": t5, "IC_20d": ic20, "t_20d": t20})
    rep = pd.DataFrame(rows)
    rep.to_csv(DATA / "features_rapport.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n================ KENMERKEN ================")
    print(f"{len(feats)} kenmerken, {F['isin'].nunique()} effecten, {F['date'].nunique()} dagen, "
          f"{len(F):,} rijen -> data/features/")
    print(rep.round(3).to_string(index=False))
    print("\nIC = gemiddelde rangcorrelatie met het rendement over 5 of 20 dagen (niet-overlappend).")
    print("BESCHRIJVEND, geen modelkeuze: hele periode, inclusief survivorship bias.")
    print("============================================\n")


def run():
    log("Invoer laden ...")
    inp = load_inputs()
    log("Kenmerken berekenen ...")
    F, T = build(inp)
    OUT.mkdir(parents=True, exist_ok=True)
    F.to_parquet(OUT / "features.parquet", index=False)
    T.to_parquet(OUT / "targets.parquet", index=False)
    lf = live_features()
    if not lf.empty:
        lf.to_parquet(OUT / "live_only.parquet", index=False)
    report(F, T)


# --------------------------------------------------------------------------- lektest
def tamper(inp: dict, d: pd.Timestamp, rng) -> dict:
    """Alles wat na het beslismoment van dag d bekend werd, willekeurig veranderen."""
    out = {}
    for k, v in inp.items():
        if isinstance(v, pd.DataFrame) and isinstance(v.index, pd.DatetimeIndex):
            w = v.copy()
            fut = w.index > d
            w.loc[fut] = w.loc[fut].to_numpy() * rng.uniform(0.5, 1.5, size=w.loc[fut].shape)
            out[k] = w
        elif isinstance(v, pd.DataFrame) and not v.empty and "known_at" in v.columns:
            # tabellen met een publicatietijdstip: alles wat na het beslismoment gepubliceerd werd, wijzigen
            w = v.copy()
            fut = (w["known_at"] > d + pd.Timedelta(days=1, hours=DECISION_HOUR)).to_numpy()
            for c in ("surprisepct", "shares_adj", "short_adj", "days_to_cover"):
                if c in w.columns:
                    w.loc[fut, c] = w.loc[fut, c].to_numpy(float) * rng.uniform(0.3, 3.0, fut.sum())
            out[k] = w
        else:
            out[k] = v
    return out


def run_leak_test(n_dates: int = 3, seed: int = 1) -> bool:
    inp = load_inputs()
    rng = np.random.default_rng(seed)
    cal = inp["close_eur"].index
    dates = sorted(rng.choice(cal[400:-60], size=n_dates, replace=False))
    sub = list(inp["close_eur"].columns[rng.choice(len(inp["close_eur"].columns), 40, replace=False)])

    def small(i):                                    # alleen 40 effecten: sneller, zelfde logica
        j = dict(i)
        for k in ("close_eur", "open_eur", "logret", "raw_volume", "raw_high_local", "raw_low_local",
                  "raw_close_local", "raw_close_eur"):
            j[k] = i[k][[c for c in sub if c in i[k].columns]]
        return j

    base_F, base_T = build(small(inp))
    ok = True
    for d in dates:
        d = pd.Timestamp(d)
        F2, T2 = build(small(tamper(inp, d, rng)))
        a = base_F[base_F["date"] == d].set_index("isin").sort_index()
        b = F2[F2["date"] == d].set_index("isin").sort_index()
        cols = [c for c in a.columns if c != "date"]
        same = np.allclose(a[cols].to_numpy(float), b[cols].to_numpy(float), equal_nan=True, rtol=1e-5, atol=1e-6)
        ta = base_T[base_T["date"] == d].set_index("isin").sort_index()["y_5"]
        tb = T2[T2["date"] == d].set_index("isin").sort_index()["y_5"]
        target_changed = not np.allclose(ta.to_numpy(float), tb.to_numpy(float), equal_nan=True)
        print(f"  {d.date()}: kenmerken onveranderd {'OK' if same else 'LEK'};  "
              f"doelvariabele wel veranderd {'OK' if target_changed else 'FOUT (test niet gevoelig)'}")
        if not same:
            diff = [c for c in cols if not np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), equal_nan=True)]
            print(f"     lekkende kenmerken: {diff}")
        ok &= same and target_changed
    print(f"\nLektest kenmerken: {'GESLAAGD' if ok else 'MISLUKT'}\n")
    return ok


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("\n================ LEKTEST KENMERKEN ================")
        sys.exit(0 if run_leak_test() else 1)
    run()


if __name__ == "__main__":
    main()
