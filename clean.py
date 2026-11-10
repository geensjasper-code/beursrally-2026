#!/usr/bin/env python3
"""
Beursrally 2026 - opschonen en point-in-time-toegang tot de data.

Gebruik
    python clean.py          # data opschonen -> data/clean/ + rapport
    python clean.py test     # lektest: verandert een kenmerk op t als je de toekomst na t wijzigt?

Wat het doet
    1. Onvolledige dagen weg: cellen die tijdens de beurs zijn opgehaald (nog geen slotkoers).
    2. Artefacten uit artefacten.csv toepassen (start_vanaf: alles ervoor weg; dag_weg: rendement van die dag weg).
    3. Dagen zonder volume en met een ongewijzigde koers weg (geen echte handel, kunstmatig 0%).
    4. De eerste 10 handelsdagen na een beursintroductie weg.
    5. Rendementen opnieuw berekenen op de eigen geldige dagen; rentes als verandering in niveau.
    6. Nieuwe artefact-kandidaten zoeken en melden (niet automatisch toepassen).
    7. Aantal aandelen en shortposities: omzetten naar dezelfde basis als de koersen (splitsingen) en
       een publicatietijdstip (known_at) geven.

Alle modellen halen hun data via PointInTime.as_of(t): alleen wat op tijdstip t bekend was.
"""
from __future__ import annotations

import sys
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PROC = DATA / "processed"
RAW = DATA / "raw"
CLEAN = DATA / "clean"
ARTEFACTS = HERE / "artefacten.csv"

IPO_SKIP_DAYS = 10
JUMP = np.log(1.5)          # |log-rendement| boven ln(1,5): grote sprong

# Wanneer is de slotkoers van dag d bekend? (Belgische tijd, met een kleine marge)
# Conservatief gekozen: in de weken met verschillende zomertijd sluit New York om 21u, hier rekenen we 22u15.
ZONES = {
    "EU":  (0, time(17, 45)),     # Euronext-slot
    "US":  (0, time(22, 15)),     # Amerikaanse slot
    "FUT": (0, time(23, 15)),     # settlement futures
    "24":  (1, time(1, 15)),      # bitcoin, wisselkoersen: slot rond middernacht UTC = de nacht erna
}
MARKET_ZONE = {
    "eurusd": "24", "bitcoin": "24", "stoxx600": "EU", "msci_world": "EU",
    "sp500": "US", "nasdaq100": "US", "vix": "US", "vxn": "US",
    "us_rate_3m": "US", "us_rate_10y": "US", "gold": "FUT", "brent": "FUT",
}
RATE_SERIES = {"us_rate_3m", "us_rate_10y"}           # niveaus in %, geen prijzen
LIVE_ONLY = {"info"}                                  # momentopnames: nooit in een backtest
SHARES_LAG_DAYS = 45        # aantal aandelen: datum = einde kwartaal, publicatie pas weken later (10-Q)
SHORT_LAG_BDAYS = 9         # FINRA publiceert ~7 werkdagen na de afrekendatum; 2 dagen marge voor feestdagen
KNOWN_AT_TABLES = ("earnings", "shares", "short_interest")


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _rp(name: str) -> pd.DataFrame:
    f = PROC / f"{name}.parquet"
    if not f.exists():
        sys.exit(f"{f} ontbreekt. Eerst:  python data.py process")
    return pd.read_parquet(f)


def zone_of(ticker: str, currency: str = "EUR") -> str:
    """Wanneer is de EURO-slotkoers van dit effect bekend?

    Euronext in euro: na de Euronext-slot. Alles in dollar: de euro-prijs gebruikt de EUR/USD-koers van die dag,
    en die is pas rond middernacht UTC definitief -> zone '24' (01u15 de nacht erna), ook al sluit de beurs om 22u.
    """
    eu = str(ticker).endswith((".AS", ".PA", ".BR"))
    if currency == "USD":
        return "24"
    return "EU" if eu else "US"


def known_at(dates: pd.DatetimeIndex, zone: str) -> pd.DatetimeIndex:
    """Tijdstip waarop de slotkoers van elke datum bekend is."""
    add_days, t = ZONES[zone]
    return dates + pd.Timedelta(days=add_days) + pd.Timedelta(hours=t.hour, minutes=t.minute)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log-rendement per kolom op de eigen geldige dagen: na een ontbrekende dag loopt het rendement verder."""
    out = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for c in prices.columns:
        s = prices[c].dropna()
        s = s[s > 0]
        if len(s) > 1:
            out.loc[s.index[1:], c] = np.log(s.values[1:] / s.values[:-1])
    return out


# --------------------------------------------------------------------------- opschonen
def incomplete_cells(tick2isin: dict) -> pd.DataFrame:
    """(datum, isin)-paren waarvan de recentste download gebeurde voor hun slotkoers bekend was."""
    files = sorted((RAW / "prices").glob("*.parquet"))
    p = pd.concat([pd.read_parquet(f, columns=["date", "ticker", "downloaded"]) for f in files])
    p = p.sort_values("downloaded").drop_duplicates(["ticker", "date"], keep="last")
    p = p[p["ticker"].isin(tick2isin)]
    exch_zone = p["ticker"].map(lambda k: "EU" if str(k).endswith((".AS", ".PA", ".BR")) else "US")
    ready = pd.Series(pd.NaT, index=p.index, dtype="datetime64[ns]")
    for z in ("EU", "US"):                     # hier gaat het om de beurs zelf: is de slotkoers er al?
        m = exch_zone == z
        ready[m] = known_at(pd.DatetimeIndex(p.loc[m, "date"]), z)
    bad = p[p["downloaded"] < ready]
    return pd.DataFrame({"date": bad["date"], "isin": bad["ticker"].map(tick2isin)})


def load_artefacts() -> pd.DataFrame:
    if not ARTEFACTS.exists():
        return pd.DataFrame(columns=["isin", "datum", "actie", "reden"])
    a = pd.read_csv(ARTEFACTS, dtype=str)
    a["datum"] = pd.to_datetime(a["datum"])
    bad = a[~a["actie"].isin(["start_vanaf", "dag_weg"])]
    if len(bad):
        sys.exit(f"Onbekende actie in {ARTEFACTS.name}: {bad['actie'].tolist()} (gebruik start_vanaf of dag_weg)")
    return a


def find_artefact_candidates(close: pd.DataFrame, open_: pd.DataFrame, lr: pd.DataFrame,
                             names: dict) -> pd.DataFrame:
    """Sprongen die bijna volledig 'overnight' gebeuren, met een gewone beweging tijdens de dag.

    Ligt er een splitsing of dividend in de buurt, dan is het waarschijnlijk boekhouding en geen handel.
    """
    acts = pd.read_parquet(PROC / "actions.parquet") if (PROC / "actions.parquet").exists() else pd.DataFrame()
    rows = []
    for c in lr.columns:
        x = lr[c].dropna()
        big = x[x.abs() > JUMP]
        if big.empty:
            continue
        cl = close[c].dropna()
        for d, total in big.items():
            if d not in open_.index or pd.isna(open_.at[d, c]):
                continue
            prev = cl[cl.index < d]
            if prev.empty:
                continue
            overnight = np.log(open_.at[d, c] / prev.iloc[-1])
            intraday = np.log(close.at[d, c] / open_.at[d, c])
            if abs(intraday) > 0.10 or abs(overnight) < 0.8 * abs(total):
                continue
            near = []
            if not acts.empty:
                a = acts[(acts["isin"] == c) & ((acts["date"] - d).abs() <= pd.Timedelta(days=5))]
                if "stock_splits" in a and (a["stock_splits"] > 0).any():
                    near.append("splitsing")
                if "dividends" in a and (a["dividends"] > 0).any():
                    near.append("dividend")
            rows.append({"isin": c, "naam": names.get(c, ""), "datum": d.date(),
                         "totaal": f"{np.exp(total) - 1:+.0%}", "overnight": f"{np.exp(overnight) - 1:+.0%}",
                         "tijdens_dag": f"{np.exp(intraday) - 1:+.0%}",
                         "in_de_buurt": ", ".join(near) or "-",
                         "oordeel": "waarschijnlijk artefact" if near else "overnight sprong (nieuws?)"})
    return pd.DataFrame(rows)


def run_clean() -> None:
    CLEAN.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(DATA / "meta" / "tickers.csv", dtype=str)
    t = t[t["status"] == "ok"]
    tick2isin = dict(zip(t["ticker"], t["isin"]))
    names = dict(zip(t["isin"], t["name"]))
    close, open_ = _rp("close_eur"), _rp("open_eur")
    vol = _rp("volume").reindex(index=close.index, columns=close.columns)
    lr_raw = _rp("logret_eur")
    stats = {}

    # 1. onvolledige dagen
    inc = incomplete_cells(tick2isin)
    inc = inc[inc["isin"].isin(close.columns) & inc["date"].isin(close.index)]
    for d, i in zip(inc["date"], inc["isin"]):
        close.at[d, i] = np.nan
        open_.at[d, i] = np.nan
    stats["onvolledige dagen (tijdens de beurs opgehaald)"] = len(inc)

    # 2a. artefacten: start_vanaf
    art = load_artefacts()
    applied = []
    for _, a in art[art["actie"] == "start_vanaf"].iterrows():
        if a["isin"] in close.columns:
            n = int(close.loc[close.index < a["datum"], a["isin"]].notna().sum())
            close.loc[close.index < a["datum"], a["isin"]] = np.nan
            open_.loc[open_.index < a["datum"], a["isin"]] = np.nan
            applied.append(f"{names.get(a['isin'], a['isin'])}: {n} dagen voor {a['datum'].date()} weg")

    # 3. geen volume en ongewijzigde koers
    same = close.eq(close.ffill().shift(1))
    novol = (vol == 0) & same
    stats["dagen zonder handel (volume 0, koers ongewijzigd)"] = int(novol.sum().sum())
    close = close.mask(novol)

    # 5. rendementen op de eigen geldige dagen
    lr = log_returns(close)

    # 2b. artefacten: dag_weg (het rendement van die dag, niet de koers)
    for _, a in art[art["actie"] == "dag_weg"].iterrows():
        if a["isin"] in lr.columns and a["datum"] in lr.index:
            old = lr.at[a["datum"], a["isin"]]
            lr.at[a["datum"], a["isin"]] = np.nan
            applied.append(f"{names.get(a['isin'], a['isin'])}: rendement {a['datum'].date()} "
                           f"({np.exp(old) - 1:+.0%}) weg" if pd.notna(old) else
                           f"{names.get(a['isin'], a['isin'])}: {a['datum'].date()} had geen rendement")

    # 4. eerste dagen na beursintroductie (alleen als de reeks echt later begint dan de data)
    first_day = close.index.min()
    n_ipo = 0
    for c in lr.columns:
        s = close[c].dropna()
        if len(s) and s.index[0] > first_day + timedelta(days=10):
            idx = lr[c].dropna().index[:IPO_SKIP_DAYS]
            lr.loc[idx, c] = np.nan
            n_ipo += 1
    stats[f"beursintroducties (eerste {IPO_SKIP_DAYS} dagen weg)"] = n_ipo

    # marktreeksen: log-rendement voor prijzen, verandering in niveau voor rentes
    mk = _rp("market_close")
    mk_ret = pd.DataFrame(index=mk.index)
    for c in mk.columns:
        s = mk[c].dropna()
        r = s.diff() if c in RATE_SERIES else np.log(s[s > 0]).diff()
        mk_ret[c] = r.reindex(mk.index)

    # 6. kandidaten voor nieuwe artefacten
    cand = find_artefact_candidates(close, open_, lr, names)
    known = set(zip(art["isin"], art["datum"].dt.date)) if len(art) else set()
    if len(cand):
        cand = cand[[(i, d) not in known for i, d in zip(cand["isin"], cand["datum"])]]

    # beschikbaarheidszone per kolom
    ccy = dict(zip(t["ticker"], t["currency"]))
    zone = pd.Series({tick2isin[k]: zone_of(k, ccy.get(k, "EUR")) for k in tick2isin
                      if tick2isin[k] in close.columns}, name="zone")

    close.to_parquet(CLEAN / "close_eur.parquet")
    open_.to_parquet(CLEAN / "open_eur.parquet")
    lr.to_parquet(CLEAN / "logret.parquet")
    mk.to_parquet(CLEAN / "market_level.parquet")
    mk_ret.to_parquet(CLEAN / "market_ret.parquet")
    zone.to_frame().to_csv(CLEAN / "zones.csv")
    cand.to_csv(DATA / "artefact_kandidaten.csv", index=False)
    prepare_earnings()
    sh_stats = prepare_shares_and_short()

    removed = int(lr_raw.notna().sum().sum() - lr.notna().sum().sum())
    print("\n================ OPSCHONEN ================")
    for k, v in stats.items():
        print(f"{k:<52} {v}")
    print(f"{'artefacten toegepast uit artefacten.csv':<52} {len(applied)}")
    for a in applied:
        print(f"   - {a}")
    print(f"{'rendementen verwijderd in totaal':<52} {removed} van {int(lr_raw.notna().sum().sum())}")
    print(f"\nNieuwe artefact-kandidaten: {len(cand)} (volledige lijst: data/artefact_kandidaten.csv)")
    if len(cand):
        likely = cand[cand["oordeel"] == "waarschijnlijk artefact"]
        show = pd.concat([likely, cand[cand["oordeel"] != "waarschijnlijk artefact"]]).head(15)
        print(show[["naam", "datum", "totaal", "overnight", "tijdens_dag", "in_de_buurt", "oordeel"]]
              .to_string(index=False))
        print("\nEen echt artefact? Voeg het toe aan artefacten.csv (dag_weg) en draai clean.py opnieuw.")
    for k, v in sh_stats.items():
        print(f"{k:<52} {v}")
    print(f"\nOpgeschoonde data: {lr.shape[1]} effecten, {lr.notna().any(axis=1).sum()} dagen -> data/clean/")
    print("===========================================\n")


# --------------------------------------------------------------------------- point-in-time
class PointInTime:
    """Geeft alleen data terug die op tijdstip t (Belgische tijd) bekend was.

    backtest=True: momentopnames (info, huidige analistenverwachtingen) zijn verboden.
    """

    def __init__(self, backtest: bool = True, data: dict | None = None):
        self.backtest = backtest
        if data is None:
            data = {"close": pd.read_parquet(CLEAN / "close_eur.parquet"),
                    "open": pd.read_parquet(CLEAN / "open_eur.parquet"),
                    "logret": pd.read_parquet(CLEAN / "logret.parquet"),
                    "market_level": pd.read_parquet(CLEAN / "market_level.parquet"),
                    "market_ret": pd.read_parquet(CLEAN / "market_ret.parquet")}
            z = pd.read_csv(CLEAN / "zones.csv", index_col=0)["zone"]
            data["zones"] = z.to_dict()
            for k in KNOWN_AT_TABLES:
                f = CLEAN / f"{k}.parquet"
                data[k] = pd.read_parquet(f) if f.exists() else pd.DataFrame()
        self.d = data

    def _mask(self, df: pd.DataFrame, t: pd.Timestamp, zones: dict) -> pd.DataFrame:
        out = df[df.index <= t.normalize() + pd.Timedelta(days=1)].copy()   # snel voorfilter
        for z in ZONES:
            cols = [c for c in out.columns if zones.get(c, "US") == z]
            if cols:
                ok = known_at(out.index, z) <= t
                out.loc[~ok, cols] = np.nan
        return out.dropna(how="all")

    def as_of(self, table: str, t) -> pd.DataFrame:
        if table in LIVE_ONLY and self.backtest:
            raise PermissionError(f"'{table}' is een momentopname van vandaag en mag niet in een backtest.")
        t = pd.Timestamp(t)
        if table == "earnings":
            return self.earnings_as_of(t)
        if table in ("shares", "short_interest"):
            e = self.d.get(table, pd.DataFrame())
            return e[e["known_at"] <= t].copy() if not e.empty else e
        if table.startswith("market"):
            return self._mask(self.d[table], t, MARKET_ZONE)
        return self._mask(self.d[table], t, self.d["zones"])


    def earnings_as_of(self, t: pd.Timestamp, ahead_days: int = 30) -> pd.DataFrame:
        """Gepubliceerde cijfers vanaf het moment van publicatie; komende datums tot ahead_days vooruit.

        Bedrijven kondigen hun publicatiedatum doorgaans enkele weken vooraf aan. We nemen aan dat een datum
        30 dagen vooraf bekend is; de cijfers zelf (gerapporteerde winst, verrassing) blijven tot dan leeg.
        """
        e = self.d.get("earnings", pd.DataFrame())
        if e.empty:
            return e
        past = e[e["known_at"] <= t].copy()
        coming = e[(e["known_at"] > t) & (e["known_at"] <= t + pd.Timedelta(days=ahead_days))].copy()
        for c in ("reported_eps", "surprisepct"):
            if c in coming:
                coming[c] = np.nan
        coming["gepubliceerd"] = False
        past["gepubliceerd"] = True
        return pd.concat([past, coming], ignore_index=True)


def prepare_earnings() -> None:
    """Publicatietijdstip omzetten van New Yorkse naar Belgische tijd; dat is het moment waarop het bekend is."""
    f = PROC / "earnings.parquet"
    if not f.exists():
        return
    e = pd.read_parquet(f)
    if e.empty or "earnings_ts" not in e:
        return
    ts = pd.to_datetime(e["earnings_ts"])
    e["known_at"] = (ts.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
                       .dt.tz_convert("Europe/Brussels").dt.tz_localize(None))
    e = e.dropna(subset=["known_at"])
    e.to_parquet(CLEAN / "earnings.parquet", index=False)


def split_factor_after(isin_dates: pd.DataFrame, actions: pd.DataFrame) -> np.ndarray:
    """Product van alle splitsingsverhoudingen NA elke datum: zo zet je een ruw aantal om naar de basis van
    de (voor splitsingen gecorrigeerde) koersen. Ruwe koers op dag s = gecorrigeerde koers x deze factor."""
    out = np.ones(len(isin_dates))
    if actions.empty or "stock_splits" not in actions:
        return out
    sp = actions[actions["stock_splits"] > 0][["isin", "date", "stock_splits"]]
    if sp.empty:
        return out
    by = {i: g for i, g in sp.groupby("isin")}
    for n, (i, d) in enumerate(zip(isin_dates["isin"], isin_dates["date"])):
        g = by.get(i)
        if g is not None:
            out[n] = float(np.prod(g.loc[g["date"] > d, "stock_splits"].to_numpy()))
    return out


def prepare_shares_and_short() -> dict:
    """Aantal aandelen (Yahoo) en shortposities (FINRA) -> data/clean/, met known_at en splitsingscorrectie."""
    stats = {}
    act = pd.read_parquet(PROC / "actions.parquet") if (PROC / "actions.parquet").exists() else pd.DataFrame()
    if not act.empty:
        act["date"] = pd.to_datetime(act["date"])

    f = PROC / "shares.parquet"
    if f.exists():
        sh = pd.read_parquet(f).dropna(subset=["isin"])
        sh = sh[pd.to_numeric(sh["shares"], errors="coerce") > 0].copy()
        sh["ts"] = pd.to_datetime(sh["ts"])
        sh = sh.sort_values(["isin", "ts"]).drop_duplicates(["isin", "ts"], keep="last")
        sh["shares_adj"] = sh["shares"].astype(float) * split_factor_after(
            sh[["isin", "ts"]].rename(columns={"ts": "date"}), act)
        # uitschieters: afwijking > 50% van de mediaan van de VORIGE 5 waarnemingen (alleen terugkijken)
        med = sh.groupby("isin")["shares_adj"].transform(lambda x: x.shift(1).rolling(5, min_periods=3).median())
        out = (np.log(sh["shares_adj"] / med).abs() > np.log(1.5)) & med.notna()
        stats["aantal aandelen: uitschieters verwijderd"] = int(out.sum())
        sh = sh[~out]
        sh["known_at"] = sh["ts"].dt.normalize() + pd.Timedelta(days=SHARES_LAG_DAYS)
        sh[["isin", "ts", "known_at", "shares", "shares_adj"]].to_parquet(CLEAN / "shares.parquet", index=False)
        stats["aantal aandelen: effecten met historiek"] = sh["isin"].nunique()

    f = PROC / "short_interest.parquet"
    if f.exists():
        si = pd.read_parquet(f).dropna(subset=["isin", "settlement_date", "short_qty"]).copy()
        si["settlement_date"] = pd.to_datetime(si["settlement_date"])
        si = si.sort_values(["isin", "settlement_date"])
        fac = split_factor_after(si[["isin", "settlement_date"]].rename(columns={"settlement_date": "date"}), act)
        si["short_adj"] = si["short_qty"] * fac
        si["adv_adj"] = si["avg_daily_volume"] * fac
        si["known_at"] = si["settlement_date"] + pd.offsets.BDay(SHORT_LAG_BDAYS)
        si[["isin", "settlement_date", "known_at", "short_qty", "short_adj", "adv_adj", "days_to_cover",
            "split_flag", "revision_flag"]].to_parquet(CLEAN / "short_interest.parquet", index=False)
        stats["shortposities: VS-aandelen / afrekendatums"] = f"{si['isin'].nunique()} / {si['settlement_date'].nunique()}"
    return stats


# --------------------------------------------------------------------------- lektest
def example_features(pit: PointInTime, t) -> pd.DataFrame:
    """Voorbeeldkenmerken zoals een model ze zou maken: volatiliteit 20 dagen en momentum 6-1 maanden."""
    lr = pit.as_of("logret", t)
    px = pit.as_of("close", t)
    vol20 = lr.tail(20).std() * np.sqrt(252)
    mom = np.log(px.ffill().iloc[-21] / px.ffill().iloc[-126]) if len(px) > 126 else pd.Series(dtype=float)
    vix = pit.as_of("market_level", t)["vix"].dropna().iloc[-1] if "vix" in pit.d["market_level"] else np.nan
    return pd.DataFrame({"vol20": vol20, "mom_6_1": mom, "vix": vix})


def run_leak_test(n_dates: int = 5, seed: int = 0) -> bool:
    """Wijzig alle data na t willekeurig; de kenmerken op t mogen niet veranderen."""
    base = PointInTime(backtest=True)
    rng = np.random.default_rng(seed)
    idx = base.d["logret"].index
    dates = rng.choice(idx[300:-5], size=n_dates, replace=False)
    ok = True
    for d in sorted(dates):
        for hh, label in ((18, "18u (na Euronext, voor VS)"), (23, "23u (na VS)")):
            t = pd.Timestamp(d) + pd.Timedelta(hours=hh)
            f1 = example_features(base, t)
            tampered = {}
            for k, v in base.d.items():
                if isinstance(v, pd.DataFrame) and isinstance(v.index, pd.DatetimeIndex):
                    w = v.copy()
                    fut = w.index > pd.Timestamp(d) - pd.Timedelta(days=1)   # vanaf de dag zelf alles wijzigen
                    w.loc[fut] = w.loc[fut] * rng.uniform(0.5, 1.5, size=w.loc[fut].shape)
                    tampered[k] = w
                else:
                    tampered[k] = v
            # wat op t al bekend was, terugzetten: zo test je precies de grens
            for k in ("close", "open", "logret", "market_level", "market_ret"):
                known = base.as_of(k, t)
                tampered[k].loc[known.index, known.columns] = tampered[k].loc[known.index, known.columns] \
                    .where(known.isna(), known)
            f2 = example_features(PointInTime(backtest=True, data=tampered), t)
            same = np.allclose(f1.fillna(-9).values, f2.fillna(-9).values, equal_nan=True)
            print(f"  {pd.Timestamp(d).date()} {label:<28} {'OK' if same else 'LEK'}")
            ok &= same
    try:
        base.as_of("info", pd.Timestamp(dates[0]))
        print("  momentopname in backtest: NIET geblokkeerd (LEK)")
        ok = False
    except PermissionError:
        print("  momentopname 'info' in backtest: geblokkeerd (OK)")
    print(f"  -> {'geslaagd' if ok else 'MISLUKT'}\n")
    return ok


def run_fixed_checks() -> bool:
    """Controles met verwachtingen die met de hand zijn vastgelegd, dus niet via as_of zelf.

    Op een gewone werkdag d:
      18u00  -> Euronext-aandelen hebben een koers voor d; dollar-aandelen (in euro) en de S&P 500 niet.
      23u00  -> de S&P 500 wel; dollar-aandelen in euro nog niet (wisselkoers nog niet definitief); bitcoin niet.
      d+1 01u30 -> dollar-aandelen in euro en bitcoin wel.
    Kwartaalcijfers: een publicatie om 16u30 New Yorkse tijd is om 22u00 Belgische tijd nog niet bekend, om 22u45 wel.
    """
    pit = PointInTime(backtest=True)
    close, zones = pit.d["close"], pit.d["zones"]
    eu = [c for c, z in zones.items() if z == "EU" and c in close.columns]
    usd = [c for c, z in zones.items() if z == "24" and c in close.columns]
    days = close.index[(close[eu].notna().sum(axis=1) > 10) & (close[usd].notna().sum(axis=1) > 10)]
    winter = days[(days.month == 1) & (days.day_of_week < 4)]       # januari: geen gedoe met zomertijd
    d = winter[len(winter) // 2] if len(winter) else days[len(days) // 2]
    ok = True

    def check(label, cond):
        nonlocal ok
        print(f"  {label:<70} {'OK' if cond else 'FOUT'}")
        ok &= bool(cond)

    def row(table, t):
        df = pit.as_of(table, t)
        return df.loc[d] if d in df.index else pd.Series(dtype=float)

    r18, r23, r01 = row("close", d + pd.Timedelta(hours=18)), row("close", d + pd.Timedelta(hours=23)), \
        row("close", d + pd.Timedelta(days=1, hours=1, minutes=30))
    m18 = row("market_level", d + pd.Timedelta(hours=18))
    m23 = row("market_level", d + pd.Timedelta(hours=23))
    print(f"Controledag: {d.date()}")
    check("18u00: Euronext-aandelen hebben een slotkoers", r18.reindex(eu).notna().any())
    check("18u00: geen enkel dollar-aandeel heeft al een euro-koers", r18.reindex(usd).isna().all())
    check("18u00: S&P 500 nog niet bekend", pd.isna(m18.get("sp500", np.nan)))
    check("23u00: S&P 500 wel bekend", pd.notna(m23.get("sp500", np.nan)) or "sp500" not in pit.d["market_level"])
    check("23u00: dollar-aandelen in euro nog niet (wisselkoers niet definitief)", r23.reindex(usd).isna().all())
    check("23u00: bitcoin nog niet", pd.isna(m23.get("bitcoin", np.nan)))
    check("d+1 01u30: dollar-aandelen in euro wel bekend", r01.reindex(usd).notna().any())

    e = pit.d.get("earnings", pd.DataFrame())
    if not e.empty:
        ny = pd.Timestamp(d) + pd.Timedelta(hours=16, minutes=30)
        fake = pd.DataFrame({"isin": ["TEST"], "earnings_ts": [ny], "reported_eps": [1.0], "surprisepct": [5.0]})
        fake["known_at"] = fake["earnings_ts"].dt.tz_localize("America/New_York").dt.tz_convert(
            "Europe/Brussels").dt.tz_localize(None)
        pit.d["earnings"] = pd.concat([e, fake], ignore_index=True)
        at22 = pit.as_of("earnings", d + pd.Timedelta(hours=22))
        at2245 = pit.as_of("earnings", d + pd.Timedelta(hours=22, minutes=45))
        t22 = at22[at22["isin"] == "TEST"]
        t2245 = at2245[at2245["isin"] == "TEST"]
        check("cijfers 16u30 NY: om 22u00 nog niet gepubliceerd, winst nog leeg",
              len(t22) == 1 and not t22["gepubliceerd"].iloc[0] and pd.isna(t22["reported_eps"].iloc[0]))
        check("cijfers 16u30 NY: om 22u45 wel gepubliceerd", len(t2245) == 1 and t2245["gepubliceerd"].iloc[0])
        nov = pd.Timestamp("2021-11-05 16:30").tz_localize("America/New_York").tz_convert("Europe/Brussels")
        check("zomertijdverschil: 16u30 NY op 5-11-2021 = 21u30 Belgische tijd", nov.hour == 21 and nov.minute == 30)

    si = pit.d.get("short_interest", pd.DataFrame())
    if not si.empty:
        r = si.iloc[len(si) // 2]
        s0 = r["settlement_date"]
        seen = lambda t: ((pit.as_of("short_interest", t)["isin"] == r["isin"]) &
                          (pit.as_of("short_interest", t)["settlement_date"] == s0)).any()
        check(f"short afrekendatum {s0.date()}: 7 kalenderdagen later nog niet bekend",
              not seen(s0 + pd.Timedelta(days=7, hours=12)))
        check(f"short afrekendatum {s0.date()}: 16 kalenderdagen later wel bekend",
              seen(s0 + pd.Timedelta(days=16)))
    sh = pit.d.get("shares", pd.DataFrame())
    if not sh.empty:
        r = sh.iloc[len(sh) // 2]
        vis = lambda t: (pit.as_of("shares", t)["ts"] == r["ts"]).any()
        check("aantal aandelen: 30 dagen na de datum nog niet bekend, 46 dagen erna wel",
              not vis(r["ts"] + pd.Timedelta(days=30)) and vis(r["ts"].normalize() + pd.Timedelta(days=46)))
    return ok


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        if not (CLEAN / "logret.parquet").exists():
            run_clean()
        print("\n================ LEKTEST ================")
        print("1. Vaste controles (met de hand vastgelegde verwachtingen):")
        a = run_fixed_checks()
        print("\n2. Toekomst wijzigen (kenmerken mogen niet veranderen):")
        b = run_leak_test()
        print(f"Samen: {'GESLAAGD' if a and b else 'MISLUKT'}\n")
        sys.exit(0 if a and b else 1)
    run_clean()


if __name__ == "__main__":
    main()
