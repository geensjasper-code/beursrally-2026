#!/usr/bin/env python3
"""   python data.py init --synthetic
Beursrally 2026 - stap 1: data.

Haalt alle data op via Yahoo Finance (yfinance), bewaart de ruwe downloads
ongewijzigd in data/raw/ en bouwt daaruit nette tabellen in data/processed/.

Gebruik
    python data.py init        # eerste keer: alles ophalen (15-30 min) + verwerken + rapport
    python data.py update      # dagelijks: nieuwe koersen + opties; wekelijks ook cijfers en splitsingen
    python data.py report      # alleen het kwaliteitsrapport opnieuw maken
    python data.py <taak>      # een losse taak: resolve | prices | actions | earnings | info | options | process

    --synthetic                # nepdata (fake_yf.py) om de pipeline zonder internet te testen

Een onderbroken run hervat automatisch: wat al binnen is, wordt niet opnieuw opgehaald.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- instellingen
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
META = DATA / "meta"

START = "2016-11-01"            # 10 jaar historiek
COMPETITION_END = "2027-01-22"  # einde wedstrijd (voor de keuze van optievervaldagen)
CHUNK = 50                      # tickers per koers-opvraging
PAUSE = float(os.environ.get("BR_PAUSE", "0.8"))      # seconden tussen opvragingen per ticker
BACKOFF = float(os.environ.get("BR_BACKOFF", "1"))    # schaal van de wachttijden na een fout (tests: 0)
MAX_CONSECUTIVE_FAILS = 12      # zoveel mislukte opvragingen na elkaar = waarschijnlijk geblokkeerd
OVERLAP_DAYS = 7                # bij een update de laatste dagen opnieuw ophalen (correcties van Yahoo)
WEEKLY_EVERY_DAYS = 7           # cijfers, splitsingen en alle opties: om de zoveel dagen

# Markt- en regime-reeksen (Yahoo-ticker -> korte naam)
MARKET = {
    "EURUSD=X": "eurusd",
    "^GSPC": "sp500",
    "^NDX": "nasdaq100",
    "^STOXX": "stoxx600",
    "IWDA.AS": "msci_world",
    "^VIX": "vix",
    "^VXN": "vxn",
    "BTC-USD": "bitcoin",
}
US_EXCHANGES = {"NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "BTS", "NYS", "NAS", "NasdaqGS", "NYSE", "NasdaqGM"}
EU_EXCHANGES = {"AMS", "PAR", "BRU"}

TODAY = date.today().strftime("%Y%m%d")


class BlockedError(RuntimeError):
    """Te veel mislukte opvragingen na elkaar: Yahoo blokkeert waarschijnlijk tijdelijk."""


def get_yf(synthetic: bool):
    if synthetic:
        import fake_yf as yf
    else:
        try:
            import yfinance as yf
        except ImportError:
            sys.exit("Installeer eerst de afhankelijkheden:  pip install -r requirements.txt")
    return yf


# --------------------------------------------------------------------------- hulpfuncties
def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_state() -> dict:
    f = META / "state.json"
    return json.loads(f.read_text()) if f.exists() else {}


def save_state(state: dict) -> None:
    META.mkdir(parents=True, exist_ok=True)
    (META / "state.json").write_text(json.dumps(state, indent=2, default=str))


class Fetcher:
    """Roept Yahoo aan met pauzes, opnieuw proberen en detectie van een blokkade."""

    def __init__(self, pause: float = PAUSE):
        self.pause = pause
        self.fails_in_row = 0

    def call(self, fn, *args, tries: int = 4, empty_is_fail: bool = False, **kw):
        last_err = None
        for k in range(tries):
            try:
                out = fn(*args, **kw)
                empty = out is None or (hasattr(out, "empty") and out.empty) or (isinstance(out, (dict, tuple, list)) and len(out) == 0)
                if empty and empty_is_fail:
                    raise ValueError("leeg resultaat")
                self.fails_in_row = 0
                time.sleep(self.pause)
                return out
            except Exception as e:  # noqa: BLE001 - yfinance gooit allerlei fouten
                last_err = e
                time.sleep(min(2 ** (k + 1), 30) * BACKOFF)
        self.fails_in_row += 1
        if self.fails_in_row >= MAX_CONSECUTIVE_FAILS:
            raise BlockedError(
                f"{self.fails_in_row} opvragingen na elkaar mislukt (laatste fout: {last_err}).\n"
                "Yahoo blokkeert waarschijnlijk tijdelijk. Wacht een uur en draai hetzelfde commando opnieuw;\n"
                "wat al binnen is, wordt overgeslagen.")
        return None


def universe() -> pd.DataFrame:
    u = pd.read_csv(HERE / "universe.csv", dtype=str)
    return u[u["type"].isin(["stock", "etf"])].reset_index(drop=True)


def tickers_table() -> pd.DataFrame:
    f = META / "tickers.csv"
    if not f.exists():
        sys.exit("Eerst de tickers oplossen:  python data.py resolve")
    return pd.read_csv(f, dtype=str)


def ok_tickers(t: pd.DataFrame) -> pd.DataFrame:
    return t[t["status"] == "ok"]


# --------------------------------------------------------------------------- 1. tickers
def task_resolve(yf, force: bool = False) -> None:
    """Controleert elke ticker (beurs, munt) en zoekt ontbrekende op via ISIN."""
    META.mkdir(parents=True, exist_ok=True)
    out_f = META / "tickers.csv"
    done = pd.read_csv(out_f, dtype=str).set_index("isin") if out_f.exists() and not force else None
    u = universe()
    fx = Fetcher()
    rows = []
    log(f"Tickers controleren voor {len(u)} effecten ...")
    for _, r in u.iterrows():
        if done is not None and r["isin"] in done.index and done.at[r["isin"], "status"] == "ok":
            rows.append(done.loc[r["isin"]].to_dict() | {"isin": r["isin"]})
            continue
        cands = [r["ticker"]] if isinstance(r["ticker"], str) and r["ticker"] else []
        res = fx.call(lambda: yf.Search(r["isin"], max_results=10).quotes)
        for q in res or []:
            s = q.get("symbol")
            if s and s not in cands:
                cands.append(s)
        found, fetch_failed = None, res is None
        for t in cands:
            info = fx.call(_fast_info, yf, t)
            if info is None:                       # opvraging mislukt: niet stilletjes een andere notering kiezen
                fetch_failed = True
                break
            if not info.get("currency") or not info.get("lastPrice"):
                continue
            exch = str(info.get("exchange", ""))
            # voorkeur: de beurs die bij de munt van de Beursrally past
            if r["ccy"] == "EUR" and exch not in EU_EXCHANGES and t != cands[-1]:
                continue
            found = (t, info["currency"], exch, info.get("quoteType", ""))
            break
        if found:
            rows.append({"isin": r["isin"], "name": r["name"], "type": r["type"], "ccy_beursrally": r["ccy"],
                         "ticker": found[0], "currency": found[1], "exchange": found[2],
                         "quote_type": found[3], "status": "ok"})
        else:
            status = "opvraging mislukt" if fetch_failed else "niet gevonden"
            rows.append({"isin": r["isin"], "name": r["name"], "type": r["type"], "ccy_beursrally": r["ccy"],
                         "ticker": cands[0] if cands else "", "currency": "", "exchange": "",
                         "quote_type": "", "status": status})
            log(f"  {status}: {r['name']} ({r['isin']})")
    t = pd.DataFrame(rows)
    dup = t["status"].eq("ok") & t["ticker"].duplicated(keep="first")
    if dup.any():                                          # twee ISINs op dezelfde ticker: tweede niet gebruiken
        for _, r in t[dup].iterrows():
            log(f"  dubbele ticker {r['ticker']} voor {r['name']} ({r['isin']}): nakijken in universe.csv")
        t.loc[dup, "status"] = "dubbele ticker"
    t.to_csv(out_f, index=False)
    bad = t[t["status"] != "ok"]
    mism = t[(t["status"] == "ok") & (t["currency"] != t["ccy_beursrally"])]
    log(f"{(t['status'] == 'ok').sum()} van {len(t)} tickers ok -> data/meta/tickers.csv")
    nf, ff = (bad["status"] == "niet gevonden").sum(), (bad["status"] == "opvraging mislukt").sum()
    if nf:
        log(f"{nf} niet gevonden. Vul de juiste Yahoo-ticker in universe.csv in en draai 'resolve' opnieuw.")
    if ff:
        log(f"{ff} opvragingen mislukt. Draai 'resolve' later opnieuw; die worden dan opnieuw geprobeerd.")
    if len(mism):
        log(f"Let op: {len(mism)} effecten noteren bij Yahoo in een andere munt dan op de Beursrally:")
        for _, r in mism.iterrows():
            log(f"  {r['name']}: Yahoo {r['currency']} ({r['ticker']}), Beursrally {r['ccy_beursrally']}")


def _fast_info(yf, t: str) -> dict:
    """Alleen de velden die we nodig hebben: fast_info haalt elke waarde apart (lui) op."""
    fi = yf.Ticker(t).fast_info
    out = {}
    for k in ("currency", "exchange", "quoteType", "lastPrice"):
        try:
            out[k] = fi[k]
        except Exception:  # noqa: BLE001
            out[k] = None
    return out


# --------------------------------------------------------------------------- 2. koersen
def _download_long(yf, tickers: list[str], start: str) -> pd.DataFrame:
    """yfinance.download -> lange tabel (date, ticker, open, high, low, close, adj_close, volume).

    auto_adjust=False: 'Close' is dan alleen voor splitsingen gecorrigeerd, niet voor dividenden.
    """
    df = yf.download(tickers, start=start, auto_adjust=False, actions=False, progress=False,
                     group_by="column", threads=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if not isinstance(df.columns, pd.MultiIndex):           # een enkele ticker
        df.columns = pd.MultiIndex.from_product([df.columns, tickers])
    long = df.stack(level=1, future_stack=True).reset_index()
    long.columns = [str(c).lower().replace(" ", "_") for c in long.columns]
    long = long.rename(columns={"price": "ticker", "level_1": "ticker", "date": "date"})
    if "ticker" not in long.columns:                       # naam van het tickerniveau verschilt per versie
        long = long.rename(columns={long.columns[1]: "ticker"})
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None).dt.normalize()
    long = long.dropna(subset=["close"])
    keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    return long[[c for c in keep if c in long.columns]]


def task_prices(yf, full: bool = False, only: list[str] | None = None) -> None:
    """Koersen ophalen. Eerste keer of full=True: vanaf START; anders vanaf de laatste datum - OVERLAP_DAYS."""
    t = ok_tickers(tickers_table())
    tickers = only or (t["ticker"].tolist() + list(MARKET))
    state = load_state()
    last = state.get("prices_last_date")
    start = START if (full or not last or only) else (pd.Timestamp(last) - timedelta(days=OVERLAP_DAYS)).strftime("%Y-%m-%d")
    out_dir = RAW / "prices"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{TODAY}_{'refetch' if only else 'full' if start == START else 'upd'}"
    # Hervatten: welke tickers zitten al in de downloads van vandaag met dezelfde startdatum?
    run_start = pd.Timestamp.now()
    # Alleen de lange volledige download hervatten; een update (8 opvragingen) gewoon opnieuw doen.
    resume = tag.endswith("_full")
    got = [pd.read_parquet(f) for f in sorted(out_dir.glob(f"{tag}_*.parquet"))] if resume else []
    have = set(pd.concat(got)["ticker"]) if got else set()
    todo = [x for x in tickers if x not in have]
    if have:
        log(f"Koersen: {len(have)} tickers al binnen vandaag, {len(todo)} nog op te halen.")
    fx = Fetcher(pause=PAUSE * 2.5)
    n_files = len(list(out_dir.glob(f"{TODAY}_*.parquet")))
    failed = 0
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        log(f"Koersen {i + 1}-{i + len(chunk)} van {len(todo)} (vanaf {start}) ...")
        df = fx.call(_download_long, yf, chunk, start, empty_is_fail=True)
        if df is None or df.empty:
            log("  mislukt; wordt bij de volgende run opnieuw geprobeerd")
            failed += 1
            continue
        df["downloaded"] = pd.Timestamp.now()
        df.to_parquet(out_dir / f"{tag}_{n_files:03d}_{run_start:%H%M%S}.parquet", index=False)
        n_files += 1
        got.append(df)
    if not got:
        return
    allp = pd.concat(got)
    missing = sorted(set(tickers) - set(allp["ticker"]))
    if missing:
        log(f"Geen koersen voor {len(missing)} tickers: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if failed:
        log(f"{failed} groepjes mislukt: draai hetzelfde commando opnieuw om aan te vullen.")
    if start != START and not only:
        changed = _changed_history(allp, before=run_start)
        if changed:
            log(f"Oude koersen gewijzigd bij Yahoo (splitsing?) voor {changed}: volledige historiek opnieuw ophalen.")
            task_prices(yf, full=True, only=changed)
    if not only and not failed:
        state["prices_last_date"] = str(allp["date"].max().date())
        if start == START:
            state["prices_full_date"] = TODAY
        save_state(state)
    log(f"Koersen binnen tot {allp['date'].max().date()}.")


def _changed_history(new: pd.DataFrame, before: pd.Timestamp, tol: float = 0.01) -> list[str]:
    """Tickers waarvan de overlappende dagen in de nieuwe download meer dan tol afwijken van de vorige downloads.

    Gebeurt na een splitsing: Yahoo herschaalt dan de hele historiek, en oude en nieuwe koersen passen niet meer.
    """
    out_dir = RAW / "prices"
    old_files = sorted(out_dir.glob("*.parquet"))
    if not old_files:
        return []
    old = pd.concat([pd.read_parquet(f, columns=["date", "ticker", "close", "downloaded"]) for f in old_files])
    old = old[old["downloaded"] < before]                  # alleen wat er voor deze run al was
    if old.empty:
        return []
    old = old.sort_values("downloaded").drop_duplicates(["ticker", "date"], keep="last")
    m = new[["date", "ticker", "close"]].merge(old[["date", "ticker", "close"]], on=["date", "ticker"],
                                                suffixes=("_new", "_old"))
    if m.empty:
        return []
    dev = np.log(m["close_new"] / m["close_old"]).abs().groupby(m["ticker"]).median()
    return sorted(dev[dev > tol].index)


# --------------------------------------------------------------------------- 3-5. per-ticker data
def _per_ticker(yf, kind: str, tickers: list[str], fn, refresh_days: int) -> None:
    """Haalt per ticker iets op en bewaart het als data/raw/<kind>/<datum>/<ticker>.parquet (hervatbaar)."""
    base = RAW / kind
    prev = sorted(p for p in base.glob("*") if p.is_dir()) if base.exists() else []
    if prev and refresh_days and (datetime.now() - datetime.strptime(prev[-1].name, "%Y%m%d")).days < refresh_days \
            and prev[-1].name != TODAY:
        log(f"{kind}: laatst opgehaald {prev[-1].name}, nog niet aan vernieuwing toe.")
        return
    d = base / TODAY
    d.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tickers if not (d / f"{_safe(t)}.parquet").exists() and not (d / f"{_safe(t)}.empty").exists()]
    log(f"{kind}: {len(todo)} van {len(tickers)} tickers nog op te halen ...")
    fx = Fetcher()
    for n, t in enumerate(todo, 1):
        df = fx.call(fn, yf, t)
        if df is None:
            continue                                   # mislukt: volgende run opnieuw
        if df.empty:
            (d / f"{_safe(t)}.empty").touch()          # echt leeg (bv. geen opties): niet opnieuw vragen
        else:
            df.insert(0, "ticker", t)
            df["downloaded"] = pd.Timestamp.now()
            df.to_parquet(d / f"{_safe(t)}.parquet", index=False)
        if n % 50 == 0:
            log(f"  {n}/{len(todo)}")
    log(f"{kind}: klaar.")


def _safe(t: str) -> str:
    return t.replace("^", "_").replace("/", "_").replace("=", "_")


def _actions(yf, t):
    a = yf.Ticker(t).actions
    if a is None or a.empty:
        return pd.DataFrame()
    a = a.reset_index()
    a.columns = [str(c).lower().replace(" ", "_") for c in a.columns]
    a["date"] = pd.to_datetime(a["date"]).dt.tz_localize(None).dt.normalize()
    return a


def _earnings(yf, t):
    e = yf.Ticker(t).get_earnings_dates(limit=40)
    if e is None or e.empty:
        return pd.DataFrame()
    e = e.reset_index()
    e.columns = [str(c).lower().replace(" ", "_").replace("(%)", "pct") for c in e.columns]
    col = [c for c in e.columns if "date" in c][0]
    ts = pd.to_datetime(e[col])
    e["earnings_ts"] = ts.dt.tz_convert("America/New_York").dt.tz_localize(None) if ts.dt.tz is not None else ts
    # voor 12u NY-tijd = voor de opening; anders na de sluiting (reactie de volgende beursdag)
    e["timing"] = np.where(e["earnings_ts"].dt.hour < 12, "voor_opening", "na_sluiting")
    return e.drop(columns=[col])


def _info(yf, t):
    i = yf.Ticker(t).info or {}
    keys = ["longName", "quoteType", "sector", "industry", "country", "marketCap", "averageVolume",
            "sharesOutstanding", "floatShares", "shortPercentOfFloat", "beta", "currency", "exchange"]
    return pd.DataFrame([{k: i.get(k) for k in keys}])


def _options(yf, t):
    """Momentopname: opties rond de koers voor de vervaldagen net voor en net na het einde van de wedstrijd."""
    tk = yf.Ticker(t)
    exps = list(tk.options or [])
    if not exps:
        return pd.DataFrame()
    end = pd.Timestamp(COMPETITION_END)
    ex = pd.to_datetime(pd.Series(exps))
    before = ex[ex <= end]
    after = ex[ex > end]
    chosen = ([before.max()] if len(before) else []) + ([after.min()] if len(after) else [])
    try:
        spot = tk.fast_info["lastPrice"]
    except Exception:  # noqa: BLE001
        spot = None
    rows = []
    for e in chosen:
        ch = tk.option_chain(e.strftime("%Y-%m-%d"))
        for side, df in (("call", ch.calls), ("put", ch.puts)):
            if df is None or df.empty:
                continue
            df = df.copy()
            df["side"], df["expiry"], df["spot"] = side, e, spot
            if spot:
                df = df[(df["strike"] / spot).between(0.7, 1.3)]       # alleen rond de koers bewaren
            rows.append(df)
        time.sleep(PAUSE / 2)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    keep = ["side", "expiry", "spot", "strike", "lastPrice", "bid", "ask", "volume", "openInterest",
            "impliedVolatility", "lastTradeDate"]
    return out[[c for c in keep if c in out.columns]]


def us_stocks(t: pd.DataFrame) -> list[str]:
    s = ok_tickers(t)
    s = s[(s["type"] == "stock") & (s["currency"] == "USD") & (~s["ticker"].str.contains(r"\."))]
    return s["ticker"].tolist()


def focus_list(t: pd.DataFrame) -> list[str]:
    """Tickers voor de dagelijkse optie-momentopname: data/meta/focus.txt (een per lijn), anders alle VS-aandelen."""
    f = META / "focus.txt"
    if f.exists():
        wanted = {l.strip() for l in f.read_text().splitlines() if l.strip() and not l.startswith("#")}
        return [x for x in us_stocks(t) if x in wanted]
    return us_stocks(t)


def task_actions(yf, refresh_days=0):
    _per_ticker(yf, "actions", ok_tickers(tickers_table())["ticker"].tolist(), _actions, refresh_days)


def task_earnings(yf, refresh_days=0):
    t = ok_tickers(tickers_table())
    _per_ticker(yf, "earnings", t[t["type"] == "stock"]["ticker"].tolist(), _earnings, refresh_days)


def task_info(yf, refresh_days=0):
    _per_ticker(yf, "info", ok_tickers(tickers_table())["ticker"].tolist(), _info, refresh_days)


def task_options(yf, focus: bool = False):
    t = tickers_table()
    _per_ticker(yf, "options", focus_list(t) if focus else us_stocks(t), _options, 0)


# --------------------------------------------------------------------------- 6. verwerken
def _latest_per_ticker(kind: str) -> pd.DataFrame:
    """Voegt alle downloads van een soort samen; per ticker telt de recentste download."""
    files = sorted((RAW / kind).glob("*/*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    last = df.groupby("ticker")["downloaded"].transform("max")
    return df[df["downloaded"] == last]


def task_process() -> None:
    t = ok_tickers(tickers_table())
    files = sorted((RAW / "prices").glob("*.parquet"))
    if not files:
        sys.exit("Nog geen koersen. Eerst:  python data.py prices")
    log(f"Koersen verwerken uit {len(files)} downloads ...")
    p = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # Per ticker: is er een volledige download, dan geldt die (Yahoo past oude koersen aan na een splitsing);
    # daarna vullen latere updates de recentere dagen aan.
    p = p.sort_values("downloaded").drop_duplicates(["ticker", "date"], keep="last")
    PROC.mkdir(parents=True, exist_ok=True)

    tick2isin = dict(zip(t["ticker"], t["isin"]))
    ccy = dict(zip(t["isin"], t["currency"]))

    def wide(col, tickers):
        w = p[p["ticker"].isin(tickers)].pivot(index="date", columns="ticker", values=col).sort_index()
        return w

    inst = wide("close", list(tick2isin)).rename(columns=tick2isin)
    # gemeenschappelijke kalender = alle dagen waarop minstens een effect verhandeld werd (geen weekends van bitcoin)
    cal = inst.index[inst.notna().sum(axis=1) >= max(5, 0.05 * inst.shape[1])]

    mk = wide("close", list(MARKET)).rename(columns=MARKET)
    # wisselkoers: vooruit invullen mag hier, het is alleen een omrekening en geen rendement
    eurusd = mk["eurusd"].reindex(cal.union(mk.index)).ffill().reindex(cal)
    mk = mk.reindex(cal)                                    # indices: NaN op dagen dat hun beurs dicht is

    def to_eur(w):
        w = w.reindex(cal)
        for c in w.columns:
            if ccy.get(c) == "USD":
                w[c] = w[c] / eurusd
            elif ccy.get(c) not in ("EUR", None, ""):
                log(f"  let op: munt {ccy.get(c)} voor {c} niet omgezet")
        return w

    close_local = inst.reindex(cal)
    close_eur = to_eur(inst.copy())
    open_eur = to_eur(wide("open", list(tick2isin)).rename(columns=tick2isin))
    high = wide("high", list(tick2isin)).rename(columns=tick2isin).reindex(cal)
    low = wide("low", list(tick2isin)).rename(columns=tick2isin).reindex(cal)
    vol = wide("volume", list(tick2isin)).rename(columns=tick2isin).reindex(cal)

    # Log-rendementen per effect op zijn EIGEN handelsdagen: geen ingevulde 0%-dagen.
    # Na een gesloten dag loopt het rendement van de vorige koers tot de volgende.
    lr = pd.DataFrame(index=cal, columns=close_eur.columns, dtype=float)
    for c in close_eur.columns:
        s = close_eur[c].dropna()
        lr.loc[s.index[1:], c] = np.log(s.values[1:] / s.values[:-1])
    mk_lr = pd.DataFrame(index=cal, columns=mk.columns, dtype=float)
    for c in mk.columns:
        s = mk[c].dropna()
        mk_lr.loc[s.index[1:], c] = np.log(s.values[1:] / s.values[:-1])

    for name, df in [("close_eur", close_eur), ("open_eur", open_eur), ("close_local", close_local),
                     ("high_local", high), ("low_local", low), ("volume", vol), ("logret_eur", lr),
                     ("market_close", mk), ("market_logret", mk_lr)]:
        df.to_parquet(PROC / f"{name}.parquet")

    for kind in ("actions", "earnings", "info"):
        d = _latest_per_ticker(kind)
        if not d.empty:
            d.insert(0, "isin", d["ticker"].map(tick2isin))
            d.to_parquet(PROC / f"{kind}.parquet", index=False)
    opt_days = sorted(p.name for p in (RAW / "options").glob("*") if p.is_dir()) if (RAW / "options").exists() else []
    for day in opt_days:                                   # een bestand per momentopname
        fs = list((RAW / "options" / day).glob("*.parquet"))
        if fs:
            o = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
            o.insert(0, "isin", o["ticker"].map(tick2isin))
            o.to_parquet(PROC / f"options_{day}.parquet", index=False)
    log(f"Verwerkt: {close_eur.shape[1]} effecten, {len(cal)} beursdagen "
        f"({cal.min().date()} tot {cal.max().date()}) -> data/processed/")


# --------------------------------------------------------------------------- 7. kwaliteitsrapport
def task_report() -> pd.DataFrame:
    t = tickers_table()
    close = pd.read_parquet(PROC / "close_eur.parquet")
    lr = pd.read_parquet(PROC / "logret_eur.parquet")
    act = pd.read_parquet(PROC / "actions.parquet") if (PROC / "actions.parquet").exists() else pd.DataFrame()
    earn = pd.read_parquet(PROC / "earnings.parquet") if (PROC / "earnings.parquet").exists() else pd.DataFrame()
    opt_files = sorted(PROC.glob("options_*.parquet"))
    opt = pd.read_parquet(opt_files[-1]) if opt_files else pd.DataFrame()
    last_day = close.index.max()
    jump = math.log(1.5)
    rows = []
    for _, r in t.iterrows():
        i = r["isin"]
        row = {"isin": i, "name": r["name"], "ticker": r["ticker"], "status": r["status"], "currency": r.get("currency")}
        if r["status"] != "ok" or i not in close.columns or close[i].notna().sum() == 0:
            row["probleem"] = "geen koersen"
            rows.append(row)
            continue
        s = close[i].dropna()
        x = lr[i].dropna()
        bdays = np.busday_count(s.index.min().date(), s.index.max().date()) + 1
        row.update({
            "eerste_dag": s.index.min().date(), "laatste_dag": s.index.max().date(),
            "jaren": round(len(s) / 252, 1), "dekking": round(len(s) / (bdays * 250 / 261), 2),
            "max_dagrendement": round(float(x.abs().max()), 3) if len(x) else np.nan,
        })
        problems = []
        if (last_day - s.index.max()).days > 5:
            problems.append("verouderd")
        if row["dekking"] < 0.9:
            problems.append("veel ontbrekende dagen")
        if len(s) < 60:
            problems.append("minder dan 60 dagen")
        big = x[x.abs() > jump]
        splits = act.loc[(act["isin"] == i) & (act["stock_splits"] > 0), "date"].tolist() \
            if not act.empty and "stock_splits" in act else []
        for d in big.index:
            near = any(abs((d - sd).days) <= 3 for sd in splits)
            problems.append(f"{'SPLITSING NIET GECORRIGEERD' if near else 'sprong'} {d.date()} ({math.exp(big[d]) - 1:+.0%})")
        if not earn.empty and r["type"] == "stock":
            e = earn[earn["isin"] == i]
            row["kwartaalcijfers"] = len(e)
            fut = e[e["earnings_ts"] > pd.Timestamp.now()]["earnings_ts"]
            row["volgende_cijfers"] = fut.min().date() if len(fut) else None
        if not opt.empty:
            row["opties"] = int((opt["isin"] == i).sum())
        row["probleem"] = "; ".join(problems)
        rows.append(row)
    rep = pd.DataFrame(rows)
    rep.to_csv(DATA / "kwaliteitsrapport.csv", index=False)

    ok = rep[rep["probleem"].fillna("") == ""]
    print("\n================ KWALITEITSRAPPORT ================")
    print(f"Effecten in universe:        {len(t)}")
    print(f"Met koersen:                 {(rep['probleem'] != 'geen koersen').sum()}")
    print(f"Zonder opmerkingen:          {len(ok)}")
    print(f"Laatste beursdag in de data: {last_day.date()}")
    hist = rep["jaren"].dropna()
    if len(hist):
        print(f"Historiek: mediaan {hist.median():.1f} jaar, {int((hist < 1).sum())} effecten korter dan 1 jaar")
    if "kwartaalcijfers" in rep:
        st = rep[t["type"].values == "stock"] if len(rep) == len(t) else rep
        print(f"Kwartaalcijfers: {int((st['kwartaalcijfers'].fillna(0) > 0).sum())} van {len(st)} aandelen; "
              f"{int(st['volgende_cijfers'].notna().sum())} met een datum in de toekomst")
    if "opties" in rep:
        print(f"Opties (laatste momentopname): {int((rep['opties'].fillna(0) > 0).sum())} aandelen")
    prob = rep[rep["probleem"].fillna("") != ""]
    if len(prob):
        print(f"\n{len(prob)} effecten met opmerkingen (volledige lijst: data/kwaliteitsrapport.csv):")
        for _, r in prob.head(40).iterrows():
            print(f"  {str(r['name'])[:32]:<32} {str(r['ticker']):<10} {r['probleem']}")
        if len(prob) > 40:
            print(f"  ... en {len(prob) - 40} meer")
    print("====================================================\n")
    return rep


# --------------------------------------------------------------------------- samengestelde taken
def task_init(yf):
    task_resolve(yf)
    task_prices(yf, full=True)
    task_actions(yf)
    task_earnings(yf)
    task_info(yf)
    task_options(yf)
    task_process()
    task_report()
    st = load_state()
    st["weekly_last"] = TODAY
    save_state(st)


def task_update(yf):
    st = load_state()
    if not st.get("prices_last_date"):
        sys.exit("Nog geen data. Eerst:  python data.py init")
    task_prices(yf)
    weekly = not st.get("weekly_last") or \
        (datetime.now() - datetime.strptime(st["weekly_last"], "%Y%m%d")).days >= WEEKLY_EVERY_DAYS
    if weekly:
        log("Wekelijkse vernieuwing: splitsingen, kwartaalcijfers, alle opties.")
        task_actions(yf)
        task_earnings(yf)
        task_options(yf, focus=False)
        refetch_split_tickers(yf)
        st = load_state()
        st["weekly_last"] = TODAY
        save_state(st)
    else:
        task_options(yf, focus=True)
    task_process()
    task_report()


def refetch_split_tickers(yf):
    """Na een nieuwe splitsing past Yahoo de hele historiek aan: die ticker volledig opnieuw ophalen."""
    st = load_state()
    full = st.get("prices_full_date")
    if not full:
        return
    act = _latest_per_ticker("actions")
    if act.empty or "stock_splits" not in act:
        return
    new = act[(act["stock_splits"] > 0) & (act["date"] > pd.Timestamp(full))]["ticker"].unique().tolist()
    if new:
        log(f"Nieuwe splitsing bij {new}: volledige historiek opnieuw ophalen.")
        task_prices(yf, full=True, only=new)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["init", "update", "report", "resolve", "prices", "actions", "earnings",
                                     "info", "options", "process"])
    ap.add_argument("--synthetic", action="store_true", help="nepdata via fake_yf.py (test zonder internet)")
    ap.add_argument("--force", action="store_true", help="resolve: alle tickers opnieuw controleren")
    ap.add_argument("--full", action="store_true", help="prices: volledige historiek opnieuw ophalen")
    a = ap.parse_args()
    yf = get_yf(a.synthetic)
    try:
        {
            "init": lambda: task_init(yf),
            "update": lambda: task_update(yf),
            "report": task_report,
            "resolve": lambda: task_resolve(yf, force=a.force),
            "prices": lambda: task_prices(yf, full=a.full),
            "actions": lambda: task_actions(yf),
            "earnings": lambda: task_earnings(yf),
            "info": lambda: task_info(yf),
            "options": lambda: task_options(yf),
            "process": task_process,
        }[a.task]()
    except BlockedError as e:
        log(str(e))
        sys.exit(2)


if __name__ == "__main__":
    main()
