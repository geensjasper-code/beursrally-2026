"""
Nagemaakte yfinance voor tests zonder internet:  python br.py data init --synthetic

Geeft dezelfde vormen terug als yfinance (download, Ticker, Search) met willekeurige maar
reproduceerbare data, en bevat bewust een paar problemen om de controles te testen:
- een ISIN die niet gevonden wordt (Energyvision),
- af en toe een mislukte opvraging (zoals bij Yahoo),
- een splitsing die NIET gecorrigeerd is in de koersen (SMCI),
- recente beursintroducties (CRCL, CRWV, NBIS),
- aantal aandelen met een uitschieter en een splitsing (NVDA: ruw aantal x10 na 10-6-2024),
- FakeFinra: shortposities twee keer per maand vanaf juni 2021.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd

import os
FAIL_RATE = float(os.environ.get("FAKE_FAIL", "0.03"))
_rng_fail = np.random.default_rng(0)
_END = pd.Timestamp(os.environ.get("FAKE_END", "2026-11-06"))
_IPO = {"CRCL": "2025-06-05", "CRWV": "2025-03-28", "NBIS": "2024-10-21", "SNDK": "2025-02-24"}
_UNADJUSTED_SPLIT = {"SMCI": "2024-10-01"}
_ADJUSTED_SPLIT = {"NVDA": "2024-06-10"}
_SPLIT_AFTER_FULL = {"COIN": "2026-11-10"}   # splitsing NA de eerste download: Yahoo herschaalt de historiek
_NOT_FOUND_ISIN = {"BE0974499312"}


def _seed(t: str) -> int:
    return int(hashlib.md5(t.encode()).hexdigest()[:8], 16)


def _maybe_fail():
    if _rng_fail.random() < FAIL_RATE:
        raise ConnectionError("429 Too Many Requests (nep)")


_GEN_END = pd.Timestamp("2027-12-31")          # reeksen altijd tot hier maken, dan afknippen: consistent


def _calendar(t: str) -> pd.DatetimeIndex:
    days = pd.bdate_range("2014-01-01", _GEN_END)
    if t == "BTC-USD":
        return pd.date_range("2014-01-01", _GEN_END)
    eu = any(t.endswith(s) for s in (".AS", ".PA", ".BR"))
    hol = ["05-01", "12-26"] if eu else ["07-04", "11-26"]       # een paar beursspecifieke feestdagen
    days = days[~days.strftime("%m-%d").isin(hol + ["12-25", "01-01"])]
    if t in _IPO:
        days = days[days >= pd.Timestamp(_IPO[t])]
    return days


def _series(t: str) -> pd.DataFrame:
    rng = np.random.default_rng(_seed(t))
    idx = _calendar(t)
    vol = 0.008 if t.startswith("^") or t == "EURUSD=X" else rng.uniform(0.012, 0.045)
    r = np.clip(rng.standard_t(4, len(idx)) / np.sqrt(2) * vol, -0.25, 0.25)
    close = (1.1 if t == "EURUSD=X" else rng.uniform(5, 400)) * np.exp(np.cumsum(r))
    if t in _UNADJUSTED_SPLIT:                       # koers valt 90%: splitsing niet gecorrigeerd
        close = np.where(idx >= pd.Timestamp(_UNADJUSTED_SPLIT[t]), close / 10, close)
    gap = rng.normal(0, vol / 2, len(idx))
    open_ = close * np.exp(gap)
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, vol / 2, len(idx))))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, vol / 2, len(idx))))
    df = pd.DataFrame({"Open": open_, "High": hi, "Low": lo, "Close": close, "Adj Close": close * 0.97,
                       "Volume": rng.integers(1e4, 5e6, len(idx))}, index=idx)
    if t in _SPLIT_AFTER_FULL and _END < pd.Timestamp(_SPLIT_AFTER_FULL[t]):
        # voor een 4:1-splitsing noteert het aandeel 4x hoger; daarna herschaalt Yahoo de hele historiek
        df[["Open", "High", "Low", "Close"]] *= 4
    return df[df.index <= _END]


def download(tickers, start=None, auto_adjust=True, actions=False, progress=True, group_by="column",
             threads=True, **kw):
    _maybe_fail()
    tickers = [tickers] if isinstance(tickers, str) else list(tickers)
    frames = {}
    for t in tickers:
        df = _series(t)
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        frames[t] = df
    out = pd.concat(frames, axis=1)                 # kolommen (Ticker, Price)
    out = out.swaplevel(0, 1, axis=1).sort_index(axis=1)
    out.columns.names = ["Price", "Ticker"]
    out.index.name = "Date"
    return out


class _FastInfo(dict):
    pass


class Ticker:
    def __init__(self, t: str):
        self.t = t

    @property
    def fast_info(self):
        _maybe_fail()
        eu = any(self.t.endswith(s) for s in (".AS", ".PA", ".BR"))
        ex = {".AS": "AMS", ".PA": "PAR", ".BR": "BRU"}.get(self.t[-3:], "NMS")
        return _FastInfo(currency="EUR" if eu else "USD", exchange=ex, quoteType="EQUITY",
                         lastPrice=float(_series(self.t)["Close"].iloc[-1]))

    @property
    def actions(self):
        _maybe_fail()
        rows = []
        for d in pd.date_range("2017-03-15", _END, freq="QS-MAR")[::2]:
            rows.append((d, 0.25, 0.0))
        for m in (_UNADJUSTED_SPLIT, _ADJUSTED_SPLIT):
            if self.t in m:
                rows.append((pd.Timestamp(m[self.t]), 0.0, 10.0))
        df = pd.DataFrame(rows, columns=["Date", "Dividends", "Stock Splits"]).set_index("Date")
        df.index = df.index.tz_localize("America/New_York")
        return df

    def get_earnings_dates(self, limit=12):
        _maybe_fail()
        if any(self.t.endswith(s) for s in (".BR",)):
            return None                              # Europese cijfers ontbreken vaak
        rng = np.random.default_rng(_seed(self.t) + 1)
        ds = pd.date_range(_END - pd.DateOffset(years=4), _END + pd.DateOffset(months=4), freq="QS-FEB")[-limit:]
        ds = ds + pd.to_timedelta(rng.integers(0, 20, len(ds)), unit="D") + pd.to_timedelta(
            rng.choice([7, 16], len(ds)), unit="h")
        est = rng.normal(1, 0.5, len(ds))
        rep = est * (1 + rng.normal(0.03, 0.1, len(ds)))
        fut = ds > _END
        rep[fut] = np.nan
        df = pd.DataFrame({"EPS Estimate": est, "Reported EPS": rep,
                           "Surprise(%)": (rep / est - 1) * 100}, index=ds.tz_localize("America/New_York"))
        df.index.name = "Earnings Date"
        return df.sort_index(ascending=False)

    @property
    def info(self):
        _maybe_fail()
        rng = np.random.default_rng(_seed(self.t) + 2)
        return {"longName": self.t, "quoteType": "EQUITY", "sector": rng.choice(["Technology", "Financials", "Energy"]),
                "industry": "x", "country": "x", "marketCap": int(rng.uniform(1e9, 3e12)),
                "averageVolume": int(rng.uniform(1e5, 5e7)), "shortPercentOfFloat": rng.uniform(0, 0.2)}

    def get_shares_full(self, start=None, end=None):
        _maybe_fail()
        if self.t.startswith("^") or self.t in ("EURUSD=X", "BTC-USD"):
            return None
        idx = _calendar(self.t)
        idx = idx[(idx >= pd.Timestamp(start or "2016-01-01")) & (idx <= _END)]
        rng = np.random.default_rng(_seed(self.t) + 4)
        obs = idx[rng.random(len(idx)) < 0.08]             # onregelmatige waarnemingen, zoals bij Yahoo
        if len(obs) == 0:
            return None
        sh = _raw_shares(self.t, obs)
        if len(sh) > 20:
            sh[len(sh) // 2] *= 3                          # een foute waarneming (komt bij Yahoo voor)
        s = pd.Series(sh, index=obs.tz_localize("America/New_York") + pd.Timedelta(hours=9))
        return s

    @property
    def options(self):
        _maybe_fail()
        if "." in self.t or self.t.startswith("^"):
            return ()
        return ("2026-11-20", "2026-12-18", "2027-01-15", "2027-02-19", "2027-06-18")

    def option_chain(self, exp):
        _maybe_fail()
        spot = float(_series(self.t)["Close"].iloc[-1])
        strikes = spot * np.linspace(0.5, 1.5, 21)
        rng = np.random.default_rng(_seed(self.t) + 3)
        iv = rng.uniform(0.3, 1.2)

        def mk():
            return pd.DataFrame({"contractSymbol": [f"{self.t}{exp}{k:.0f}" for k in strikes], "strike": strikes,
                                 "lastPrice": spot * 0.1, "bid": spot * 0.09, "ask": spot * 0.11, "volume": 10,
                                 "openInterest": 100, "impliedVolatility": iv * (1 + 0.2 * np.abs(strikes / spot - 1)),
                                 "lastTradeDate": pd.Timestamp("2026-11-06")})
        return SimpleNamespace(calls=mk(), puts=mk())


class Search:
    def __init__(self, query, max_results=8):
        _maybe_fail()
        self.quotes = [] if query in _NOT_FOUND_ISIN else [{"symbol": _isin_to_symbol(query), "exchange": "X"}]


def _isin_to_symbol(isin: str) -> str:
    known = {"US15675D1037": "CBRS", "US74768A1043": "QNTM", "US78392B2060": "SKHY", "US84615Q1031": "SPCX",
             "NL0015073TS8": "CSG.AS"}
    return known.get(isin, "Z" + isin[2:9])


def _raw_shares(t: str, dates: pd.DatetimeIndex) -> np.ndarray:
    """Ruw aantal aandelen: langzaam veranderend, x10 na een (in de koersen gecorrigeerde) splitsing."""
    rng = np.random.default_rng(_seed(t) + 5)
    base = rng.uniform(5e7, 5e9)
    years = (dates - pd.Timestamp("2016-01-01")).days / 365.25
    sh = base * np.exp(rng.uniform(-0.03, 0.03) * years)
    if t in _ADJUSTED_SPLIT:
        sh = np.where(dates >= pd.Timestamp(_ADJUSTED_SPLIT[t]), sh * 10, sh)
    return np.asarray(sh, dtype=float)


class FakeFinra:
    """Nagemaakte FINRA-API: afrekendatums rond de 15e en op de laatste werkdag van elke maand."""

    def month(self, symbols, start, end):
        _maybe_fail()
        bd = pd.bdate_range(start, min(pd.Timestamp(end), _END - pd.Timedelta(days=10)))
        if len(bd) == 0:
            return []
        dates = sorted({bd[bd.day <= 15][-1], bd[-1]} if (bd.day <= 15).any() else {bd[-1]})
        dates = [d for d in dates if d >= pd.Timestamp("2021-06-01")]
        rows = []
        for sym in symbols:
            if sym.startswith("^"):
                continue
            for d in dates:
                ix = pd.DatetimeIndex([d - pd.Timedelta(days=14), d])
                sh = _raw_shares(sym, ix)
                r = np.random.default_rng(_seed(sym + str(d.date())))
                r0 = np.random.default_rng(_seed(sym + str((d - pd.Timedelta(days=14)).date())))
                pct = 0.02 + 0.10 * (_seed(sym) % 100) / 100
                cur, prev = sh[1] * pct * r.uniform(0.8, 1.2), sh[0] * pct * r0.uniform(0.8, 1.2)
                adv = sh[1] * 0.005
                rows.append({"symbolCode": sym, "settlementDate": d.strftime("%Y-%m-%d"),
                             "currentShortPositionQuantity": int(cur), "previousShortPositionQuantity": int(prev),
                             "averageDailyVolumeQuantity": int(adv), "daysToCoverQuantity": round(cur / adv, 2),
                             "stockSplitFlag": None, "revisionFlag": None, "marketClassCode": "NNM"})
        return rows
