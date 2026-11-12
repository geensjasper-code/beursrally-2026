#!/usr/bin/env python3
"""
Beursrally 2026 - scenariogenerator (stap 4), deel 4a: de kern.

Wat dit doet
    Uit de voorspellingen van stap 2 en 3 de bouwstenen maken om duizenden mogelijke verlopen van de
    competitie te simuleren: voor alle aandelen tegelijk, dag per dag.

    verwacht rendement   rendement_vandaag.parquet  mu_h (log-rendement over h dagen, h = 1 ... 48)
    variantie            vol_vandaag.parquet         v_h  (variantie van het log-rendement over h dagen)
    samenhang            corr_vandaag.parquet        correlatiematrix (factormodel)

    Per aandeel interpoleren we de CUMULATIEVE mu en variantie tussen de horizonnen (0, 1, 3, 5, 10, 20, 48)
    en nemen de verschillen per dag. Zo heeft elke horizon in de simulatie precies de voorspelde variantie en
    het voorspelde rendement, met de looptijdstructuur (bv. nu onrustig, later rustiger).

    Dagrendement aandeel i op dag t:   r_it = m_it + sqrt(v_it) * z_it,   z_t ~ N(0, R)
    (Deel 4a: normale schokken. Dikke staarten, crashdagen en onzekerheid over de vol volgen in deel 4c,
    alleen als de backtest van deel 4b toont dat ze helpen.)

Gebruik
    python br.py scen_core test        # zelftest: bekende verdeling erin -> komt dezelfde eruit?

    Andere scripts gebruiken:
        params = build_params(...)                    # bouwstenen
        for chunk in generate(params, n, seed): ...   # cumulatieve log-rendementen (paden, dagen, aandelen)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HORIZONS = [1, 3, 5, 10, 20, 48]
END = pd.Timestamp("2027-01-22")
CLOSED = pd.to_datetime(["2026-12-25", "2027-01-01"])        # beurzen in Europa en VS dicht
US_CLOSED = pd.to_datetime(["2026-11-26", "2027-01-18"])     # alleen VS dicht (Thanksgiving, Martin Luther King)
MIN_DAY_VAR = 1e-8
DEVICE = os.environ.get("BR_SCEN_DEVICE", "auto")             # auto: GPU als PyTorch er een ziet
MAX_GAIN = 15.0                                               # +1500% (x16) vanaf de start: daarboven blijft het staan
CAP_LOG = float(np.log1p(MAX_GAIN))


def trading_days(after: pd.Timestamp, until: pd.Timestamp = END) -> pd.DatetimeIndex:
    """Werkdagen na 'after' tot en met 'until', zonder de gemeenschappelijke sluitingsdagen."""
    d = pd.bdate_range(after + pd.Timedelta(days=1), until)
    return d[~d.isin(CLOSED)]


def cumulative_curve(values: np.ndarray, T: int, monotone: bool, own_days: np.ndarray | None = None) -> np.ndarray:
    """values: (N, 6) waarden op de horizonnen 1,3,5,10,20,48 (geteld in EIGEN beursdagen van het aandeel).
    Geeft (N, T) cumulatieve waarden op kalenderdag 1..T: lineair tussen de knopen (dag 0 = 0), na dag 48
    doorgetrokken met de helling van het laatste stuk. own_days (N, T): aantal eigen beursdagen tot en met elke
    kalenderdag (standaard 1..T). Op een dag dat het aandeel niet noteert blijft de waarde gelijk."""
    knots = np.array([0] + HORIZONS, float)
    vals = np.c_[np.zeros(len(values)), values]
    if own_days is None:
        own_days = np.broadcast_to(np.arange(1, T + 1, dtype=float), (len(values), T))
    out = np.empty((len(values), T))
    slope = (vals[:, -1] - vals[:, -2]) / (knots[-1] - knots[-2])
    for i in range(len(values)):
        d = own_days[i].astype(float)
        out[i] = np.interp(d, knots, vals[i])
        beyond = d > knots[-1]
        out[i, beyond] = vals[i, -1] + slope[i] * (d[beyond] - knots[-1])
    if monotone:
        out = np.maximum.accumulate(out, axis=1)
    return out


def complete_corr(R: pd.DataFrame, isins: list[str]) -> tuple[np.ndarray, list[str]]:
    """Correlatiematrix voor ALLE isins. Wie ontbreekt (recente beursgang: te weinig historiek) krijgt de
    gemiddelde rij (de 'gewone' samenhang met de markt), maal 0,8 (we weten minder: iets eigener)."""
    have = [i for i in isins if i in R.index]
    miss = [i for i in isins if i not in R.index]
    A = R.loc[have, have].to_numpy(float)
    if not miss:
        return nearest_psd(A), miss
    n, m = len(have), len(miss)
    avg = (A.sum(1) - 1.0) / (n - 1)                           # gemiddelde correlatie van elk aandeel
    rho_mm = float(avg.mean()) * 0.8 ** 2
    full = np.eye(n + m)
    full[:n, :n] = A
    full[:n, n:] = (avg * 0.8)[:, None]
    full[n:, :n] = full[:n, n:].T
    full[n:, n:] = rho_mm
    np.fill_diagonal(full, 1.0)
    idx = {k: j for j, k in enumerate(have + miss)}
    order = [idx[i] for i in isins]
    return nearest_psd(full[np.ix_(order, order)]), miss


def nearest_psd(R: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    val, vec = np.linalg.eigh((R + R.T) / 2)
    if val.min() >= floor:
        return R
    S = (vec * np.maximum(val, floor)) @ vec.T
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


def corr_root(R: np.ndarray) -> np.ndarray:
    """Matrix C met C @ C.T = R (via eigenwaarden: werkt ook als R net niet strikt positief is)."""
    val, vec = np.linalg.eigh(R)
    return vec * np.sqrt(np.maximum(val, 0.0))


@dataclass(frozen=True)
class Shape:
    """Vorm van de verdeling (deel 4c). Elke uitbreiding laat de VERWACHTE variantie per aandeel ongemoeid
    (alle vermenigvuldigers hebben gemiddelde 1); ze veranderen alleen de staarten en de samenhang.
      nu           stressdagen: per dag een gemeenschappelijke schaal W = (nu-2)/chi2(nu) voor alle aandelen
                   (meerdimensionale t-verdeling). 0 = uit (normaal). Geheel getal >= 4.
      vol_markt    onzekerheid over de vol, gemeenschappelijk: per scenario variantie x exp(s*g - s^2/2)
      vol_aandeel  idem per aandeel per scenario: dit aandeel is de hele periode wilder of rustiger"""
    nu: int = 0
    vol_markt: float = 0.0
    vol_aandeel: float = 0.0

    def label(self) -> str:
        return f"nu={self.nu or 'normaal'}, vol_markt={self.vol_markt:.2f}, vol_aandeel={self.vol_aandeel:.2f}"


@dataclass
class Params:
    isins: list[str]
    dates: pd.DatetimeIndex            # de dagen van de competitie die we teruggeven
    skip: int                          # gesimuleerde dagen VOOR de start (data loopt niet tot de start)
    drift: np.ndarray                  # (T, N) verwacht log-rendement per dag
    sd: np.ndarray                     # (T, N) standaardafwijking per dag
    root: np.ndarray                   # (N, N) wortel van de correlatiematrix
    info: dict = field(default_factory=dict)
    shape: Shape = field(default_factory=Shape)

    @property
    def T(self) -> int:
        return self.drift.shape[0]


def build_params(mu: pd.DataFrame, var: pd.DataFrame, R: pd.DataFrame, data_date: pd.Timestamp,
                 start: pd.Timestamp | None = None, end: pd.Timestamp = END,
                 us: set[str] | None = None) -> Params:
    """mu, var: index isin, kolommen 1,3,5,10,20,48 (cumulatief). R: correlatiematrix (index = kolommen = isin).
    data_date: laatste dag met koersen. Dag 1 van de simulatie = de beursdag daarna (zoals de voorspellingen).
    start: eerste competitiedag; dagen tussen data_date en start worden gesimuleerd maar niet teruggegeven.
    us: isins die op een Amerikaanse beurs noteren: op US_CLOSED-dagen bewegen ze niet (rendement 0)."""
    isins = list(var.index)
    days = trading_days(data_date, end)
    if start is None:
        start = days[0]
    skip = int((days < start).sum())
    T = len(days)
    us = us or set()
    open_ = np.ones((len(isins), T), bool)
    us_rows = [j for j, i in enumerate(isins) if i in us]
    open_[np.ix_(us_rows, np.where(days.isin(US_CLOSED))[0])] = False
    own = np.cumsum(open_, axis=1)                                 # eigen beursdagen tot en met elke dag
    V = cumulative_curve(var.loc[isins, HORIZONS].to_numpy(float), T, monotone=True, own_days=own)
    M = cumulative_curve(mu.reindex(isins)[HORIZONS].to_numpy(float), T, monotone=False, own_days=own)
    dv = np.diff(np.c_[np.zeros(len(isins)), V], axis=1)
    dm = np.diff(np.c_[np.zeros(len(isins)), M], axis=1)
    sd = np.where(open_, np.sqrt(np.maximum(dv, MIN_DAY_VAR)), 0.0)
    dm = np.where(open_, dm, 0.0)
    Rf, miss = complete_corr(R, isins)
    return Params(isins=isins, dates=days[skip:], skip=skip, drift=dm.T.astype(np.float32),
                  sd=sd.T.astype(np.float32), root=corr_root(Rf).astype(np.float32),
                  info={"zonder_correlaties": miss, "R": Rf, "open": open_[:, skip:]})


def save_params(p: Params, path) -> None:
    np.savez_compressed(path, isins=np.array(p.isins), dates=p.dates.values.astype("datetime64[D]"),
                        skip=p.skip, drift=p.drift, sd=p.sd, root=p.root, R=p.info["R"], open=p.info["open"],
                        shape=np.array([p.shape.nu, p.shape.vol_markt, p.shape.vol_aandeel], float),
                        zonder_correlaties=np.array(p.info["zonder_correlaties"], dtype=str))


def load_params(path) -> Params:
    z = np.load(path)
    return Params(isins=list(z["isins"]), dates=pd.DatetimeIndex(z["dates"]), skip=int(z["skip"]),
                  drift=z["drift"], sd=z["sd"], root=z["root"],
                  info={"R": z["R"], "zonder_correlaties": list(z["zonder_correlaties"]), "open": z["open"]},
                  shape=Shape(int(z["shape"][0]), float(z["shape"][1]), float(z["shape"][2]))
                  if "shape" in z else Shape())


# --------------------------------------------------------------------------- simulatie
_T = {}


def backend(device: str = DEVICE):
    """('torch', apparaat) als PyTorch beschikbaar is, anders ('numpy', None)."""
    if device == "numpy":
        return "numpy", None
    if "torch" not in _T:
        try:
            import torch
            _T["torch"] = torch
        except ImportError:
            _T["torch"] = None
    torch = _T["torch"]
    if torch is None:
        return "numpy", None
    dev = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    return "torch", torch.device(dev)


def generate(p: Params, n: int, seed: int = 0, chunk: int = 2000, device: str = DEVICE,
             shape: Shape | None = None):
    """Levert stukken van (paden, dagen, aandelen): CUMULATIEF log-rendement vanaf de start van de competitie
    (float32). Als numpy-array (ook als de berekening op de GPU gebeurde). Zelfde seed -> zelfde paden op
    hetzelfde apparaat. shape: vorm van de verdeling (standaard die van p; Shape() = normaal)."""
    sh = p.shape if shape is None else shape
    kind, dev = backend(device)
    T, N = p.drift.shape
    if kind == "torch":
        torch = _T["torch"]
        g = torch.Generator(device=dev).manual_seed(seed)
        randn = lambda sz: torch.randn(sz, generator=g, device=dev)
        drift = torch.as_tensor(p.drift, device=dev)
        sd = torch.as_tensor(p.sd, device=dev)
        rootT = torch.as_tensor(p.root.T.copy(), device=dev)
        exp, sqrt, cumsum = torch.exp, torch.sqrt, (lambda x: torch.cumsum(x, dim=1))
        out = lambda c: c.cpu().numpy()
    else:
        rng = np.random.default_rng(seed)
        randn = lambda sz: rng.standard_normal(sz, dtype=np.float32)
        drift, sd, rootT = p.drift, p.sd, p.root.T
        exp, sqrt, cumsum = np.exp, np.sqrt, (lambda x: np.cumsum(x, axis=1))
        out = lambda c: c.astype(np.float32, copy=False)
    done = 0
    while done < n:
        k = min(chunk, n - done)
        z = randn((k, T, N)) @ rootT
        if sh.nu:                                              # stressdagen: gemeenschappelijke schaal per dag
            chi = (randn((k, T, sh.nu)) ** 2).sum(-1)
            z = z * sqrt((sh.nu - 2) / chi)[:, :, None]
        mult = None
        if sh.vol_markt > 0:                                   # hele periode wilder/rustiger, voor iedereen
            s_ = sh.vol_markt
            mult = exp(s_ * randn((k, 1, 1)) - s_ * s_ / 2)
        if sh.vol_aandeel > 0:                                 # per aandeel wilder/rustiger
            s_ = sh.vol_aandeel
            m2 = exp(s_ * randn((k, 1, N)) - s_ * s_ / 2)
            mult = m2 if mult is None else mult * m2
        if mult is not None:
            z = z * sqrt(mult)
        c = cumsum(drift + sd * z)
        if p.skip:
            c = c[:, p.skip:, :] - c[:, p.skip - 1:p.skip, :]
        # plafond: een aandeel dat x16 bereikt, blijft daar (hoogste echte 48-daagse stijging 2019-2021: +1201%)
        c = c.clamp(max=CAP_LOG) if kind == "torch" else np.minimum(c, CAP_LOG)
        yield out(c)
        done += k


# --------------------------------------------------------------------------- zelftest
def selftest() -> bool:
    """Bekende bouwstenen erin; komen gemiddelde, variantie, correlatie en onafhankelijkheid tussen dagen
    terug? Op elk beschikbaar apparaat (numpy, en PyTorch/GPU als die er is)."""
    rng = np.random.default_rng(1)
    N = 25
    isins = [f"X{i:02d}" for i in range(N)]
    B = rng.normal(0, 0.45, (N, 3))                             # 3 factoren
    B *= np.sqrt(rng.uniform(0.2, 0.8, N) / (B ** 2).sum(1))[:, None]   # gemeenschappelijk deel 20-80%
    R = B @ B.T
    np.fill_diagonal(R, 1.0)
    R = pd.DataFrame(R, isins, isins)
    daily = rng.uniform(0.0002, 0.002, N)                       # dagvariantie
    hump = np.array([1.6, 1.4, 1.3, 1.15, 1.05, 1.0])           # nu onrustig, later rustiger
    var_all = pd.DataFrame(daily[:, None] * np.array(HORIZONS) * hump, isins, HORIZONS)
    mu_all = pd.DataFrame(rng.normal(0, 0.0005, N)[:, None] * np.array(HORIZONS), isins, HORIZONS)
    us = set(isins[:5])                                         # de eerste 5 noteren in de VS
    p = build_params(mu_all, var_all, R, pd.Timestamp("2026-11-06"), start=pd.Timestamp("2026-11-16"), us=us)
    ok_all = True

    def check(name, ok):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"  {'OK  ' if ok else 'FOUT'} {name}")

    print(f"Zelftest scenariogenerator: {N} aandelen, {p.skip} dagen voor de start, {len(p.dates)} competitiedagen")
    check("aantal competitiedagen = 48 (16 nov - 22 jan)", len(p.dates) == 48)
    kinds = ["numpy"] + (["auto"] if backend("auto")[0] == "torch" else [])
    n = 40000
    for kind in kinds:
        name = "numpy" if kind == "numpy" else f"PyTorch ({backend(kind)[1]})"
        a = np.concatenate(list(generate(p, n, seed=3, device=kind)))
        b = np.concatenate(list(generate(p, 2000, seed=3, device=kind)))
        check(f"{name}: zelfde seed -> zelfde paden", np.array_equal(a[:2000], b))
        # vergelijken met de VOLLEDIGE simulatie (inclusief de dagen voor de start)
        # vergelijken met de VOLLEDIGE simulatie (inclusief de dagen voor de start), alleen EU-aandelen:
        # bij hen valt kalenderdag h samen met eigen beursdag h
        full = np.concatenate(list(generate(Params(p.isins, p.dates, 0, p.drift, p.sd, p.root), n, seed=5,
                                               device=kind)))
        worst_m = worst_v = 0.0
        eu = [j for j, i in enumerate(isins) if i not in us]
        full_eu = full[:, :, eu]
        mu_eu, var_eu = mu_all.iloc[eu], var_all.iloc[eu]
        for j, h in enumerate(HORIZONS):
            x = full_eu[:, h - 1, :].astype(np.float64)
            se_m = np.sqrt(var_eu[h].to_numpy() / n)
            worst_m = max(worst_m, np.max(np.abs(x.mean(0) - mu_eu[h].to_numpy()) / se_m))
            worst_v = max(worst_v, np.max(np.abs(x.var(0) / var_eu[h].to_numpy() - 1)))
        check(f"{name}: gemiddelde op elke horizon = voorspelling (grootste afwijking {worst_m:.1f} standaardfouten)",
              worst_m < 4.5)
        check(f"{name}: variantie op elke horizon = voorspelling (grootste afwijking {worst_v:.1%})", worst_v < 0.05)
        x20 = full[:, 19, :].astype(np.float64)
        dc = np.abs(np.corrcoef(x20[:, eu].T) - R.to_numpy()[np.ix_(eu, eu)]).max()
        check(f"{name}: correlatie 20-daagse rendementen = R (grootste afwijking {dc:.3f})", dc < 0.03)
        d = np.diff(full, axis=1)[:, :, eu[0]].astype(np.float64)
        ac = np.corrcoef(d[:, :-1].ravel(), d[:, 1:].ravel())[0, 1]
        check(f"{name}: dagen onafhankelijk (autocorrelatie {ac:+.4f})", abs(ac) < 0.01)
        # VS-aandelen: geen beweging op Thanksgiving en MLK; tot 22 jan variantie van 2 eigen dagen minder
        step = np.diff(a, axis=1)
        closed = np.where(p.dates.isin(US_CLOSED))[0]
        still = all(np.all(step[:, c - 1, :5] == 0) for c in closed) and np.all(step[:, closed[0] - 1, 5:] != 0)
        check(f"{name}: VS-aandelen bewegen niet op {', '.join(str(d.date()) for d in US_CLOSED)}; EU wel", still)
        open_all = np.ones((N, p.T), bool)
        open_all[:5][:, np.where(np.r_[np.zeros(p.skip, bool), p.dates.isin(US_CLOSED)])[0]] = False
        Vt = cumulative_curve(var_all[HORIZONS].to_numpy(), p.T, True, own_days=np.cumsum(open_all, axis=1))
        exp_v = Vt[:, -1] - Vt[:, p.skip - 1]
        dv = np.max(np.abs(a[:, -1, :].var(0) / exp_v - 1))
        us_days = np.cumsum(open_all, axis=1)[0, -1] - np.cumsum(open_all, axis=1)[0, p.skip - 1]
        check(f"{name}: na overslaan van {p.skip} dagen: variantie tot 22 jan klopt, VS over {us_days} eigen "
              f"dagen, EU over {p.T - p.skip} (afwijking {dv:.1%})", dv < 0.05 and us_days == p.T - p.skip - 2)
    # vorm: variantie blijft gelijk, staarten en spreiding worden breder
    for kind in kinds:
        name = "numpy" if kind == "numpy" else f"PyTorch ({backend(kind)[1]})"
        base = Params(p.isins, p.dates, 0, p.drift, p.sd, p.root)
        sims = {}
        for lab, sh in (("normaal", Shape()), ("vol", Shape(0, 0.4, 0.4)), ("stress", Shape(4, 0, 0)),
                        ("alles", Shape(5, 0.3, 0.4))):
            sims[lab] = np.concatenate(list(generate(base, 30000, seed=7, device=kind, shape=sh))).astype(np.float64)
        e = [j for j, i in enumerate(isins) if i not in us]
        worst = 0.0
        for lab in ("vol", "stress", "alles"):
            for h in (5, 48):
                v0 = sims["normaal"][:, h - 1, e].var(0)
                v1 = sims[lab][:, h - 1, e].var(0)
                worst = max(worst, float(np.median(np.abs(v1 / v0 - 1))))
        check(f"{name}: vorm laat de variantie gelijk (grootste mediaanafwijking {worst:.1%})", worst < 0.04)
        def kurt(x):
            x = x - x.mean(0)
            return float(np.median((x ** 4).mean(0) / (x ** 2).mean(0) ** 2 - 3))
        d1 = {lab: kurt(np.diff(sims[lab], axis=1)[:, 10, :][:, e]) for lab in sims}
        k48 = {lab: kurt(sims[lab][:, 47, e]) for lab in sims}
        check(f"{name}: stressdagen geven dikke staarten per dag (extra kurtosis {d1['normaal']:+.2f} -> "
              f"{d1['stress']:+.2f})", d1["stress"] > d1["normaal"] + 1)
        check(f"{name}: vol-onzekerheid geeft dikke staarten over 48 dagen ({k48['normaal']:+.2f} -> "
              f"{k48['vol']:+.2f})", k48["vol"] > k48["normaal"] + 0.3)
        disp = {lab: np.std(sims[lab][:, 47, e].std(1)) / np.mean(sims[lab][:, 47, e].std(1)) for lab in sims}
        check(f"{name}: spreiding tussen aandelen wisselt meer tussen scenario's (variatie {disp['normaal']:.2f} -> "
              f"{disp['vol']:.2f})", disp["vol"] > 1.25 * disp["normaal"])
    # plafond
    wild = Params(p.isins, p.dates, 0, p.drift * 0 + 0.08, p.sd, p.root)   # sterke stijging: plafond wordt bereikt
    top = max(float(c.max()) for c in generate(wild, 500, seed=1))
    check(f"plafond: geen aandeel boven +{MAX_GAIN:.0%} (hoogste {np.expm1(top):+.0%})", top <= CAP_LOG + 1e-6)
    # een ongeldige matrix (negatieve eigenwaarde) wordt hersteld: diagonaal 1, positief
    Rbad = R.to_numpy().copy()
    Rbad[0, 1] = Rbad[1, 0] = -0.99
    Rbad[0, 2] = Rbad[2, 0] = 0.99
    Rbad[1, 2] = Rbad[2, 1] = 0.99
    Rok, _ = complete_corr(pd.DataFrame(Rbad, isins, isins), isins)
    C = corr_root(Rok)
    check("ongeldige correlatiematrix hersteld (variantie blijft 1 per aandeel)",
          np.allclose(np.diag(C @ C.T), 1, atol=1e-6) and np.linalg.eigvalsh(Rok).min() > 0)
    # ontbrekende aandelen
    R2 = R.drop(index="X00", columns="X00")
    Rf, miss = complete_corr(R2, isins)
    ev = np.linalg.eigvalsh(Rf).min()
    check(f"aandeel zonder correlaties aangevuld (ontbrekend: {miss}), matrix geldig (kleinste eigenwaarde {ev:.4f})",
          miss == ["X00"] and ev > -1e-9 and np.allclose(np.diag(Rf), 1))
    print("ZELFTEST GESLAAGD" if ok_all else "ZELFTEST MISLUKT")
    return ok_all


def main():
    if sys.argv[1:] == ["test"]:
        sys.exit(0 if selftest() else 1)
    print(__doc__)


if __name__ == "__main__":
    main()
