#!/usr/bin/env python3
"""
Beursrally 2026 - risicomodel deel 2 en 3: correlaties en thema's.

Gebruik
    python risk_corr.py          # correlatiemodellen vergelijken, thema's bouwen, theme_mom_21
    python risk_corr.py test     # zelftest van Ledoit-Wolf en de clustering (bekende antwoorden)
    python risk_corr.py --snel   # dagelijks: alleen thema's en theme_mom_21 bijwerken (bewaarde K en
                                 # correlatiekeuze; zonder portefeuilletest en stabiliteitskeuze)

Correlaties (altijd op 2-daagse rendementen: Europa en de VS sluiten op verschillende uren)
    sample       gewone correlaties over de laatste 500 beursdagen (250 tweedaagse rendementen)
    lw           Ledoit-Wolf: sample, getrokken naar de gemiddelde correlatie; de trekkracht wordt berekend
                 uit de data (Ledoit & Wolf 2003, 'constant correlation'-doel)
    fac_k        factormodel: k hoofdcomponenten + een eigen deel per aandeel (k = 1, 3, 5, 10)

Test: willekeurige portefeuilles van 10 aandelen (gelijk gewicht), elke 10 beursdagen in 2019-2021.
    Voorspelde variantie over de volgende 20 dagen = w' D R D w, met D = de GEREALISEERDE volatiliteit van
    elk aandeel (zo beoordelen we alleen de correlatie R) en de gerealiseerde portefeuillevariantie.
    Maatstaf: QLIKE, en Diebold-Mariano over de testdatums.

Thema's
    Hierarchische clustering (gemiddelde koppeling) op d = sqrt(2(1 - rho)), met rho = de correlaties NA het
    verwijderen van de 1, 2 of 3 sterkste gemeenschappelijke factoren (wereldmarkt, Europa tegenover VS, ...).
    Zonder die stap klonteren bijna alle aandelen samen in een reuzenthema.
    Aantal verwijderde factoren en K: samen gekozen op de stabiliteit (2019-2021).
    Elk kwartaal opnieuw, alleen met de 500 beursdagen ervoor (geen lek in theme_mom_21).
    Aantal thema's K: het meest stabiele (twee helften van het venster apart clusteren; overeenkomst gemeten
    met de adjusted Rand index), gekozen op de kwartalen van 2019-2021.
    theme_mom_21: gemiddeld 21-dagenrendement van de ANDERE aandelen in hetzelfde thema (minstens 2).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from risk_common import EVAL_START, FEAT, RISK, SPLIT, dm_test, load_returns, log, names, nw_tstat

WINDOW = 500                     # beursdagen voor correlaties en thema's
MIN_OBS_SHARE = 0.8              # een aandeel moet in minstens 80% van het venster een koers hebben
FAC_KS = [1, 3, 5, 10]
K_GRID = [8, 10, 12, 15, 20, 25, 30]
NREM_GRID = [1, 2, 3]            # aantal gemeenschappelijke factoren dat we verwijderen voor het clusteren
N_PORT, PORT_SIZE, PORT_H = 200, 10, 20


# --------------------------------------------------------------------------- bouwstenen
def two_day_returns(lr: pd.DataFrame) -> pd.DataFrame:
    """Niet-overlappende 2-daagse log-rendementen op de gemeenschappelijke kalender (ontbrekende dag = 0)."""
    x = lr.fillna(0.0)
    g = np.arange(len(x)) // 2
    out = x.groupby(g).sum()
    cnt = lr.notna().groupby(g).sum()
    out = out.where(cnt > 0)
    out.index = lr.index[::2][:len(out)]
    return out


def window_matrix(r2: pd.DataFrame, end: pd.Timestamp, n2: int = WINDOW // 2) -> pd.DataFrame:
    """Tweedaagse rendementen strikt voor 'end', laatste n2 waarnemingen; alleen aandelen met genoeg data."""
    w = r2[r2.index < end].tail(n2)
    keep = w.notna().mean() >= MIN_OBS_SHARE
    w = w.loc[:, keep]
    return w.fillna(0.0) if len(w) >= 0.8 * n2 else pd.DataFrame()


def standardize(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return X / sd


def corr_sample(X: np.ndarray) -> np.ndarray:
    Z = standardize(X)
    return Z.T @ Z / len(Z)


def corr_lw(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2003) met constante-correlatie-doel, toegepast op gestandaardiseerde rendementen.

    S = steekproefcorrelatie, F = alle correlaties gelijk aan het gemiddelde; R = delta F + (1 - delta) S,
    met delta = max(0, min(1, (pi - rho) / gamma / T)) (formules uit Ledoit & Wolf, 'Honey, I shrunk...').
    """
    Z = standardize(X)
    T, N = Z.shape
    S = Z.T @ Z / T
    rbar = (S.sum() - N) / (N * (N - 1))
    F = np.full((N, N), rbar)
    np.fill_diagonal(F, 1.0)
    Z2 = Z ** 2
    pi_mat = Z2.T @ Z2 / T - S ** 2
    pi_hat = pi_mat.sum()
    theta = (Z ** 3).T @ Z / T - S          # theta[i, j] = mean[(z_i^2 - 1)(z_i z_j - s_ij)] met s_ii = 1
    rho_off = rbar * (theta.sum() - np.trace(theta))   # sqrt(s_jj/s_ii) = 1 na standaardiseren
    rho_hat = np.trace(pi_mat) + rho_off
    gamma = ((F - S) ** 2).sum()
    delta = float(np.clip((pi_hat - rho_hat) / gamma / T, 0, 1)) if gamma > 0 else 1.0
    return delta * F + (1 - delta) * S, delta


def corr_factor(X: np.ndarray, k: int) -> np.ndarray:
    S = corr_sample(X)
    val, vec = np.linalg.eigh(S)
    idx = np.argsort(val)[::-1][:k]
    B = vec[:, idx] * np.sqrt(np.maximum(val[idx], 0))
    R = B @ B.T
    np.fill_diagonal(R, 1.0)
    return R


def corr_resid(X: np.ndarray, n_remove: int = 1) -> np.ndarray:
    """Correlaties NA het verwijderen van de marktfactor (eerste hoofdcomponent): wat aandelen gemeen hebben
    bovenop 'alles beweegt met de markt'. Daarop clusteren we de thema's."""
    Z = standardize(X)
    S = Z.T @ Z / len(Z)
    val, vec = np.linalg.eigh(S)
    V = vec[:, np.argsort(val)[::-1][:n_remove]]
    E = Z - (Z @ V) @ V.T
    return corr_sample(E)


# --------------------------------------------------------------------------- clustering
def cluster(R: np.ndarray, K: int) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    D = np.sqrt(np.clip(2 * (1 - R), 0, None))
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, t=K, criterion="maxclust")


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index: 1 = identieke indeling, ~0 = toeval."""
    from scipy.special import comb
    ct = pd.crosstab(a, b).to_numpy()
    sum_c = comb(ct, 2).sum()
    sa, sb = comb(ct.sum(1), 2).sum(), comb(ct.sum(0), 2).sum()
    n2 = comb(len(a), 2)
    exp = sa * sb / n2
    mx = (sa + sb) / 2
    return float((sum_c - exp) / (mx - exp)) if mx != exp else 1.0


# --------------------------------------------------------------------------- correlatietest
def portfolio_test(lr: pd.DataFrame, r2: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    cal = lr.index
    test_days = cal[(cal >= EVAL_START) & (cal < SPLIT)][::10]
    models = ["sample", "lw"] + [f"fac_{k}" for k in FAC_KS]
    rows = []
    for d in test_days:
        W = window_matrix(r2, d + pd.Timedelta(days=1))          # data tot en met dag d
        if W.empty:
            continue
        fut = r2[(r2.index > d)].head(PORT_H // 2)              # volgende 20 dagen = 10 tweedaagse rendementen
        fut = fut.loc[:, fut.columns.isin(W.columns)]
        ok = fut.notna().all() & (fut.notna().sum() == len(fut))
        cols = [c for c in W.columns if c in ok.index and ok[c]]
        if len(cols) < 50 or len(fut) < PORT_H // 2:
            continue
        X = W[cols].to_numpy()
        Rs = {"sample": corr_sample(X), "lw": corr_lw(X)[0]}
        for k in FAC_KS:
            Rs[f"fac_{k}"] = corr_factor(X, k)
        F = fut[cols].to_numpy()
        vol = np.sqrt((F ** 2).sum(0))                           # gerealiseerde volatiliteit per aandeel
        for p in range(N_PORT):
            idx = rng.choice(len(cols), PORT_SIZE, replace=False)
            w = np.full(PORT_SIZE, 1 / PORT_SIZE)
            real = float(((F[:, idx] @ w) ** 2).sum())
            dv = w * vol[idx]
            rec = {"date": d, "real": real}
            for m in models:
                rec[m] = float(dv @ Rs[m][np.ix_(idx, idx)] @ dv)
            rows.append(rec)
    return pd.DataFrame(rows)


def report_corr(res: pd.DataFrame) -> str:
    models = [c for c in res.columns if c not in ("date", "real")]
    loss = {m: pd.Series(np.maximum(res["real"], 1e-12) / res[m] - np.log(np.maximum(res["real"], 1e-12) / res[m]) - 1)
            for m in models}
    best = min(models, key=lambda m: loss[m].mean())
    rows = []
    for m in models:
        d = (loss[m] - loss[best]).groupby(res["date"]).mean()
        rows.append({"model": m, "QLIKE": loss[m].mean(), "bias": res["real"].mean() / res[m].mean(),
                     "t_vs_beste": np.nan if m == best else nw_tstat(d.to_numpy(), 2)})
    t = pd.DataFrame(rows).sort_values("QLIKE")
    print("\n================ CORRELATIES: PORTEFEUILLETEST (keuzeperiode 2019-2021) ================")
    print(f"{res['date'].nunique()} testdatums x {N_PORT} willekeurige portefeuilles van {PORT_SIZE} aandelen, 20 dagen vooruit.")
    print("Alleen de correlatie wordt beoordeeld (elk aandeel krijgt zijn gerealiseerde volatiliteit).")
    print("bias = gerealiseerde / voorspelde portefeuillevariantie (1 = juist; > 1 = samenhang onderschat).")
    show = t.copy()
    show["QLIKE"] = show["QLIKE"].round(4)
    show = show.round({"bias": 2, "t_vs_beste": 2})
    print(show.to_string(index=False, na_rep="-"))
    print("t_vs_beste > 2: duidelijk slechter dan de beste.")
    print("=========================================================================================\n")
    return best


# --------------------------------------------------------------------------- thema's door de tijd
def themes_over_time(lr: pd.DataFrame, r2: pd.DataFrame, K: int | None = None,
                     n_remove: int | None = None) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    """Thema's per kwartaal. K = None: kies K op de stabiliteit in 2019-2021 (een instelling die, net als de
    HAR-blokken, op de keuzeperiode gekozen wordt). De indeling zelf gebruikt alleen data van voor het kwartaal."""
    quarters = pd.date_range("2018-01-01", lr.index.max() + pd.Timedelta(days=1), freq="QS")
    stab = []
    windows = {}
    for q in quarters:
        W = window_matrix(r2, q)
        if W.empty or W.shape[1] < 50:
            continue
        windows[q] = W
        if K is None and EVAL_START <= q < SPLIT:
            X = W.to_numpy()
            half = len(X) // 2
            for nr in NREM_GRID:
                R1, R2 = corr_resid(X[:half], nr), corr_resid(X[half:], nr)
                for k_try in K_GRID:                  # eigen naam: K zelf mag hier niet overschreven worden
                    stab.append({"kwartaal": q, "factoren_weg": nr, "K": k_try,
                                 "ARI": adjusted_rand(cluster(R1, k_try), cluster(R2, k_try))})
    st = pd.DataFrame(stab)
    if K is None:
        if st.empty:
            K, n_remove = 15, 1
        else:
            n_remove, K = (int(v) for v in st.groupby(["factoren_weg", "K"])["ARI"].mean().idxmax())
    n_remove = 1 if n_remove is None else n_remove
    rows = []
    for q, W in windows.items():
        lab = cluster(corr_resid(W.to_numpy(), n_remove), K)
        rows.append(pd.DataFrame({"kwartaal": q, "isin": W.columns, "thema": lab, "factoren_weg": n_remove}))
    th = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["kwartaal", "isin", "thema"])
    return th, K, st


def theme_momentum(th: pd.DataFrame) -> pd.DataFrame:
    """theme_mom_21 op dag d: gemiddelde r_21 van de andere leden van het thema (thema's van het kwartaal van d)."""
    F = pd.read_parquet(FEAT / "features.parquet", columns=["date", "isin", "r_21"])
    F["kwartaal"] = F["date"].dt.to_period("Q").dt.start_time
    M = F.merge(th, on=["kwartaal", "isin"], how="inner")
    g = M.groupby(["date", "thema"])["r_21"]
    s, n = g.transform("sum"), g.transform("count")
    own = M["r_21"].notna().astype(float)
    others = n - own
    M["theme_mom_21"] = ((s - M["r_21"].fillna(0)) / others).where(others >= 2)
    return M[["date", "isin", "theme_mom_21"]]


def describe_themes(th: pd.DataFrame, K: int, st: pd.DataFrame) -> None:
    nm = names()
    last = th[th["kwartaal"] == th["kwartaal"].max()]
    print("================ THEMA'S ================")
    nrem = int(last["factoren_weg"].iloc[0]) if "factoren_weg" in last else 1
    if not st.empty and "factoren_weg" in st:
        ari = st.groupby(["factoren_weg", "K"])["ARI"].mean().unstack().round(3)
        print("Stabiliteit (gemiddelde ARI over de kwartalen van 2019-2021; 1 = identiek),")
        print("per aantal verwijderde gemeenschappelijke factoren (rijen) en aantal thema's K (kolommen):")
        print(ari.to_string())
    print(f"Gekozen: {nrem} factor(en) verwijderd, K = {K}. "
          f"Thema's van het laatste kwartaal ({last['kwartaal'].max().date()}), grootste eerst:")
    sizes = last["thema"].value_counts()
    for t, n in sizes.items():
        mem = [str(nm.get(i, i))[:20] for i in last.loc[last["thema"] == t, "isin"]]
        shown = ", ".join(mem[:12]) + (f" ... (+{len(mem) - 12})" if len(mem) > 12 else "")
        print(f"  thema {t:>2} ({n:>3}): {shown}")
    print("=========================================\n")


# --------------------------------------------------------------------------- zelftest
def self_test() -> bool:
    rng = np.random.default_rng(1)
    ok = True
    print("\n================ ZELFTEST CORRELATIES EN THEMA'S ================")
    # 1. Ledoit-Wolf: bij echte constante correlatie moet het veel krimpen; bij duidelijke structuur weinig
    T, N = 250, 60
    f = rng.standard_normal((T, 1))
    X_cc = 0.5 * f + rng.standard_normal((T, N))
    d_cc = corr_lw(X_cc)[1]
    G = rng.standard_normal((T, 3))
    load = np.zeros((N, 3))
    for j in range(N):
        load[j, j % 3] = 1.5
    X_st = G @ load.T + rng.standard_normal((T, N))
    d_st = corr_lw(X_st)[1]
    good = d_cc > 0.5 and d_st < 0.2
    ok &= good
    print(f"1. Ledoit-Wolf trekkracht: echte constante correlatie {d_cc:.2f} (hoog verwacht), "
          f"duidelijke structuur {d_st:.2f} (laag verwacht)   {'OK' if good else 'FOUT'}")
    # 2. LW is beter dan sample bij weinig data (afstand tot de echte correlatie)
    true = np.full((N, N), 0.2)
    np.fill_diagonal(true, 1)
    L = np.linalg.cholesky(true)
    err_s, err_l = [], []
    for _ in range(20):
        X = rng.standard_normal((80, N)) @ L.T
        err_s.append(((corr_sample(X) - true) ** 2).sum())
        err_l.append(((corr_lw(X)[0] - true) ** 2).sum())
    good = np.mean(err_l) < 0.5 * np.mean(err_s)
    ok &= good
    print(f"2. Fout tegenover de echte correlatie: sample {np.mean(err_s):.1f}, Ledoit-Wolf {np.mean(err_l):.1f}"
          f"   {'OK' if good else 'FOUT'}")
    # 3. clustering vindt drie bekende groepen terug; ARI = 1 bij identieke indeling
    lab = cluster(corr_lw(X_st)[0], 3)
    truth = np.arange(N) % 3
    ari = adjusted_rand(lab, truth)
    good = ari > 0.95 and abs(adjusted_rand(truth, truth) - 1) < 1e-12
    ok &= good
    print(f"3. Clustering vindt 3 echte groepen terug: ARI = {ari:.2f} (1 = perfect)   {'OK' if good else 'FOUT'}")
    # 4. marktfactor met verschillende gevoeligheid per aandeel (zoals in het echt: Nvidia beweegt sterker mee
    #    dan Ageas) + 4 thema's: op gewone correlaties klonteren de gevoelige aandelen samen (reuzenthema);
    #    na het verwijderen van de marktfactor komen de thema's terug
    N4 = 80
    m = rng.standard_normal((T, 1))
    G = rng.standard_normal((T, 4))
    beta = rng.uniform(0.3, 2.5, N4)
    truth4 = np.arange(N4) % 4
    load = np.zeros((N4, 4))
    load[np.arange(N4), truth4] = 0.7
    X_m = m * beta + G @ load.T + rng.standard_normal((T, N4))
    ari_raw = adjusted_rand(cluster(corr_sample(X_m), 4), truth4)
    ari_res = adjusted_rand(cluster(corr_resid(X_m), 4), truth4)
    good = ari_res > 0.8 and ari_raw < 0.5
    ok &= good
    print(f"4. Markt (ongelijke gevoeligheid) + 4 thema's: ARI zonder marktfactor {ari_res:.2f}, op gewone correlaties "
          f"{ari_raw:.2f}   {'OK' if good else 'FOUT'}")
    print(f"\nZelftest: {'GESLAAGD' if ok else 'MISLUKT'}")
    print("=================================================================\n")
    return ok


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        sys.exit(0 if self_test() else 1)
    quick = "--snel" in sys.argv and (RISK / "themas_stabiliteit.csv").exists() and (RISK / "corr_keuze.txt").exists()
    log("Rendementen laden, tweedaagse rendementen maken ...")
    lr = load_returns()
    r2 = two_day_returns(lr)
    if quick:
        best = (RISK / "corr_keuze.txt").read_text().strip()
        st = pd.read_csv(RISK / "themas_stabiliteit.csv")
        nr, K = (int(v) for v in st.groupby(["factoren_weg", "K"])["ARI"].mean().idxmax())
        log(f"Snel: bewaarde keuzes (correlatie '{best}', {nr} factor(en) weg, K = {K}). Thema's per kwartaal ...")
        th, K, _ = themes_over_time(lr, r2, K=K, n_remove=nr)
        th.to_parquet(RISK / "themas.parquet", index=False)
    else:
        log("Portefeuilletest correlatiemodellen (2019-2021) ...")
        res = portfolio_test(lr, r2)
        res.to_parquet(RISK / "corr_test.parquet", index=False)
        best = report_corr(res)
        log("Thema's per kwartaal en stabiliteitskeuze ...")
        th, K, st = themes_over_time(lr, r2)
        th.to_parquet(RISK / "themas.parquet", index=False)
        st.to_csv(RISK / "themas_stabiliteit.csv", index=False)
        describe_themes(th, K, st)
    log("theme_mom_21 berekenen ...")
    tm = theme_momentum(th)
    tm.to_parquet(RISK / "theme_mom.parquet", index=False)
    log(f"theme_mom_21: {tm['theme_mom_21'].notna().sum():,} waarden -> data/risk/theme_mom.parquet")
    (RISK / "corr_keuze.txt").write_text(best)
    log(f"Beste correlatiemodel: {best} (opgeslagen in data/risk/corr_keuze.txt). "
        "Volgende:  python features.py  en daarna  python risk_build.py")


if __name__ == "__main__":
    main()
