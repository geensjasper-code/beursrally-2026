#!/usr/bin/env python3
"""
Beursrally 2026 - spelmotor (stap 4b): speelt een strategie volgens de wedstrijdregels, in duizenden
scenario's tegelijk.

Regels (wedstrijdreglement 2026 + veelgestelde vragen; zie de projectnotitie 'rules for model')
    Startkapitaal 50.000 euro. Quiz: 100 euro per goed antwoord, 46 vragen op 46 van de 48 beursdagen.
    Finance Avenue: +500 euro (uiterlijk 4 december bijgeschreven).
    Max 5 orders per beursdag; per effect per dag max 1 aankoop en 1 verkoop.
    Geweigerde orders tellen niet mee.
    Een positie mag bij AANKOOP niet boven 10% van de totale portefeuillewaarde komen; door koersstijging
    mag ze erboven groeien (geen gedwongen verkoop), maar dan geen bijkopen.
    Cash nooit negatief, geen gedeeltelijke uitvoering: te groot aankooporder -> volledig geweigerd.
    Kosten aandelen en ETF's: 5 euro + 0,35% beurstaks, bij aan- en verkoop.
    Eindwaarde: cash + posities aan de slotkoers van 22 januari (geen verkoopkosten).
    Weekprijs: groei van de waarde van vrijdagavond tot vrijdagavond (extra quizkapitaal telt mee).

Tijdlijn per beursdag d in de motor
    1. De strategie plaatst orders VOOR de opening, met wat bekend is tot de slotkoers van d-1.
    2. Uitvoering aan de openingskoers van dag d: eerst verkopen (maken cash vrij), dan aankopen, in de
       volgorde van de strategie. Elk order wordt getoetst aan alle regels.
    3. Slotkoers van dag d; quiz en Finance Avenue worden bijgeschreven; waarde van de dag vastgelegd.
    Benadering: openingskoers van dag d = slotkoers van dag d-1 (de scenario's hebben slotkoersen).

Aannames die je kunt aanpassen (Rules)
    Geen quizvraag op 24 en 31 december (halve beursdagen; het reglement zegt niet welke 2 dagen).
    Quizgeld en Finance Avenue zijn pas bruikbaar vanaf de volgende opening.
    Hele aandelen: een aankoop van X euro koopt floor(X / koers) aandelen.

Gebruik
    python game.py test        # test per regel
    python game.py demo        # een paar eenvoudige strategieen op de scenario's van vandaag (scen_build.py)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Rules:
    start_capital: float = 50_000.0
    quiz_amount: float = 100.0
    no_quiz: tuple = ("2026-12-24", "2026-12-31")
    fa_bonus: float = 500.0
    fa_credit_date: str = "2026-12-04"        # bijgeschreven op deze dag -> bruikbaar vanaf de volgende opening
    max_orders: int = 5
    cap: float = 0.10
    fixed_cost: float = 5.0
    tax: float = 0.0035
    whole_shares: bool = True


@dataclass
class State:
    """Wat een strategie ziet voor de opening van dag 'day' (alles tot en met de slotkoers van day-1)."""
    day: int
    dates: pd.DatetimeIndex
    cash: np.ndarray            # (B,)
    units: np.ndarray           # (B, N) aantal aandelen
    price: np.ndarray           # (B, N) laatst gekende koers in euro (slot van day-1; dag 0: startkoers)
    values: np.ndarray          # (B, day) waarde aan elke slotkoers tot nu
    cum: np.ndarray             # (B, day, N) cumulatief log-rendement tot nu (voor strategieen met momentum)

    @property
    def holdings_value(self) -> np.ndarray:
        return (self.units * self.price).sum(1)

    @property
    def total(self) -> np.ndarray:
        return self.cash + self.holdings_value


@dataclass
class Result:
    final: np.ndarray           # (B,) eindwaarde
    values: np.ndarray          # (B, T) waarde aan elke slotkoers
    executed: np.ndarray        # (B,) uitgevoerde orders
    rejected: dict = field(default_factory=dict)   # reden -> (B,) aantal
    costs: np.ndarray = None    # (B,) betaalde kosten
    units: np.ndarray = None    # (B, N) eindposities
    cash: np.ndarray = None     # (B,) eindcash

    def weekly_growth(self, dates: pd.DatetimeIndex, start_capital: float = 50_000.0) -> np.ndarray:
        """(B, weken): groei van vrijdagavond tot vrijdagavond. Is vrijdag geen beursdag (25 dec, 1 jan),
        dan telt de laatste beursdag van die week. Week 1 start bij het startkapitaal (vrijdag 13 nov)."""
        week = dates.to_period("W-FRI")
        last = pd.Series(np.arange(len(dates)), index=dates).groupby(np.asarray(week.asi8)).max().to_numpy()
        v = self.values[:, last]
        prev = np.c_[np.full(len(v), start_capital), v[:, :-1]]
        return v / prev - 1


# --------------------------------------------------------------------------- de motor
def play(policy, cum: np.ndarray, price0: np.ndarray, dates: pd.DatetimeIndex, rules: Rules = Rules(),
         quiz_correct: np.ndarray | float = 1.0, finance_avenue: bool = True, seed: int = 0) -> Result:
    """policy: object met .orders(state) -> lijst van (sec (B,) int, kant +1 koop / -1 verkoop, bedrag (B,)).
         koop: bedrag in euro (0 = geen order); verkoop: fractie van de positie (1 = alles, 0 = geen order).
       cum: (B, T, N) cumulatief log-rendement vanaf de start (scen_core.generate); price0: (N,) startkoers euro.
       quiz_correct: kans op een goed antwoord per vraag (getal of (B,)); 1 = alles juist.
       finance_avenue: aanwezig op Finance Avenue (+500 euro)."""
    B, T, N = cum.shape
    rng = np.random.default_rng(seed)
    cash = np.full(B, rules.start_capital)
    units = np.zeros((B, N))
    values = np.zeros((B, T))
    executed = np.zeros(B, int)
    costs = np.zeros(B)
    rej = {k: np.zeros(B, int) for k in ("cash", "cap", "max_orders", "dubbel", "niets")}
    quiz_day = ~dates.isin(pd.to_datetime(list(rules.no_quiz)))
    fa_day = np.where(dates == pd.Timestamp(rules.fa_credit_date))[0]
    p_ok = np.broadcast_to(np.asarray(quiz_correct, float), (B,))
    rows = np.arange(B)
    for d in range(T):
        open_px = price0 * (np.exp(cum[:, d - 1, :]) if d > 0 else np.ones((B, N)))
        st = State(d, dates, cash.copy(), units.copy(), open_px, values[:, :d], cum[:, :d, :])
        orders = policy.orders(st) or []
        sells = [o for o in orders if o[1] < 0]
        buys = [o for o in orders if o[1] > 0]
        n_today = np.zeros(B, int)
        sold = np.zeros((B, N), bool)
        bought = np.zeros((B, N), bool)
        for sec, side, amt in sells + buys:
            sec = np.broadcast_to(np.asarray(sec, int), (B,))
            amt = np.broadcast_to(np.asarray(amt, float), (B,))
            want = amt > 0
            px = open_px[rows, sec]
            held = units[rows, sec]
            if side < 0:
                q = np.where(amt >= 1, held, np.floor(amt * held) if rules.whole_shares else amt * held)
                q = np.minimum(q, held)
                gross = q * px
                fee = np.where(q > 0, rules.fixed_cost + rules.tax * gross, 0.0)
                dup = sold[rows, sec]
                ok_basic = want & (q > 0)
                ok = ok_basic & ~dup & (n_today < rules.max_orders) & (cash + gross - fee >= 0)
                rej["niets"] += want & ~(q > 0)
                rej["dubbel"] += ok_basic & dup
                rej["max_orders"] += ok_basic & ~dup & (n_today >= rules.max_orders)
                units[rows, sec] -= np.where(ok, q, 0)
                cash += np.where(ok, gross - fee, 0)
                sold[rows, sec] |= ok
            else:
                q = np.floor(amt / px) if rules.whole_shares else amt / px
                gross = q * px
                fee = np.where(q > 0, rules.fixed_cost + rules.tax * gross, 0.0)
                total = cash + (units * open_px).sum(1)
                dup = bought[rows, sec]
                ok_basic = want & (q > 0)
                cap_ok = held * px + gross <= rules.cap * total + 1e-9
                cash_ok = gross + fee <= cash + 1e-9
                ok = ok_basic & ~dup & (n_today < rules.max_orders) & cap_ok & cash_ok
                base = ok_basic & ~dup & (n_today < rules.max_orders)
                rej["niets"] += want & ~(q > 0)
                rej["dubbel"] += ok_basic & dup
                rej["max_orders"] += ok_basic & ~dup & (n_today >= rules.max_orders)
                rej["cap"] += base & ~cap_ok
                rej["cash"] += base & cap_ok & ~cash_ok
                units[rows, sec] += np.where(ok, q, 0)
                cash -= np.where(ok, gross + fee, 0)
                bought[rows, sec] |= ok
            n_today += ok
            executed += ok
            costs += np.where(ok, fee, 0)
        # einde van de dag: quiz en Finance Avenue, dan de waarde aan de slotkoers
        if quiz_day[d]:
            cash += rules.quiz_amount * (rng.random(B) < p_ok)
        if finance_avenue and d in fa_day:
            cash += rules.fa_bonus
        values[:, d] = cash + (units * price0 * np.exp(cum[:, d, :])).sum(1)
    return Result(final=values[:, -1].copy(), values=values, executed=executed, rejected=rej, costs=costs,
                  units=units, cash=cash)


# --------------------------------------------------------------------------- eenvoudige strategieen
class BuyHold:
    """Koop de doelaandelen aan 'weight' van de totale waarde (5 per dag: de motor beperkt), hou ze tot het
    einde. Quizgeld: 'cash' (laten staan) of 'bijkopen': om de 'every' beursdagen de posities die het verst
    onder 'weight' zitten aanvullen, met orders van minstens 'min_order' euro."""

    def __init__(self, targets, weight: float = 0.097, quiz: str = "cash", every: int = 5,
                 min_order: float = 500.0, whole_shares: bool = True):
        self.targets = np.atleast_2d(np.asarray(targets, int))   # (1, K) of (B, K)
        self.weight, self.quiz, self.every, self.min_order = weight, quiz, every, min_order
        self.whole_shares = whole_shares

    def orders(self, st: State):
        B = len(st.cash)
        tg = np.broadcast_to(self.targets, (B, self.targets.shape[1]))
        rows = np.arange(B)[:, None]
        held = st.units[rows, tg] > 0
        amt = self.weight * st.total
        # een doel waarvan 1 aandeel duurder is dan de positie (hele aandelen) is onkoopbaar: overslaan
        buyable = st.price[rows, tg] <= amt[:, None] if self.whole_shares else np.ones_like(held)
        todo = ~held & buyable
        out = []
        if todo.any():                                        # opbouwen: elk nog niet gekocht doel
            for k in range(tg.shape[1]):
                out.append((tg[:, k], +1, np.where(todo[:, k], amt, 0.0)))
            return out
        if self.quiz == "bijkopen" and st.day % self.every == 0:
            val = st.units[rows, tg] * st.price[rows, tg]
            gap = np.where(held, self.weight * st.total[:, None] - val, -np.inf)   # alleen gekochte posities
            order = np.argsort(-gap, axis=1)
            budget = st.cash.copy()
            for k in range(min(5, tg.shape[1])):
                j = order[:, k]
                g = gap[np.arange(B), j]
                a = np.minimum(g, (budget - 5.0) / (1 + 0.0035))
                a = np.where(a >= self.min_order, a, 0.0)
                budget -= np.where(a > 0, a * 1.0035 + 5.0, 0.0)
                out.append((tg[np.arange(B), j], +1, a))
        return out


class Scripted:
    """Voor tests: een vaste lijst orders per dag {dag: [(sec, kant, bedrag), ...]}."""

    def __init__(self, plan: dict):
        self.plan = plan

    def orders(self, st: State):
        return self.plan.get(st.day, [])


# --------------------------------------------------------------------------- tests
def selftest() -> bool:
    from scen_core import trading_days
    dates = trading_days(pd.Timestamp("2026-11-13"))
    T = len(dates)
    ok_all = True

    def check(name, ok):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"  {'OK  ' if ok else 'FOUT'} {name}")

    def flat(B=1, N=12):
        return np.zeros((B, T, N))

    px = np.full(12, 100.0)
    print(f"Test spelmotor: {T} beursdagen ({dates[0].date()} - {dates[-1].date()})")

    # 1. kosten
    r = play(Scripted({0: [(0, +1, 4900.0)]}), flat(), px, dates, quiz_correct=0.0, finance_avenue=False)
    check(f"kosten aankoop 4.900 euro = 5 + 0,35% = 22,15 (betaald {r.costs[0]:.2f})", abs(r.costs[0] - 22.15) < 1e-9)
    check("cash na aankoop = 50.000 - 4.900 - 22,15", abs(r.cash[0] - (50000 - 4900 - 22.15)) < 1e-9)
    # 2. max 5 orders per dag
    plan = {0: [(i, +1, 4000.0) for i in range(10)]}
    plan[1] = [(i, +1, 4000.0) for i in range(10)]
    r = play(Scripted(plan), flat(), px, dates, quiz_correct=0.0)
    check(f"10 aankopen op dag 1 en 2: 5 + 5 uitgevoerd (totaal {r.executed[0]})", r.executed[0] == 10)
    r1 = play(Scripted({0: [(i, +1, 4000.0) for i in range(10)]}), flat(), px, dates, quiz_correct=0.0)
    check(f"maar op 1 dag slechts 5 ({r1.executed[0]} uitgevoerd, {r1.rejected['max_orders'][0]} te veel)",
          r1.executed[0] == 5 and r1.rejected["max_orders"][0] == 5)
    # 3. per effect per dag 1 aankoop
    r = play(Scripted({0: [(0, +1, 2000.0), (0, +1, 2000.0), (1, +1, 2000.0)]}), flat(), px, dates,
             quiz_correct=0.0)
    check("tweede aankoop van hetzelfde effect op dezelfde dag geweigerd, telt niet mee",
          r.executed[0] == 2 and r.rejected["dubbel"][0] == 1)
    # 4. 10%-grens
    r = play(Scripted({0: [(0, +1, 5500.0)]}), flat(), px, dates, quiz_correct=0.0)
    check("aankoop van 11% geweigerd", r.executed[0] == 0 and r.rejected["cap"][0] == 1)
    cum = flat()
    cum[:, 1:, 0] = np.log(2.5)                                   # effect 0 x2,5 na dag 1
    cum[:, 30:, 0] = np.log(0.5)                                  # en zakt later terug naar de helft
    plan = {0: [(0, +1, 4900.0)], 5: [(0, +1, 500.0)], 31: [(0, +1, 500.0)]}
    r = play(Scripted(plan), cum, px, dates, quiz_correct=0.0)
    big = r.values[0, 2]
    check("9,8% kopen OK; positie groeit door koersstijging boven 10% en blijft staan (geen gedwongen verkoop)",
          r.units[0, 0] > 49 and 4900 * 2.5 / big > 0.10)
    check("bijkopen boven 10% geweigerd; onder 10% weer toegestaan",
          r.rejected["cap"][0] == 1 and r.executed[0] == 2)
    # 5. cash: te groot order geweigerd, telt niet mee; geen negatieve cash
    plan = {0: [(i, +1, 4999.0) for i in range(5)], 1: [(i, +1, 4999.0) for i in range(5, 9)],
            2: [(9, +1, 4999.0), (10, +1, 4999.0), (11, +1, 100.0)]}
    r = play(Scripted(plan), flat(), px, dates, quiz_correct=0.0, finance_avenue=False)
    check(f"te groot aankooporder volledig geweigerd (cash), volgend order wel uitgevoerd "
          f"(geweigerd {r.rejected['cash'][0]}, cash nooit negatief: {r.cash[0]:.2f})",
          r.rejected["cash"][0] >= 1 and r.cash[0] >= 0 and r.units[0, 11] == 1)
    # 6. verkopen eerst: met de opbrengst kan dezelfde dag gekocht worden
    plan = {0: [(i, +1, 4999.0) for i in range(5)], 1: [(i, +1, 4999.0) for i in range(5, 10)],
            2: [(11, +1, 4999.0), (0, -1, 1.0)]}
    r = play(Scripted(plan), flat(), px, dates, quiz_correct=0.0)
    check("verkoop wordt voor de aankoop uitgevoerd (aankoop met de opbrengst lukt dezelfde dag)",
          r.units[0, 0] == 0 and r.units[0, 11] > 0)
    # 7. hele aandelen
    p2 = px.copy()
    p2[3] = 333.0
    r = play(Scripted({0: [(3, +1, 5000.0)]}), flat(), p2, dates, quiz_correct=0.0)
    check(f"5.000 euro aan 333 euro = 15 aandelen ({r.units[0, 3]:.0f})", r.units[0, 3] == 15)
    # 8. quiz en Finance Avenue
    r = play(Scripted({}), flat(), px, dates, quiz_correct=1.0)
    check(f"zonder handel en alle quizvragen juist: eindwaarde 55.100 ({r.final[0]:,.0f})", abs(r.final[0] - 55100) < 1e-6)
    fa = np.where(dates == pd.Timestamp("2026-12-04"))[0][0]
    check("Finance Avenue op 4 december in de waarde, pas daarna bruikbaar",
          abs(r.values[0, fa] - r.values[0, fa - 1] - 600) < 1e-6)
    # 9. eindwaarde zonder verkoopkosten
    cum = flat()
    cum[:, :, 0] = np.log(1.2)
    r = play(Scripted({0: [(0, +1, 4900.0)]}), cum, px, dates, quiz_correct=0.0, finance_avenue=False)
    exp_final = 50000 - 4900 - 22.15 + 49 * 100 * 1.2
    check(f"eindwaarde = cash + aandelen x slotkoers, geen verkoopkosten ({r.final[0]:,.2f})",
          abs(r.final[0] - exp_final) < 1e-6)
    # 10. weekprijs: quizgeld telt mee, 25 dec en 1 jan zijn geen beursdag
    r = play(Scripted({}), flat(), px, dates, quiz_correct=1.0)
    wg = r.weekly_growth(dates)
    check(f"{wg.shape[1]} weken; week 1 zonder beleggingen = 5 x 100 / 50.000 = 1,0% ({wg[0, 0]:.2%})",
          wg.shape[1] == 10 and abs(wg[0, 0] - 0.01) < 1e-9)
    # 11. verkoop: kosten en gedeeltelijke verkoop
    plan = {0: [(0, +1, 4900.0)], 1: [(0, -1, 0.5)]}
    r = play(Scripted(plan), flat(), px, dates, quiz_correct=0.0)
    check(f"helft verkopen: 24 van 49 aandelen verkocht, kosten 5 + 0,35% ({r.units[0, 0]:.0f} over)",
          r.units[0, 0] == 25 and abs(r.costs[0] - (22.15 + 5 + 0.0035 * 2400)) < 1e-9)
    # 12. per scenario onafhankelijk (vectorisatie = losse berekening)
    rng = np.random.default_rng(0)
    cumB = np.cumsum(rng.normal(0, 0.03, (6, T, 12)), axis=1)
    pol = BuyHold(np.arange(10), quiz="bijkopen")
    rb = play(pol, cumB, px, dates)
    single = np.array([play(BuyHold(np.arange(10), quiz="bijkopen"), cumB[[b]], px, dates).final[0] for b in range(6)])
    check("6 scenario's tegelijk = 6 keer apart (zelfde eindwaarden)", np.allclose(rb.final, single))
    # 13. BuyHold: 10 posities in 2 dagen; quizgeld gaat naar posities die onder het doel zakten
    cum = flat()
    cum[:, 3:, 0] = np.log(0.6)                                   # effect 0 zakt 40%: ruimte om bij te kopen
    rc = play(BuyHold(np.arange(10), quiz="cash"), cum, px, dates)
    rq = play(BuyHold(np.arange(10), quiz="bijkopen"), cum, px, dates)
    check(f"BuyHold: 10 posities in 2 dagen ({rc.executed[0]} orders); bijkopen vult de gezakte positie aan "
          f"(cash {rc.cash[0]:,.0f} -> {rq.cash[0]:,.0f}, aandelen effect 0: {rc.units[0, 0]:.0f} -> {rq.units[0, 0]:.0f})",
          rc.executed[0] == 10 and rq.units[0, 0] > rc.units[0, 0] and rq.cash[0] < rc.cash[0] - 1000)
    tot = rq.values[0, -1]
    check(f"na bijkopen geen positie boven 10% door aankopen (grootste {(rq.units[0] * px * np.exp(cum[0, -1])).max() / tot:.1%})",
          (rq.units[0] * px * np.exp(cum[0, -1])).max() / tot <= 0.10 + 1e-9)
    print("ZELFTEST GESLAAGD" if ok_all else "ZELFTEST MISLUKT")
    return ok_all


def demo(n: int = 4000) -> None:
    """Eenvoudige strategieen in dezelfde scenario's: werkt alles samen, en hoe zien de uitkomsten eruit?
    (Nog zonder concurrenten: dat is stap 5. Hier alleen de verdeling van de eigen eindwaarde.)"""
    import time
    from pathlib import Path
    from scen_core import generate, load_params
    here = Path(__file__).resolve().parent
    f = here / "data" / "scenarios" / "params_vandaag.npz"
    if not f.exists():
        sys.exit("Eerst:  python scen_build.py")
    p = load_params(f)
    close = pd.read_parquet(here / "data" / "processed" / "close_eur.parquet")
    price0 = close.ffill().iloc[-1].reindex(p.isins).to_numpy(float)
    vol = pd.read_parquet(here / "data" / "risk" / "vol_vandaag.parquet").set_index("isin").reindex(p.isins)
    names = vol["naam"].fillna("").to_numpy()
    ok = np.isfinite(price0) & (price0 > 0)
    order = np.argsort(-np.where(ok, vol["v_48"].to_numpy(), -1))
    rng = np.random.default_rng(1)
    dear = np.where(ok & (price0 > 0.097 * 50_000))[0]
    strategies = {
        "10 meest beweeglijke, quizgeld in cash": (BuyHold(order[:10], quiz="cash"), Rules()),
        "10 meest beweeglijke, quizgeld bijkopen": (BuyHold(order[:10], quiz="bijkopen"), Rules()),
        "idem, met fracties van aandelen": (BuyHold(order[:10], quiz="bijkopen", whole_shares=False),
                                           Rules(whole_shares=False)),
        "10 willekeurige aandelen": (BuyHold(rng.choice(np.where(ok)[0], 10, replace=False), quiz="bijkopen"),
                                     Rules()),
        "niets doen (alleen quiz)": (Scripted({}), Rules()),
    }
    print(f"Spelmotor op {n:,} scenario's van vandaag ({p.dates[0].date()} - {p.dates[-1].date()})")
    print("10 meest beweeglijke: " + ", ".join(names[order[:10]]))
    print(f"Duurder dan een positie van 9,7% (4.850 euro) per aandeel: {len(dear)} -> " +
          (", ".join(f"{names[i]} ({price0[i]:,.0f} euro)" for i in dear[:12]) or "geen"))
    rows = []
    t0 = time.time()
    finals = {k: [] for k in strategies}
    info = {k: [] for k in strategies}
    for chunk_i, cum in enumerate(generate(p, n, seed=0, chunk=1000)):
        cum = cum.astype(np.float64)
        for k, (pol, rules) in strategies.items():
            r = play(pol, cum, price0, p.dates, rules=rules, seed=chunk_i)
            finals[k].append(r.final)
            info[k].append((r.executed.mean(), r.costs.mean(), r.cash.mean(),
                            sum(v.mean() for v in r.rejected.values())))
    for k in strategies:
        f_ = np.concatenate(finals[k]) / 50_000 - 1
        ex, co, ca, rj = np.mean(info[k], axis=0)
        rows.append({"strategie": k, "p5": np.quantile(f_, 0.05),
                     "mediaan": np.median(f_), "p95": np.quantile(f_, 0.95), "p99": np.quantile(f_, 0.99),
                     "orders": ex, "kosten": co, "eindcash": ca, "geweigerd": rj})
    t = pd.DataFrame(rows).set_index("strategie")
    for c in ("p5", "mediaan", "p95", "p99"):
        t[c] = t[c].map(lambda v: f"{v:+.1%}")
    for c in ("kosten", "eindcash"):
        t[c] = t[c].map(lambda v: f"{v:,.0f}")
    t["orders"] = t["orders"].map(lambda v: f"{v:.1f}")
    t["geweigerd"] = t["geweigerd"].map(lambda v: f"{v:.1f}")
    pd.set_option("display.width", 200)
    print("\nRendement op 50.000 euro startkapitaal (quiz en Finance Avenue inbegrepen):")
    print(t.to_string())
    print(f"\n({time.time() - t0:.0f} s) Wie de top 3 haalt hangt af van de concurrenten: dat is stap 5.")


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        sys.exit(0 if selftest() else 1)
    if sys.argv[1:] == ["demo"]:
        demo()
        sys.exit(0)
    print(__doc__)
