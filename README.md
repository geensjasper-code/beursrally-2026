# Beursrally 2026


About this project. This is my first fully independent project of this scale. It is vibe coded: the code was written with an AI coding assistant. I set the goal, the constraints and the validation rules, ran and checked every step, and made the design decisions. The accompanying paper explains and defends each part.


A quantitative strategy for **Beursrally 2026**, the virtual stock-market competition of the Belgian newspaper
De Tijd (about 10,000–35,000 players, €50,000 virtual starting capital, prizes for the top 3 of the general
ranking and of the student ranking).

**Goal:** maximise the chance of a top-3 finish, measured as **expected prize money**, not expected return.
The system simulates thousands of possible futures for ~370 stocks, plays the exact game rules in each one,
and compares our portfolio with a simulated field of tens of thousands of competitors.

Competition: **Monday 16 November 2026 – Friday 22 January 2027** (48 trading days).

> Script output and comments are in Dutch; this README explains every step and command in English.

---

## Fixed settings

| What | Value |
|---|---|
| Exclusions (own judgment) | `Bpost,Sif Holding,Avantium,Alstom` |
| Candidates in strategy.py | `--kandidaten alle` (all affordable stocks) |
| Processes | `$env:BR_WORKERS = "20"` (all 20 searches at the same time) |
| Starting amount per position | €4,750 (9.5%: room below the 10% cap) |
| Decision rule during the competition | rollout, weekly, max 2 actions (= 4 orders) + topping up with quiz money |

**Not changed during the competition:** the model choice per horizon, w and b (measured over 2019 – now, after
the final test), the shape of the scenarios, the decision rule. The models are refitted each week on new data;
*how* we model stays fixed.

### When to exclude a stock (the only rule)

Only for a **concrete, public, dated fact** that the model cannot see and that pushes the price in a predictable
direction during the competition, or makes the model's inputs unreliable:

- an announced share issue, or a shareholder meeting to authorise one (e.g. Sif, 22 December);
- a takeover bid at a fixed price (the price freezes, the upside disappears);
- an unadjusted rights issue or split in the price history (Avantium: volatility partly an artefact);
- a suspension, delisting, or not tradable on Beursrally.

**Not a reason:** "bad company", a bad feeling, weak results, bad news already in the price, analyst price
targets. Decide **before** seeing the model's result, and write down the reason.

---

## The start

### Friday 13 November, after 22:30 CET (after the US close)

```powershell
python br.py data update
```

Fetch prices and **options** on a trading day: on weekends option quotes are often empty, and the risk model for
US stocks then falls back to its base version without options.

### Sunday 15 November (evening)

```powershell
python br.py clean
python br.py features
python br.py risk_har
python br.py risk_boost --alleen
python br.py risk_build
python br.py ret_build
python br.py scen_build
```

Check: `risk_build.py` reports "Opties van <date>: ... van 150 aandelen door de kwaliteitsfilter" (options from
<date>: ... of 150 stocks pass the quality filter). Normal is about 145 of 150, dated Friday.

```powershell
$env:BR_WORKERS = "20"
python br.py strategy --kandidaten alle --uitsluiten "Bpost,Sif Holding,Avantium,Alstom"
```

Takes about 40–60 minutes. Output: `data/strategy/orders_start_eigen_uitsluiting_kalle.csv`.

Then:
1. **News check** on the 10 names (see "Every week", step 3). If one fails, add it to `--uitsluiten` and run again.
2. If the result looks very different from Thursday 12/11 (a contrarian portfolio of "AI losers", luxury,
   consulting; about €300 expected prize money), review it before placing orders.

### Monday 16 and Tuesday 17 November

- **Monday:** the 5 orders marked "maandag 16 nov" in the order list. **Tuesday:** the 5 marked "dinsdag 17 nov".
- Market orders. Enter the **amount** if possible (€4,750); otherwise the number of shares, computed with that
  morning's price (the `ca_aantal` column uses Friday's price).
- Whole shares only. The 10% cap applies at purchase. Max 5 orders per day.
- Check that all orders were executed; a rejected order does not count towards the 5.

---

## Every day

- **Answer the quiz** (€100 per correct answer). The model assumes there is no quiz on 24 and 31 December.
- Quiz money stays in cash until the weekly decision (`week.py` proposes "bijkopen", topping up positions).

## Once: Finance Avenue

- **Register and attend on 28 November** → €500, credited on 4 December.
- Not registered? Then use `python br.py week ... --geen-fa`.

---

## Every week (Friday evening after 22:30 CET; first time Friday 20 November)

### 1. Update the data

```powershell
python br.py data update
python br.py clean
python br.py features
python br.py risk_har
python br.py risk_boost --alleen
python br.py risk_build
python br.py ret_build
python br.py scen_build
```

`scen_build.py` automatically starts the futures on the next trading day and runs them to 22 January.
Run `data.py update` on Friday evening (options, see above). If it has to be on the weekend, check that
`risk_build.py` still reports about 145 of 150 stocks with options.

### 2. Read in the standings (~7 minutes, no typing)

```powershell
python br.py stand_inlezen
```

On the Beursrally website, press **Ctrl+A, Ctrl+C** on each page (any order):
1. Mijn Portefeuille (your own portfolio, all pages)
2. The student ranking
3. The general ranking
4. The portfolios the script lists under **"nog:"** (still needed: top 5 students + top 5 general; every page)

When it says **"nog: niets – druk Enter"**, press Enter. Check for **"Controle: alles klopt"** (all checks pass).
If it reports a problem (missing page, totals don't add up), copy that page again with the same command;
copies from the same date are merged.

This only reads pages you open and copy yourself: no scraping, no automated orders.

### 3. News check (~5 minutes)

Search the news for **your own 10 stocks** and for every stock `week.py` wants to **buy**: share issues,
takeover bids, suspensions, problems with banks or debt covenants. Apply the rule above.

Points of attention at the start (if they are in the portfolio):
- **Ontex:** negotiating a temporary covenant waiver with its banks (aim: before year-end). Waiver with a share
  issue, or refused → sell.
- **Ubisoft:** half-year results on 19 November; suspended for a week around its results last year.
- **Lucid:** raises capital regularly; new share offering → sell.

### 4. The decision

```powershell
python br.py week --uitsluiten "Bpost,Sif Holding,Avantium,Alstom"
```

Output: **"NIETS DOEN deze week"** (do nothing this week), or the orders for Monday (also saved in
`data/stand/<date>/orders_maandag.csv`): sells first, then buys, then quiz-money top-ups.

Watch the warnings at the top: if your value or a competitor's differs more than 5% from their page, the prices
are off (update the data again).

### 5. Monday: enter the orders

Same rules as at the start. Max 5 orders per day; per security max 1 buy and 1 sell per day.
Sells are executed before buys.

---

## Special dates

| Date | What |
|---|---|
| 26 November | Thanksgiving: US markets closed (US stocks don't move) |
| 28 November | Finance Avenue |
| 4 December | €500 from Finance Avenue credited |
| 24 and 31 December | (assumed) no quiz |
| 18 January | Martin Luther King Day: US markets closed |
| 22 January | last day; no selling costs at the end |

---

## Project layout

```
br.py                 one entry point: python br.py <command> [options]   (python br.py = list of commands)
config/               universe.csv (all Beursrally securities), artefacten.csv (known data errors)
beursrally/
  paths.py            project folders (data/, config/)
  data/               1. data.py, clean.py, features.py, check_data.py, fake_yf.py (fake data for tests)
  risk/               2. risk_common, risk_simple, risk_garch, risk_earn, risk_har, risk_boost, risk_iv,
                         risk_corr, risk_compare, risk_build, risk_forward
  returns/            3. ret_models, ret_compare, ret_build
  scenarios/          4. scen_core, scen_build, scen_backtest
  game/               5. game.py (game rules), field.py (simulated competitors)
  strategy/           6. strategy.py, vergelijk_velden.py, rules.py, rollout.py
  weekly/             stand_inlezen.py, week.py
  tests/              risk_test.py, ret_test.py (leakage tests)
docs/                 paper; notes/ = step notes (data, risk, returns) and the original design notes (PDF, Dutch)
data/                 not in git: Yahoo prices, forecasts, standings (other players' names)
```

| Step | Command | Output |
|---|---|---|
| Prices, options, earnings | `data`, `clean`, `check_data` (readable check per stock) | `data/raw`, `data/processed`, `data/meta` |
| Features | `features` (`features test` = leakage test) | `data/features` |
| Risk (volatility) | `risk_simple`, `risk_garch`, `risk_earn`, `risk_har`, `risk_boost`, `risk_iv` | forecasts per model |
| Risk (co-movement) | `risk_corr` | correlation models |
| Risk: selection and today | `risk_compare` (selection/check/final test), `risk_build` | `data/risk/vol_vandaag`, `corr_vandaag` |
| Risk: forward test | `risk_forward` (optional, weekly: does the option blend lambda really work?) | on the logs of `risk_build` |
| Returns | `ret_models`, `ret_compare` (selection; `--alles`: w and b), `ret_build` | `data/returns/rendement_vandaag` |
| Scenarios | `scen_core` (`test`), `scen_build`, `scen_backtest` (choose and test the shape) | `data/scenarios/params_vandaag.npz` |
| Game | `game` (`test`, `demo`), `field` (`test`) | |
| Starting portfolio | `strategy` | `data/strategy/orders_start*.csv` |
| Comparison per field type | `vergelijk_velden` | (screen) |
| Trading rules during the competition | `rules` (fixed rules, no gain), `rollout` (the chosen rollout) | `data/strategy` |
| Reading in the standings | `stand_inlezen` | `data/stand/<date>/` |
| Weekly decision | `week` | `data/stand/<date>/orders_maandag.csv` |
| Tests | `risk_test`, `ret_test`; `data ... --synthetic` runs on fake data | |

## Validation (summary)

Three periods, each time choosing first and testing afterwards: selection 2019–2021, check 2022 – mid 2025,
final test mid 2025 – November 2026 (looked at once).

| Component | Final test |
|---|---|
| Volatility (average of HAR + boosting) | better than EWMA 0.94 at every horizon (t_DM −3.0 to −8.5), bias 0.99–1.06 |
| Returns (rank IC) | 1–5 days significant (t 2.2–3.4); 10–48 days positive but not significant (t 0.7–1.0). Pooled over 2019 – now: every horizon t ≥ 2.95 → w ≈ 0.9 |
| Scenarios | deviation 26 vs 76 for a normal distribution; percentiles match |

## Troubleshooting

- **`data.py update` fails:** internet? Try again later; Yahoo is sometimes slow. An interrupted run resumes
  automatically.
- **"niet in onze data"** (not in our data) for a competitor: that position is held constant (not an error).
  For your own portfolio: add the security to `config/universe.csv`, then `python br.py data resolve` and rerun the pipeline.
- **The clipboard script doesn't work:** paste everything into one text file and run
  `python br.py stand_inlezen --bestand stand.txt`.
- **Reprocess a set of standings:** `python br.py stand_inlezen --opnieuw 2026-11-20`.

---

## What the model does and doesn't know

**Does:** prices, volatility (including from options), co-movement between stocks, a small return forecast,
the game rules, costs, the quiz, the real positions of the top players.

**Doesn't:** news, balance sheets, share issues, takeovers, suspensions. That's what the news check is for.

**Starting portfolio (computed on Thursday 12/11):** contrarian relative to the field, which is heavily invested
in AI and chips. It wins mainly in futures where the AI winners lag the rest. It won in all six field types
(+€57 to +€74 expected prize money compared with a volatile portfolio).
