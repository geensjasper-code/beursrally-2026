# Step 2 — Risk model

All risk scripts read from `data/clean/` and `data/features/` and write to `data/risk/`.

## Once: build everything and choose

```
python br.py risk_simple           # models 1 and 2: rolling 60 days, EWMA
python br.py risk_garch test       # GARCH self-test
python br.py risk_garch            # model 3: GARCH per stock
python br.py risk_har              # model 4: HAR (with block selection)
python br.py risk_boost            # model 5a: boosting as a correction on HAR
python br.py risk_boost --alleen   # model 5b: boosting on its own
python br.py risk_earn             # earnings jump on top of EWMA/GARCH/rolling
python br.py risk_compare          # comparison on 2019-2021 -> winner per horizon
python br.py risk_iv               # implied volatility: index backtest 2002-2021, c and lambda
python br.py risk_corr test        # self-test Ledoit-Wolf and clustering
python br.py risk_corr             # choose the correlation model; themes (for information only)
python br.py risk_build            # risk for today
python br.py risk_test             # leakage test of the risk model
```

## Weekly (after the US close, with the rest of the pipeline)

```
python br.py data update
python br.py clean
python br.py features
python br.py risk_har
python br.py risk_boost --alleen
python br.py risk_build
```

Optionally also `python br.py risk_forward`: the forward test of the option blend (lambda).

## Choices

| Component | Choice | Why |
|---|---|---|
| Volatility | All horizons (1-48 days): average of HAR and boosting on its own | Best in the comparison on 2019-2021 for 1-10 days; 20 and 48 days added after the check period 2022 – mid 2025; confirmed in the final test |
| Options (US stocks) | c = 0.9, lambda = 0.7 | c from the index backtest 2002-2021 (stable); lambda 0.7 as a cautious transfer to single stocks |
| Correlations | factor model `fac_10` on the last 500 trading days (see `data/risk/corr_keuze.txt`); recent listings get loadings from a regression on the factors, shrunk with weight n/(n+60) | Portfolio test 2019-2021 |
| Themes | For information only; not a feature (theme_mom_21 dropped) | Themes too unstable (ARI ~0.24), and for Europe mostly "the European market" |

Validation periods: selection 2019-2021, check 2022 – mid 2025 (`python br.py risk_compare --controle`),
final test mid 2025 – now, looked at once (`python br.py risk_compare --final`, `python br.py risk_iv --final`).
