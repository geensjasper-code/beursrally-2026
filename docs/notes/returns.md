# Step 3 — Return model

Forecasts the STOCK-SPECIFIC return of each stock (better or worse than the average stock) over 1, 3, 5, 10, 20 and
48 trading days. The market part is fixed: 7% per year.

## Once

```
pip install torch --index-url https://download.pytorch.org/whl/cu132   # PyTorch with GPU (driver with CUDA 13)
python br.py ret_models nntest   # network self-test (GPU or CPU: does it learn a known relationship?)
python br.py ret_models          # walk-forward for zero, capm, composite, enet, boost, nn (nn on the GPU if available)
python br.py ret_compare         # comparison 2019-2021, choice per horizon, w, b, cost test
python br.py ret_build           # expected return today
python br.py ret_test            # leakage test
```

## Weekly (after `features`)

```
python br.py ret_build
```

## Choices

| Component | Choice |
|---|---|
| Candidates | all equal (zero), CAPM, academic composite, elastic net, gradient boosting, neural network (32-16-8, ensemble of 3) |
| Features | ranked per day (-0.5 to +0.5); market-wide features not used |
| Test | walk-forward with embargo (h + 5 days); choose on 2019-2021 |
| Main metric | rank IC (t-value with Newey-West) |
| Confidence | w = max(0, 1 - 1/t²); w = 0 if t < 2 or IC ≤ 0 |
| Costs | €5 + 0.35% per order (rules): ~0.9% per round trip |
| Return used | 7%/year × h/252 + w × b × forecast |

Chosen per horizon (on 2019-2021): 1 and 3 days `gem:nn+enet`, 5 days `gem:boost+nn`, 10 days `gem:capm+nn`,
20 and 48 days `gem:capm+boost` (`gem:` = average of the two, each standardised per day).

w and b: measured over 2019 – now after the final test (`python br.py ret_compare --alles`), written to
`data/returns/w_controle.json` and used by `ret_build`. Final test (mid 2025 – now): 1-5 days significant;
10-48 days positive IC but not significant on their own; pooled over 2019 – now every horizon has t ≥ 2.95.
