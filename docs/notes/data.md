# Step 1 — Data

`python br.py data` fetches all data from Yahoo Finance (and short interest from FINRA), stores the raw downloads
unchanged in `data/raw/`, and builds clean tables from them in `data/processed/`.

## Install

```bash
pip install -r requirements.txt
```

The universe of securities is `config/universe.csv`. Make sure `data/` is in `.gitignore`.

## Test first without internet

```bash
python br.py data init --synthetic
```

This runs the whole pipeline on fake data (`beursrally/data/fake_yf.py`). The report deliberately shows a few
problems (an ISIN that can't be found, an uncorrected split at SMCI). Delete the `data/` folder afterwards, before
fetching the real data.

## Real data

```bash
python br.py data init      # first time: 15-30 minutes
python br.py data update    # afterwards, after the US close (~1-8 minutes)
python br.py data report    # only show the quality report again
```

- **Interrupted or blocked?** Run the same command again; whatever is already in is skipped.
  If Yahoo blocks you, wait an hour.
- **Not found?** Enter the right Yahoo ticker in the `ticker` column of `config/universe.csv` and run
  `python br.py data resolve`, then `python br.py data prices` and `python br.py data process`.
- **Daily options for a selection only:** put tickers (one per line) in `data/meta/focus.txt`.
  Without that file all US stocks are fetched daily. Weekly always all of them.

## What happens

| Task | What | How often (with `update`) |
|---|---|---|
| `resolve` | Check ticker, exchange and currency per security; look up missing ones via ISIN | Only with `init` |
| `prices` | Open, High, Low, Close (adjusted for splits only), Volume + market series + EUR/USD | Daily |
| `actions` | Splits and dividends | Weekly |
| `earnings` | Dates and earnings surprises of quarterly results | Weekly |
| `info` | Sector, market cap, short interest (snapshot, live only) | Only with `init` |
| `shares` | Shares outstanding over time (for historical market cap) | Weekly |
| `shortinterest` | FINRA short interest of US stocks, twice a month, since June 2021 | Daily (unfinished months only) |
| `options` | Snapshot of options around the price, expiries around 22 January 2027 | Daily (focus), weekly (all) |
| `process` | Combine everything into tables in euro | Daily |
| `report` | Quality report | Daily |

Built-in checks:
- Returns per security on its **own** trading days: no filled-in 0% days.
- With every update, overlapping days are compared with earlier downloads. If they differ (Yahoo rescales the
  history after a split), that ticker is fetched again in full.
- Jumps above 50% are reported, noting whether there was a split around that date.

## Output in `data/processed/`

| File | Content |
|---|---|
| `close_eur`, `open_eur` | Close and open prices in euro (days × ISIN) |
| `close_local`, `high_local`, `low_local`, `volume` | In the local currency (for high/low estimators) |
| `logret_eur` | Daily log returns in euro |
| `market_close`, `market_logret` | S&P 500, Nasdaq 100, STOXX 600, MSCI World, VIX, VXN, bitcoin, EUR/USD, US rates |
| `actions`, `earnings`, `info`, `shares` | Long tables per ISIN |
| `short_interest` | Per ISIN and settlement date: short interest, average volume, days to cover |
| `options_<date>` | One options snapshot per day |

Plus `data/kwaliteitsrapport.csv` (quality report) and `data/meta/tickers.csv`.

Next: `python br.py clean` (splits, known data errors from `config/artefacten.csv`, euro, returns) and
`python br.py features` (features per stock and day; `python br.py features test` = leakage test).
