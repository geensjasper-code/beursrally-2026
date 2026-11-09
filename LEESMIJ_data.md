# Stap 1 — Data

`data.py` haalt alle data op via Yahoo Finance, bewaart de ruwe downloads ongewijzigd in `data/raw/`
en bouwt daaruit nette tabellen in `data/processed/`.

## Installeren

```bash
pip install -r requirements.txt
```

Zet `universe.csv`, `data.py` en `fake_yf.py` in dezelfde map. Zorg dat `data/` in je `.gitignore` staat.

## Eerst testen zonder internet

```bash
python data.py init --synthetic
```

Dat draait de hele pipeline met nepdata (`fake_yf.py`). Het rapport toont bewust een paar problemen
(een ISIN die niet gevonden wordt, een splitsing die niet gecorrigeerd is bij SMCI). Verwijder daarna
de map `data/` voor je echte data ophaalt.

## Echte data

```bash
python data.py init      # eerste keer: 15-30 minuten
python data.py update    # daarna dagelijks, na de Amerikaanse sluiting (~1-8 minuten)
python data.py report    # alleen het kwaliteitsrapport opnieuw tonen
```

- **Onderbroken of geblokkeerd?** Draai hetzelfde commando opnieuw; wat al binnen is, wordt overgeslagen.
  Bij een blokkade door Yahoo: wacht een uur.
- **Niet gevonden?** Vul de juiste Yahoo-ticker in de kolom `ticker` van `universe.csv` in en draai
  `python data.py resolve`, daarna `python data.py prices` en `python data.py process`.
- **Dagelijkse opties alleen voor een selectie:** zet tickers (een per lijn) in `data/meta/focus.txt`.
  Zonder dat bestand worden dagelijks alle Amerikaanse aandelen opgehaald. Wekelijks altijd alle.

## Wat er gebeurt

| Taak | Wat | Hoe vaak (bij `update`) |
|---|---|---|
| `resolve` | Ticker, beurs en munt per effect controleren; ontbrekende via ISIN zoeken | Alleen bij `init` |
| `prices` | Open, High, Low, Close (alleen gecorrigeerd voor splitsingen), Volume + marktreeksen + EUR/USD | Dagelijks |
| `actions` | Splitsingen en dividenden | Wekelijks |
| `earnings` | Datums en winstverrassingen van kwartaalcijfers | Wekelijks |
| `info` | Sector, marktkapitalisatie, shortpositie | Alleen bij `init` |
| `options` | Momentopname van opties rond de koers, vervaldagen rond 22 januari 2027 | Dagelijks (focus), wekelijks (alle) |
| `process` | Alles samenvoegen tot tabellen in euro | Dagelijks |
| `report` | Kwaliteitsrapport | Dagelijks |

Ingebouwde controles:
- Rendementen per effect op zijn **eigen** beursdagen: geen ingevulde 0%-dagen.
- Bij elke update worden de overlappende dagen vergeleken met eerdere downloads. Wijken ze af
  (Yahoo herschaalt de historiek na een splitsing), dan wordt die ticker volledig opnieuw opgehaald.
- Sprongen boven 50% worden gemeld, met de vermelding of er rond die datum een splitsing was.

## Uitvoer in `data/processed/`

| Bestand | Inhoud |
|---|---|
| `close_eur`, `open_eur` | Slot- en openingskoersen in euro (dagen × ISIN) |
| `close_local`, `high_local`, `low_local`, `volume` | In de eigen munt (voor schatters op hoog/laag) |
| `logret_eur` | Dagelijkse log-rendementen in euro |
| `market_close`, `market_logret` | S&P 500, Nasdaq 100, STOXX 600, MSCI World, VIX, VXN, bitcoin, EUR/USD |
| `actions`, `earnings`, `info` | Lange tabellen per ISIN |
| `options_<datum>` | Een momentopname van de opties per dag |

Plus `data/kwaliteitsrapport.csv` en `data/meta/tickers.csv`.
