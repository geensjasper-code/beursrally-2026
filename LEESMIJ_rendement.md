# Stap 2: Rendementsmodel

Voorspelt per aandeel het AANDEELSPECIFIEKE rendement (beter of slechter dan het gemiddelde aandeel) over
1, 3, 5, 10, 20 en 48 beursdagen. Het marktdeel is vast: 7% per jaar.

## Eenmalig

```
python ret_models.py      # walk-forward voor zero, capm, composite, enet, boost (enkele minuten)
python ret_compare.py     # vergelijking 2019-2021, keuze per horizon, w, b, kostentoets
python ret_build.py       # verwacht rendement vandaag
python ret_test.py        # lektest
```

## Dagelijks (na features.py)

```
python ret_build.py
```

## Keuzes

| Onderdeel | Keuze |
|---|---|
| Kandidaten | alles gelijk, CAPM, academische samenstelling, elastic net, gradient boosting |
| Kenmerken | per dag gerangschikt (-0,5 tot +0,5); marktbrede kenmerken niet gebruikt |
| Toets | walk-forward met embargo (h + 5 dagen); kiezen op 2019-2021 |
| Hoofdmaatstaf | rang-IC (t-waarde met Newey-West) |
| Vertrouwen | w = max(0, 1 - 1/t^2) |
| Kosten | 5 euro + 0,35% per order (reglement): ~0,9% per heen-en-terug |
| Gebruikt rendement | 7%/jaar x h/252 + w x b x voorspelling |

De testperiode 2022-2026: een keer, aan het einde (`python ret_compare.py --final`).
