# Stap 3: Risicomodel

Alle scripts lezen uit `data/clean/` en `data/features/` en schrijven naar `data/risk/`.

## Eenmalig: alles opbouwen en kiezen

```
python risk_simple.py          # model 1 en 2: voortschrijdend 60 dagen, EWMA
python risk_garch.py test      # zelftest GARCH
python risk_garch.py           # model 3: GARCH per aandeel
python risk_har.py             # model 4: HAR (met blokselectie)
python risk_boost.py           # model 5a: boosting als correctie op HAR
python risk_boost.py --alleen  # model 5b: boosting op zichzelf
python risk_earn.py            # cijfersprong bovenop EWMA/GARCH/voortschrijdend
python risk_compare.py         # vergelijking op 2019-2021 -> winnaar per horizon
python risk_iv.py              # implied volatility: indexbacktest 2002-2021, c en lambda
python risk_corr.py test       # zelftest Ledoit-Wolf en clustering
python risk_corr.py            # correlatiemodellen kiezen; thema's (alleen ter informatie)
python risk_build.py           # risico voor vandaag
python risk_test.py            # lektest risicomodel
```

## Dagelijks (na de Amerikaanse slot)

```
python data.py update
python clean.py
python features.py
python risk_har.py
python risk_boost.py --alleen
python risk_build.py
```

Wekelijks ook `python risk_forward.py`: de voorwaartse test van de optiemenging (lambda).

## Keuzes tot nu toe

| Onderdeel | Keuze | Waarom |
|---|---|---|
| Volatiliteit | 1-10 dagen: gemiddelde HAR en boosting (op zichzelf); 20-48 dagen: HAR | Beste in de vergelijking 2019-2021 |
| Opties (VS-aandelen) | c = 0,9, lambda = 0,7 | c uit de indexbacktest 2002-2021 (stabiel); lambda 0,7 als voorzichtige overdracht naar aandelen |
| Correlaties | zie `data/risk/corr_keuze.txt` | Portefeuilletest 2019-2021 |
| Thema's | Alleen ter informatie; geen kenmerk (theme_mom_21 geschrapt) | Thema's te instabiel (ARI ~0,24) en voor Europa vooral "de Europese markt" |

De testperiode 2022-2026 wordt één keer bekeken, aan het einde:
`python risk_compare.py --final` en `python risk_iv.py --final`.
