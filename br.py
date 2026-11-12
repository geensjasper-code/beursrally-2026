#!/usr/bin/env python3
"""
Beursrally 2026 - een startpunt voor alle stappen.

Gebruik
    python br.py                        # lijst van alle commando's
    python br.py <commando> [opties]    # bv. python br.py strategy --kandidaten alle

De code staat in de map beursrally/ (per stap een submap); data/ en config/ staan naast dit bestand.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

GROUPS = {
    "1. Data": {
        "data": ("beursrally.data.data", "koersen, opties, cijfers ophalen (init | update | resolve | report ...)"),
        "clean": ("beursrally.data.clean", "opkuisen: splitsingen, artefacten, euro, rendementen"),
        "features": ("beursrally.data.features", "kenmerken per aandeel en dag ('test' = lektest)"),
        "check_data": ("beursrally.data.check_data", "leesbare controle per aandeel"),
    },
    "2. Risico": {
        "risk_simple": ("beursrally.risk.risk_simple", "voortschrijdend 60 dagen en EWMA"),
        "risk_garch": ("beursrally.risk.risk_garch", "GARCH(1,1) per aandeel"),
        "risk_earn": ("beursrally.risk.risk_earn", "cijfersprong bovenop EWMA/GARCH"),
        "risk_har": ("beursrally.risk.risk_har", "HAR-regressie voor alle aandelen samen"),
        "risk_boost": ("beursrally.risk.risk_boost", "gradient boosting (--alleen: zonder HAR)"),
        "risk_iv": ("beursrally.risk.risk_iv", "opties (implied volatility) gemengd met het basismodel"),
        "risk_corr": ("beursrally.risk.risk_corr", "correlatiemodellen"),
        "risk_compare": ("beursrally.risk.risk_compare", "modellen vergelijken (--controle, --final)"),
        "risk_build": ("beursrally.risk.risk_build", "volatiliteit en correlaties van vandaag"),
        "risk_forward": ("beursrally.risk.risk_forward", "voorwaartse test van de optiemenging"),
    },
    "3. Rendement": {
        "ret_models": ("beursrally.returns.ret_models", "zes rendementsmodellen, walk-forward"),
        "ret_compare": ("beursrally.returns.ret_compare", "vergelijken; --controle, --final, --alles (w en b)"),
        "ret_build": ("beursrally.returns.ret_build", "verwacht rendement van vandaag"),
    },
    "4. Scenario's": {
        "scen_core": ("beursrally.scenarios.scen_core", "de generator ('test' = zelftest)"),
        "scen_build": ("beursrally.scenarios.scen_build", "parameters van vandaag tot 22 januari"),
        "scen_backtest": ("beursrally.scenarios.scen_backtest", "vorm kiezen en toetsen (--grid, --controle, --final)"),
    },
    "5. Spel": {
        "game": ("beursrally.game.game", "spelmotor ('test' = zelftest, 'demo')"),
        "field": ("beursrally.game.field", "gesimuleerde concurrenten ('test', 'report')"),
    },
    "6. Strategie": {
        "strategy": ("beursrally.strategy.strategy", "startportefeuille (--kandidaten, --uitsluiten)"),
        "vergelijk_velden": ("beursrally.strategy.vergelijk_velden", "portefeuilles per veldtype vergelijken"),
        "rules": ("beursrally.strategy.rules", "vaste handelsregels testen"),
        "rollout": ("beursrally.strategy.rollout", "rollout testen (de gekozen beslisregel)"),
    },
    "Wekelijks": {
        "stand_inlezen": ("beursrally.weekly.stand_inlezen", "stand inlezen via het klembord"),
        "week": ("beursrally.weekly.week", "beslissing voor maandag (--uitsluiten, --geen-fa)"),
    },
    "Tests": {
        "risk_test": ("beursrally.tests.risk_test", "lektest risicomodel"),
        "ret_test": ("beursrally.tests.ret_test", "lektest rendementsmodel"),
    },
}
COMMANDS = {k: v for g in GROUPS.values() for k, v in g.items()}


def usage() -> None:
    print(__doc__)
    for g, cmds in GROUPS.items():
        print(f"{g}")
        for k, (_, help_) in cmds.items():
            print(f"    {k:<18} {help_}")
        print()


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        usage()
        return
    cmd = sys.argv[1].removesuffix(".py")
    if cmd not in COMMANDS:
        print(f"Onbekend commando: {cmd}\n")
        usage()
        sys.exit(2)
    mod = importlib.import_module(COMMANDS[cmd][0])
    sys.argv = [f"br.py {cmd}"] + sys.argv[2:]          # de module ziet alleen haar eigen opties
    (getattr(mod, "cli", None) or mod.main)()         # cli(): modules met een zelftest-optie


if __name__ == "__main__":
    main()
