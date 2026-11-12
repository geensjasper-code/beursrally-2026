"""Vaste mappen van het project. Alle scripts gebruiken deze, waar ze ook staan.

    ROOT    de projectmap (waar br.py staat)
    DATA    ROOT/data     (niet in git: koersen, voorspellingen, standen)
    CONFIG  ROOT/config   (universe.csv, artefacten.csv)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CONFIG = ROOT / "config"
