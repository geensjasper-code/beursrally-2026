#!/usr/bin/env python3
"""
Beursrally 2026 - de stand van de week inlezen zonder te typen.

Gebruik
    python stand_inlezen.py --ik "Mijn Spelernaam"     # een keer: je eigen spelernaam (bewaard)
    python stand_inlezen.py                            # klembord-modus
        Open op de Beursrally-site elke pagina en doe Ctrl+A, Ctrl+C:
          - je eigen portefeuille (alle pagina's: 1 of 2, 2 of 2, ...)
          - het klassement (studenten en algemeen)
          - de portefeuilles van de top 5 studenten en top 5 algemeen (alle pagina's)
        Het script ziet elke nieuwe kopie, herkent wat het is en bewaart ze. Druk Enter als je klaar bent.
    python stand_inlezen.py --opnieuw 2026-11-20       # bewaarde kopieen van die datum opnieuw verwerken
    python stand_inlezen.py --bestand kopie.txt         # een tekstbestand verwerken (om te testen)

Dit leest alleen pagina's die je zelf opent en kopieert: geen scraping, geen automatische orders.

Uitvoer (data/stand/<datum>/):
    raw/NNN.txt           elke kopie, ongewijzigd
    portefeuilles.json    per speler: totaal, cash, posities (isin, aantal, waarde)
    posities.csv          dezelfde posities als tabel, met 'in_data' (zit het effect in onze data?)
    klassement.csv        rang, speler, waarde, student ja/nee (zodra het klassement-formaat bekend is)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STAND = DATA / "stand"
ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
NUM_RE = re.compile(r"[-+]?\d{1,3}(?:\.\d{3})*(?:,\d+)?|[-+]?\d+(?:,\d+)?")


def num(s: str) -> float:
    """Belgisch getal ('5.012,65 €', '+1.202,38 €', '(-3,71%)', '6,00') -> float."""
    m = NUM_RE.search(s.replace("\xa0", " ").replace(" ", ""))
    if not m:
        raise ValueError(f"geen getal in {s!r}")
    return float(m.group(0).replace(".", "").replace(",", "."))


def lines_of(text: str) -> list[str]:
    return [ln.strip() for ln in text.replace("\r", "").split("\n")]


# --------------------------------------------------------------------------- portefeuille
@dataclass
class Portfolio:
    speler: str
    datum: str                      # 'YYYY-MM-DD' van 'Laatst bijgewerkt'
    totaal: float
    belegd: float
    cash: float
    posities: dict = field(default_factory=dict)    # isin -> {naam, gewicht, waarde, aantal, wv}
    paginas: set = field(default_factory=set)
    n_paginas: int = 1
    eigen: bool = False

    def to_json(self):
        return {"speler": self.speler, "datum": self.datum, "totaal": self.totaal, "belegd": self.belegd,
                "cash": self.cash, "paginas": f"{len(self.paginas)}/{self.n_paginas}",
                "posities": [{"isin": k, **v} for k, v in self.posities.items()]}


def is_portfolio(text: str) -> bool:
    return "Huidige posities" in text and "Laatst bijgewerkt" in text


def parse_portfolio(text: str) -> Portfolio:
    L = [ln for ln in lines_of(text)]
    nz = [ln for ln in L if ln]
    i = next(k for k, ln in enumerate(nz) if ln.startswith("Laatst bijgewerkt"))
    d = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", nz[i]).groups()
    datum = f"{d[2]}-{d[1]}-{d[0]}"
    j = i + 1
    if len(nz[j]) == 1 and "€" not in nz[j + 1]:    # eigen pagina: eerst een initiaal ('J'), dan de naam
        j += 1
    speler = nz[j]
    totaal = num(nz[j + 1])
    # cash: de rij 'CashEUR' (gewicht, dan bedrag); belegd = totaal - cash
    kc = next((k for k, ln in enumerate(nz) if ln.replace(" ", "").startswith("CashEUR")), None)
    if kc is not None:
        cash = num(next(ln for ln in nz[kc + 1:kc + 4] if "€" in ln))
    elif "CASH" in nz:
        cash = num(nz[nz.index("CASH") - 1])
    else:
        cash = 0.0
    belegd = totaal - cash
    pf = Portfolio(speler, datum, totaal, belegd, cash)
    pf.eigen = "Mijn Portefeuille" in text
    m = re.search(r"(\d+)\s+of\s+(\d+)", text)
    if m:
        pf.paginas.add(int(m.group(1)))
        pf.n_paginas = int(m.group(2))
    else:
        pf.paginas.add(1)
    # posities: een ISIN-regel; naam ervoor; daarna gewicht, 'waarde<TAB>aantal', W/V, (W/V%)
    for k, ln in enumerate(nz):
        if not ISIN_RE.match(ln):
            continue
        try:
            naam = nz[k - 1]
            gewicht = num(nz[k + 1]) / 100
            raw_vals = nz[k + 2]
            cells = [c for c in re.split(r"\t", raw_vals) if c.strip()]
            if len(cells) >= 2:
                waarde, aantal = num(cells[0]), num(cells[1])
            else:                                   # tab verloren: 'waarde €  aantal'
                nums = NUM_RE.findall(raw_vals.replace(" ", " "))
                waarde, aantal = num(nums[0]), num(nums[1])
            wv = num(nz[k + 3]) if k + 3 < len(nz) and "€" in nz[k + 3] else float("nan")
        except (ValueError, IndexError) as e:
            raise ValueError(f"positie bij {ln} ({pf.speler}) niet te lezen: {e}")
        pf.posities[ln] = {"naam": naam, "gewicht": gewicht, "waarde": waarde, "aantal": aantal, "wv": wv}
    return pf


def split_pages(text: str) -> list[str]:
    """Een kopie kan meerdere pagina's bevatten (bv. alles in een keer geplakt): splits op 'Orderoverzicht'."""
    parts = re.split(r"(?m)^\s*(?=(?:Orderoverzicht|Mijn Portefeuille)\s*$)", text)
    return [p for p in parts if p.strip() and is_portfolio(p)]


def merge(pfs: list[Portfolio]) -> dict[str, Portfolio]:
    out: dict[str, Portfolio] = {}
    for pf in pfs:
        key = pf.speler
        if key not in out:
            out[key] = pf
            continue
        o = out[key]
        if pf.datum != o.datum:
            print(f"  LET OP: {key}: pagina's van verschillende datums ({o.datum} en {pf.datum}); de nieuwste telt")
            if pf.datum > o.datum:
                out[key] = pf
            continue
        o.posities.update(pf.posities)
        o.paginas |= pf.paginas
        o.n_paginas = max(o.n_paginas, pf.n_paginas)
    return out


# --------------------------------------------------------------------------- klassement
START = 50_000.0
TOP_N = 5                     # zoveel portefeuilles per klassement nodig


def is_ranking(text: str) -> bool:
    return "Uw huidige positie in het klassement" in text and not is_portfolio(text)


@dataclass
class Ranking:
    soort: str                # 'student' of 'algemeen'
    datum: str
    eigen_rang: int
    deelnemers: int
    rijen: dict = field(default_factory=dict)       # rang -> (speler, resultaat)
    eigen: tuple | None = None


def parse_ranking(text: str) -> Ranking:
    nz = [ln for ln in lines_of(text) if ln]
    i = next(k for k, ln in enumerate(nz) if ln.startswith("Laatst bijgewerkt"))
    d = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", nz[i]).groups()
    pos = next(ln for ln in nz if ln.startswith("Uw huidige positie"))
    soort = "student" if re.search(r"klassement Student\b", pos) else "algemeen"
    m = re.search(r"is ([\d.]+) op een totaal van ([\d.]+)", pos)
    eigen, totaal = int(m.group(1).replace(".", "")), int(m.group(2).replace(".", ""))
    r = Ranking(soort, f"{d[2]}-{d[1]}-{d[0]}", eigen, totaal)
    k0 = nz.index("Resultaat") + 1
    rows = []
    k = k0
    while k + 3 < len(nz) + 1 and k < len(nz):
        # rij = rang / initiaal / naam / resultaat ('+8.004,74 €')
        if re.fullmatch(r"[\d.]+", nz[k]) and k + 3 < len(nz) and "€" in nz[k + 3]:
            rows.append((int(nz[k].replace(".", "")), nz[k + 2], num(nz[k + 3])))
            k += 4
        else:
            k += 1
    # de eerste rij is je eigen plaats (staat bovenaan); die telt niet als klassementrij
    if rows and rows[0][0] == eigen:
        r.eigen = rows[0]
        rows = rows[1:]
    for rang, sp, res in rows:
        r.rijen[rang] = (sp, res)
    return r


def merge_rankings(rks: list[Ranking]) -> dict[str, Ranking]:
    out: dict[str, Ranking] = {}
    for r in rks:
        o = out.get(r.soort)
        if o is None or r.datum > o.datum:
            out[r.soort] = r
        elif r.datum == o.datum:
            o.rijen.update(r.rijen)
    return out


def todo(rks: dict[str, Ranking], pfs: dict, ik: str | None) -> list[str]:
    """Welke portefeuilles moet je nog kopieren? Top 5 van elk klassement + jezelf."""
    need = []
    for soort in ("student", "algemeen"):
        r = rks.get(soort)
        if r is None:
            need.append(f"klassement {soort}")
            continue
        for rang in range(1, TOP_N + 1):
            if rang in r.rijen and r.rijen[rang][0] not in pfs and r.rijen[rang][0] not in need:
                need.append(r.rijen[rang][0])
    if ik and ik not in pfs:
        need.append(f"{ik} (jezelf)")
    return need


# --------------------------------------------------------------------------- controle en opslaan
def check(pfs: dict[str, Portfolio], known: set[str]) -> list[str]:
    msgs = []
    for sp, pf in pfs.items():
        tot_pos = sum(v["waarde"] for v in pf.posities.values())
        if abs(tot_pos - pf.belegd) > max(2.0, 0.002 * pf.belegd):
            msgs.append(f"{sp}: posities samen {tot_pos:,.2f} EUR, pagina zegt belegd {pf.belegd:,.2f} EUR "
                        f"(pagina's {sorted(pf.paginas)} van {pf.n_paginas})")
        if abs(pf.belegd + pf.cash - pf.totaal) > 1.0:
            msgs.append(f"{sp}: belegd + cash = {pf.belegd + pf.cash:,.2f} EUR, totaal {pf.totaal:,.2f} EUR")
        missing = set(range(1, pf.n_paginas + 1)) - pf.paginas
        if missing:
            msgs.append(f"{sp}: pagina('s) {sorted(missing)} van {pf.n_paginas} ontbreken -> kopieer die nog")
        onb = [f"{v['naam']} ({k})" for k, v in pf.posities.items() if k not in known]
        if onb:
            msgs.append(f"{sp}: niet in onze data: " + ", ".join(onb))
    return msgs


def known_isins() -> set[str]:
    f = DATA / "meta" / "tickers.csv"
    return set(pd.read_csv(f, dtype=str)["isin"]) if f.exists() else set()


def process(texts: list[str], ik: str | None, out_dir: Path | None = None, quiet=False):
    pfs, rankings, other = [], [], 0
    for t in texts:
        if is_portfolio(t):
            for page in split_pages(t) or [t]:
                pfs.append(parse_portfolio(page))
        elif is_ranking(t):
            rankings.append(parse_ranking(t))
        else:
            other += 1
    merged = merge(pfs)
    rks = merge_rankings(rankings)
    own = [pf.speler for pf in pfs if pf.eigen]
    if own:
        ik = own[-1]
        (STAND / "ik.txt").write_text(ik, encoding="utf-8")
    known = known_isins()
    datum = max([p.datum for p in merged.values()] + [r.datum for r in rks.values()],
                default=time.strftime("%Y-%m-%d"))
    if not quiet:
        print(f"\n================ STAND {datum} ================")
        for sp, pf in merged.items():
            tag = "  <- IK" if ik and sp == ik else ""
            print(f"  {sp:<28} {pf.totaal:>11,.2f} EUR | cash {pf.cash:>9,.2f} | {len(pf.posities):>2} posities | "
                  f"pagina's {len(pf.paginas)}/{pf.n_paginas}{tag}")
        for soort, r in rks.items():
            top3 = ", ".join(f"{sp} {START + res:,.0f}" for rang, (sp, res) in sorted(r.rijen.items())[:3])
            print(f"  klassement {soort:<8}: jij {r.eigen_rang:,} van {r.deelnemers:,} | top 3: {top3}")
        print(f"  niet herkend: {other}")
        nog = todo(rks, merged, ik)
        if nog:
            print("  NOG TE KOPIEREN: " + ", ".join(nog))
        if ik and ik not in merged:
            print(f"  LET OP: je eigen portefeuille ('{ik}') ontbreekt")
        msgs = check(merged, known)
        print("Controle: " + ("alles klopt" if not msgs else ""))
        for m in msgs:
            print("  - " + m)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        js = {"datum": datum, "ik": ik, "spelers": [pf.to_json() for pf in merged.values()]}
        (out_dir / "portefeuilles.json").write_text(json.dumps(js, indent=2, ensure_ascii=False), encoding="utf-8")
        rows = [{"speler": sp, "ik": sp == ik, "isin": k, "naam": v["naam"], "aantal": v["aantal"],
                 "waarde": v["waarde"], "in_data": k in known}
                for sp, pf in merged.items() for k, v in pf.posities.items()]
        rows += [{"speler": sp, "ik": sp == ik, "isin": "CASH", "naam": "cash", "aantal": 0, "waarde": pf.cash,
                  "in_data": True} for sp, pf in merged.items()]
        pd.DataFrame(rows).to_csv(out_dir / "posities.csv", index=False)
        kr = [{"soort": s_, "rang": rang, "speler": sp, "resultaat": res, "waarde": START + res,
               "deelnemers": r.deelnemers, "eigen_rang": r.eigen_rang}
              for s_, r in rks.items() for rang, (sp, res) in sorted(r.rijen.items())]
        pd.DataFrame(kr).to_csv(out_dir / "klassement.csv", index=False)
        if not quiet:
            print(f"-> {out_dir / 'portefeuilles.json'} en posities.csv")
    return merged, rks


# --------------------------------------------------------------------------- klembord
IK_NOW = [None]


def watch_clipboard() -> list[str]:
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()

    def get():
        try:
            return root.clipboard_get()
        except tk.TclError:
            return ""

    import threading
    stop = threading.Event()
    threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
    last, texts = get(), []
    print("Klaar om te kopieren: open elke pagina en doe Ctrl+A, Ctrl+C. Druk Enter als je klaar bent.\n")
    while not stop.is_set():
        root.update()
        t = get()
        if t and t != last:
            last = t
            texts.append(t)
            if is_portfolio(t):
                try:
                    for pf in (parse_portfolio(p) for p in (split_pages(t) or [t])):
                        print(f"  [{len(texts):>2}] portefeuille {pf.speler}: pagina {sorted(pf.paginas)} van "
                              f"{pf.n_paginas}, {len(pf.posities)} posities")
                except Exception as e:
                    print(f"  [{len(texts):>2}] portefeuille, maar niet te lezen: {e}")
            elif is_ranking(t):
                try:
                    r = parse_ranking(t)
                    print(f"  [{len(texts):>2}] klassement {r.soort}: rangen {min(r.rijen)}-{max(r.rijen)}")
                except Exception as e:
                    print(f"  [{len(texts):>2}] klassement, maar niet te lezen: {e}")
            else:
                print(f"  [{len(texts):>2}] niet herkend (wordt toch bewaard)")
            try:
                m_, r_ = process(texts, IK_NOW[0], None, quiet=True)
                nog = todo(r_, m_, IK_NOW[0])
                print("       nog: " + (", ".join(nog) if nog else "niets - druk Enter"))
            except Exception:
                pass
        time.sleep(0.3)
    root.destroy()
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ik", help="je eigen spelernaam (wordt bewaard)")
    ap.add_argument("--opnieuw", help="datum-map in data/stand opnieuw verwerken")
    ap.add_argument("--bestand", help="een tekstbestand verwerken")
    a = ap.parse_args()
    STAND.mkdir(parents=True, exist_ok=True)
    f_ik = STAND / "ik.txt"
    if a.ik:
        f_ik.write_text(a.ik.strip(), encoding="utf-8")
        print(f"Eigen spelernaam bewaard: {a.ik.strip()}")
        return
    ik = f_ik.read_text(encoding="utf-8").strip() if f_ik.exists() else None
    if ik is None:
        print("Tip: zet eerst je spelernaam: python stand_inlezen.py --ik \"Naam\"")

    if a.bestand:
        process([Path(a.bestand).read_text(encoding="utf-8")], ik)
        return
    if a.opnieuw:
        d = STAND / a.opnieuw
        texts = [p.read_text(encoding="utf-8") for p in sorted((d / "raw").glob("*.txt"))]
        process(texts, ik, d)
        return

    IK_NOW[0] = ik
    texts = watch_clipboard()
    if not texts:
        print("Niets gekopieerd.")
        return
    merged, _ = process(texts, ik, None, quiet=True)
    r_ = _
    datum = max([p.datum for p in merged.values()] + [r.datum for r in r_.values()],
                default=time.strftime("%Y-%m-%d"))
    d = STAND / datum
    (d / "raw").mkdir(parents=True, exist_ok=True)
    start = len(list((d / "raw").glob("*.txt")))
    for k, t in enumerate(texts):
        (d / "raw" / f"{start + k + 1:03d}.txt").write_text(t, encoding="utf-8")
    all_texts = [p.read_text(encoding="utf-8") for p in sorted((d / "raw").glob("*.txt"))]
    process(all_texts, ik, d)                     # alles van die datum samen (ook eerdere sessies)


if __name__ == "__main__":
    main()
