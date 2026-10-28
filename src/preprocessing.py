"""Czyszczenie i filtrowanie danych CNN/DailyMail.

Te same funkcje są używane w treningu, ewaluacji i w API, żeby model zawsze
dostawał tekst przygotowany w ten sam sposób (decyzje z EDA, tydzień 2, E9).
"""

import re

import pandas as pd

# --- Wzorce śmieci w artykułach (EDA, E6) ---

# Daty publikacji Daily Mail: "PUBLISHED: . 15:54 EST, 13 March 2012 . | . UPDATED: . 19:50 EST, 15 March 2012 ."
# W starszych artykułach występuje samo "UPDATED: . 19:03 EST, 29 December 2011 ."
DATY_PUBLIKACJI = re.compile(r"(?:PUBLISHED|UPDATED): \. [^|.]{1,40}?\d{4} \. (?:\| \. )?")
# Najstarszy format: "Last updated at 11:05 PM on 17th June 2011 ." lub "Created 5:33 PM on 10th July 2011 ."
OSTATNIA_AKTUALIZACJA = re.compile(r"(?:Last updated at|Created) [^.]{1,40}? on [^.]{1,30}?\d{4} \. ")
# Podpis autora na początku artykułu: "By . Daily Mail Reporter ."
PODPIS_NA_POCZATKU = re.compile(r"^By \. [^.]{1,80}? \. ")
# Podpis autora w środku nagłówka, tuż przed datą publikacji
PODPIS_PRZED_DATA = re.compile(r"By \. [^.]{1,80}? \. (?=PUBLISHED: \.|UPDATED: \.|Last updated at|Created \d)")
# Napis przy przycisku wideo
PRZEWIN_DO_WIDEO = re.compile(r"\s*Scroll down for video\s*\.?")

# --- Wzorce w streszczeniach (tydzień 1 i EDA, E6) ---

# Krótki punkt zaczynający się od WARNING, np. "WARNING: GRAPHIC CONTENT ." albo "WARNING, CONTAINS SPOILERS ."
DOPISEK_WARNING = re.compile(r"^WARNING\b[^\n]{0,60}(\n|$)", flags=re.MULTILINE)
SPACJA_PRZED_KROPKA = re.compile(r" \.(?=\n|$)", flags=re.MULTILINE)


def wyczysc_artykul(tekst: str) -> str:
    """Usuwa z artykułu podpis autora, daty publikacji i napis 'Scroll down for video'."""
    tekst = PODPIS_PRZED_DATA.sub("", tekst)
    tekst = PODPIS_NA_POCZATKU.sub("", tekst)
    tekst = DATY_PUBLIKACJI.sub("", tekst)
    tekst = OSTATNIA_AKTUALIZACJA.sub("", tekst)
    tekst = PRZEWIN_DO_WIDEO.sub(" ", tekst)
    return re.sub(r"\s+", " ", tekst).strip()


def wyczysc_streszczenie(tekst: str) -> str:
    """Usuwa dopiski 'WARNING: ...' i spację przed kropką na końcu punktów streszczenia."""
    tekst = DOPISEK_WARNING.sub("", tekst)
    tekst = SPACJA_PRZED_KROPKA.sub(".", tekst)
    return tekst.strip()


def filtruj_train(train: pd.DataFrame, val: pd.DataFrame, min_slow: int = 50) -> tuple[pd.DataFrame, dict]:
    """Usuwa z train przykłady, które EDA (E7, E8) uznała za szkodliwe.

    Zwraca przefiltrowany zbiór i słownik z liczbą przykładów usuniętych przez każdy filtr.
    Filtry działają po kolei, więc każdy przykład jest liczony tylko raz.
    """
    raport = {}
    przed = len(train)

    train = train.drop_duplicates(subset=["article", "highlights"])
    raport["powtórzone pary"] = przed - len(train)

    art_slowa = train["article"].str.split().str.len()
    hl_slowa = train["highlights"].str.split().str.len()

    krotkie = art_slowa < min_slow
    raport[f"artykuł < {min_slow} słów"] = int(krotkie.sum())
    train, art_slowa, hl_slowa = train[~krotkie], art_slowa[~krotkie], hl_slowa[~krotkie]

    dluzsze = hl_slowa > art_slowa
    raport["streszczenie dłuższe niż artykuł"] = int(dluzsze.sum())
    train = train[~dluzsze]

    wspolne_z_val = train["article"].isin(set(val["article"]))
    raport["wspólne z validation"] = int(wspolne_z_val.sum())
    train = train[~wspolne_z_val]

    puste_streszczenia = train["highlights"].str.strip() == ""
    raport["puste streszczenie po czyszczeniu"] = int(puste_streszczenia.sum())
    train = train[~puste_streszczenia]

    return train.reset_index(drop=True), raport
