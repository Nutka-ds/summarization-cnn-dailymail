"""Testy API (lekcja 40). Uruchomienie z katalogu projektu: python3 -m pytest tests/ -v"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

ARTYKUL = (
    "By . Daily Mail Reporter . PUBLISHED: . 10:00 EST, 1 May 2013 . | . UPDATED: . 11:00 EST, 1 May 2013 . "
    "The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, "
    "a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The formal accession "
    "was marked with a ceremony at The Hague, in the Netherlands, where the court is based. The Palestinians signed "
    "the ICC's founding Rome Statute in January, when they also accepted its jurisdiction over alleged crimes "
    "committed in the occupied Palestinian territory. Israel and the United States opposed the Palestinians' "
    "efforts to join the body. Scroll down for video . Later that month, the ICC opened a preliminary examination "
    "into the situation in Palestinian territories, paving the way for possible war crimes investigations."
)


@pytest.fixture(scope="module")
def klient():
    with TestClient(app) as k:      # "with" uruchamia lifespan, czyli ładuje model
        yield k


def test_health(klient):
    odp = klient.get("/health")
    assert odp.status_code == 200
    assert odp.json()["model_loaded"] is True


def test_streszczenie(klient):
    odp = klient.post("/summarize", json={"text": ARTYKUL})
    assert odp.status_code == 200
    dane = odp.json()
    assert len(dane["summary"].split()) >= 10
    assert "ICC" in dane["summary"] or "Criminal Court" in dane["summary"]
    assert dane["truncated"] is False


def test_czyszczenie_wejscia(klient):
    """Śmieci z nagłówka Daily Mail nie trafiają do streszczenia."""
    dane = klient.post("/summarize", json={"text": ARTYKUL}).json()
    for smiec in ["PUBLISHED", "UPDATED", "Daily Mail Reporter", "Scroll down"]:
        assert smiec not in dane["summary"]
    assert dane["input_words"] < len(ARTYKUL.split())


def test_za_krotki_tekst(klient):
    odp = klient.post("/summarize", json={"text": "Too short to summarize."})
    assert odp.status_code == 422


def test_brak_tekstu(klient):
    assert klient.post("/summarize", json={}).status_code == 422


def test_zly_max_length(klient):
    assert klient.post("/summarize", json={"text": ARTYKUL, "max_length": 5}).status_code == 422
