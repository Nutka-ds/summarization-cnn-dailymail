"""API do streszczania artykułów prasowych modelem bart-base fine-tunowanym na CNN/DailyMail.

Uruchomienie (z katalogu projektu):
    uvicorn api.main:app --port 8000

Dokumentacja interaktywna: http://localhost:8000/docs
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

KATALOG_PROJEKTU = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KATALOG_PROJEKTU / "src"))
from preprocessing import wyczysc_artykul  # noqa: E402  (ta sama funkcja co w treningu)

KATALOG_MODELU = Path(os.environ.get("MODEL_DIR", KATALOG_PROJEKTU / "models" / "bart-base_najlepszy"))
MAX_INPUT_LENGTH = 1024        # limit BART (EDA, E3)
MIN_SLOW = 50                  # krótsze teksty usuwaliśmy z treningu (EDA, E7)
MAX_ZNAKOW = 30_000            # ochrona przed bardzo długimi zapytaniami

# Najlepsze parametry generowania ze strojenia na walidacji (04_trening_colab.ipynb, K8)
PARAMETRY_GENEROWANIA = dict(num_beams=4, length_penalty=1.0, no_repeat_ngram_size=3, min_length=30)

model_ml = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Model ładuje się raz, przy starcie aplikacji, a nie przy każdym zapytaniu."""
    if not KATALOG_MODELU.exists():
        raise RuntimeError(f"Nie znaleziono modelu w {KATALOG_MODELU}")
    urzadzenie = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model_ml["tokenizer"] = AutoTokenizer.from_pretrained(KATALOG_MODELU)
    model_ml["model"] = AutoModelForSeq2SeqLM.from_pretrained(KATALOG_MODELU).to(urzadzenie).eval()
    model_ml["urzadzenie"] = urzadzenie
    yield
    model_ml.clear()


app = FastAPI(
    title="Streszczanie artykułów",
    description="Generuje streszczenie anglojęzycznego artykułu prasowego modelem `bart-base` "
                "fine-tunowanym na CNN/DailyMail (ROUGE-2 = 18,63 na zbiorze testowym).",
    version="1.0",
    lifespan=lifespan,
)


class Zapytanie(BaseModel):
    text: str = Field(..., max_length=MAX_ZNAKOW, description="Treść artykułu po angielsku (min. 50 słów)")
    max_length: int = Field(128, ge=30, le=256, description="Maksymalna długość streszczenia w tokenach")

    model_config = {"json_schema_extra": {"examples": [{
        "text": "The Palestinian Authority officially became the 123rd member of the International Criminal Court "
                "on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian "
                "territories. The formal accession was marked with a ceremony at The Hague, in the Netherlands, "
                "where the court is based. The Palestinians signed the ICC's founding Rome Statute in January, "
                "when they also accepted its jurisdiction over alleged crimes committed in the occupied "
                "Palestinian territory. Israel and the United States opposed the Palestinians' efforts to join.",
        "max_length": 128,
    }]}}


class Odpowiedz(BaseModel):
    summary: str = Field(description="Streszczenie, każdy punkt w osobnej linii")
    input_words: int = Field(description="Liczba słów w artykule po oczyszczeniu")
    truncated: bool = Field(description="Czy artykuł był dłuższy niż 1024 tokeny i został obcięty")
    seconds: float = Field(description="Czas generowania w sekundach")


@app.get("/health")
def health():
    """Sprawdzenie, czy API działa i model jest załadowany."""
    return {"status": "ok", "model_loaded": "model" in model_ml, "device": model_ml.get("urzadzenie")}


@app.post("/summarize", response_model=Odpowiedz)
def summarize(zapytanie: Zapytanie):
    """Streszcza artykuł. Tekst jest czyszczony tą samą funkcją co dane treningowe."""
    tekst = wyczysc_artykul(zapytanie.text)
    liczba_slow = len(tekst.split())
    if liczba_slow < MIN_SLOW:
        raise HTTPException(status_code=422, detail=f"Tekst jest za krótki: {liczba_slow} słów, wymagane co najmniej {MIN_SLOW}")

    tokenizer, model = model_ml["tokenizer"], model_ml["model"]
    start = time.time()
    wejscie = tokenizer(tekst, max_length=MAX_INPUT_LENGTH, truncation=True, return_tensors="pt").to(model_ml["urzadzenie"])
    obciety = len(tokenizer(tekst)["input_ids"]) > MAX_INPUT_LENGTH
    with torch.no_grad():
        wyjscie = model.generate(**wejscie, max_new_tokens=zapytanie.max_length, **PARAMETRY_GENEROWANIA)

    return Odpowiedz(
        summary=tokenizer.decode(wyjscie[0], skip_special_tokens=True),
        input_words=liczba_slow,
        truncated=obciety,
        seconds=round(time.time() - start, 2),
    )
