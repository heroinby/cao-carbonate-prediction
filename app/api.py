"""REST API для предсказания CaO (FastAPI).

Запуск: ``uvicorn app.api:app --reload``; документация — http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.predict import load_model, predict_cao

# uvicorn запускает именно этот объект: uvicorn app.api:app
app = FastAPI(title="CaO predictor", description="Оценка CaO в карбонатном сырье по примесям")


class Sample(BaseModel):
    """Состав пробы, масс. %. Обязательны SiO2, Al2O3 и MgO.

    Field(ge=0, le=100) сам проверяет, что значение от 0 до 100. Необязательные
    оксиды по умолчанию None, в модель они уйдут как пропуски.
    """

    sio2: float = Field(ge=0, le=100)
    al2o3: float = Field(ge=0, le=100)
    mgo: float = Field(ge=0, le=100)
    fe2o3_tot: float | None = Field(None, ge=0, le=100)
    k2o: float | None = Field(None, ge=0, le=100)
    na2o: float | None = Field(None, ge=0, le=100)
    tio2: float | None = Field(None, ge=0, le=100)
    p2o5: float | None = Field(None, ge=0, le=100)
    mno: float | None = Field(None, ge=0, le=100)


@app.get("/health")
def health() -> dict:
    """Жив ли сервис, плюс паспорт модели."""
    return {"status": "ok", **load_model()["meta"]}


@app.post("/predict")
def predict(sample: Sample) -> dict:
    """Предсказать CaO и 90 % интервал."""
    try:
        # типы и диапазоны уже проверил pydantic, тут ловим остальное (например, отрицательные числа)
        return predict_cao(sample.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
