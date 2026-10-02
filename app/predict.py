"""Предсказание CaO по химическому составу пробы.

Использование как библиотеки::

    from app.predict import predict_cao
    predict_cao({"sio2": 8.5, "al2o3": 2.1, "fe2o3_tot": 0.9, "mgo": 1.2})

Использование из командной строки (CSV со столбцами-оксидами)::

    python -m app.predict samples.csv predictions.csv
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.features import FEATURES

# путь от самого файла, чтобы работало из любой папки
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cao_model.joblib"
REQUIRED = ("sio2", "al2o3", "mgo")  # без них оценка ненадёжна, остальные оксиды можно не указывать


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH) -> dict:
    """Загрузить модель и метаданные. lru_cache, чтобы читать файл с диска один раз."""
    return joblib.load(path)


def predict_frame(samples: pd.DataFrame) -> pd.DataFrame:
    """Предсказать CaO для таблицы проб.

    Отсутствующие оксиды допускаются (NaN), кроме обязательных ``sio2``,
    ``al2o3`` и ``mgo``. Возвращает таблицу с колонками ``cao_pred``,
    ``cao_low`` и ``cao_high`` (90 % интервал).
    """
    bundle = load_model()
    # те же 9 столбцов в том же порядке, что при обучении; чего нет - станет NaN
    data = samples.reindex(columns=FEATURES).astype(float)
    missing = [c for c in REQUIRED if data[c].isna().any()]
    if missing:
        raise ValueError(f"Обязательные оксиды не заданы: {missing}")
    if (data.fillna(0) < 0).any().any():
        raise ValueError("Содержания оксидов должны быть неотрицательными (масс. %)")
    # полуширина интервала посчитана при обучении (см. ноутбук 02, раздел 7)
    half = bundle["meta"]["interval_halfwidth_90"]
    # CaO бывает только от 0 до 56.1 %
    pred = bundle["model"].predict(data).clip(0, 56.1)
    return pd.DataFrame({"cao_pred": pred.round(2),
                         "cao_low": (pred - half).clip(0).round(2),
                         "cao_high": (pred + half).clip(max=56.1).round(2)},
                        index=samples.index)


def predict_cao(sample: dict[str, float]) -> dict[str, float]:
    """Предсказать CaO для одной пробы, заданной словарём оксидов (масс. %)."""
    return predict_frame(pd.DataFrame([sample])).iloc[0].to_dict()


def main(argv: list[str]) -> None:
    """``python -m app.predict вход.csv выход.csv``"""
    if len(argv) != 2:
        raise SystemExit(__doc__)
    src, dst = argv
    result = predict_frame(pd.read_csv(src))
    result.to_csv(dst, index=False)
    print(f"Предсказания для {len(result)} проб сохранены в {dst}")


if __name__ == "__main__":
    main(sys.argv[1:])
