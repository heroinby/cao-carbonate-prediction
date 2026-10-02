"""Сборка датасета карбонатных пород (известняк / мергель / мел) из глобальной
геохимической базы (Zenodo, DOI 10.5281/zenodo.2592822, Gard et al., 2019).

Запуск:
    python -m src.build_dataset

Скрипт скачивает три таблицы (major, sample, rockgroup, ~340 МБ) в ``data/raw``
(если их там ещё нет), отбирает карбонатные осадочные породы, очищает химические
анализы и сохраняет результат в ``data/carbonates_cao.csv``.

По шагам:
    1. скачать базу, если её ещё нет;
    2. оставить осадочные карбонатные породы;
    3. привести анализы к одному виду (пределы обнаружения, форма железа);
    4. выкинуть явно битые анализы;
    5. сохранить CSV, с которым дальше работают ноутбуки.
"""
from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ZENODO_RECORD = "3359791"  # версия записи 10.5281/zenodo.2592822
# major - химия, sample - описание пробы (порода, координаты, статья), rockgroup - справочник групп
RAW_FILES = ("major.csv", "sample.csv", "rockgroup.csv")

# признаки - оксиды-примеси, цель - CaO (всё в масс. %)
FEATURES = ["sio2", "al2o3", "fe2o3_tot", "mgo", "k2o", "na2o", "tio2", "p2o5", "mno"]
TARGET = "cao"
OXIDES = FEATURES + [TARGET]

# железо в базе записано то как FeO, то как Fe2O3; всё переводим в Fe2O3
# (1.1113 - отношение молярных масс Fe2O3 / 2FeO)
FE_FACTOR = 1.1113

# класс породы берём из названия. Порядок важен: побеждает первое совпадение,
# поэтому "marl" проверяется раньше, чем общее "carbonate"
ROCK_PATTERNS = {
    "marl": r"marl",
    "chalk": r"chalk",
    "limestone": r"limestone|calcilutite|micrite|calcarenite|calcisiltite",
    "calcareous_other": r"calcareous|carbonate",
}
# доломиты и обломочные породы - не то сырьё, которое нам нужно
EXCLUDE_PATTERN = r"dolo|sandstone|greywacke|conglomerate|breccia|tuff|chert|siltstone"

MIN_CAO = 5.0  # меньше - уже не карбонат, скорее ошибка в названии породы
MAX_CAO = 56.1  # в чистом кальците CaCO3 ровно столько, больше быть не может
MAX_OXIDE_SUM = 105.0  # больше 100 % быть не должно, 5 % оставляем на погрешность


def download_raw(raw_dir: Path) -> None:
    """Скачать недостающие файлы базы с Zenodo в ``raw_dir``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        target = raw_dir / name
        if target.exists():
            continue  # уже скачано, 340 МБ второй раз не качаем
        url = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{name}/content"
        print(f"Скачиваю {name} ...")
        urllib.request.urlretrieve(url, target)


def classify_rock(name: str) -> str | None:
    """Вернуть класс породы по названию или None, если порода не подходит."""
    name = name.lower()
    if re.search(EXCLUDE_PATTERN, name):
        return None  # сначала отсеиваем доломиты и обломочные
    for cls, pattern in ROCK_PATTERNS.items():
        if re.search(pattern, name):
            return cls
    return None


def handle_detection_limit(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Обработать значения ниже предела обнаружения.

    В базе такие измерения записаны отрицательным числом (-предел обнаружения).
    Стандартная геохимическая замена — половина предела: ``|x| / 2``.
    Для нулей значение неоднозначно, поэтому они считаются пропусками.
    Добавляется счётчик ``n_censored``.
    """
    df = df.copy()
    df["n_censored"] = 0  # в скольких оксидах пробы было "ниже предела"
    for col in columns:
        censored = df[col] < 0
        df["n_censored"] += censored.astype(int)
        # модуль - это сам предел, а реальное значение между 0 и пределом, берём середину
        df.loc[censored, col] = df.loc[censored, col].abs() / 2
        df.loc[df[col] == 0, col] = np.nan  # ноль в базе значит "не измеряли"
    return df


def total_iron(df: pd.DataFrame) -> pd.Series:
    """Общее железо в форме Fe2O3: fe2o3_tot, иначе feo_tot, иначе fe2o3 + FeO."""
    iron = df["fe2o3_tot"].copy()  # лучший случай: общее железо уже дано
    iron = iron.fillna(df["feo_tot"] * FE_FACTOR)
    # в крайнем случае складываем Fe2O3 и FeO, если измерено хоть что-то
    both = df["fe2o3"].fillna(0) + df["feo"].fillna(0) * FE_FACTOR
    iron = iron.fillna(both.where((df["fe2o3"].notna()) | (df["feo"].notna())))
    return iron


def build(raw_dir: Path) -> pd.DataFrame:
    """Собрать и очистить датасет (шаги 2-4 из описания выше)."""
    sample_cols = [
        "sample_id", "latitude", "longitude", "rock_name", "sample_description",
        "major_id", "rgroup_id", "ref_id", "country_id",
    ]
    # в sample.csv есть символы не из UTF-8, с latin-1 читается без ошибок
    sample = pd.read_csv(raw_dir / "sample.csv", usecols=sample_cols,
                         encoding="latin-1", low_memory=False)
    rockgroup = pd.read_csv(raw_dir / "rockgroup.csv")
    major = pd.read_csv(raw_dir / "major.csv")

    # Шаг 2. Отбор: осадочные породы, потом по названию
    sample = sample.merge(rockgroup, on="rgroup_id", how="left")
    sample = sample[sample["rock_group"] == "sedimentary"].copy()
    sample["rock_class"] = sample["rock_name"].fillna("").map(classify_rock)
    sample = sample.dropna(subset=["rock_class"])  # None = не подошла

    # Шаг 3. Подтягиваем анализы и приводим к одному виду
    df = sample.merge(major, on="major_id")
    df["fe2o3_tot"] = total_iron(df)
    # у CaO пределы не обрабатываем: такие пробы нам не нужны, они уйдут на следующей строке
    df = handle_detection_limit(df, [c for c in OXIDES if c != TARGET])
    df = df[df[TARGET] > 0]

    # Шаг 4. Чистка. CaO, SiO2, Al2O3 и MgO обязательны, остальные оксиды могут быть пустыми
    df = df.dropna(subset=[TARGET, "sio2", "al2o3", "mgo"])
    df = df[(df[TARGET] >= MIN_CAO) & (df[TARGET] <= MAX_CAO)]
    df = df[df[OXIDES].sum(axis=1, min_count=1) <= MAX_OXIDE_SUM]
    # одна проба могла попасть в базу из разных статей, оставляем одну копию
    df = df.drop_duplicates(subset=OXIDES + ["latitude", "longitude"])

    # loi оставляем только для EDA, в модель он не пойдёт
    keep = ["sample_id", "rock_class", "rock_name", "latitude", "longitude",
            "ref_id", "country_id", *FEATURES, "loi", TARGET, "n_censored"]
    return df[keep].reset_index(drop=True)


def main() -> None:
    """Скачать данные, собрать датасет, сохранить CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/carbonates_cao.csv"))
    args = parser.parse_args()

    download_raw(args.raw_dir)
    df = build(args.raw_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Сохранено {len(df)} проб в {args.out}")
    print(df["rock_class"].value_counts())


if __name__ == "__main__":
    main()
