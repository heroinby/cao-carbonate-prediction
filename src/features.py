"""Загрузка данных, группы для валидации и признаки.

Три идеи, на которых всё держится:
    * группы: пробы из одной статьи похожи, поэтому при проверке их нельзя
      делить между train и test (make_groups);
    * физическая формула: CaO можно прикинуть из баланса масс без обучения
      (stoichiometric_cao), это сильный baseline;
    * признаки, выведенные из той же физики (OxideFeatures).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

from src.build_dataset import FEATURES, TARGET

DATA_PATH = Path("data/carbonates_cao.csv")
RANDOM_STATE = 42  # один сид везде, чтобы результаты повторялись

# Сколько CO2 уходит при прокаливании на единицу оксида (CaCO3 -> CaO + CO2,
# MgCO3 -> MgO + CO2). Это просто отношения молярных масс: CO2 44.01, CaO 56.08, MgO 40.30.
CO2_PER_CAO = 44.01 / 56.08
CO2_PER_MGO = 44.01 / 40.30


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Прочитать готовый датасет."""
    return pd.read_csv(path)


def make_groups(df: pd.DataFrame, n_blocks: int = 8, seed: int = RANDOM_STATE) -> pd.Series:
    """Метка группы для GroupKFold: номер публикации (ref_id).

    Одна публикация даёт больше половины проб. Если оставить её одной группой,
    в какой-то фолд попадёт больше половины данных, поэтому её дополнительно
    режем на n_blocks географических блоков (KMeans по координатам).
    """
    groups = df["ref_id"].astype(str)
    big = df["ref_id"] == df["ref_id"].value_counts().idxmax()  # самая большая публикация
    # широту и долготу переводим в точки на сфере, чтобы KMeans не путался на ±180°
    lat, lon = np.radians(df.loc[big, "latitude"]), np.radians(df.loc[big, "longitude"])
    xyz = np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    blocks = KMeans(n_clusters=n_blocks, n_init=10, random_state=seed).fit_predict(xyz)
    groups = groups.copy()
    # к номеру публикации дописываем номер блока: "123" -> "123_b4"
    groups.loc[big] = groups.loc[big] + "_b" + pd.Series(blocks, index=groups.index[big]).astype(str)
    return groups


def stoichiometric_cao(X: pd.DataFrame) -> pd.Series:
    """Физический baseline: CaO из баланса масс, без обучения.

    Сумма анализа = 100 %: CaO + MgO + прочие оксиды + ППП, где потери при
    прокаливании (ППП) — это CO2 карбонатов: ``0.785*CaO + 1.092*MgO``. Отсюда
    ``CaO = (100 - (1 + 1.092)*MgO - прочие) / (1 + 0.785)``. Пропущенные оксиды
    считаются нулевыми.
    """
    rest = X[[c for c in FEATURES if c != "mgo"]].fillna(0).sum(axis=1)  # всё, кроме MgO
    mgo = X["mgo"].fillna(0)
    cao = (100 - (1 + CO2_PER_MGO) * mgo - rest) / (1 + CO2_PER_CAO)
    return cao.clip(lower=0, upper=56.1)  # 56.1 = чистый кальцит, выше не бывает


class OxideFeatures(BaseEstimator, TransformerMixin):
    """Добавляет к девяти оксидам производные признаки. Пропуски не трогает.

    - ``oxide_sum`` — сумма оксидов без CaO (пропуски как 0);
    - ``stoich_cao`` — оценка CaO из баланса масс (см. ``stoichiometric_cao``);
    - ``log_si_al`` — log(SiO2 / Al2O3), намёк на глинистость;
    - ``mg_share`` — MgO / (MgO + SiO2 + Al2O3), намёк на доломитизацию;
    - ``n_missing`` — сколько оксидов не измерено (косвенно говорит о методе анализа).

    Ставится первым шагом в Pipeline. Ничего не учит по данным, так что утечки нет.
    """

    def fit(self, X: pd.DataFrame, y=None):
        # учить нечего, запоминаем только имена столбцов (так хочет sklearn)
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[FEATURES].copy()
        out["oxide_sum"] = X[FEATURES].fillna(0).sum(axis=1)
        out["stoich_cao"] = stoichiometric_cao(X)
        out["log_si_al"] = np.log(X["sio2"] / X["al2o3"].clip(lower=0.01))  # clip: не делить на 0
        denom = X["mgo"] + X["sio2"] + X["al2o3"]
        out["mg_share"] = X["mgo"] / denom
        out["n_missing"] = X[FEATURES].isna().sum(axis=1)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURES + ["oxide_sum", "stoich_cao", "log_si_al", "mg_share", "n_missing"])


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Признаки отдельно, целевая переменная отдельно."""
    return df[FEATURES].copy(), df[TARGET].copy()
