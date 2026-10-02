"""Модели, метрики и групповая кросс-валидация.

В этом файле:
    * простые модели для сравнения (медиана, физическая формула, Ridge);
    * деревья и бустинги (RandomForest, ExtraTrees, LightGBM, CatBoost);
    * два ансамбля: стекинг и гибрид «формула + бустинг на остатках»;
    * кросс-валидация, где пробы одной публикации не делятся между train и test.
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import RANDOM_STATE, OxideFeatures, stoichiometric_cao

warnings.filterwarnings("ignore", category=UserWarning)  # иначе вывод ноутбука засоряется


class MedianBaseline(BaseEstimator, RegressorMixin):
    """Всегда отвечает медианой из обучающей выборки. Хуже быть уже нельзя."""

    def fit(self, X, y):
        self.median_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.median_)


class StoichBaseline(BaseEstimator, RegressorMixin):
    """Формула баланса масс из features.py. Ничему не учится."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return stoichiometric_cao(X).to_numpy()


def linear_pipeline(with_features: bool = True) -> Pipeline:
    """Ridge: пропуски -> медиана, масштаб -> StandardScaler, alpha подбирает RidgeCV."""
    steps = [OxideFeatures()] if with_features else []
    steps += [SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
              RidgeCV(alphas=np.logspace(-3, 3, 25))]
    return make_pipeline(*steps)


def tree_pipeline(model) -> Pipeline:
    """Новые признаки + модель. LightGBM и CatBoost сами умеют работать с NaN."""
    return make_pipeline(OxideFeatures(), model)


def forest_pipeline(model) -> Pipeline:
    """То же, но леса из sklearn NaN не понимают, поэтому сначала импутация."""
    return make_pipeline(OxideFeatures(), SimpleImputer(strategy="median"), model)


def default_models(seed: int = RANDOM_STATE) -> dict[str, Callable[[], BaseEstimator]]:
    """Словарь «название -> функция, которая создаёт свежую модель».

    Именно функции, а не готовые модели: в каждом фолде нужна новая, чтобы
    она не помнила прошлый фолд.
    """
    return {
        "Median": lambda: MedianBaseline(),
        "Stoichiometry": lambda: StoichBaseline(),
        "Ridge (9 оксидов)": lambda: linear_pipeline(with_features=False),
        "Ridge (+признаки)": lambda: linear_pipeline(),
        "RandomForest": lambda: forest_pipeline(
            RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=seed)),
        "ExtraTrees": lambda: forest_pipeline(
            ExtraTreesRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=seed)),
        "LightGBM": lambda: tree_pipeline(LGBMRegressor(
            n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=15,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
            verbose=-1, n_jobs=1)),
        "CatBoost": lambda: tree_pipeline(CatBoostRegressor(
            iterations=600, learning_rate=0.05, depth=6, random_seed=seed,
            verbose=False, thread_count=1, allow_writing_files=False)),
    }


def _group_splits(X: pd.DataFrame, n_splits: int):
    """Разбиения GroupKFold. Группы берём из индекса X (см. with_groups)."""
    n_groups = X.index.nunique()
    return list(GroupKFold(n_splits=min(n_splits, n_groups)).split(X, groups=X.index))


def with_groups(X: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Копия X, где индекс = метка группы. Так модель внутри fit знает группы."""
    return X.set_axis(groups.to_numpy())


class GroupStack(BaseEstimator, RegressorMixin):
    """Стекинг с групповой валидацией внутри.

    Базовые модели предсказывают пробы, которых не видели (out-of-fold, группы
    берём из индекса X). Поверх этих предсказаний учится LinearRegression с
    положительными весами, то есть просто взвешенное среднее моделей.
    """

    def __init__(self, estimators: dict, n_splits: int = 4):
        self.estimators = estimators
        self.n_splits = n_splits

    def fit(self, X, y):
        names = list(self.estimators)
        # oof = out-of-fold: предсказание для пробы делает модель, которая её группу не видела
        oof = np.zeros((len(y), len(names)))
        for tr, va in _group_splits(X, self.n_splits):
            for j, name in enumerate(names):
                oof[va, j] = clone(self.estimators[name]).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
        self.names_ = names
        # а сами базовые модели для предсказаний обучаем на всех данных
        self.models_ = [clone(self.estimators[n]).fit(X, y) for n in names]
        # positive=True: отрицательные веса на таких данных только переобучают
        self.meta_ = LinearRegression(positive=True).fit(oof, y)
        return self

    def _base_predictions(self, X):
        # строки - пробы, столбцы - базовые модели
        return np.column_stack([m.predict(X) for m in self.models_])

    def predict(self, X):
        return self.meta_.predict(self._base_predictions(X))


class ResidualBoost(BaseEstimator, RegressorMixin):
    """Гибрид: простая модель даёт основу, бустинг доучивает её ошибки.

    Остатки считаем по out-of-fold предсказаниям. Если брать предсказания на
    обучающих данных, остатки получатся слишком маленькими и бустингу нечему учиться.
    """

    def __init__(self, base, booster, n_splits: int = 5):
        self.base = base
        self.booster = booster
        self.n_splits = n_splits

    def fit(self, X, y):
        self.base_ = clone(self.base).fit(X, y)
        oof = np.zeros(len(y))
        for tr, va in _group_splits(X, self.n_splits):
            oof[va] = clone(self.base).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
        # бустинг учим предсказывать ошибку базовой модели, а не сам CaO
        self.booster_ = clone(self.booster).fit(_with_base(X, oof), y - oof)
        return self

    def predict(self, X):
        base = self.base_.predict(X)
        return base + self.booster_.predict(_with_base(X, base))


def _with_base(X: pd.DataFrame, base_pred: np.ndarray) -> pd.DataFrame:
    """Дописать ответ базовой модели как ещё один признак."""
    out = X.copy()
    out["base_pred"] = base_pred
    return out


def lgbm_model(seed: int = RANDOM_STATE, **params) -> LGBMRegressor:
    """LightGBM с небольшими деревьями: данных мало, большие сразу переобучатся."""
    base = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=15,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
    return LGBMRegressor(**{**base, **params, "random_state": seed, "verbose": -1, "n_jobs": 1})


def catboost_model(seed: int = RANDOM_STATE, **params) -> CatBoostRegressor:
    """CatBoost, настройки умеренные."""
    base = dict(iterations=600, learning_rate=0.05, depth=6)
    return CatBoostRegressor(**{**base, **params, "random_seed": seed, "verbose": False,
                                "thread_count": 1, "allow_writing_files": False})


def build_stack(seed: int = RANDOM_STATE, lgbm_params: dict | None = None,
                cat_params: dict | None = None) -> GroupStack:
    """Стекинг из Ridge, LightGBM, CatBoost и ExtraTrees.

    Модели разные по устройству, поэтому ошибаются по-разному и частично
    компенсируют друг друга.
    """
    return GroupStack({
        "ridge": linear_pipeline(with_features=False),
        "lgbm": tree_pipeline(lgbm_model(seed, **(lgbm_params or {}))),
        "cat": tree_pipeline(catboost_model(seed, **(cat_params or {}))),
        "et": forest_pipeline(ExtraTreesRegressor(300, min_samples_leaf=2, n_jobs=-1,
                                                  random_state=seed)),
    })


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """RMSE и MAE (в п.п. CaO, меньше = лучше) и R2 (1 = идеально, 0 = как медиана)."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def group_cv_predict(factory: Callable[[], BaseEstimator], X: pd.DataFrame, y: pd.Series,
                     groups: pd.Series, n_splits: int = 5, seed: int = RANDOM_STATE) -> np.ndarray:
    """Out-of-fold предсказания для всех проб: GroupKFold с перемешиванием групп."""
    oof = np.zeros(len(y))
    Xg = with_groups(X, groups)  # группы в индексе, их использует GroupStack
    cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in cv.split(X, y, groups):
        model = factory()
        model.fit(Xg.iloc[tr], y.iloc[tr])
        oof[va] = model.predict(Xg.iloc[va])
    return oof


def repeated_group_cv(factories: dict[str, Callable[[], BaseEstimator]], X: pd.DataFrame,
                      y: pd.Series, groups: pd.Series, seeds=(0, 1, 2),
                      n_splits: int = 5) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """CV несколько раз с разными seed. Возвращает метрики и oof-предсказания первого seed."""
    rows, oofs = [], {}
    for name, factory in factories.items():
        for seed in seeds:
            oof = group_cv_predict(factory, X, y, groups, n_splits, seed)
            rows.append({"model": name, "seed": seed, **regression_metrics(y, oof)})
            if seed == seeds[0]:
                oofs[name] = oof
    return pd.DataFrame(rows), oofs


def summarize(cv_table: pd.DataFrame) -> pd.DataFrame:
    """Среднее и std метрик по seed."""
    agg = cv_table.groupby("model", sort=False)[["RMSE", "MAE", "R2"]].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.round(3)
