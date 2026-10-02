"""Базовые тесты: признаки, физический baseline, отсутствие утечки групп и API.

Запуск из корня проекта: ``pytest -q``.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from src.build_dataset import FEATURES, classify_rock, handle_detection_limit
from src.features import OxideFeatures, make_groups, stoichiometric_cao, load_data


def test_classify_rock_excludes_dolomite_and_clastics():
    # нужные породы распознаются, доломиты и песчаники отбрасываются
    assert classify_rock("Marl") == "marl"
    assert classify_rock("limestone") == "limestone"
    assert classify_rock("dolostone") is None
    assert classify_rock("carbonate sandstone") is None


def test_detection_limit_replaced_by_half():
    # -0.2 ("ниже предела 0.2") -> 0.1, ноль -> пропуск, обычное значение не меняется
    df = pd.DataFrame({"k2o": [-0.2, 0.0, 1.0]})
    out = handle_detection_limit(df, ["k2o"])
    assert out["k2o"].iloc[0] == pytest.approx(0.1)
    assert np.isnan(out["k2o"].iloc[1])
    assert out["n_censored"].tolist() == [1, 0, 0]


def test_stoichiometry_pure_calcite_is_about_56():
    # без примесей это чистый кальцит, CaO около 56.1 %
    pure = pd.DataFrame([{c: 0.0 for c in FEATURES}])
    assert stoichiometric_cao(pure).iloc[0] == pytest.approx(56.1, abs=0.1)


def test_oxide_features_shape_and_nan_kept():
    # 9 исходных + 5 новых; пропуски не должны исчезнуть
    row = pd.DataFrame([{c: np.nan for c in FEATURES}]).assign(sio2=5.0, al2o3=1.0, mgo=1.0)
    out = OxideFeatures().fit(row).transform(row)
    assert out.shape[1] == len(FEATURES) + 5
    assert out["k2o"].isna().all()


def test_group_split_has_no_leakage():
    # самый важный тест: одна группа не может быть и в train, и в test
    df = load_data()
    groups = make_groups(df)
    for tr, va in GroupKFold(n_splits=5).split(df, groups=groups):
        assert not set(groups.iloc[tr]) & set(groups.iloc[va])


def test_predict_interval_contains_prediction():
    # low <= pred <= high и всё в пределах 0...56.1
    from app.predict import predict_cao

    res = predict_cao({"sio2": 5, "al2o3": 1.5, "mgo": 1.0})
    assert 0 <= res["cao_low"] <= res["cao_pred"] <= res["cao_high"] <= 56.1


def test_predict_rejects_negative_values():
    # отрицательный процент - ошибка ввода, предсказывать по нему нельзя
    from app.predict import predict_cao

    with pytest.raises(ValueError):
        predict_cao({"sio2": -1, "al2o3": 1.5, "mgo": 1.0})


def test_predict_api():
    # сервер не поднимаем, TestClient дёргает приложение напрямую
    from fastapi.testclient import TestClient
    from app.api import app

    client = TestClient(app)
    ok = client.post("/predict", json={"sio2": 5, "al2o3": 1.5, "mgo": 1.0})
    assert ok.status_code == 200
    assert 40 < ok.json()["cao_pred"] < 56.1
    # без обязательных оксидов должна быть ошибка 422
    assert client.post("/predict", json={"sio2": 5}).status_code == 422
