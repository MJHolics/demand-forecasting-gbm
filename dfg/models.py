"""예측 모델들. baseline 2종(순수 함수) + GBM 2종(학습 필요, 얇은 래퍼).

baseline은 학습 없이 결정적으로 답이 나오는 순수 함수라 리그레션 테스트가 쉽다.
GBM은 sklearn 스타일 fit/predict 래퍼 — 라이브러리 자체 로직은 신뢰하고, 우리가 만든
특징 파이프라인·입출력 계약만 테스트한다(원칙: 남의 라이브러리 내부는 안 믿고 재구현하지 않되,
우리 코드와의 경계는 검증한다).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def seasonal_naive_predict(df: pd.DataFrame, lag_col: str = "lag_7") -> np.ndarray:
    """"오늘 수요 = 7일 전 수요" — 가장 값싼 베이스라인. lag_7가 NaN이면(초반 구간) 0으로 대체."""
    return df[lag_col].fillna(0).to_numpy()


def fit_linear_baseline(X_train: pd.DataFrame, y_train: np.ndarray):
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train.fillna(0), y_train)
    return model


def fit_lightgbm_point(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
                        n_estimators: int = 400, seed: int = 42):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, random_state=seed, verbosity=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return model


def fit_lightgbm_quantile(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
                           quantile: float, n_estimators: int = 400, seed: int = 42):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=quantile,
        n_estimators=n_estimators, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, random_state=seed, verbosity=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="quantile",
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return model


class DetrendedGBM:
    """트리 모델은 학습 타깃 범위 밖을 외삽 못 한다(리프 평균 예측) — 선형 추세를 먼저 떼어내고
    잔차만 GBM에 맡긴 뒤 다시 더한다. real 데이터(항당뇨병 판매량, 꾸준한 성장 추세)에서
    LightGBM 단독이 베이스라인보다 나빴던 문제(demo_real.py)를 고치려고 도입했다.
    """

    def __init__(self, trend_model, gbm_model):
        self.trend_model = trend_model
        self.gbm_model = gbm_model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        trend = self.trend_model.predict(X[["days_since_start"]].fillna(0))
        residual = self.gbm_model.predict(X)
        return trend + residual


def fit_lightgbm_detrended(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
                            n_estimators: int = 400, seed: int = 42) -> DetrendedGBM:
    from sklearn.linear_model import LinearRegression
    trend_model = LinearRegression()
    trend_model.fit(X_train[["days_since_start"]].fillna(0), y_train)

    train_resid = y_train - trend_model.predict(X_train[["days_since_start"]].fillna(0))
    val_resid = y_val - trend_model.predict(X_val[["days_since_start"]].fillna(0))

    gbm = fit_lightgbm_point(X_train, train_resid, X_val, val_resid, n_estimators=n_estimators, seed=seed)
    return DetrendedGBM(trend_model, gbm)


def fit_xgboost_point(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
                       n_estimators: int = 400, seed: int = 42):
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=n_estimators, learning_rate=0.05, max_depth=6,
        min_child_weight=5, random_state=seed, early_stopping_rounds=30,
        eval_metric="mae", verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model
