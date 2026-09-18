"""DeepAR/TFT(신경망 기반 확률적 시계열 예측) — neuralforecast(Nixtla) 래퍼.

기존 dfg/models.py(LightGBM/XGBoost/Ridge)와의 근본적 차이 하나를 반드시 이해하고 써야 한다:
LightGBM 계열은 dfg/features.py의 lag_*/roll_* 특징을 test 구간에서도 **실측(oracle) 과거값**으로
계산한다(build_feature_frame이 train/val/test 분할 전에 전체 시리즈에 shift로 특징을 만든다) —
즉 "test 구간의 각 시점을, 그 직전 실측값들이 이미 알려져 있다고 가정하고 1스텝씩 예측"하는 과제다.
반면 여기 DeepAR/TFT는 **진짜 다중 스텝 예측**이다 — 마지막 관측(cutoff) 이후로는 실측값을 전혀
보지 않고 h스텝을 한 번에 예측한다. 후자가 더 어려운(더 현실적인) 과제라, 신경망이 GBM보다
못해도 "신경망이 열등하다"가 아니라 "다른(더 어려운) 과제를 풀었다"일 수 있다는 걸 README에서
반드시 같이 말한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_long_format(df: pd.DataFrame, id_col: str = "series_id", date_col: str = "date",
                    value_col: str = "y") -> pd.DataFrame:
    """neuralforecast가 요구하는 long format(unique_id, ds, y)으로 변환."""
    out = df.rename(columns={id_col: "unique_id", date_col: "ds", value_col: "y"})
    return out[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"]).reset_index(drop=True)


def fit_deepar(train_long: pd.DataFrame, h: int, freq: str, input_size: int,
                max_steps: int = 300, seed: int = 42):
    """DistributionLoss(StudentT) + level=[80] -> 예측 시 점추정(분포 평균)과 80% 구간(~P10/P90)을 준다."""
    from neuralforecast import NeuralForecast
    from neuralforecast.models import DeepAR
    from neuralforecast.losses.pytorch import DistributionLoss

    model = DeepAR(
        h=h, input_size=input_size,
        loss=DistributionLoss(distribution="StudentT", level=[80]),
        max_steps=max_steps, random_seed=seed, scaler_type="standard",
        enable_progress_bar=False, logger=False,
    )
    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(df=train_long)
    return nf


def fit_tft(train_long: pd.DataFrame, h: int, freq: str, input_size: int,
            max_steps: int = 300, seed: int = 42):
    """MQLoss(quantiles=[0.1,0.5,0.9]) -> 예측 시 P10/P50/P90을 직접 컬럼으로 준다."""
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TFT
    from neuralforecast.losses.pytorch import MQLoss

    model = TFT(
        h=h, input_size=input_size,
        loss=MQLoss(quantiles=[0.1, 0.5, 0.9]),
        max_steps=max_steps, random_seed=seed, scaler_type="standard",
        enable_progress_bar=False, logger=False,
    )
    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(df=train_long)
    return nf


def extract_deepar_quantiles(pred_df: pd.DataFrame) -> dict:
    """DeepAR 예측 df에서 (점추정, P10, P90) 배열을 뽑는다. 컬럼명은 'DeepAR'/'DeepAR-lo-80'/'DeepAR-hi-80'."""
    return {
        "point": pred_df["DeepAR"].to_numpy(),
        "p10": pred_df["DeepAR-lo-80"].to_numpy(),
        "p90": pred_df["DeepAR-hi-80"].to_numpy(),
    }


def extract_tft_quantiles(pred_df: pd.DataFrame) -> dict:
    """TFT MQLoss(quantiles=[0.1,0.5,0.9]) 예측 df 실측 컬럼명(neuralforecast 3.2.2 기준):
    'TFT-median' / 'TFT-lo-80.0' / 'TFT-hi-80.0' (level 80% 표기로 환산되고 소수점이 붙는다 — 첫
    실행에서 확인했다, endswith가 아니라 부분일치로 잡아야 한다)."""
    cols = list(pred_df.columns)
    median_col = next(c for c in cols if c == "TFT" or "median" in c or "-q-50" in c)
    lo_col = next(c for c in cols if "-lo-80" in c or "-q-10" in c)
    hi_col = next(c for c in cols if "-hi-80" in c or "-q-90" in c)
    return {
        "point": pred_df[median_col].to_numpy(),
        "p10": pred_df[lo_col].to_numpy(),
        "p90": pred_df[hi_col].to_numpy(),
    }
