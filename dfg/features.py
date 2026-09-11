"""시계열 -> 표 형태(tabular) 특징 변환. 전부 순수 함수, 미래 정보 누출 없음.

입력 규약: DataFrame에 최소 [series_id, date, y] 컬럼이 있고 (series_id, date) 오름차순 정렬.
모든 lag/rolling 특징은 t 시점 예측에 t-1 이하 값만 쓴다(shift(1) 이후 계산) — 리키지 회귀 테스트로 고정.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAG_DAYS_DEFAULT = (1, 7, 14, 28)
ROLL_WINDOWS_DEFAULT = (7, 28)


def add_lag_features(df: pd.DataFrame, lags=LAG_DAYS_DEFAULT, group_col: str = "series_id", value_col: str = "y") -> pd.DataFrame:
    """series_id별로 groupby.shift(k)만 쓴다 — 다른 시리즈의 값이 섞이지 않는다."""
    out = df.copy()
    g = out.groupby(group_col)[value_col]
    for k in lags:
        out[f"lag_{k}"] = g.shift(k)
    return out


def add_rolling_features(df: pd.DataFrame, windows=ROLL_WINDOWS_DEFAULT, group_col: str = "series_id", value_col: str = "y") -> pd.DataFrame:
    """rolling은 shift(1) 뒤에 계산한다 — t 시점 값 자신이 자기 통계에 들어가면 리키지."""
    out = df.copy()
    shifted = out.groupby(group_col)[value_col].shift(1)
    for w in windows:
        out[f"roll_mean_{w}"] = shifted.groupby(out[group_col]).transform(lambda s: s.rolling(w, min_periods=max(2, w // 3)).mean())
        out[f"roll_std_{w}"] = shifted.groupby(out[group_col]).transform(lambda s: s.rolling(w, min_periods=max(2, w // 3)).std())
    return out


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    d = pd.to_datetime(out[date_col])
    out["dow"] = d.dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["month"] = d.dt.month
    out["day_of_year"] = d.dt.dayofyear
    out["year"] = d.dt.year
    return out


def add_trend_index(df: pd.DataFrame, group_col: str = "series_id", date_col: str = "date") -> pd.DataFrame:
    """시리즈 시작일 기준 경과일수 — 선형 트렌드를 모델이 직접 배울 수 있게 하는 스칼라 특징."""
    out = df.copy()
    d = pd.to_datetime(out[date_col])
    start = d.groupby(out[group_col]).transform("min")
    out["days_since_start"] = (d - start).dt.days
    return out


FEATURE_COLUMNS = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_std_7", "roll_mean_28", "roll_std_28",
    "dow", "is_weekend", "month", "day_of_year", "days_since_start",
    "is_promo",
]


def build_feature_frame(df: pd.DataFrame, lags=LAG_DAYS_DEFAULT, windows=ROLL_WINDOWS_DEFAULT,
                         group_col: str = "series_id", date_col: str = "date", value_col: str = "y") -> pd.DataFrame:
    """전체 특징 파이프라인. is_promo가 원본에 없으면 0으로 채운다(실데이터 검증용 — 프로모션 정보가 없는 시리즈)."""
    out = df.sort_values([group_col, date_col]).reset_index(drop=True)
    out = add_lag_features(out, lags, group_col, value_col)
    out = add_rolling_features(out, windows, group_col, value_col)
    out = add_calendar_features(out, date_col)
    out = add_trend_index(out, group_col, date_col)
    if "is_promo" not in out.columns:
        out["is_promo"] = 0
    return out


def usable_feature_columns(df: pd.DataFrame) -> list[str]:
    """FEATURE_COLUMNS(일별 기본값) 우선이되, lag_*/roll_*처럼 build_feature_frame이 실제로
    만들어낸(커스텀 lags/windows 포함) 특징 컬럼을 전부 동적으로 잡는다.
    버그 이력: 이 함수가 FEATURE_COLUMNS 고정 목록만 봤을 때, 월별 데이터용 커스텀 lag(예: lag_12)가
    조용히 빠져 LightGBM이 계절 신호 없이 학습되는 사고가 있었다(demo_real.py 최초 실행에서 발견).
    """
    dynamic = [c for c in df.columns if c.startswith("lag_") or c.startswith("roll_mean_") or c.startswith("roll_std_")]
    calendar = ["dow", "is_weekend", "month", "day_of_year", "year", "days_since_start", "is_promo"]
    ordered = dynamic + [c for c in calendar if c in df.columns]
    seen = set()
    result = []
    for c in ordered:
        if c in df.columns and c not in seen:
            seen.add(c)
            result.append(c)
    return result
