import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dfg.features import add_lag_features, add_rolling_features, build_feature_frame, usable_feature_columns


def _toy_df():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame({
        "series_id": ["A"] * 10,
        "date": dates,
        "y": np.arange(10, dtype=float),  # 0,1,2,...,9
    })


def test_lag_1_shifts_by_one_no_lookahead():
    df = _toy_df()
    out = add_lag_features(df, lags=(1,))
    # t=5(y=5)의 lag_1은 t=4의 y=4여야 한다 — 자기 자신(5)이 섞이면 리키지
    assert out.loc[5, "lag_1"] == 4
    assert out.loc[0, "lag_1"] != out.loc[0, "y"]
    assert pd.isna(out.loc[0, "lag_1"])  # 첫 행은 과거가 없어 NaN


def test_lag_does_not_mix_across_series():
    df = pd.concat([_toy_df().assign(series_id="A"), _toy_df().assign(series_id="B", y=lambda d: d.y + 100)])
    df = df.sort_values(["series_id", "date"]).reset_index(drop=True)
    out = add_lag_features(df, lags=(1,))
    b_first_row = out[out.series_id == "B"].iloc[0]
    assert pd.isna(b_first_row["lag_1"])  # B의 첫 행이 A의 마지막 값을 lag로 가져오면 리키지


def test_rolling_mean_excludes_current_value():
    df = _toy_df()
    out = add_rolling_features(df, windows=(3,))
    # t=5(y=5) 기준 roll_mean_3은 y[2],y[3],y[4] = (2+3+4)/3 = 3.0 이어야 한다(자기 자신 5 미포함)
    assert abs(out.loc[5, "roll_mean_3"] - 3.0) < 1e-9


def test_build_feature_frame_runs_end_to_end_and_fills_is_promo():
    df = _toy_df()
    out = build_feature_frame(df)
    assert "is_promo" in out.columns
    assert (out["is_promo"] == 0).all()
    assert "days_since_start" in out.columns
    assert out["days_since_start"].iloc[0] == 0
    assert out["days_since_start"].iloc[-1] == 9


def test_usable_feature_columns_picks_up_custom_lags_not_in_daily_defaults():
    """회귀 테스트 — demo_real.py 최초 실행에서 lag_12(월별 계절 lag)가 커스텀 인자로 만들어졌는데도
    이 함수가 일별 기본값(lag_7/28)만 찾아서 조용히 빠지는 사고가 있었다."""
    df = _toy_df()
    out = build_feature_frame(df, lags=(1, 2, 3, 12), windows=(3, 12))
    cols = usable_feature_columns(out)
    assert "lag_12" in cols
    assert "lag_2" in cols
    assert "lag_3" in cols
    assert "roll_mean_12" in cols
    assert "lag_7" not in cols  # 애초에 안 만들어졌으니 당연히 없어야 함(존재하지 않는 컬럼을 만들지 않는다)
