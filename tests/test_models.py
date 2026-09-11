"""GBM 자체 로직은 라이브러리를 신뢰한다 — 우리 코드(경계: 입출력 계약)만 검증."""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dfg import models as M


def _linear_trend_df(n=60):
    days = np.arange(n)
    y = 5.0 + 0.5 * days  # 순수 선형 추세(노이즈 없음)
    X = pd.DataFrame({"days_since_start": days, "month": (days % 12) + 1})
    return X, y


def test_seasonal_naive_predict_uses_lag_column():
    df = pd.DataFrame({"lag_7": [1.0, np.nan, 3.0]})
    pred = M.seasonal_naive_predict(df)
    assert list(pred) == [1.0, 0.0, 3.0]  # NaN은 0으로 대체


def test_detrended_gbm_recovers_pure_linear_trend_better_than_plain_gbm():
    """순수 선형 추세(트리 모델이 취약한 극단 케이스)에서, 추세 제거가 진짜 도움이 되는지 확인.
    demo_real.py에서 발견한 '트리 모델 외삽 실패'의 최소 재현 케이스."""
    Xtr, ytr = _linear_trend_df(60)
    Xval, yval = _linear_trend_df(60)  # 같은 분포(오버피팅 여부는 이 테스트의 관심사가 아님)
    Xte = pd.DataFrame({"days_since_start": np.arange(60, 80), "month": ((np.arange(60, 80)) % 12) + 1})
    yte = 5.0 + 0.5 * np.arange(60, 80)  # train 범위(0~59일치 y값) 밖으로 외삽해야 하는 구간

    plain = M.fit_lightgbm_point(Xtr, ytr, Xval, yval, n_estimators=50)
    plain_pred = plain.predict(Xte)
    plain_mae = np.mean(np.abs(yte - plain_pred))

    detrended = M.fit_lightgbm_detrended(Xtr, ytr, Xval, yval, n_estimators=50)
    detrended_pred = detrended.predict(Xte)
    detrended_mae = np.mean(np.abs(yte - detrended_pred))

    assert detrended_mae < plain_mae * 0.5  # 순수 선형 추세에서는 극적으로 나아야 정상
