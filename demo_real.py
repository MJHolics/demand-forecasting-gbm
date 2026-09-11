"""실데이터 검증 — 호주 항당뇨병 약품 판매량(a10, 1991-07~2008-06, 월별 204건, PBS 공식 통계).
출처: https://github.com/selva86/datasets/blob/master/a10.csv (Hyndman fpp2 교재에서 쓰는 표준 벤치마크)
python demo_real.py

합성 데이터(40개 시리즈, 일별)와 달리 이 데이터는 시리즈가 1개뿐이고 월별이다 — 같은 파이프라인이
"여러 시리즈에서 배우는" 이점 없이도 단일 실제 시계열에 통하는지가 관전 포인트.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dfg.features import build_feature_frame, usable_feature_columns
from dfg import models as M
from dfg import evaluate as E

DATA_PATH = "data/a10_antidiabetic_sales.csv"
TEST_MONTHS = 24


def load_real_series() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["series_id"] = "a10_antidiabetic"
    df = df.rename(columns={"value": "y"})
    return df[["series_id", "date", "y"]].sort_values("date").reset_index(drop=True)


def main():
    df = load_real_series()
    print(f"=== 실데이터: {DATA_PATH} ===")
    print(f"    {len(df)}개월 · {df.date.min().date()} ~ {df.date.max().date()} (월별, 단일 시리즈)")

    # 월별 데이터라 lag/window를 개월 단위로 다시 잡는다(일별 기본값 lag_7/28은 의미 없음)
    feat = build_feature_frame(df, lags=(1, 2, 3, 12), windows=(3, 12))
    feat = feat.dropna(subset=["lag_12"]).reset_index(drop=True)
    cols = [c for c in usable_feature_columns(feat) if c not in ("dow", "is_weekend", "day_of_year")]
    cols = [c for c in cols if c in feat.columns]
    print(f"    특징 {len(cols)}개: {cols}")

    test_start = feat["date"].max() - pd.DateOffset(months=TEST_MONTHS - 1)
    val_start = test_start - pd.DateOffset(months=TEST_MONTHS)
    train = feat[feat.date < val_start]
    val = feat[(feat.date >= val_start) & (feat.date < test_start)]
    test = feat[feat.date >= test_start]
    print(f"    train {len(train)}개월 / val {len(val)}개월 / test {len(test)}개월")

    if len(train) < 20:
        print("    경고: train 구간이 너무 짧습니다(원본이 204개월뿐이라 lag_12 워밍업 후 여유가 적음).")

    Xtr, ytr = train[cols], train["y"].to_numpy()
    Xval, yval = val[cols], val["y"].to_numpy()
    Xte, yte = test[cols], test["y"].to_numpy()

    naive_pred = M.seasonal_naive_predict(test, lag_col="lag_12")  # 계절 주기=12개월
    linear = M.fit_linear_baseline(Xtr, ytr)
    linear_pred = np.clip(linear.predict(Xte.fillna(0)), 0, None)
    lgb_model = M.fit_lightgbm_point(Xtr, ytr, Xval, yval, n_estimators=200)
    lgb_pred = np.clip(lgb_model.predict(Xte), 0, None)

    print("\n=== 결과 (MAE, 실제 판매량 단위) ===")
    print(f"    seasonal-naive(12개월전) MAE={E.mae(yte, naive_pred):.3f}  RMSE={E.rmse(yte, naive_pred):.3f}  MAPE={E.mape(yte, naive_pred):.1f}%")
    print(f"    Ridge 선형회귀           MAE={E.mae(yte, linear_pred):.3f}  RMSE={E.rmse(yte, linear_pred):.3f}  MAPE={E.mape(yte, linear_pred):.1f}%")
    print(f"    LightGBM                 MAE={E.mae(yte, lgb_pred):.3f}  RMSE={E.rmse(yte, lgb_pred):.3f}  MAPE={E.mape(yte, lgb_pred):.1f}%  best_iter={lgb_model.best_iteration_}")

    r = E.wilcoxon_paired_test(E.paired_abs_error(yte, lgb_pred), E.paired_abs_error(yte, naive_pred))
    print(f"\n    LightGBM vs seasonal-naive: p={r['p_value']:.3f}  n={r['n']}  (n이 {TEST_MONTHS}뿐이라 검정력이 낮음 — 정직하게 명시)")

    detrended = M.fit_lightgbm_detrended(Xtr, ytr, Xval, yval, n_estimators=200)
    detrended_pred = np.clip(detrended.predict(Xte), 0, None)
    print(f"    LightGBM(추세 제거)     MAE={E.mae(yte, detrended_pred):.3f}  RMSE={E.rmse(yte, detrended_pred):.3f}  MAPE={E.mape(yte, detrended_pred):.1f}%")

    print("\n=== 진단: 왜 LightGBM이 이 실데이터에서 베이스라인보다 나쁜가 ===")
    print(f"    train y 범위: [{ytr.min():.2f}, {ytr.max():.2f}]")
    print(f"    LightGBM 예측 범위: [{lgb_pred.min():.2f}, {lgb_pred.max():.2f}]")
    print(f"    실제 test y 범위: [{yte.min():.2f}, {yte.max():.2f}]")
    print("    -> 트리 기반 모델은 리프의 학습 타깃 평균으로 예측하므로 train에서 본 값 범위를")
    print("       벗어나 외삽하지 못한다. 이 시리즈는 항당뇨병 약품 사용량이 꾸준히 늘어난 실제")
    print("       추세라 test 구간(최근 24개월)이 train 최댓값을 넘어서는데, LightGBM은 그 추세를")
    print("       따라가지 못하고 train 근처에 눌러앉는다. 반대로 Ridge(선형)는 추세를 직선으로")
    print("       외삽할 수 있어 이 시리즈에서 더 낫다 — '합성 데이터에서 GBM이 항상 이긴다'는")
    print("       결론이 실데이터에서 뒤집히는 지점이다(트리 모델의 알려진 한계, no free lunch).")
    print(f"\n    수정: 선형 추세를 먼저 떼어내고 잔차만 LightGBM에 맡기면 MAE {E.mae(yte, lgb_pred):.2f} -> "
          f"{E.mae(yte, detrended_pred):.2f}로 개선(베이스라인 Ridge {E.mae(yte, linear_pred):.2f}와 비교).")


if __name__ == "__main__":
    main()
