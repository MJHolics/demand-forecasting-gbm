"""합성 다품목(40개 series x 730일) 수요 데이터 — 전 구간 비교 데모.
python demo.py

시간순 분할(shuffle 없음): 마지막 28일 = test, 그 앞 28일 = val(조기종료용), 나머지 = train.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from dfg.synth_data import generate_demand_panel
from dfg.features import build_feature_frame, usable_feature_columns
from dfg import models as M
from dfg import evaluate as E


def time_based_split(df: pd.DataFrame, test_days: int, val_days: int):
    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)
    train = df[df["date"] < val_start]
    val = df[(df["date"] >= val_start) & (df["date"] < test_start)]
    test = df[df["date"] >= test_start]
    return train, val, test


def main():
    print("=== 1) 합성 데이터 생성 (40 series x 730일, seed=42, 결정적) ===")
    panel = generate_demand_panel(n_series=40, n_days=730, seed=42)
    feat = build_feature_frame(panel)
    feat = feat.dropna(subset=["lag_28"]).reset_index(drop=True)  # 워밍업 구간(최대 lag) 제거
    cols = usable_feature_columns(feat)
    print(f"    특징 {len(cols)}개: {cols}")

    train, val, test = time_based_split(feat, test_days=28, val_days=28)
    print(f"    train {len(train)}행 / val {len(val)}행 / test {len(test)}행"
          f" (기간: {train.date.min().date()} ~ {test.date.max().date()})")

    Xtr, ytr = train[cols], train["y"].to_numpy()
    Xval, yval = val[cols], val["y"].to_numpy()
    Xte, yte = test[cols], test["y"].to_numpy()

    print("\n=== 2) 베이스라인 ===")
    naive_pred = M.seasonal_naive_predict(test)
    print(f"    seasonal-naive(7일전 값)   MAE={E.mae(yte, naive_pred):.3f}  RMSE={E.rmse(yte, naive_pred):.3f}")

    t0 = time.time()
    linear = M.fit_linear_baseline(Xtr, ytr)
    linear_pred = np.clip(linear.predict(Xte.fillna(0)), 0, None)
    linear_time = time.time() - t0
    print(f"    Ridge 선형회귀            MAE={E.mae(yte, linear_pred):.3f}  RMSE={E.rmse(yte, linear_pred):.3f}  ({linear_time:.2f}s)")

    print("\n=== 3) LightGBM (점 예측) ===")
    t0 = time.time()
    lgb_model = M.fit_lightgbm_point(Xtr, ytr, Xval, yval)
    lgb_time = time.time() - t0
    lgb_pred = np.clip(lgb_model.predict(Xte), 0, None)
    print(f"    LightGBM 점 예측          MAE={E.mae(yte, lgb_pred):.3f}  RMSE={E.rmse(yte, lgb_pred):.3f}"
          f"  best_iter={lgb_model.best_iteration_}  ({lgb_time:.2f}s)")

    print("\n=== 4) XGBoost (점 예측, LightGBM과 비교) ===")
    t0 = time.time()
    xgb_model = M.fit_xgboost_point(Xtr, ytr, Xval, yval)
    xgb_time = time.time() - t0
    xgb_pred = np.clip(xgb_model.predict(Xte), 0, None)
    print(f"    XGBoost 점 예측           MAE={E.mae(yte, xgb_pred):.3f}  RMSE={E.rmse(yte, xgb_pred):.3f}"
          f"  best_iter={xgb_model.best_iteration}  ({xgb_time:.2f}s)")

    print("\n=== 5) 통계적 유의성 — Wilcoxon 짝지음 검정 (LightGBM vs 베이스라인) ===")
    err_naive = E.paired_abs_error(yte, naive_pred)
    err_linear = E.paired_abs_error(yte, linear_pred)
    err_lgb = E.paired_abs_error(yte, lgb_pred)
    err_xgb = E.paired_abs_error(yte, xgb_pred)

    r1 = E.wilcoxon_paired_test(err_lgb, err_naive)
    print(f"    LightGBM vs seasonal-naive: p={r1['p_value']:.2e}  median(|err_naive|-|err_lgb|)={-r1['median_diff']:.3f}  n={r1['n']}")
    r2 = E.wilcoxon_paired_test(err_lgb, err_linear)
    print(f"    LightGBM vs Ridge 선형:     p={r2['p_value']:.2e}  median(|err_linear|-|err_lgb|)={-r2['median_diff']:.3f}  n={r2['n']}")
    r3 = E.wilcoxon_paired_test(err_lgb, err_xgb)
    print(f"    LightGBM vs XGBoost:        p={r3['p_value']:.2e}  median(|err_xgb|-|err_lgb|)={-r3['median_diff']:.3f}  n={r3['n']}")

    print("\n=== 6) 분위수 회귀 (P10/P50/P90) — 확률적 예측 ===")
    q_models = {}
    for q in (0.1, 0.5, 0.9):
        t0 = time.time()
        q_models[q] = M.fit_lightgbm_quantile(Xtr, ytr, Xval, yval, quantile=q)
        print(f"    LightGBM quantile(alpha={q}) 학습 완료 ({time.time()-t0:.2f}s)")

    p10 = np.clip(q_models[0.1].predict(Xte), 0, None)
    p50 = np.clip(q_models[0.5].predict(Xte), 0, None)
    p90 = np.clip(q_models[0.9].predict(Xte), 0, None)
    # 교차 위반(crossing) 보정: 이론상 P10<=P50<=P90 이어야 하는데 개별 모델이라 어긋날 수 있음
    crossing_violations = int(np.sum((p10 > p50) | (p50 > p90)))
    p10_fixed = np.minimum(p10, np.minimum(p50, p90))
    p90_fixed = np.maximum(p90, np.maximum(p50, p10))

    coverage_raw = E.interval_coverage(yte, p10, p90)
    coverage_fixed = E.interval_coverage(yte, p10_fixed, p90_fixed)
    pin10 = E.pinball_loss(yte, p10, 0.1)
    pin50 = E.pinball_loss(yte, p50, 0.5)
    pin90 = E.pinball_loss(yte, p90, 0.9)
    print(f"    교차 위반(P10>P50 또는 P50>P90): {crossing_violations}/{len(yte)}건"
          f" ({100*crossing_violations/len(yte):.1f}%) — 보정 후 재계산")
    print(f"    P10~P90 구간 커버리지: 보정전 {coverage_raw*100:.1f}%  보정후 {coverage_fixed*100:.1f}%  (이론값 80%)")
    print(f"    pinball loss  P10={pin10:.3f}  P50={pin50:.3f}  P90={pin90:.3f}  (P50 pinball == MAE/2 = {E.mae(yte, p50)/2:.3f})")

    print("\n=== 요약 표 ===")
    print(f"{'모델':<20}{'MAE':>8}{'RMSE':>8}{'학습시간(s)':>12}")
    print(f"{'seasonal-naive':<20}{E.mae(yte, naive_pred):>8.3f}{E.rmse(yte, naive_pred):>8.3f}{'-':>12}")
    print(f"{'Ridge 선형회귀':<20}{E.mae(yte, linear_pred):>8.3f}{E.rmse(yte, linear_pred):>8.3f}{linear_time:>12.2f}")
    print(f"{'LightGBM':<20}{E.mae(yte, lgb_pred):>8.3f}{E.rmse(yte, lgb_pred):>8.3f}{lgb_time:>12.2f}")
    print(f"{'XGBoost':<20}{E.mae(yte, xgb_pred):>8.3f}{E.rmse(yte, xgb_pred):>8.3f}{xgb_time:>12.2f}")


if __name__ == "__main__":
    main()
