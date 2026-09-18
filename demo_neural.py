"""DeepAR/TFT(neuralforecast) vs 기존 LightGBM/XGBoost/Ridge/naive — 같은 데이터·같은 test 구간 비교.
실행: .venv_neural/Scripts/python.exe demo_neural.py
(neuralforecast·torch는 이 프로젝트 전용 격리 venv에만 설치했다 — 저장소 공유 파이썬 환경을
건드리지 않기 위해서다. 상세는 README "에러 대처 기록".)

중요한 방법론 차이(README에서 자세히 다룸): 기존 LightGBM 계열은 dfg/features.py의 lag/rolling
특징을 test 구간에서도 실측(oracle) 과거값으로 계산한다 — "직전 실측값이 이미 알려져 있다"고
가정한 1스텝 예측이다. 여기 DeepAR/TFT는 cutoff 이후 실측값을 전혀 보지 않고 h스텝을 한 번에
예측하는 **진짜 다중 스텝 예측**이다 — 더 어려운 과제라는 걸 비교 시 반드시 같이 밝힌다.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from dfg.synth_data import generate_demand_panel
from dfg.features import build_feature_frame, usable_feature_columns
from dfg import models as M
from dfg import evaluate as E
from dfg.neural_models import (
    to_long_format, fit_deepar, fit_tft,
    extract_deepar_quantiles, extract_tft_quantiles,
)

warnings.filterwarnings("ignore")


def time_based_split(df, test_days, val_days):
    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)
    train = df[df["date"] < val_start]
    val = df[(df["date"] >= val_start) & (df["date"] < test_start)]
    test = df[df["date"] >= test_start]
    return train, val, test


def run_synthetic():
    print("\n" + "=" * 70)
    print("[A] 합성 데이터 (40 series x 730일) — DeepAR/TFT vs 기존 GBM/baseline")
    print("=" * 70)

    panel = generate_demand_panel(n_series=40, n_days=730, seed=42)
    feat = build_feature_frame(panel)
    feat = feat.dropna(subset=["lag_28"]).reset_index(drop=True)
    cols = usable_feature_columns(feat)

    train, val, test = time_based_split(feat, test_days=28, val_days=28)
    Xtr, ytr = train[cols], train["y"].to_numpy()
    Xval, yval = val[cols], val["y"].to_numpy()
    Xte, yte_df = test[cols], test[["series_id", "date", "y"]].reset_index(drop=True)

    print("\n--- 기존 GBM/baseline 재계산 (README 표와 동일 파이프라인, 정렬 일치용) ---")
    naive_pred = M.seasonal_naive_predict(test)
    linear = M.fit_linear_baseline(Xtr, ytr)
    linear_pred = np.clip(linear.predict(Xte.fillna(0)), 0, None)
    lgb_model = M.fit_lightgbm_point(Xtr, ytr, Xval, yval)
    lgb_pred = np.clip(lgb_model.predict(Xte), 0, None)
    xgb_model = M.fit_xgboost_point(Xtr, ytr, Xval, yval)
    xgb_pred = np.clip(xgb_model.predict(Xte), 0, None)
    q_models = {q: M.fit_lightgbm_quantile(Xtr, ytr, Xval, yval, quantile=q) for q in (0.1, 0.9)}
    lgb_p10 = np.clip(q_models[0.1].predict(Xte), 0, None)
    lgb_p90 = np.clip(q_models[0.9].predict(Xte), 0, None)

    base = yte_df.copy()
    base["naive"] = naive_pred
    base["ridge"] = linear_pred
    base["lgb"] = lgb_pred
    base["xgb"] = xgb_pred
    base["lgb_p10"] = lgb_p10
    base["lgb_p90"] = lgb_p90

    print(f"    seasonal-naive MAE={E.mae(base.y, base.naive):.3f}  Ridge MAE={E.mae(base.y, base.ridge):.3f}"
          f"  LightGBM MAE={E.mae(base.y, base.lgb):.3f}  XGBoost MAE={E.mae(base.y, base.xgb):.3f}")

    print("\n--- DeepAR/TFT 학습 (train+val 구간 전체를 cutoff까지 context로, h=28 다중스텝 예측) ---")
    fit_data = train_val_long = to_long_format(pd.concat([train, val], ignore_index=True))
    test_long = to_long_format(test)

    t0 = time.time()
    nf_deepar = fit_deepar(fit_data, h=28, freq="D", input_size=90, max_steps=300, seed=42)
    deepar_pred = nf_deepar.predict(df=fit_data, level=[80])
    deepar_time = time.time() - t0
    print(f"    DeepAR 학습+예측 완료 ({deepar_time:.1f}s), 컬럼: {list(deepar_pred.columns)}")

    t0 = time.time()
    nf_tft = fit_tft(fit_data, h=28, freq="D", input_size=90, max_steps=300, seed=42)
    tft_pred = nf_tft.predict(df=fit_data)
    tft_time = time.time() - t0
    print(f"    TFT 학습+예측 완료 ({tft_time:.1f}s), 컬럼: {list(tft_pred.columns)}")

    deepar_q = extract_deepar_quantiles(deepar_pred)
    tft_q = extract_tft_quantiles(tft_pred)
    deepar_pred = deepar_pred.assign(deepar_point=deepar_q["point"], deepar_p10=deepar_q["p10"], deepar_p90=deepar_q["p90"])
    tft_pred = tft_pred.assign(tft_point=tft_q["point"], tft_p10=tft_q["p10"], tft_p90=tft_q["p90"])

    merged = base.merge(
        deepar_pred[["unique_id", "ds", "deepar_point", "deepar_p10", "deepar_p90"]],
        left_on=["series_id", "date"], right_on=["unique_id", "ds"], how="inner",
    ).merge(
        tft_pred[["unique_id", "ds", "tft_point", "tft_p10", "tft_p90"]],
        left_on=["series_id", "date"], right_on=["unique_id", "ds"], how="inner",
    )
    n_lost = len(base) - len(merged)
    print(f"    조인 후 표본 {len(merged)}/{len(base)} (미매칭 {n_lost}건)")

    print("\n--- 점 예측 비교 (MAE/RMSE, 동일 1120행 기준) ---")
    rows = [
        ("seasonal-naive", merged.naive), ("Ridge", merged.ridge),
        ("LightGBM", merged.lgb), ("XGBoost", merged.xgb),
        ("DeepAR(분포평균)", merged.deepar_point), ("TFT(중앙값)", merged.tft_point),
    ]
    for name, pred in rows:
        print(f"    {name:<18} MAE={E.mae(merged.y, pred):.3f}  RMSE={E.rmse(merged.y, pred):.3f}")

    print("\n--- Wilcoxon 짝지음 검정 (vs LightGBM) ---")
    err_lgb = E.paired_abs_error(merged.y, merged.lgb)
    for name, pred in [("DeepAR", merged.deepar_point), ("TFT", merged.tft_point)]:
        err = E.paired_abs_error(merged.y, pred)
        r = E.wilcoxon_paired_test(err, err_lgb)
        direction = "LightGBM이 더 나음" if r["median_diff"] > 0 else f"{name}이 더 나음"
        print(f"    {name} vs LightGBM: p={r['p_value']:.2e}  median_diff={r['median_diff']:.3f}  n={r['n']}  ({direction})")

    print("\n--- 확률적 예측(P10/P90 구간 커버리지, 이론값 80%) ---")
    for name, p10, p90 in [
        ("LightGBM quantile", merged.lgb_p10, merged.lgb_p90),
        ("DeepAR(80% 구간)", merged.deepar_p10, merged.deepar_p90),
        ("TFT(P10/P90)", merged.tft_p10, merged.tft_p90),
    ]:
        cov = E.interval_coverage(merged.y, p10, p90)
        print(f"    {name:<20} 커버리지={cov*100:.1f}%")

    merged.to_csv("results_neural_synthetic.csv", index=False) if False else None
    return merged, {"deepar_time": deepar_time, "tft_time": tft_time}


def run_real():
    print("\n" + "=" * 70)
    print("[B] 실데이터 a10(항당뇨병 판매량, 월별 204건) — DeepAR/TFT vs 기존")
    print("=" * 70)

    df = pd.read_csv("data/a10_antidiabetic_sales.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["series_id"] = "a10_antidiabetic"
    df = df.rename(columns={"value": "y"})[["series_id", "date", "y"]].sort_values("date").reset_index(drop=True)

    TEST_MONTHS = 24
    feat = build_feature_frame(df, lags=(1, 2, 3, 12), windows=(3, 12))
    feat = feat.dropna(subset=["lag_12"]).reset_index(drop=True)
    cols = [c for c in usable_feature_columns(feat) if c not in ("dow", "is_weekend", "day_of_year")]
    cols = [c for c in cols if c in feat.columns]

    test_start = feat["date"].max() - pd.DateOffset(months=TEST_MONTHS - 1)
    val_start = test_start - pd.DateOffset(months=TEST_MONTHS)
    train = feat[feat.date < val_start]
    val = feat[(feat.date >= val_start) & (feat.date < test_start)]
    test = feat[feat.date >= test_start]

    Xtr, ytr = train[cols], train["y"].to_numpy()
    Xval, yval = val[cols], val["y"].to_numpy()
    Xte = test[cols]

    naive_pred = M.seasonal_naive_predict(test, lag_col="lag_12")
    linear = M.fit_linear_baseline(Xtr, ytr)
    linear_pred = np.clip(linear.predict(Xte.fillna(0)), 0, None)
    lgb_model = M.fit_lightgbm_point(Xtr, ytr, Xval, yval, n_estimators=200)
    lgb_pred = np.clip(lgb_model.predict(Xte), 0, None)
    detrended = M.fit_lightgbm_detrended(Xtr, ytr, Xval, yval, n_estimators=200)
    detrended_pred = np.clip(detrended.predict(Xte), 0, None)

    base = test[["series_id", "date", "y"]].reset_index(drop=True)
    base["naive"] = naive_pred
    base["ridge"] = linear_pred
    base["lgb"] = lgb_pred
    base["lgb_detrended"] = detrended_pred

    fit_cutoff = pd.concat([train, val], ignore_index=True)[["series_id", "date", "y"]]
    fit_long = to_long_format(fit_cutoff)

    t0 = time.time()
    nf_deepar = fit_deepar(fit_long, h=TEST_MONTHS, freq="MS", input_size=24, max_steps=300, seed=42)
    deepar_pred = nf_deepar.predict(df=fit_long, level=[80])
    deepar_time = time.time() - t0
    print(f"    DeepAR 학습+예측 완료 ({deepar_time:.1f}s)")

    t0 = time.time()
    nf_tft = fit_tft(fit_long, h=TEST_MONTHS, freq="MS", input_size=24, max_steps=300, seed=42)
    tft_pred = nf_tft.predict(df=fit_long)
    tft_time = time.time() - t0
    print(f"    TFT 학습+예측 완료 ({tft_time:.1f}s)")

    deepar_q = extract_deepar_quantiles(deepar_pred)
    tft_q = extract_tft_quantiles(tft_pred)
    deepar_pred = deepar_pred.assign(deepar_point=deepar_q["point"], deepar_p10=deepar_q["p10"], deepar_p90=deepar_q["p90"])
    tft_pred = tft_pred.assign(tft_point=tft_q["point"], tft_p10=tft_q["p10"], tft_p90=tft_q["p90"])

    merged = base.merge(
        deepar_pred[["unique_id", "ds", "deepar_point", "deepar_p10", "deepar_p90"]],
        left_on=["series_id", "date"], right_on=["unique_id", "ds"], how="inner",
    ).merge(
        tft_pred[["unique_id", "ds", "tft_point", "tft_p10", "tft_p90"]],
        left_on=["series_id", "date"], right_on=["unique_id", "ds"], how="inner",
    )
    print(f"    조인 후 표본 {len(merged)}/{len(base)}")

    print("\n--- 점 예측 비교 (MAE/RMSE/MAPE) ---")
    rows = [
        ("seasonal-naive", merged.naive), ("Ridge", merged.ridge),
        ("LightGBM(단독)", merged.lgb), ("LightGBM(추세제거)", merged.lgb_detrended),
        ("DeepAR(분포평균)", merged.deepar_point), ("TFT(중앙값)", merged.tft_point),
    ]
    for name, pred in rows:
        print(f"    {name:<20} MAE={E.mae(merged.y, pred):.3f}  RMSE={E.rmse(merged.y, pred):.3f}  MAPE={E.mape(merged.y, pred):.1f}%")

    print("\n--- 확률적 예측(P10/P90 구간 커버리지, 이론값 80%, n=24라 참고용) ---")
    for name, p10, p90 in [
        ("DeepAR(80% 구간)", merged.deepar_p10, merged.deepar_p90),
        ("TFT(P10/P90)", merged.tft_p10, merged.tft_p90),
    ]:
        cov = E.interval_coverage(merged.y, p10, p90)
        print(f"    {name:<20} 커버리지={cov*100:.1f}%")

    print(f"\n    train y 범위: [{ytr.min():.2f}, {ytr.max():.2f}]  실제 test y 범위: [{merged.y.min():.2f}, {merged.y.max():.2f}]")
    print(f"    DeepAR 예측 범위: [{merged.deepar_point.min():.2f}, {merged.deepar_point.max():.2f}]")
    print(f"    TFT 예측 범위:    [{merged.tft_point.min():.2f}, {merged.tft_point.max():.2f}]")

    return merged, {"deepar_time": deepar_time, "tft_time": tft_time}


if __name__ == "__main__":
    syn_merged, syn_times = run_synthetic()
    real_merged, real_times = run_real()

    print("\n" + "=" * 70)
    print("요약 저장 -> results_neural/results_neural_summary.txt")
    print("=" * 70)
    import os
    os.makedirs("results_neural", exist_ok=True)
    with open("results_neural/results_neural_summary.txt", "w", encoding="utf-8") as f:
        f.write("[A] 합성 데이터(40 series x 730일, test 28일 x 40 series = 1120행)\n")
        f.write(f"  DeepAR 학습+예측 {syn_times['deepar_time']:.1f}s, TFT {syn_times['tft_time']:.1f}s\n")
        for name, col in [("naive", "naive"), ("ridge", "ridge"), ("lgb", "lgb"), ("xgb", "xgb"),
                           ("deepar", "deepar_point"), ("tft", "tft_point")]:
            f.write(f"  {name:<10} MAE={E.mae(syn_merged.y, syn_merged[col]):.3f}  RMSE={E.rmse(syn_merged.y, syn_merged[col]):.3f}\n")
        f.write("\n[B] 실데이터 a10(월별, test 24개월)\n")
        f.write(f"  DeepAR 학습+예측 {real_times['deepar_time']:.1f}s, TFT {real_times['tft_time']:.1f}s\n")
        for name, col in [("naive", "naive"), ("ridge", "ridge"), ("lgb", "lgb"),
                           ("lgb_detrended", "lgb_detrended"), ("deepar", "deepar_point"), ("tft", "tft_point")]:
            f.write(f"  {name:<14} MAE={E.mae(real_merged.y, real_merged[col]):.3f}  RMSE={E.rmse(real_merged.y, real_merged[col]):.3f}  MAPE={E.mape(real_merged.y, real_merged[col]):.1f}%\n")
    print("완료.")
