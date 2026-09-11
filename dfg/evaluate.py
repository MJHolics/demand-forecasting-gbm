"""평가 지표 — 전부 순수 함수(numpy 배열 in/out), side effect 없음."""
from __future__ import annotations

import numpy as np
from scipy import stats


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """0에 가까운 실측값이 있어 eps로 분모를 바닥 처리(카운트성 수요 데이터 특성)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """분위수 회귀 손실(quantile loss). quantile=0.5면 MAE의 절반과 같다."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """P10~P90 구간에 실제값이 들어온 비율. 보정이 잘 된 모델이면 이론상 80%에 가까워야 한다."""
    y_true = np.asarray(y_true)
    inside = (y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))
    return float(np.mean(inside))


def paired_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(y_true) - np.asarray(y_pred))


def wilcoxon_paired_test(errors_a: np.ndarray, errors_b: np.ndarray) -> dict:
    """모델 A/B의 (같은 시점·시리즈) 절대오차 쌍을 짝지어 Wilcoxon 부호순위검정.
    McNemar가 분류 문제의 짝지음 검정이듯, 회귀에서는 비모수 짝지음 검정으로 Wilcoxon을 쓴다.
    정규분포 가정이 없어 카운트성 수요 오차(치우친 분포)에 적합하다.
    """
    a = np.asarray(errors_a)
    b = np.asarray(errors_b)
    diff = a - b
    if np.allclose(diff, 0):
        return {"statistic": float("nan"), "p_value": 1.0, "median_diff": 0.0, "n": len(diff)}
    statistic, p_value = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "median_diff": float(np.median(diff)),
        "n": int(len(diff)),
    }
