import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dfg.evaluate import mae, rmse, pinball_loss, interval_coverage, wilcoxon_paired_test


def test_mae_rmse_basic():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 33])
    assert abs(mae(y_true, y_pred) - 7 / 3) < 1e-9
    assert rmse(y_true, y_pred) >= mae(y_true, y_pred)  # RMSE >= MAE는 항상 성립(Jensen 부등식)


def test_pinball_loss_median_equals_half_mae():
    y_true = np.array([10.0, 20.0, 5.0])
    y_pred = np.array([12.0, 18.0, 8.0])
    assert abs(pinball_loss(y_true, y_pred, 0.5) - mae(y_true, y_pred) / 2) < 1e-9


def test_pinball_loss_penalizes_underprediction_more_at_high_quantile():
    y_true = np.array([100.0])
    under = np.array([80.0])   # 과소예측
    over = np.array([120.0])   # 과대예측
    # P90 예측이 과소예측이면 더 크게 벌점을 받아야 한다
    assert pinball_loss(y_true, under, 0.9) > pinball_loss(y_true, over, 0.9)


def test_interval_coverage_all_inside():
    y_true = np.array([5, 10, 15])
    lower = np.array([0, 0, 0])
    upper = np.array([20, 20, 20])
    assert interval_coverage(y_true, lower, upper) == 1.0


def test_interval_coverage_none_inside():
    y_true = np.array([100, 200])
    lower = np.array([0, 0])
    upper = np.array([1, 1])
    assert interval_coverage(y_true, lower, upper) == 0.0


def test_wilcoxon_identical_errors_gives_pvalue_one():
    errs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = wilcoxon_paired_test(errs, errs.copy())
    assert result["p_value"] == 1.0
    assert result["median_diff"] == 0.0


def test_wilcoxon_detects_systematic_difference():
    rng = np.random.default_rng(0)
    errors_a = rng.uniform(5, 10, size=200)   # 항상 더 큰 오차
    errors_b = rng.uniform(1, 4, size=200)    # 항상 더 작은 오차
    result = wilcoxon_paired_test(errors_a, errors_b)
    assert result["p_value"] < 0.01
    assert result["median_diff"] > 0
