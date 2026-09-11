"""결정적 합성 다품목 수요 데이터 생성기.

가정(문서화):
- N개 매장×품목 조합(series_id)마다 서로 다른 기본 수요·트렌드·계절성 진폭을 랜덤 배정(seed 고정).
- 요일 계절성(주말 +40%), 연 계절성(12월 성수기 사인 곡선), 선형 트렌드, 랜덤 프로모션(수요 +50~90%),
  포아송 노이즈(카운트성 수요를 흉내). 전부 재현 가능(seed=42).
- 실무 수요예측 데이터의 전형적 구조(추세+주기성+프로모션+정수 노이즈)를 재현하되, 실제 판매 기록은
  아니다 — README '정직한 범위'에 명시.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_demand_panel(
    n_series: int = 40,
    n_days: int = 730,
    start_date: str = "2024-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    day_of_year = dates.dayofyear.values
    dow = dates.dayofweek.values

    rows = []
    for i in range(n_series):
        base = rng.uniform(15, 120)            # 시리즈별 기본 수요 수준
        trend_per_day = rng.uniform(-0.03, 0.08)  # 완만한 추세(감소하는 시리즈도 일부 포함)
        yearly_amp = rng.uniform(0.1, 0.5) * base   # 연 계절성 진폭(12월 성수기)
        weekend_boost = rng.uniform(1.2, 1.6)       # 주말 배수
        promo_boost = rng.uniform(1.5, 1.9)         # 프로모션 배수
        promo_prob = rng.uniform(0.03, 0.10)        # 하루에 프로모션이 걸릴 확률

        t = np.arange(n_days)
        seasonal = yearly_amp * np.sin(2 * np.pi * (day_of_year - 300) / 365.25) ** 2  # 12월 근처 피크
        level = base + trend_per_day * t + seasonal
        level = np.where(dow >= 5, level * weekend_boost, level)

        is_promo = (rng.random(n_days) < promo_prob).astype(int)
        level = np.where(is_promo == 1, level * promo_boost, level)
        level = np.clip(level, 0.5, None)

        y = rng.poisson(level).astype(float)

        rows.append(pd.DataFrame({
            "series_id": f"S{i:03d}",
            "date": dates,
            "y": y,
            "is_promo": is_promo,
        }))

    panel = pd.concat(rows, ignore_index=True)
    return panel.sort_values(["series_id", "date"]).reset_index(drop=True)
