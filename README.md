# Demand Forecasting GBM — LightGBM/XGBoost + DeepAR/TFT 수요 예측·확률적 예측(P10/P50/P90)

> 2026-09-11. 캐롯아이(AI Engineer, AI Agent & Forecasting) JD의 필수요건 "LightGBM, XGBoost 등
> 머신러닝 모델을 학습/평가 경험"과 우대사항 "DeepAR, TFT 등 시계열 수요/매출 예측·확률적 예측
> (P10/P50/P90)"을 직접 겨냥해 신규로 만들었다. 지금까지 프로젝트는 전부 딥러닝·LLM이었고
> **정형 데이터·그래디언트 부스팅·시계열 예측은 이 프로젝트가 처음**이다. 전부 무료·결정적·Windows.
> **(2026-09-18 확장)** 우대사항의 DeepAR/TFT(신경망 확률적 예측)까지 구현해 같은 데이터로
> head-to-head 비교했다 — 맨 아래 "2026-09-18 확장" 섹션. 합성 데이터에선 GBM이 압승했지만
> 실데이터(외삽이 필요한 트렌드 시리즈)에서는 TFT가 역전해 Ridge와 동률 1위를 냈다.

## 목표·이유
채용 공고가 요구하는 구체 스킬(LightGBM/XGBoost, 확률적 예측)에 실측 근거가 없었다. "배울 수 있다"가
아니라 실제로 학습·평가·비교해서 수치로 보여주는 프로젝트가 필요했다.

## 상품성
수요/매출 예측은 재고·인력·마케팅 예산 배분에 바로 쓰인다. 점 추정(얼마 팔릴까) 하나로는 재고
리스크(과잉/품절)를 다룰 수 없어 실무는 P10(비관)~P90(낙관) 구간을 함께 본다 — 이 프로젝트가
점 예측과 확률적 예측을 모두 구현한 이유.

## 모델링 채용 근거
- **LightGBM/XGBoost**: 정형 특징(lag·rolling·달력) 기반 예측에서 딥러닝보다 적은 데이터로도
  강하고, 학습이 빨라 반복 실험에 유리하다 — 업계 표준(Kaggle 시계열 대회 대부분의 우승 스택).
- **선형회귀(Ridge)/계절성 naive를 베이스라인으로**: "그래디언트 부스팅이 무조건 낫다"고 주장하지
  않기 위해 반드시 비교 대상을 둔다.
- **LightGBM quantile objective**로 P10/P50/P90: 별도 확률분포 가정(DeepAR류) 없이 손실함수만
  바꿔 분위수 예측을 얻는 가장 값싼 방법 — quantile objective 3개를 독립 학습.

## 핵심 결과 (합성 데이터, `python demo.py`, 40개 시리즈 x 730일, 결정적 seed=42)

| 모델 | MAE | RMSE | 학습시간 |
|---|---:|---:|---:|
| seasonal-naive(7일전 값) | 21.180 | 35.924 | - |
| Ridge 선형회귀 | 12.411 | 17.992 | 0.23s |
| **LightGBM** | **10.353** | **13.861** | 1.60s |
| XGBoost | 10.405 | 14.180 | 0.61s |

- LightGBM vs seasonal-naive: **p=4.94e-42**(Wilcoxon 짝지음 검정, n=1120) — 압도적으로 유의.
- LightGBM vs Ridge: **p=6.06e-13** — 유의하게 더 나음.
- **LightGBM vs XGBoost: p=0.358 — 유의한 차이 없음.** 두 GBM 라이브러리가 이 데이터에서는
  통계적으로 동등하다고 정직하게 보고한다("LightGBM이 낫다"고 우기지 않는다).
- 분위수 예측(P10/P50/P90) 구간 커버리지: **74.8%**(이론값 80%) — 약간 과신(overconfident) 경향,
  교차 위반(P10>P50 등)은 0.4%(5/1120)뿐이라 순서는 대체로 지켜짐.

## 실데이터 검증 (2026-09-11, `python demo_real.py`) — 합성 데이터 결론이 뒤집힌 지점

Hyndman *fpp2* 교재 표준 벤치마크 **a10**(호주 PBS 항당뇨병 약품 판매량, 1991-07~2008-06,
월별 204건, 출처: `github.com/selva86/datasets/a10.csv`)로 같은 파이프라인을 검증했다.

| 모델 | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| seasonal-naive(12개월전) | 3.267 | 3.670 | 15.0% |
| **Ridge 선형회귀** | **2.228** | **2.661** | **10.5%** |
| LightGBM(단독) | 7.202 | 7.916 | 31.6% |
| LightGBM(추세 제거 후) | 4.294 | 5.204 | 18.2% |

**LightGBM 단독이 naive보다도 나쁘다(7.20 vs 3.27).** 합성 데이터에서 봤던 "GBM이 항상 이긴다"는
결론이 실데이터에서 완전히 뒤집힌다 — `predictive_maintenance`가 C-MAPSS에서 겪은 것과 같은
패턴(합성 데이터의 낙관이 실데이터에서 깨짐)이 이번엔 반대 방향(GBM이 진 쪽)으로 나왔다.

### 원인 진단
```
train y 범위: [3.45, 18.00]
LightGBM 예측 범위: [14.25, 15.43]
실제 test y 범위: [16.43, 29.67]
```
이 시리즈는 항당뇨병 약품 사용량이 실제로 꾸준히 늘어난 추세(1991년 대비 2008년 약 5배)라,
test 구간(가장 최근 24개월)이 train에서 본 값의 최댓값을 넘어선다. **트리 기반 모델은 리프의
학습 타깃 평균으로 예측하므로 train 범위 밖을 원리적으로 외삽하지 못한다** — 잘 알려진 GBM의
구조적 한계다. 반대로 Ridge는 추세를 직선으로 외삽할 수 있어 이 시리즈에서 이긴다.

### 교체 기록 (LightGBM 단독 → 추세 제거 후 LightGBM)
선형 추세를 먼저 떼어내고(`days_since_start`만으로 LinearRegression) **잔차만 LightGBM에
맡긴 뒤 다시 더하는** `DetrendedGBM`으로 바꿨다. MAE **7.20 → 4.29(40% 개선)**. 다만 여전히
Ridge 단독(2.23)에는 못 미친다 — 이 시리즈는 추세+계절성이 거의 전부라 선형모델이 이미 정답에
가깝고, GBM은 144개월뿐인 학습 데이터에서 잔차의 잡음까지 일부 배우는 손해가 남기 때문으로
해석한다. **"항상 GBM으로 바꿔라"가 아니라 "시리즈 성격(추세 강도·데이터량)을 보고 고른다"가
정직한 결론이다.**

## 개발 순서
1. 특징 엔지니어링(`dfg/features.py`) — lag/rolling/달력, 리키지 방지(shift 순서) 단위테스트부터.
2. 합성 다품목 데이터(`dfg/synth_data.py`)로 전체 파이프라인(베이스라인·LightGBM·XGBoost·분위수) 완성.
3. **버그**: `usable_feature_columns`가 일별 기본 lag 이름(`lag_7`, `lag_28`)만 하드코딩돼 있어서,
   실데이터(월별, `lag_12` 등 커스텀)를 쓴 첫 실행에서 계절 특징이 조용히 빠졌다 — LightGBM MAE가
   8.15로 비정상적으로 나쁘게 나온 게 단서였다. 컬럼을 동적으로 잡도록 고치고
   회귀 테스트(`test_usable_feature_columns_picks_up_custom_lags_not_in_daily_defaults`)로 고정.
4. 실데이터(a10) 검증 → 고친 뒤에도 LightGBM이 베이스라인보다 나쁨을 발견 → 원인(외삽 실패) 진단
   → `DetrendedGBM`으로 교체 → 개선 확인, 그러나 완전히 따라잡지는 못함(정직하게 보고).

## 구성
```
dfg/
  features.py    lag/rolling/달력/추세 특징 — 순수 함수, 리키지 없음(shift 순서 테스트로 고정)
  synth_data.py   결정적 합성 다품목 수요 생성기(seed=42)
  models.py       naive/Ridge/LightGBM(점+분위수)/XGBoost/DetrendedGBM
  evaluate.py     MAE/RMSE/MAPE/pinball loss/구간 커버리지/Wilcoxon 짝지음 검정 — 순수 함수
demo.py           합성 데이터 전체 비교(6단계)
demo_real.py      실데이터(a10) 검증 + 원인 진단 + DetrendedGBM 개선
tests/            14개 단위·회귀 테스트(리키지 방지, 지표, GBM 경계 계약)
data/a10_antidiabetic_sales.csv   실데이터 캐시(출처 상단 주석)
```

## 정직한 범위
- 합성 데이터는 실제 판매 기록이 아니다(트렌드+주말+연계절성+프로모션+포아송 노이즈로 생성,
  `synth_data.py`에 가정 전부 문서화). 구조·지표·비교 방법론을 시연하는 용도다.
- 실데이터 검증은 **단일 월별 시리즈(204건)** 뿐이다 — "여러 매장/품목에서 배우는" GBM의 장점이
  발휘될 조건이 아니다. 다품목 실데이터(예: 여러 매장 POS)로는 검증하지 못했다.
- 클라우드 배포·실시간 서빙은 다루지 않았다. 로컬 배치 학습·평가까지가 이 프로젝트의 범위다.

---

## 2026-09-18 확장 — DeepAR/TFT(신경망 기반 확률적 예측)

> 캐롯아이(AI Engineer, AI Agent & Forecasting) JD 우대사항이 "DeepAR, TFT 등 시계열 수요/매출
> 예측·확률적 예측(P10/P50/P90)"이고, 위 "정직한 범위"에 **"DeepAR·TFT는 구현하지 않았다"**고
> 명시돼 있던 갭이었다. 새 프로젝트를 만들지 않고 이 프로젝트를 확장해, 기존 표와 완전히 같은 두
> 데이터셋(합성 40시리즈·실데이터 a10)으로 신경망 모델을 재서 나란히 비교했다.

### 라이브러리 선택
`neuralforecast`(Nixtla)를 택했다. DeepAR·TFT를 sklearn 스타일 `NeuralForecast(models=[...]).fit/predict`
하나로 다루고, long-format(`unique_id/ds/y`) 판다스 입력을 그대로 받아 다중 시리즈를 한 번에
학습한다 — 40개 시리즈를 개별 `TimeSeriesDataSet`으로 손으로 구성해야 하는 `pytorch-forecasting`
보다 이 데이터 구조에 적은 코드로 맞았다. **`pytorch-forecasting`을 실제로 설치·비교해보지는
않았다** — API 문서·요구 보일러플레이트만 보고 판단한 선택이라는 점을 밝힌다.

### 측정 설계 — 반드시 먼저 말해야 하는 비대칭
기존 LightGBM/XGBoost/Ridge는 `dfg/features.py`가 lag/rolling 특징을 **train/val/test 분할 전
전체 시리즈**에서 계산한다 — 즉 test 구간의 각 시점도 "직전 실측값을 이미 안다"고 가정한 **1스텝
예측**이다. 반면 DeepAR/TFT는 `NeuralForecast.predict()`로 마지막 cutoff 이후 실측값을 전혀 보지
않고 h스텝(합성 28일·실데이터 24개월)을 **한 번에** 예측하는 **진짜 다중 스텝 예측**이다. 후자가
명백히 더 어려운 과제라, 신경망이 GBM보다 못하더라도 "모델이 열등하다"가 아니라 "더 어려운 과제를
풀었다"일 수 있다 — 아래 결과를 읽을 때 이 비대칭을 계속 염두에 둬야 한다.

두 모델 다 train+val 구간 전체(합성 702일·실데이터 180개월)를 cutoff까지 컨텍스트로 학습했고,
`max_steps=300`(둘 다 CPU, `.venv_neural/`), 합성은 `input_size=90`, 실데이터는 `input_size=24`
(계절 주기 2회분)를 썼다. DeepAR는 `DistributionLoss(StudentT, level=[80])`로 분포 평균(점추정)과
80% 구간을, TFT는 `MQLoss(quantiles=[0.1,0.5,0.9])`로 P10/중앙값/P90을 직접 냈다.

### 핵심 결과 (실측, `python demo_neural.py`)

**합성 데이터(40시리즈, test 28일×40 = 1120행) — 신경망이 전부 졌다**

| 모델 | MAE | RMSE | 학습+예측 시간 |
|---|---:|---:|---:|
| seasonal-naive | 21.180 | 35.924 | - |
| Ridge | 12.411 | 17.992 | 0.23s |
| **LightGBM** | **10.353** | **13.861** | 1.60s |
| XGBoost | 10.405 | 14.180 | 0.61s |
| DeepAR | 18.821 | 30.639 | 160.9s |
| TFT | 21.915 | 33.700 | 811.9s |

DeepAR vs LightGBM: Wilcoxon **p=8.93e-45**(LightGBM이 유의하게 나음). TFT vs LightGBM: **p=1.49e-63**
(LightGBM이 유의하게 나음). TFT는 naive와 사실상 동률(21.915 vs 21.180)이다. **GBM은 학습 1.6초,
TFT는 812초(약 500배)** — 시간까지 감안하면 이 조건에서 신경망을 쓸 이유가 없다.

확률적 예측 커버리지(이론값 80%)는 반대로 **TFT가 가장 잘 보정됐다** — LightGBM quantile 74.8%,
DeepAR 68.5%, **TFT 79.8%**. 점추정은 가장 나쁜 모델이 구간 보정은 가장 좋다는, 점 정확도와 분포
보정이 서로 다른 축이라는 걸 보여주는 결과다.

**실데이터 a10(월별, test 24개월) — TFT가 역전해 최상위권**

| 모델 | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| seasonal-naive | 3.267 | 3.670 | 15.0% |
| **Ridge** | **2.228** | **2.661** | **10.5%** |
| LightGBM(단독) | 7.202 | 7.916 | 31.6% |
| LightGBM(추세제거) | 4.294 | 5.204 | 18.2% |
| DeepAR | 3.942 | 4.607 | 17.6% |
| **TFT** | **2.317** | **2.626** | **10.5%** |

**TFT(MAE 2.317)가 Ridge(2.228)와 사실상 동률로 1위**이고, LightGBM 두 버전을 큰 폭으로 앞선다.
이유를 예측값 범위로 확인했다: train y 범위 `[3.45, 18.00]`, 실제 test y 범위 `[16.43, 29.67]`
(위 §"실데이터 검증"에서 이미 확인한, train 범위를 크게 벗어나는 지속 성장 추세)인데 —

```
DeepAR 예측 범위: [14.36, 24.02]
TFT   예측 범위: [17.00, 25.54]
```

**TFT는 train 범위 밖으로 외삽했다.** 이 프로젝트가 이미 진단한 "트리 모델은 리프 평균으로
예측하므로 train 범위 밖을 원리적으로 외삽 못한다"는 한계(위 §"원인 진단")를, TFT의 attention
기반 디코더는 갖고 있지 않다는 뜻이다 — Ridge가 선형 외삽으로 이겼던 바로 그 지점에서 TFT도
같은 이유로 이긴다. DeepAR는 부분적으로만 외삽했고(24.02는 test 최댓값 29.67에 못 미침) 그만큼
MAE도 TFT보다 나쁘다.

단, **확률적 예측 커버리지는 둘 다 나쁘다**(DeepAR 12.5%, TFT 37.5%, 이론값 80%, n=24라 검정력이
낮다는 건 `demo_real.py`가 이미 밝힌 한계와 동일) — 추세가 지속 상승하는 구간에서 두 모델 다
불확실성 구간을 충분히 넓게 못 잡았다(실제값이 P90을 반복해서 뚫었다).

### 해석 — no free lunch가 한 번 더, 그러나 이번엔 신경망이 이기는 쪽에서
합성 데이터(짧은 다중시리즈, 진짜 h=28 예측)에서는 GBM/Ridge가 압승했고, 실데이터(단일 트렌드
시리즈, 진짜 h=24 예측)에서는 TFT가 GBM을 크게 이기고 Ridge와 동률을 냈다 — `demand_forecasting_gbm`
프로젝트가 반복해서 확인해 온 패턴(합성과 실데이터에서 승자가 다르다, `predictive_maintenance`의
탐지-예측 트레이드오프와 같은 종류)이 신경망 대 GBM 축에서도 그대로 재현됐다. **"신경망이 최신이라
낫다"도 "GBM이 가볍고 데이터가 적어 항상 낫다"도 둘 다 이 실측 안에서는 틀렸다** — 이 시리즈가
"외삽이 필요한 트렌드"냐 아니냐가 승자를 가른다.

### 에러 대처 기록
1. **저장소 공유 파이썬 환경 오염(가장 심각) — `pip install neuralforecast`가 전역 GPU torch를
   CPU 전용으로 강제 교체했다.** `torch 2.6.0+cu124` → `torch 2.14.0+cpu`로 바뀌면서
   `torchvision`/`torchaudio`가 깨지고, numpy도 2.1.3→2.5.2로 밀려 `opencv-python`/`numba`
   버전 제약과 충돌했다. 이 저장소의 다른 프로젝트(VLM Defect Inspector 등)가 전부 이 전역
   환경을 공유해서 쓴다 — **즉시 `pip uninstall`로 되돌리고 `torch==2.6.0+cu124`를 인덱스
   지정으로 재설치, `numpy==2.1.3`도 원복**했다. 이후 **`.venv_neural/`라는 이 프로젝트 전용
   격리 venv를 새로 만들어 neuralforecast를 그 안에만 설치**하는 것으로 재발을 막았다 —
   `requirements_neural.txt` 참고. **PyTorch 생태계 라이브러리는 전역 환경에 바로 설치하지
   않는다**는 원칙을 이 사고로 새로 세웠다.
2. **`UnicodeEncodeError: 'cp949' codec can't encode character '—'`** — Windows 콘솔
   기본 코드페이지(cp949)가 한글 print문의 em dash(—)를 못 씀. `PYTHONIOENCODING=utf-8
   PYTHONUTF8=1` 환경변수로 해결.
3. **TFT 예측 컬럼명이 예상과 다름** — `MQLoss(quantiles=[0.1,0.5,0.9])`가 만드는 컬럼은
   `'TFT-median'`이 아니라 끝에 `.0`이 붙는 `'TFT-lo-80.0'`/`'TFT-hi-80.0'` 형태였다(neuralforecast
   3.2.2가 quantile을 level 80% 표기로 환산하면서 소수점을 남긴다). `endswith("-lo-80")` 매칭이
   조용히 실패할 뻔한 걸 실행 로그로 컬럼명을 직접 찍어서 잡고, 부분일치(`in`)로 바꿨다
   (`dfg/neural_models.py`).
4. **학습 시간 비대칭** — TFT(870K 파라미터)가 DeepAR(199K)보다 합성 데이터에서 약 5배 느렸다
   (812s vs 161s) — attention 기반 TFT가 LSTM 기반 DeepAR보다 이 작은 데이터·CPU 환경에서
   무거웠다.

### 재현
```bash
python -m venv .venv_neural
.venv_neural/Scripts/python.exe -m pip install -r requirements_neural.txt
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 .venv_neural/Scripts/python.exe demo_neural.py
```
결과: `results_neural/neural_run.log`(전체 로그, Wilcoxon 검정·커버리지 포함),
`results_neural/results_neural_summary.txt`(요약).

### 정직한 범위 (신경망 확장분)
- **위 "측정 설계" 비대칭이 이 비교의 가장 중요한 한계다** — GBM은 oracle 1스텝, 신경망은 진짜
  h스텝 예측이라 직접적인 "동일 과제" 비교가 아니다. 신경망에도 oracle 1스텝(rolling
  origin, `cross_validation(step_size=1)`)을 적용해 과제를 맞추는 실험은 하지 않았다.
- `pytorch-forecasting`과의 head-to-head 비교는 하지 않았다(라이브러리 선택 근거는 문서 기반).
- 하이퍼파라미터 탐색을 하지 않았다(`max_steps=300` 고정, learning rate·hidden size 등 기본값) —
  신경망 쪽 수치가 더 좋아질 여지가 남아 있다는 뜻이고, 특히 합성 데이터의 큰 격차 일부는 과소
  학습(underfit) 때문일 수 있다.
- GPU를 쓰지 않았다(`.venv_neural`는 CPU torch) — 데이터가 작아 필요 없다고 판단했지만, 더 큰
  데이터·더 많은 max_steps에서는 GPU 없이 이 학습 시간(특히 TFT 812초)이 더 늘어난다.
