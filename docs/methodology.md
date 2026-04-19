# 방법론 근거 (Methodology)

> 이 문서는 포트폴리오에서 사용한 분석 방법론의 선택 근거를 기술한다.  
> "왜 이 방법인가"를 명확히 함으로써 기술적 판단력을 증명한다.

---

## 1. 혈당 지표 선택 근거

### 기본 5종 (TIR·TAR·TBR·GMI·CV%)

국제 CGM 합의 가이드라인(Battelino et al., *Diabetes Care* 2019)에서  
권고하는 표준 지표 세트다. Dexcom Clarity, Abbott LibreView 등  
기존 CGM 앱이 공통으로 제공하므로 "기본기 검증"의 의미를 갖는다.

- **TIR 기준 (70–140 mg/dL):** 건강인 기준. 당뇨 환자는 70–180 사용  
- **CV% 기준 (<36%):** Monnier et al., *Diabetes Care* 2017  
- **GMI 공식:** Bergenstal et al., *Diabetes Care* 2018 (14일 이상 착용 시 유효)

### 확장 지표 선택 이유

| 지표 | 왜 추가했나 |
|---|---|
| PPGE | 식사 단위 혈당 영향 정량화 — 기존 앱이 제공하지 않는 "어떤 식사가 문제?" |
| iAUC | 최고점(PPGE)과 달리 총 혈당 노출량 측정 — XGBoost 예측 타겟으로도 사용 |
| Recovery Time | 인슐린 민감성의 비침습적 간접 지표 — 시간 경과 추적 가능 |
| BGI | 고혈당·저혈당 위험을 비대칭 분리 — 다이어트 중 저혈당 특이적 감지 |

---

## 2. 예측 모델: XGBoost 선택 근거

### XGBoost vs 딥러닝

**왜 XGBoost인가:**

1. **해석 가능성:** SHAP 지원으로 피처별 기여량을 직접 계산 → 코칭 메시지 자동화  
2. **소규모 데이터 적합:** 45명 × 식사당 수십 행 = ~수백~수천 행 수준.  
   딥러닝(LSTM·Transformer)은 이 규모에서 과적합 위험이 높고  
   학습 불안정성이 크다.  
3. **속도:** 하이퍼파라미터 탐색 + LOSO 45회 반복이 현실적인 시간 내 완료  
4. **선행 연구:** Zeevi et al. (*Cell* 2015), CGMacros 공식 예제 모두 XGBoost 사용

**왜 딥러닝(LSTM)을 쓰지 않았나:**

- CGM 시계열 자체를 입력으로 넣는 LSTM은 45명으로 일반화 불가  
- 식사별 집계 피처(탄/단/지 + 개인 특성)는 이미 핵심 신호를 담고 있으며  
  시계열 모델링의 추가 이득이 불명확  
- "SOTA" 모델을 쓰는 것보다 해석 가능한 모델로 비즈니스 가치를 증명하는 것이 DS 역할에 부합

### 하이퍼파라미터

```python
n_estimators=300, max_depth=4, learning_rate=0.05,
subsample=0.8, colsample_bytree=0.8,
min_child_weight=5,   # 소규모 데이터 과적합 억제
reg_alpha=0.1, reg_lambda=1.0
```

`min_child_weight=5`: 45명 소규모에서 리프 노드당 최소 샘플 수를 높여  
leaf-level 과적합을 억제하기 위한 의도적 선택.

---

## 3. 검증 전략: GroupKFold + LOSO

### 왜 일반 K-Fold가 아닌 GroupKFold인가

CGM 데이터에서 동일 피험자의 식사가 train/test에 분리되면  
피험자 개인의 특성(체질, 습관)이 사실상 누수된다.  
GroupKFold(`groups=subject_id`)는 동일 피험자가 반드시 같은 fold에 배치되도록 강제한다.

```
❌ 일반 K-Fold:  피험자 A의 아침 → train, 피험자 A의 저녁 → test  (누수)
✅ GroupKFold:   피험자 A 전체 → train or test  (clean split)
```

### Population vs Personalized 비교

| 모델 | 학습 방식 | 검증 방식 | 목적 |
|---|---|---|---|
| Population | 전체 피험자 | GroupKFold | 일반화된 혈당 반응 패턴 |
| Personalized | LOSO (Leave-One-Subject-Out) | hold-out 피험자 | 개인화 이득 정량화 |

**LOSO 편향 보정:**  
LOSO는 45명 중 1명을 빼고 학습하므로 hold-out 피험자에 대한  
예측값이 전체 평균 쪽으로 편향될 수 있다.  
`bias = mean(y_hold_out) - mean(pred_hold_out)` 을 계산해 보정 적용.

---

## 4. 세그멘테이션: K-Means 선택 근거

### K-Means vs DBSCAN vs GMM

| 알고리즘 | 45명에서의 특성 |
|---|---|
| K-Means | 빠르고 안정적, 클러스터 수 K 직접 제어 → 임상 3군 분류와 자연스럽게 일치 |
| DBSCAN | 밀도 기반 → 45명 소규모에서 대부분 noise로 분류될 위험 |
| GMM | 확률 기반 soft assignment → 45명에서 분산 추정 불안정 |

### K=3 선택 근거

1. **Elbow curve:** K=3에서 관성(inertia) 감소율이 꺾임  
2. **Silhouette score:** K=3이 K=2~6 범위에서 상위 성능  
3. **임상 의미:** Stable / Moderate / At-risk 3군 분류는  
   CGM 임상 문헌에서 반복적으로 등장하는 표준 계층화 방식

### UMAP 시각화 선택 근거

- t-SNE 대비 전역 구조 보존 우수, 재현성이 높음 (`random_state` 고정)  
- `n_neighbors=10`: 45명 소규모에서 local-global 균형  
- 설치 불가 환경을 위해 PCA fallback 구현

---

## 5. A/B 테스트 설계 (서비스 미보유 → 설계 제안)

실서비스 로그가 없으므로 A/B 실험을 직접 수행할 수 없다.  
대신 "어떻게 설계할 것인가"를 기술한다.

### 가설 예시: PPGE 기반 개인화 식단 추천의 효과

```
H0: 개인화 추천(PPGE 기반)과 일반 추천의 TIR 개선율에 차이가 없다
H1: 개인화 추천이 일반 추천보다 2주 후 TIR을 더 크게 개선한다
```

### 실험 설계

| 항목 | 내용 |
|---|---|
| 단위 | 사용자 (사용자 내 교차 오염 방지) |
| 기간 | 2주 run-in + 4주 실험 |
| 1차 지표 | TIR 변화량 (% point) |
| 2차 지표 | PPGE 분포, 앱 DAU, 사용자 만족도 |
| 통계 검정 | Two-sample t-test (정규성) or Mann-Whitney U |
| 표본 수 | 효과 크기 d=0.5, α=0.05, power=0.8 → 약 64명/그룹 이상 |

### 주의사항

- **Novelty effect:** 새 기능 사용자가 초반에 과도하게 반응 → run-in 기간 필수  
- **네트워크 효과:** CGM 사용자 커뮤니티가 있다면 cluster-level randomization 고려  
- **중도 탈락:** CGM 기기 탈착, 앱 이탈 → ITT(Intention-to-Treat) 분석 원칙 적용
