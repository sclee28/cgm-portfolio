# CGM 기반 개인화 혈당 분석 포트폴리오

> **지원 직무:** 웰다(Welda) 데이터 사이언티스트  
> **데이터셋:** [CGMacros](https://github.com/nudgebio/cgmacros) — Dexcom+Libre CGM × Fitbit × 식이 데이터, 45명, 2주간  
> **마감:** 2026-04-20

---

## 한 줄 요약

CGM 혈당 시계열 + 식이 데이터로 **식사별 혈당 반응을 예측(XGBoost)**하고,  
**SHAP 해석 → 코칭 메시지 자동 생성**까지 이어지는 개인화 분석 파이프라인을 구현했다.

---

## 디렉토리 구조

```
cgm-portfolio/
├── src/
│   ├── config.py          # 상수·컬럼 맵 중앙 관리
│   ├── data_loader.py     # CGMacros 로드 + 정제
│   ├── metrics.py         # 혈당 지표 라이브러리 (11종)
│   ├── features.py        # 식사 피처 엔지니어링
│   ├── model.py           # 베이스라인·XGBoost·LOSO 파이프라인
│   └── segments.py        # K-Means 세그멘테이션
├── notebooks/
│   ├── 01_eda.ipynb            # EDA + 그룹별 분석
│   ├── 02_ppgr_model.ipynb     # PPGR 예측 모델 + SHAP
│   └── 03_segmentation.ipynb  # 대사 유형 분류 + UMAP
├── tests/
│   └── test_metrics.py    # pytest 단위 테스트
├── docs/
│   ├── jd_mapping.md      # JD 담당업무 × 포트폴리오 커버리지
│   ├── methodology.md     # 방법론 선택 근거
│   ├── limitations.md     # 한계 + 향후 방향
│   └── product_proposal.md # 서비스 고도화 제안서
├── data/
│   ├── raw/               # git 미추적 (CGMacros 원본)
│   └── processed/         # 중간 산출물 (CSV + PNG)
└── requirements.txt
```

---

## 구현 산출물 요약

### 혈당 지표 라이브러리 (`src/metrics.py`)

업계 표준 5종과 차별화 지표를 직접 구현했다.

| 지표 | 분류 | 의미 |
|---|:---:|---|
| TIR (Time in Range) | 기본 | 혈당 70–140 mg/dL 비율 — 하루 안정성 |
| TAR / TBR | 기본 | 고혈당·저혈당 노출 시간 비율 |
| GMI | 기본 | 평균 혈당 → HbA1c 추정 (Bergenstal 2018) |
| CV% | 기본 | 혈당 변동성 표준화 지수 |
| **PPGE** | **확장** | **식후 최고 혈당 − 식전 기저 혈당** |
| **iAUC** | **확장** | **식전 기저 위 혈당 노출 면적 (총 반응량)** |
| **Recovery Time** | **확장** | **식후 혈당이 기저로 돌아오는 시간 (인슐린 민감성 간접 지표)** |
| BGI (HBGI/LBGI) | 확장 | 고혈당·저혈당 위험 비대칭 분리 |
| MAGE | 확장 | 1SD 이상의 의미 있는 혈당 변동만 추출 |

### 개인화 PPGR 예측 모델 (`02_ppgr_model.ipynb`)

**타겟:** iAUC (식후 혈당 총 노출량)  
**피처:** 탄수화물·단백질·지방·섬유 + 식사 유형·시간대 + 나이·BMI·HbA1c

```
검증 방법:  GroupKFold (n=5, subject_id 기준) — 피험자 누수 방지
비교 모델:  Mean Baseline → Ridge → XGBoost Population → XGBoost Personalized (LOSO)
해석:       SHAP 전역 피처 중요도 + 식사별 Waterfall → 코칭 메시지 자동 변환
```

**서비스 출력 시뮬레이션:**

```
📊 예측 혈당 반응(iAUC): 3,850 mg/dL·min  [매우 높음 (주의)]
주요 영향 요인 (상위 3개):
  1. 탄수화물: 탄수화물 섭취량이 혈당 반응을 ↑ 높여 기여 (SHAP=420)
  2. HbA1c: HbA1c(장기 혈당 지표)가 혈당 반응을 ↑ 높여 기여 (SHAP=180)
  3. 식이섬유: 식이섬유 섭취량이 혈당 반응을 ↓ 낮춰 기여 (SHAP=95)

💡 개선 제안:
  • 탄수화물을 10~20g 줄이면 혈당 반응이 약 10~20% 감소할 수 있습니다.
  • 식이섬유(채소, 통곡물)를 추가하면 혈당 상승을 완화할 수 있습니다.
```

### 대사 유형 세그멘테이션 (`03_segmentation.ipynb`)

K-Means K=3으로 피험자를 대사 유형 3군으로 분류했다.

| 세그먼트 | 특징 | 코칭 방향 |
|---|---|---|
| **Stable** | TIR 높음, CV% 낮음 | 현재 식습관 유지·강화 |
| **Moderate** | 일부 고반응 식사 | 탄수화물 섭취량·식사 타이밍 점검 |
| **At-risk** | TIR 낮음, 빈번한 이탈 | 전문 코칭 연결 권장 |

---

## 담당업무 커버리지

| 담당업무 | 커버리지 | 핵심 산출물 |
|---|:---:|---|
| 1. 데이터 분석 + 개인화 알고리즘 | 80% | `metrics.py` + EDA + 세그멘테이션 |
| 2. 예측/추천 모델 개발 + 검증 | 70% | XGBoost + GroupKFold + SHAP |
| 3. 서비스 분석 + 실험 | 30% | `product_proposal.md` — 설계 중심 |
| 4. 데이터 활용 협업 | 60% | 코드 품질 + 타입힌트 + pytest |

자세한 매핑: [`docs/jd_mapping.md`](docs/jd_mapping.md)

---

## 빠른 시작

```bash
# 환경 구성
pip install -r requirements.txt

# 단위 테스트
pytest tests/ -v

# 노트북 순서대로 실행
jupyter lab notebooks/01_eda.ipynb
jupyter lab notebooks/02_ppgr_model.ipynb
jupyter lab notebooks/03_segmentation.ipynb
```

**데이터 준비:**  
CGMacros 원본을 `raw_data/CGMacros/` 에 배치한 후 `01_eda.ipynb`를 먼저 실행해  
`data/processed/meal_metrics.csv`, `user_split.csv`, `subject_summary.csv`를 생성해야 한다.

---

## 의도적으로 포함하지 않은 것

| 항목 | 이유 |
|---|---|
| Collaborative Filtering | 45명으로 user-item matrix 구성 불가 |
| R² > 0.8 주장 | 소규모 데이터 + subject-level 누수 차단 → 현실적 수치 제시 |
| "최신 SOTA" 모델 | 해석 가능성 > 성능 — DS 실무 우선순위와 일치 |
| AGE 축적 추정 | CGM으로 직접 계산 불가 |

한계 상세: [`docs/limitations.md`](docs/limitations.md)

---

## 참고 문헌

- Battelino et al. (2019). *Diabetes Care* — 국제 CGM 합의 가이드라인
- Bergenstal et al. (2018). *Diabetes Care* — GMI 공식
- Zeevi et al. (2015). *Cell* — iAUC 기반 개인화 영양 연구
- Kovatchev et al. (2006). *Diabetes Technology & Therapeutics* — BGI
- Monnier et al. (2017). *Diabetes Care* — CV%
- Service et al. (1970). *Diabetes* — MAGE
