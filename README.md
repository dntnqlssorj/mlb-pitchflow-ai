# MLB PitchFlow AI

**MLB Statcast 데이터 기반 실시간 투구 구종 예측 엔진 및 서빙 플랫폼**
*(Real-time Pitch Type Prediction Engine & Serving Platform using MLB Statcast Data)*

SW중심대학 산학협력 프로젝트 | 2025~2026

---

## 프로젝트 개요 (Overview)

MLB PitchFlow AI는 MLB Statcast 약 144만 행의 정형 데이터를 기반으로, **투구가 던져지기 전 시점**의 정보만을 활용하여 다음 구종(Pitch Type)을 예측하는 ML 시스템이다. 투구 결과 시점의 사후 물리 지표(구속, 회전수, 무브먼트 등)를 학습·추론에서 전면 배제하는 **Target Leakage 차단**을 핵심 연구 과제로 설정하였으며, FastAPI 기반 실시간 서빙 플랫폼까지 포함한 End-to-End 시스템을 구현하였다.

---

## 연구 배경 및 문제 정의 (Research Background)

기존 구종 예측 연구의 대부분은 `release_speed`, `pfx_x/z`, `plate_x/z` 등 투구 결과 이후에만 확정되는 사후 물리 지표를 피처로 포함하여, 사실상 **"이미 던진 공을 역분류"**하는 구조로 설계되어 있었다. 이는 실전 예측 시나리오(Pre-Pitch Prediction)와 근본적으로 괴리된 가짜 성능을 산출한다.

본 프로젝트는 이를 **Target Leakage** 문제로 규정하고, 51개 누수 피처를 전면 제거한 후 실제 예측 가능한 피처만으로 모델을 재설계하였다.

| 구분 | 이전 (누수 포함) | 이후 (누수 제거) |
|---|---|---|
| F1-Score | 95.75% (가짜) | 44.72% (실측) |
| 학습/검증 분할 | 무작위 8:2 | Date-based Chronological Split |
| 허용 피처 수 | 112개 (누수 포함) | 84개 정의 / 75개 실사용 |
| 출력 구종 클래스 | — | 18개 (희귀 구종 → `OT` 통합) |
| 예측 구조 | 사후 역분류 | Pre-Pitch 실시간 예측 |

---

## ML 아키텍처 (ML Architecture)

```
Layer 0  입력: 9개 게임 상황 식별자 (pitcher_id, batter_id, inning, balls, strikes, ...)
   ↓
Layer 1  Inference-time Feature Enrichment
         pkl 인메모리 캐시 조회 → 파생 피처 동적 조립 (latency < 0.2ms)
   ↓
Layer 2  4-Tier Routing
         Tier 1: Per-Pitcher Model     — 763명 등록 투수 개별 모델
         Tier 2: Scouting LLM Path     — 신인/미등록 투수 (GPT-4o-mini + data/scouting/ CSV)
         Tier 3: Stacking Ensemble     — XGBoost + LightGBM + CatBoost, 5-Fold OOF Blending
         Tier 4: Deep Learning Blend   — Bi-LSTM + Transformer (find_best_weights.py 가중치 최적화)
   ↓
Layer 3  출력: 18개 구종 확률 분포 (FF, SL, CH, CU, ..., OT)
   ↓
Layer 4  FastAPI → n8n → Next.js 서빙 | 3D 투구 궤적 시각화 (Three.js)
```

### 모델 성능 비교 (Model Performance Comparison)

| 모델 | Macro F1-Score |
|---|---|
| Random Forest (Baseline) | 36.09% |
| LightGBM | 39.03% |
| CatBoost | 39.30% |
| XGBoost (Standalone) | 42.52% |
| **Stacking Ensemble** | **44.72%** |
| Per-Pitcher 평균 (763명) | 41.04% |

> 문헌 상한 참고: Bright et al. — Pre-release 예측 가능 구간 약 44~55%

### Ablation Study

| 조건 | F1 변화 |
|---|---|
| `stamina_index` 제거 | −0.85%p |
| Ball-flight features 포함 시 | ~99% (Target Leakage 확인) |

---

## 시스템 구성 (System Architecture)

```
MLB PitchFlow AI/
├── ml_engine/                   # ML 학습 파이프라인
│   ├── config.py                # ALLOWED_FEATURES(84개), LEAKAGE_FEATURES, 날짜 기반 Split 상수
│   ├── feature_engineering.py  # stamina_index, is_risp, OAA 파생 피처 생성
│   ├── train.py                 # Date-based Chronological Split 단일 모델 학습
│   ├── stacking.py              # OOF Stacking Ensemble (XGB + LGBM + CatBoost)
│   ├── per_pitcher_train.py     # 투수별 개별 모델 학습 (763명)
│   ├── find_best_weights.py     # Stacking + 딥러닝 Soft Voting 블렌딩 가중치 탐색
│   └── build_cache.py           # 추론용 pkl 인메모리 캐시 사전 생성
├── backend/                     # FastAPI 추론 서버
│   ├── main.py
│   ├── routers/predict.py       # 구종 예측 엔드포인트 (XGB/LGBM/Ensemble/DL 분기)
│   ├── services/enrichment.py   # Inference-time Feature Enrichment
│   └── services/scouting_predictor.py  # GPT-4o-mini 스카우팅 리포트
├── frontend/                    # Next.js 15 대시보드
│   └── components/mlb/
│       ├── GameSelector.tsx
│       ├── LiveResultPanel.tsx
│       ├── LiveSceneViewer.tsx  # Three.js 3D 투구 궤적
│       └── ScoreBoard.tsx
├── n8n/                         # 자동화 워크플로우 (4개)
│   ├── 1_pre_game.json
│   ├── 2_in_game.json
│   ├── 3_post_game.json
│   └── 4_custom.json
└── data/
    └── scouting/                # 신인 투수 LLM 라우팅용 CSV
```

---

## 핵심 설계 원칙 (Design Principles)

### 1. Target Leakage 원천 차단

투구 결과 시점 이후에 확정되는 51개 사후 지표(`release_speed`, `pfx_x/z`, `plate_x/z`, `bat_speed` 등)를 학습 및 추론 전 단계에서 전면 제거. `ml_engine/config.py`의 `LEAKAGE_FEATURES` 상수로 관리.

### 2. Date-based Chronological Split

MLB 투구 데이터는 시계열 데이터. 무작위 분할은 미래 정보의 학습 시점 역유입을 허용하여 검증 지표를 왜곡함.

- 학습(Train): 2024 시즌 전체 + 2025 시즌 전반기 (~ 2025-08-31)
- 검증(Holdout): 2025 시즌 후반기 (2025-09-01 ~)

### 3. pkl 인메모리 캐시 아키텍처

Supabase 실시간 쿼리는 네트워크 지연으로 인해 추론 시 300ms~1.2s 소요. 시즌 집계 피처(`stamina_index`, `team_oaa_total` 등)를 사전에 pkl 파일로 빌드하여 추론 시 인메모리에서 직접 조회 → **latency < 0.2ms** 달성.

### 4. 4-Tier Routing

데이터 희소성 문제(평균 약 1,263행/투수) 해결 및 신인 투수 커버리지 확보를 위해 4단계 Fallback 구조 채택. Per-Pitcher Model → Scouting LLM → Stacking Ensemble → Deep Learning Blend.

### 5. Hybrid Ensemble (Tree + Deep Learning)

`find_best_weights.py`를 통해 Stacking Ensemble(XGB/LGBM/CatBoost)과 딥러닝 모델(Bi-LSTM, Transformer)의 Soft Voting 블렌딩 최적 가중치를 탐색. `predict.py` 라우터에서 `model_type` 파라미터로 추론 엔진 선택 가능.

---

## 기술 스택 (Tech Stack)

| 영역 | 기술 |
|---|---|
| ML (Tree) | Python 3.11, XGBoost, LightGBM, CatBoost, scikit-learn |
| ML (Deep Learning) | PyTorch, Bi-LSTM, Transformer |
| Backend | FastAPI, joblib, uvicorn |
| Frontend | Next.js 15, Three.js (3D 투구 궤적), Tailwind CSS |
| Automation | n8n (local, port 5678) |
| AI Commentary | OpenAI GPT-4o-mini (한국어 게임 리포트) |
| Data | MLB Statcast (~1.44M rows, 2024~2025), MLB Stats API |

---

## 로컬 실행 가이드 (Quick Start)

### 사전 요구사항

- Python 3.11+
- Node.js 18+
- n8n (로컬 설치)

### 1. 의존성 설치

```bash
# Python 환경
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

### 2. 환경 변수 설정

```bash
cp .env.example .env
# OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY 입력
```

### 3. pkl 캐시 빌드 (최초 1회)

```bash
python ml_engine/build_cache.py
```

### 4. 서버 실행

```bash
# Backend (port 8000)
uvicorn backend.main:app --reload

# Frontend (port 3000)
cd frontend && npm run dev

# n8n (port 5678)
n8n start
```

### 5. 추론 API 테스트

```bash
curl -X POST "http://localhost:8000/predict/pitch" \
  -H "Content-Type: application/json" \
  -d '{
    "pitcher": 605135,
    "batter": 592518,
    "fielder_2": 663728,
    "game_pk": 745057,
    "game_year": 2025,
    "balls": 1,
    "strikes": 2,
    "outs_when_up": 1,
    "inning": 7
  }'
```

---

## 참고 문헌 (References)

- Bright et al. — *Pre-release pitch type prediction ceiling: 44–55%*
- Sidle & Tran (2018) — *Predicting Pitch Types in Baseball*
- Lee (2022) — *Benchmark comparisons for pitch classification*
- Tom Tango, " The Book" — Domain feature design reference

---

## 팀 구성 (Contributors)

| 역할 | GitHub |
|---|---|
| ML 엔지니어링 / 백엔드 | [@dntnqlssoRj](https://github.com/dntnqlssoRj) |
| 프론트엔드 / 시각화 | [@doxjsj](https://github.com/doxjsj) |
| 데이터 파이프라인 / n8n | [@Kang-chaeYeon](https://github.com/Kang-chaeYeon) |

---

*SW중심대학 산학협력 프로젝트 — MLB PitchFlow AI*
