# MLB PitchFlow AI — 팀 작업 지침

## 프로젝트 개요

MLB Statcast 데이터 기반 구종 예측 ML 시스템.
데이터는 Supabase PostgreSQL에 적재되고, FastAPI 백엔드가 enrichment 조회를 담당한다.

---

## 환경 설정

프로젝트 루트에 `.env` 파일 생성 후 아래 값 입력 (팀 노션 또는 팀장에게 발급 요청):

```
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-service-role-key
```

> `.env`는 `.gitignore`에 포함되어 있으므로 절대 커밋하지 말 것.

---

## Supabase 데이터 적재 절차

### 대상 테이블
`statcast_bat_tracking`

### 업로드 컬럼 구성 (총 41개)
`ml_engine/config.py`의 `ALLOWED_FEATURES`에 정의된 컬럼만 업로드한다.
나머지 ~88개 컬럼은 누수 피처 또는 불필요 컬럼으로 제외된다.

| 그룹 | 설명 | 컬럼 수 |
|------|------|---------|
| A | 경기 상황 (pre-pitch) | 9 |
| B | 투수 이력 | 5 |
| C | 타자 이력 | 5 |
| D | 투수 체력 파생 피처 | 6 |
| E | 포수 도메인 | 4 |
| F | 야수 OAA 도메인 | 9 |
| G | PK 식별자 (game_pk, at_bat_number, pitch_number) | 3 |

### 1단계 — 기존 데이터 삭제 (Supabase SQL Editor)

Supabase 대시보드 → SQL Editor에서 실행:

```sql
TRUNCATE TABLE statcast_bat_tracking;
```

> 중복 에러 방지를 위해 업로드 전 반드시 실행할 것.

### 2단계 — 업로드 실행

```bash
python -m ml_engine.upload_to_supabase
```

`pilot_mode` 옵션 (`upload_to_supabase.py` 하단 `__main__` 블록에서 조정):

| 값 | 동작 |
|----|------|
| `True` | 상위 1,000행만 테스트 업로드 (기본 안전값) |
| `False` | 144만 행 전체 업로드 |

> 전체 업로드 전에 반드시 `pilot_mode=True`로 먼저 검증할 것.

---

## 주요 파일 참조

| 파일 | 역할 |
|------|------|
| `ml_engine/config.py` | `ALLOWED_FEATURES` 화이트리스트, 학습/검증 연도 설정 |
| `ml_engine/upload_to_supabase.py` | Supabase 적재 스크립트 |
| `ml_engine/feature_engineering.py` | 도메인 피처 빌드 함수 |
| `ml_engine/datasets.py` | 원본 데이터 로드 및 정제 |
| `ml_engine/train.py` | 모델 학습 진입점 |
| `backend/services/enrichment.py` | Supabase 조회 enrichment 서비스 |

---

## 컬럼 변경 시 주의사항

`ALLOWED_FEATURES`를 수정하면 아래 세 곳이 모두 영향을 받는다:

1. `ml_engine/config.py` — 리스트 수정
2. `ml_engine/upload_to_supabase.py` — `bigint_cols`에 정수 컬럼 추가 여부 확인
3. Supabase 테이블 스키마 — 컬럼 추가/삭제 시 DDL 변경 필요

세 곳을 함께 업데이트하지 않으면 업로드 에러 또는 enrichment 조회 실패가 발생한다.
