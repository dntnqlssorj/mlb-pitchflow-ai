# ==============================================================================
# MLB PitchFlow AI - Inference-time Feature Enrichment 서비스
# 신설 일자: 2026-05-21
# 목적: API가 9-식별자만 수신하고, Supabase 마스터 테이블을 조회하여
#       모델 입력에 필요한 파생 피처를 동적으로 조립
#
# 아키텍처 근거 (잔여_개발_1.docx):
#   "API 호출 시 최소 식별자만 수신하고, 서버 내부에서 Supabase 고속 조회 후
#    베이스라인·OAA·블로킹 스코어를 자동 병합 및 실시간 연산하여 모델에 전달"
#
# 운영 한계 (plan.md §4.5):
#   velocity_decay_ratio, spin_decay_ratio의 실시간 rolling 값은 n8n 적재 딜레이로
#   인해 경기 중 완전 계산 불가 → 초기 구현에서 1.0 고정값 적용
#   Phase 2에서 별도 캐시 레이어(Redis 등)로 고도화 예정
# ==============================================================================

import os
import time
import logging
from typing import Optional

from dotenv import load_dotenv
from supabase import create_client, Client

logger = logging.getLogger(__name__)

load_dotenv()

# ------------------------------------------------------------------------------
# Supabase 클라이언트 싱글톤
# 모듈 임포트 시 1회 초기화, FastAPI 라이프사이클 동안 재사용
# sys.exit() 대신 RuntimeError: FastAPI 서버는 프로세스 종료 없이 에러 응답 필요
# ------------------------------------------------------------------------------
_supabase_client: Optional[Client] = None


def get_supabase() -> Client:
    """Supabase 클라이언트 싱글톤 반환. 미초기화 시 환경 변수 기반 초기화."""
    global _supabase_client
    if _supabase_client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL 또는 SUPABASE_KEY 환경 변수가 설정되지 않았습니다. "
                ".env 파일 또는 환경 설정을 확인하십시오."
            )
        _supabase_client = create_client(url, key)
        logger.info("Supabase 클라이언트 초기화 완료")
    return _supabase_client


# ------------------------------------------------------------------------------
# 조회 함수 1: 투수 시즌 집계 베이스라인
# 활용 인덱스: idx_bat_tracking_pitcher_year (pitcher, game_year)
# P2 Covering Index 적용 후: Index-only scan → heap fetch 배제
# ------------------------------------------------------------------------------
def fetch_pitcher_season_baseline(pitcher_id: int, game_year: int) -> dict:
    """
    투수의 시즌 전체 평균 구속·회전수 조회.
    train.py의 build_season_baseline() 산출값과 동일한 집계 로직.

    Returns:
        {"base_speed": float | None, "base_spin": float | None,
         "source": "db" | "fallback_zero"}
    """
    try:
        supabase = get_supabase()

        # PostgreSQL: AVG()는 supabase-py에서 직접 지원 안 되므로
        # 전체 행을 받아 Python에서 집계 (데이터 수 360~480행 수준, 허용 가능)
        # 향후 Supabase Edge Function으로 집계 로직 이전 가능
        response = (
            supabase
            .table("statcast_bat_tracking")
            .select("release_speed, release_spin_rate")
            .eq("pitcher", pitcher_id)
            .eq("game_year", game_year)
            .execute()
        )

        rows = response.data
        if not rows:
            logger.warning(
                f"투수 베이스라인 데이터 없음: pitcher={pitcher_id}, year={game_year}"
            )
            return {"base_speed": None, "base_spin": None, "source": "fallback_zero"}

        speeds = [r["release_speed"] for r in rows if r.get("release_speed") is not None]
        spins  = [r["release_spin_rate"] for r in rows if r.get("release_spin_rate") is not None]

        return {
            "base_speed": sum(speeds) / len(speeds) if speeds else None,
            "base_spin":  sum(spins)  / len(spins)  if spins  else None,
            "source": "db",
        }

    except Exception as exc:
        logger.warning(f"fetch_pitcher_season_baseline 실패: {exc}")
        return {"base_speed": None, "base_spin": None, "source": "fallback_zero"}


# ------------------------------------------------------------------------------
# 조회 함수 2: 현 경기 누적 투구 수
# 활용 인덱스 (P2 적용 후): idx_bat_tracking_sequence (pitcher, game_pk, ...)
# 운영 주의: n8n 파이프라인 적재 딜레이로 0 반환 가능 → pitch_count_override 우선
# ------------------------------------------------------------------------------
def fetch_pitch_count_in_game(pitcher_id: int, game_pk: int) -> dict:
    """
    현 경기에서 해당 투수의 현재까지 적재된 투구 수 조회.

    Returns:
        {"pitch_count_in_game": int, "source": "db"}
    """
    try:
        supabase = get_supabase()

        response = (
            supabase
            .table("statcast_bat_tracking")
            .select("pitch_number", count="exact")
            .eq("pitcher", pitcher_id)
            .eq("game_pk", game_pk)
            .execute()
        )

        count = response.count if response.count is not None else 0
        return {"pitch_count_in_game": count, "source": "db"}

    except Exception as exc:
        logger.warning(f"fetch_pitch_count_in_game 실패: {exc}")
        return {"pitch_count_in_game": 0, "source": "fallback_zero"}


# ------------------------------------------------------------------------------
# 조회 함수 3: 포수 블로킹 스코어
# 활용 인덱스: PK (player_id, game_year) → 단일 행 조회, 0.1~0.5ms
# ------------------------------------------------------------------------------
def fetch_catcher_blocking(catcher_id: int, game_year: int) -> dict:
    """
    포수의 시즌 블로킹 런 스코어 조회.

    Returns:
        {"catcher_blocking_runs": float, "source": "db" | "fallback_zero"}
    """
    try:
        supabase = get_supabase()

        response = (
            supabase
            .table("catcher_blocking")
            .select("catcher_blocking_runs")
            .eq("player_id", catcher_id)
            .eq("game_year", game_year)
            .limit(1)
            .execute()
        )

        rows = response.data
        if not rows or rows[0].get("catcher_blocking_runs") is None:
            logger.debug(
                f"포수 블로킹 데이터 없음: catcher={catcher_id}, year={game_year}"
            )
            return {"catcher_blocking_runs": 0.0, "source": "fallback_zero"}

        return {
            "catcher_blocking_runs": float(rows[0]["catcher_blocking_runs"]),
            "source": "db",
        }

    except Exception as exc:
        logger.warning(f"fetch_catcher_blocking 실패: {exc}")
        return {"catcher_blocking_runs": 0.0, "source": "fallback_zero"}


# ------------------------------------------------------------------------------
# 조회 함수 4: 야수 OAA 7명 일괄 조회
# 활용 인덱스: PK (player_id, game_year) → IN 절 Bitmap Index Scan
# 7명 개별 조회 대신 IN 절 1회 쿼리로 Supabase 라운드트립 최소화
# ------------------------------------------------------------------------------
def fetch_fielding_oaa(fielder_ids: list, game_year: int) -> dict:
    """
    출전 야수진 7명의 OAA 일괄 조회 및 합산.

    Args:
        fielder_ids: fielder_3 ~ fielder_9 ID 목록 (0값 자동 제외)
        game_year:   시즌 연도

    Returns:
        {"team_oaa_total": float, "fielder_oaa_map": dict, "source": "db" | "fallback_zero"}
    """
    valid_ids = [fid for fid in fielder_ids if fid and fid != 0]

    if not valid_ids:
        return {"team_oaa_total": 0.0, "fielder_oaa_map": {}, "source": "fallback_zero"}

    try:
        supabase = get_supabase()

        response = (
            supabase
            .table("fielding_oaa")
            .select("player_id, outs_above_average")
            .in_("player_id", valid_ids)
            .eq("game_year", game_year)
            .execute()
        )

        rows = response.data
        if not rows:
            return {"team_oaa_total": 0.0, "fielder_oaa_map": {}, "source": "fallback_zero"}

        oaa_map   = {r["player_id"]: float(r["outs_above_average"] or 0) for r in rows}
        oaa_total = sum(oaa_map.values())

        return {
            "team_oaa_total": oaa_total,
            "fielder_oaa_map": oaa_map,
            "source": "db",
        }

    except Exception as exc:
        logger.warning(f"fetch_fielding_oaa 실패: {exc}")
        return {"team_oaa_total": 0.0, "fielder_oaa_map": {}, "source": "fallback_zero"}


# ------------------------------------------------------------------------------
# 통합 enrichment 함수
# predict.py의 predict_pitch 엔드포인트가 호출하는 단일 진입점
# ------------------------------------------------------------------------------
def enrich_pitch_context(
    pitcher_id:            int,
    batter_id:             int,
    catcher_id:            int,
    fielder_ids:           list,          # fielder_3 ~ fielder_9 (총 7개, 0값 허용)
    game_pk:               int,
    game_year:             int,
    on_2b:                 int,           # 득점권 판별용 (API 입력에서 직접 수신)
    on_3b:                 int,
    pitch_count_override:  Optional[int] = None,  # n8n 딜레이 회피용 직접 입력
) -> dict:
    """
    Supabase 조회 → 인메모리 도메인 피처 연산 → enriched 딕셔너리 반환.

    Args:
        pitcher_id:           투수 MLB player_id
        batter_id:            타자 MLB player_id (현재 미사용, 향후 타자 마스터 연동 대비)
        catcher_id:           포수 MLB player_id (fielder_2)
        fielder_ids:          야수 7명 ID 리스트 [fielder_3, ..., fielder_9]
        game_pk:              경기 고유 ID
        game_year:            시즌 연도
        on_2b:                2루 주자 player_id (없으면 0)
        on_3b:                3루 주자 player_id (없으면 0)
        pitch_count_override: 직접 입력 투구 수 (None이면 DB 조회)

    Returns:
        ALLOWED_FEATURES 그룹 D·E·F 대응 딕셔너리 + enrichment_latency_ms
    """
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 단계 1. Supabase 순차 조회 (4개 함수)
    # 초기 구현: 순차 호출 (단순성 우선)
    # Phase 2: asyncio.gather 병렬화로 전환 가능
    # ------------------------------------------------------------------
    baseline_result  = fetch_pitcher_season_baseline(pitcher_id, game_year)
    blocking_result  = fetch_catcher_blocking(catcher_id, game_year)
    oaa_result       = fetch_fielding_oaa(fielder_ids, game_year)

    if pitch_count_override is not None:
        pitch_count_result = {
            "pitch_count_in_game": pitch_count_override,
            "source": "override",
        }
    else:
        pitch_count_result = fetch_pitch_count_in_game(pitcher_id, game_pk)

    # ------------------------------------------------------------------
    # 단계 2. 도메인 파생 피처 인메모리 연산
    # ------------------------------------------------------------------
    base_speed = baseline_result["base_speed"] or 0.0
    base_spin  = baseline_result["base_spin"]  or 0.0

    # velocity_decay_ratio, spin_decay_ratio:
    #   실시간 경기 중 이전 투구 rolling 구속 데이터 부재 → 1.0 초기값 (감쇠 없음)
    #   Phase 2에서 n8n 실시간 캐시 적재 구조로 대체 예정
    velocity_decay_ratio = 1.0
    spin_decay_ratio     = 1.0

    pitch_count_in_game = pitch_count_result["pitch_count_in_game"]

    # stamina_index: rolling 비율 없으므로 투구 수 기반 근사값
    # vel_drop = spin_drop = 0 → stamina_index = 0 (Phase 2 이전 근사)
    stamina_index = (pitch_count_in_game / 100.0) * 0.0

    # 득점권 여부 (on_2b, on_3b: API 입력에서 직접 수신 → DB 조회 불필요)
    is_risp = int((on_2b != 0) or (on_3b != 0))

    catcher_blocking_runs   = blocking_result["catcher_blocking_runs"]
    blocking_leverage_factor = is_risp * catcher_blocking_runs * 0.1

    team_oaa_total   = oaa_result["team_oaa_total"]
    fielding_risk_index = max(-team_oaa_total * 0.05, 0)

    # ------------------------------------------------------------------
    # 단계 3. enriched 딕셔너리 조립 및 latency 계산
    # ------------------------------------------------------------------
    enrichment_latency_ms = (time.perf_counter() - t_start) * 1000

    enriched = {
        # 그룹 D: 투수 체력 파생
        "pitch_count_in_game":   pitch_count_in_game,
        "base_speed":            base_speed,
        "base_spin":             base_spin,
        "velocity_decay_ratio":  velocity_decay_ratio,
        "spin_decay_ratio":      spin_decay_ratio,
        "stamina_index":         stamina_index,
        # 그룹 E: 포수 도메인
        "catcher_blocking_runs":    catcher_blocking_runs,
        "is_risp":                  is_risp,
        "blocking_leverage_factor": blocking_leverage_factor,
        # 그룹 F: 야수 OAA 도메인
        "team_oaa_total":      team_oaa_total,
        "fielding_risk_index": fielding_risk_index,
        # 메타데이터 (모델 입력에서는 제외, 응답 메타로만 포함)
        "enrichment_latency_ms": round(enrichment_latency_ms, 2),
        "enrichment_sources": {
            "base_speed":            baseline_result["source"],
            "pitch_count_in_game":   pitch_count_result["source"],
            "catcher_blocking_runs": blocking_result["source"],
            "team_oaa_total":        oaa_result["source"],
        },
    }

    logger.info(
        f"Enrichment 완료: pitcher={pitcher_id}, game_pk={game_pk}, "
        f"latency={enrichment_latency_ms:.1f}ms"
    )

    return enriched