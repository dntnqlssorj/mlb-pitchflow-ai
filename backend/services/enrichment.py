# ==============================================================================
# MLB PitchFlow AI - Inference-time Feature Enrichment 서비스 (pkl 캐시 버전)
# 변경 이력: 2026-05-29 Supabase → pkl 캐시 기반으로 전면 전환
# 이유: Supabase 무료 플랜 500MB 용량 한도 초과
#       pkl 캐시로 전환 시 응답 속도 향상 및 오프라인 동작 가능
# ==============================================================================

import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional
import joblib

logger = logging.getLogger(__name__)

MODEL_DIR = Path("ml_engine/models")

# ------------------------------------------------------------------------------
# pkl 캐시 싱글톤 로더
# 모듈 임포트 시 1회 로드, 이후 메모리에서 즉시 조회
# ------------------------------------------------------------------------------
_cache = {}

def _load_cache(name: str) -> dict:
    if name not in _cache:
        path = MODEL_DIR / f"{name}.pkl"
        if not path.exists():
            logger.warning(f"캐시 파일 없음: {path} — build_cache.py 먼저 실행 필요")
            _cache[name] = {}
        else:
            _cache[name] = joblib.load(path)
            logger.info(f"캐시 로드 완료: {name} ({len(_cache[name])}개)")
    return _cache[name]


# ------------------------------------------------------------------------------
# enrich_pitch_context — predict.py 단일 진입점 (인터페이스 유지)
# ------------------------------------------------------------------------------
def enrich_pitch_context(
    pitcher_id:            int,
    batter_id:             int,
    catcher_id:            int,
    fielder_ids:           list,
    game_pk:               int,
    game_year:             int,
    on_2b:                 int,
    on_3b:                 int,
    pitch_count_override:  Optional[int] = None,
    inning:                int = 1,
) -> dict:
    t_start = time.perf_counter()

    # 1. 투수 베이스라인
    baseline_cache = _load_cache("enrichment_pitcher_baseline")
    baseline = baseline_cache.get((pitcher_id, game_year), {})
    base_speed = baseline.get("base_speed") or 0.0
    base_spin  = baseline.get("base_spin")  or 0.0
    p_throws   = baseline.get("p_throws")   or "R"
    baseline_source = "cache" if baseline else "fallback_zero"

    # 2. 투수 구종 비율
    rep_cache = _load_cache("enrichment_pitcher_repertoire")
    rep = rep_cache.get((pitcher_id, game_year), {})
    rep_source = "cache" if rep else "fallback_zero"

    # 3. 포수 블로킹
    blocking_cache = _load_cache("enrichment_catcher_blocking")
    blocking = blocking_cache.get((catcher_id, game_year), {})
    catcher_blocking_runs = blocking.get("catcher_blocking_runs", 0.0)
    blocking_source = "cache" if blocking else "fallback_zero"

    # 4. 야수 OAA
    oaa_cache = _load_cache("enrichment_fielding_oaa")
    valid_ids = [fid for fid in fielder_ids if fid and fid != 0]
    oaa_map = {fid: oaa_cache.get((fid, game_year), {}).get("outs_above_average", 0.0) for fid in valid_ids}
    team_oaa_total = sum(oaa_map.values())
    oaa_source = "cache" if any(oaa_map.values()) else "fallback_zero"

    # 5. 투구 수
    if pitch_count_override is not None:
        pitch_count_in_game = pitch_count_override
        pitch_count_source = "override"
    else:
        # 이닝 기반 Heuristic 추정: 이닝당 15구 기준 (최소 1구)
        pitch_count_in_game = max((inning - 1) * 15 + 1, 1)
        pitch_count_source = "heuristic_inning"

    # 6. 파생 피처 연산 (수학적 감속 및 감쇠율 계산)
    # 선발 투수 기준: 100구당 약 1.5% 구속 저하, 1.0% 회전수 저하 반영 (하한값 0.8)
    velocity_decay_ratio = max(1.0 - (pitch_count_in_game * 0.00015), 0.8)
    spin_decay_ratio     = max(1.0 - (pitch_count_in_game * 0.00010), 0.8)

    vel_drop  = max(1.0 - velocity_decay_ratio, 0)
    spin_drop = max(1.0 - spin_decay_ratio, 0)
    stamina_index = (pitch_count_in_game / 100.0) * (vel_drop * 0.7 + spin_drop * 0.3)
    is_risp = int((on_2b != 0) or (on_3b != 0))
    blocking_leverage_factor = is_risp * catcher_blocking_runs * 0.1
    fielding_risk_index = max(-team_oaa_total * 0.05, 0)

    enrichment_latency_ms = (time.perf_counter() - t_start) * 1000

    enriched = {
        "pitch_count_in_game":   pitch_count_in_game,
        "base_speed":            base_speed,
        "base_spin":             base_spin,
        "p_throws":              p_throws,
        "velocity_decay_ratio":  velocity_decay_ratio,
        "spin_decay_ratio":      spin_decay_ratio,
        "stamina_index":         stamina_index,
        "catcher_blocking_runs":    catcher_blocking_runs,
        "is_risp":                  is_risp,
        "blocking_leverage_factor": blocking_leverage_factor,
        "team_oaa_total":      team_oaa_total,
        "fielding_risk_index": fielding_risk_index,
        **rep,  # 구종 비율 36개 직접 병합
        "enrichment_latency_ms": round(enrichment_latency_ms, 2),
        "enrichment_sources": {
            "base_speed":            baseline_source,
            "pitch_count_in_game":   pitch_count_source,
            "catcher_blocking_runs": blocking_source,
            "team_oaa_total":        oaa_source,
        },
    }

    logger.info(f"Enrichment 완료: pitcher={pitcher_id}, latency={enrichment_latency_ms:.1f}ms")
    return enriched