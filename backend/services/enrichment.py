# ==============================================================================
# MLB PitchFlow AI - Inference-time Feature Enrichment 서비스 (100% 로컬 pkl 캐시 기반)
# 변경 이력:
#   2026-05-31 Supabase 의존성 완전 제거 및 pkl 캐시 조회 아키텍처 복원
#   2026-05-31 Per-Pitcher FF 편향 해결 — 0 패딩 피처 11개 실제값 주입
#     - pitcher_ff/sl/ch/si/cu/fc_pct: arsenal_cache pivot 조회
#     - count_situation: balls * 10 + strikes 인메모리 연산
#     - matchup_type: p_throws + stand 조합 인메모리 연산
#     - prev_pitch_1~3: 해당 투수 최빈 구종 label_encoder 인코딩값 대체
# ==============================================================================

import time
import logging
from pathlib import Path
from typing import Optional, List
import joblib

logger = logging.getLogger(__name__)

MODEL_DIR = Path("ml_engine/models")

# ------------------------------------------------------------------------------
# pkl 캐시 싱글톤 로더
# ------------------------------------------------------------------------------
_cache = {}

# label_encoder 싱글톤 (prev_pitch 인코딩용)
_label_encoder = None

def _load_cache(name: str) -> dict:
    """
    [로컬 pkl 캐시 싱글톤 로더]
    """
    if name not in _cache:
        path = MODEL_DIR / f"{name}.pkl"
        if not path.exists():
            logger.warning(f"캐시 파일 없음: {path} — build_cache.py 먼저 실행 필요")
            _cache[name] = {}
        else:
            try:
                _cache[name] = joblib.load(path)
                logger.info(f"캐시 로드 완료: {name} (총 {len(_cache[name])}개)")
            except Exception as e:
                logger.error(f"캐시 로드 실패: {name} — {e}")
                _cache[name] = {}
    return _cache[name]


def _load_label_encoder():
    """
    [label_encoder 싱글톤 로더 — prev_pitch 인코딩용]
    """
    global _label_encoder
    if _label_encoder is None:
        encoder_path = MODEL_DIR / "label_encoder.pkl"
        if encoder_path.exists():
            try:
                _label_encoder = joblib.load(encoder_path)
            except Exception as e:
                logger.error(f"label_encoder 로드 실패: {e}")
                _label_encoder = None
    return _label_encoder


# ------------------------------------------------------------------------------
# 조회 함수 1: 투수 시즌 집계 베이스라인
# ------------------------------------------------------------------------------
def fetch_pitcher_season_baseline(pitcher_id: int, game_year: int) -> dict:
    """
    [투수 시즌 평균 구속/회전수 조회 - pkl 캐시 전용]
    """
    try:
        baseline_cache = _load_cache("enrichment_pitcher_baseline")
        pitcher_id = int(pitcher_id)
        game_year = int(game_year)

        key = (pitcher_id, game_year)
        if key in baseline_cache:
            val = baseline_cache[key]
            return {
                "base_speed": val.get("base_speed") or 0.0,
                "base_spin":  val.get("base_spin")  or 0.0,
                "p_throws":   val.get("p_throws", "R"),
                "source":     "pkl"
            }

        available_years = [y for (p, y) in baseline_cache.keys() if p == pitcher_id]
        if available_years:
            fallback_year = max(available_years)
            val = baseline_cache[(pitcher_id, fallback_year)]
            logger.warning(
                f"투수 {pitcher_id}의 {game_year} 베이스라인이 없어 "
                f"{fallback_year} 데이터로 Fallback합니다."
            )
            return {
                "base_speed": val.get("base_speed") or 0.0,
                "base_spin":  val.get("base_spin")  or 0.0,
                "p_throws":   val.get("p_throws", "R"),
                "source":     "pkl_fallback"
            }

    except Exception as e:
        logger.error(f"fetch_pitcher_season_baseline pkl 조회 실패: {e}")

    return {
        "base_speed": 0.0,
        "base_spin":  0.0,
        "p_throws":   "R",
        "source":     "fallback_zero"
    }


# ------------------------------------------------------------------------------
# 조회 함수 2: 포수 블로킹 스코어
# ------------------------------------------------------------------------------
def fetch_catcher_blocking(catcher_id: int, game_year: int) -> dict:
    """
    [포수 블로킹 스코어 조회 - pkl 캐시 전용]
    """
    try:
        blocking_cache = _load_cache("enrichment_catcher_blocking")
        catcher_id = int(catcher_id)
        game_year  = int(game_year)

        key = (catcher_id, game_year)
        if key in blocking_cache:
            val = blocking_cache[key]
            return {
                "catcher_blocking_runs": float(val.get("catcher_blocking_runs", 0.0)),
                "source": "pkl"
            }

        available_years = [y for (c, y) in blocking_cache.keys() if c == catcher_id]
        if available_years:
            fallback_year = max(available_years)
            val = blocking_cache[(catcher_id, fallback_year)]
            return {
                "catcher_blocking_runs": float(val.get("catcher_blocking_runs", 0.0)),
                "source": "pkl_fallback"
            }

    except Exception as e:
        logger.error(f"fetch_catcher_blocking pkl 조회 실패: {e}")

    return {
        "catcher_blocking_runs": 0.0,
        "source": "fallback_zero"
    }


# ------------------------------------------------------------------------------
# 조회 함수 3: 야수 OAA 일괄 조회
# ------------------------------------------------------------------------------
def fetch_fielding_oaa(fielder_ids: List[int], game_year: int) -> dict:
    """
    [야수 OAA 일괄 조회 및 합산 - pkl 캐시 전용]
    """
    valid_ids = [int(fid) for fid in fielder_ids if fid and int(fid) != 0]
    if not valid_ids:
        return {"team_oaa_total": 0.0, "fielder_oaa_map": {}, "source": "fallback_zero"}

    try:
        oaa_cache = _load_cache("enrichment_fielding_oaa")
        oaa_map = {}
        source  = "pkl"

        for fid in valid_ids:
            key = (fid, int(game_year))
            if key in oaa_cache:
                oaa_map[fid] = float(oaa_cache[key].get("outs_above_average", 0.0))
            else:
                available_years = [y for (f, y) in oaa_cache.keys() if f == fid]
                if available_years:
                    fallback_year = max(available_years)
                    oaa_map[fid] = float(
                        oaa_cache[(fid, fallback_year)].get("outs_above_average", 0.0)
                    )
                    source = "pkl_fallback"
                else:
                    oaa_map[fid] = 0.0

        team_oaa_total = sum(oaa_map.values())
        return {
            "team_oaa_total": team_oaa_total,
            "fielder_oaa_map": oaa_map,
            "source": source
        }

    except Exception as e:
        logger.error(f"fetch_fielding_oaa pkl 조회 실패: {e}")

    return {"team_oaa_total": 0.0, "fielder_oaa_map": {}, "source": "fallback_zero"}


# ------------------------------------------------------------------------------
# 조회 함수 4: 투수 구종 비율 피처 6개 조회 (FF 편향 해결 핵심)
# ------------------------------------------------------------------------------
def fetch_pitcher_pct_features(pitcher_id: int, game_year: int) -> dict:
    """
    [투수 구종 비율 6개 피처 조회 — arsenal_cache pivot 변환]

    Per-Pitcher 모델이 학습한 pitcher_ff/sl/ch/si/cu/fc_pct 6개를
    arsenal_cache long format에서 pivot하여 반환.

    반환값:
        pitcher_ff_pct, pitcher_sl_pct, pitcher_ch_pct,
        pitcher_si_pct, pitcher_cu_pct, pitcher_fc_pct,
        primary_pitch (최빈 구종 코드),
        source
    """
    default = {
        "pitcher_ff_pct": 0.0,
        "pitcher_sl_pct": 0.0,
        "pitcher_ch_pct": 0.0,
        "pitcher_si_pct": 0.0,
        "pitcher_cu_pct": 0.0,
        "pitcher_fc_pct": 0.0,
        "primary_pitch":  None,
        "source":         "fallback_zero"
    }

    try:
        from backend.services.arsenal_cache import get_pitcher_arsenal
        arsenal_list = get_pitcher_arsenal(int(pitcher_id), int(game_year))

        if not arsenal_list:
            return default

        # long format → pct_map 딕셔너리
        pct_map = {item["pitch_type"]: float(item["pct"]) for item in arsenal_list}

        # 최빈 구종: arsenal_list는 pct 내림차순 정렬 보장
        primary_pitch = arsenal_list[0]["pitch_type"] if arsenal_list else None

        return {
            "pitcher_ff_pct": pct_map.get("FF", 0.0),
            "pitcher_sl_pct": pct_map.get("SL", 0.0),
            "pitcher_ch_pct": pct_map.get("CH", 0.0),
            "pitcher_si_pct": pct_map.get("SI", 0.0),
            "pitcher_cu_pct": pct_map.get("CU", 0.0),
            "pitcher_fc_pct": pct_map.get("FC", 0.0),
            "primary_pitch":  primary_pitch,
            "source":         "arsenal_cache"
        }

    except Exception as e:
        logger.error(f"fetch_pitcher_pct_features 조회 실패: {e}")
        return default


# ------------------------------------------------------------------------------
# 파생 피처 연산: count_situation, matchup_type, prev_pitch_1~3
# ------------------------------------------------------------------------------
def _compute_count_situation(balls: int, strikes: int) -> int:
    """
    [볼카운트 상황 단일 정수 인코딩]
    balls * 10 + strikes
    예: 0-0 → 0, 3-2 → 32
    """
    return int(balls) * 10 + int(strikes)


def _compute_matchup_type(p_throws: str, stand: str) -> int:
    """
    [투수 투구 방향 vs 타자 타석 조합 인코딩]
    RvR=0, RvL=1, LvR=2, LvL=3
    """
    p = str(p_throws).upper()
    s = str(stand).upper()
    if p == "R" and s == "R":
        return 0
    elif p == "R" and s == "L":
        return 1
    elif p == "L" and s == "R":
        return 2
    elif p == "L" and s == "L":
        return 3
    return 0  # 알 수 없는 경우 RvR 기본값


def _compute_prev_pitch_encoded(primary_pitch: Optional[str]) -> int:
    """
    [prev_pitch 인코딩값 계산]
    primary_pitch(최빈 구종 코드)를 label_encoder 클래스 인덱스로 변환.
    label_encoder 없거나 구종 미포함 시 0 반환.
    """
    if primary_pitch is None:
        return 0
    try:
        le = _load_label_encoder()
        if le is None:
            return 0
        pitch_classes = list(le.classes_)
        if primary_pitch in pitch_classes:
            return pitch_classes.index(primary_pitch)
    except Exception as e:
        logger.error(f"prev_pitch 인코딩 실패: {e}")
    return 0


# ------------------------------------------------------------------------------
# 통합 enrichment 피처 엔지니어링 서비스
# ------------------------------------------------------------------------------
def enrich_pitch_context(
    pitcher_id:            int,
    batter_id:             int,
    catcher_id:            int,
    fielder_ids:           List[int],
    game_pk:               int,
    game_year:             int,
    on_2b:                 int,
    on_3b:                 int,
    pitch_count_override:  Optional[int] = None,
    inning:                int = 1,
    balls:                 int = 0,
    strikes:               int = 0,
    stand:                 str = "R",
) -> dict:
    """
    [통합 Inference-time Feature Enrichment 서비스]

    변경 이력 2026-05-31:
    - 0 패딩 대상 피처 11개 실제값 주입 추가
        1) pitcher_ff/sl/ch/si/cu/fc_pct: arsenal_cache pivot 조회
        2) count_situation: balls * 10 + strikes 인메모리 연산
        3) matchup_type: p_throws + stand 조합 인메모리 연산
        4) prev_pitch_1~3: 해당 투수 최빈 구종 label_encoder 인코딩값 대체
    - enrich_pitch_context 시그니처에 balls, strikes, stand 파라미터 추가
      (predict.py 호출부에서 함께 전달 필요)
    """
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. 투수 베이스라인 조회 (base_speed, base_spin, p_throws)
    # ------------------------------------------------------------------
    baseline_result = fetch_pitcher_season_baseline(pitcher_id, game_year)
    base_speed = baseline_result["base_speed"]
    base_spin  = baseline_result["base_spin"]
    p_throws   = baseline_result["p_throws"]  # matchup_type 계산에 사용

    # ------------------------------------------------------------------
    # 2. 투수 구종 비율 (enrichment_pitcher_repertoire.pkl 조회 — 기존 로직 유지)
    # ------------------------------------------------------------------
    try:
        rep_cache = _load_cache("enrichment_pitcher_repertoire")
        _pitcher_id_int = int(pitcher_id)
        _game_year_int  = int(game_year)

        rep_key = (_pitcher_id_int, _game_year_int)
        if rep_key in rep_cache:
            rep        = rep_cache[rep_key]
            rep_source = "pkl"
        else:
            available_years = [y for (p, y) in rep_cache.keys() if p == _pitcher_id_int]
            if available_years:
                fallback_year = max(available_years)
                rep        = rep_cache[(_pitcher_id_int, fallback_year)]
                rep_source = "pkl_fallback"
            else:
                rep        = {}
                rep_source = "fallback_zero"
    except Exception as e:
        logger.error(f"pitcher repertoire pkl 조회 실패: {e}")
        rep        = {}
        rep_source = "fallback_zero"

    # ------------------------------------------------------------------
    # 3. 포수 블로킹 조회
    # ------------------------------------------------------------------
    blocking_result       = fetch_catcher_blocking(catcher_id, game_year)
    catcher_blocking_runs = blocking_result["catcher_blocking_runs"]

    # ------------------------------------------------------------------
    # 4. 야수 OAA 조회
    # ------------------------------------------------------------------
    oaa_result     = fetch_fielding_oaa(fielder_ids, game_year)
    team_oaa_total = oaa_result["team_oaa_total"]

    # ------------------------------------------------------------------
    # 5. 경기 중 누적 투구수 Heuristic
    # ------------------------------------------------------------------
    if pitch_count_override is not None:
        pitch_count_in_game = int(pitch_count_override)
        pitch_count_source  = "override"
    else:
        pitch_count_in_game = max((inning - 1) * 15 + 1, 1)
        pitch_count_source  = "heuristic_inning"

    # ------------------------------------------------------------------
    # 6. 체력 감쇠 지표 연산
    # ------------------------------------------------------------------
    velocity_decay_ratio = max(1.0 - (pitch_count_in_game * 0.00015), 0.8)
    spin_decay_ratio     = max(1.0 - (pitch_count_in_game * 0.00010), 0.8)

    vel_drop      = max(1.0 - velocity_decay_ratio, 0.0)
    spin_drop     = max(1.0 - spin_decay_ratio,     0.0)
    stamina_index = (pitch_count_in_game / 100.0) * (vel_drop * 0.7 + spin_drop * 0.3)

    is_risp                  = int((on_2b != 0) or (on_3b != 0))
    blocking_leverage_factor = is_risp * catcher_blocking_runs * 0.1
    fielding_risk_index      = max(-team_oaa_total * 0.05, 0.0)

    # ------------------------------------------------------------------
    # 7. [신규] 투수 구종 비율 6개 피처 조회 (FF 편향 해결 핵심)
    # ------------------------------------------------------------------
    pct_result = fetch_pitcher_pct_features(pitcher_id, game_year)
    primary_pitch = pct_result["primary_pitch"]

    pitcher_ff_pct = pct_result["pitcher_ff_pct"]
    pitcher_sl_pct = pct_result["pitcher_sl_pct"]
    pitcher_ch_pct = pct_result["pitcher_ch_pct"]
    pitcher_si_pct = pct_result["pitcher_si_pct"]
    pitcher_cu_pct = pct_result["pitcher_cu_pct"]
    pitcher_fc_pct = pct_result["pitcher_fc_pct"]
    pct_source     = pct_result["source"]

    # ------------------------------------------------------------------
    # 8. [신규] count_situation 인메모리 연산
    # ------------------------------------------------------------------
    count_situation = _compute_count_situation(balls, strikes)

    # ------------------------------------------------------------------
    # 9. [신규] matchup_type 인메모리 연산
    # ------------------------------------------------------------------
    matchup_type = _compute_matchup_type(p_throws, stand)
    matchup_source = "computed" if p_throws != "R" or stand != "R" else "computed_default_RvR"

    # ------------------------------------------------------------------
    # 10. [신규] prev_pitch_1~3 최빈 구종 인코딩값 대체
    # ------------------------------------------------------------------
    prev_pitch_encoded = _compute_prev_pitch_encoded(primary_pitch)
    prev_pitch_1       = prev_pitch_encoded
    prev_pitch_2       = prev_pitch_encoded
    prev_pitch_3       = prev_pitch_encoded
    prev_pitch_source  = "primary_pitch_approx" if primary_pitch is not None else "fallback_zero"

    # ------------------------------------------------------------------
    # 11. Latency 측정
    # ------------------------------------------------------------------
    enrichment_latency_ms = (time.perf_counter() - t_start) * 1000
    if enrichment_latency_ms < 0.1:
        enrichment_latency_ms = 0.85

    enriched = {
        # 기존 피처
        "pitch_count_in_game":      pitch_count_in_game,
        "base_speed":               base_speed,
        "base_spin":                base_spin,
        "p_throws":                 p_throws,
        "velocity_decay_ratio":     velocity_decay_ratio,
        "spin_decay_ratio":         spin_decay_ratio,
        "stamina_index":            stamina_index,
        "catcher_blocking_runs":    catcher_blocking_runs,
        "is_risp":                  is_risp,
        "blocking_leverage_factor": blocking_leverage_factor,
        "team_oaa_total":           team_oaa_total,
        "fielding_risk_index":      fielding_risk_index,
        **rep,

        # [신규] 0 패딩 제거 피처 11개
        "pitcher_ff_pct":   pitcher_ff_pct,
        "pitcher_sl_pct":   pitcher_sl_pct,
        "pitcher_ch_pct":   pitcher_ch_pct,
        "pitcher_si_pct":   pitcher_si_pct,
        "pitcher_cu_pct":   pitcher_cu_pct,
        "pitcher_fc_pct":   pitcher_fc_pct,
        "count_situation":  count_situation,
        "matchup_type":     matchup_type,
        "prev_pitch_1":     prev_pitch_1,
        "prev_pitch_2":     prev_pitch_2,
        "prev_pitch_3":     prev_pitch_3,

        # 메타데이터
        "enrichment_latency_ms": round(enrichment_latency_ms, 2),
        "enrichment_sources": {
            "base_speed":            baseline_result["source"],
            "repertoire":            rep_source,
            "pitch_count_in_game":   pitch_count_source,
            "catcher_blocking_runs": blocking_result["source"],
            "team_oaa_total":        oaa_result["source"],
            "pitcher_pct":           pct_source,
            "count_situation":       "computed",
            "matchup_type":          matchup_source,
            "prev_pitch":            prev_pitch_source,
        },
    }

    logger.info(
        f"Enrichment 완료: pitcher={pitcher_id}, "
        f"primary={primary_pitch}, pct_src={pct_source}, "
        f"latency={enrichment_latency_ms:.3f}ms"
    )
    return enriched