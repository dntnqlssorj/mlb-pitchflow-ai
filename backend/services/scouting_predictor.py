import os
import joblib
import logging
import json
from pathlib import Path
from openai import OpenAI
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CACHE_PATH = Path("ml_engine/cache/scouting_cache.pkl")
_scouting_cache = None

def _load_cache():
    global _scouting_cache
    if _scouting_cache is None:
        if CACHE_PATH.exists():
            try:
                _scouting_cache = joblib.load(CACHE_PATH)
                logger.info(f"Scouting cache loaded: {len(_scouting_cache)} pitchers.")
            except Exception as e:
                logger.error(f"Failed to load scouting cache: {e}")
                _scouting_cache = {}
        else:
            logger.warning(f"Scouting cache not found at {CACHE_PATH}")
            _scouting_cache = {}
    return _scouting_cache

def predict_with_scouting_llm(pitcher_id: int, context: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    GPT-4o-mini를 활용한 휴리스틱 예측 로직.
    scouting_cache.pkl에 존재하는 투수라면, Base Probability와 상황(Context)을 조합해
    구종 확률(18개 클래스 맵)을 JSON으로 반환합니다.
    """
    cache = _load_cache()
    if pitcher_id not in cache:
        return None
        
    pitcher_data = cache[pitcher_id]
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment. Please add it to .env")
        return None
        
    client = OpenAI(api_key=api_key)
    
    prompt = f"""You are an expert MLB pitching analyst.
Given the following pitcher's scouting grades and base probabilities for each pitch, and the current game context, adjust the pitch probabilities.

Pitcher Name: {pitcher_data['name']}
Base Pitch Probabilities (derived from scouting grades):
{json.dumps(pitcher_data['base_probs'], indent=2)}

Game Context:
Balls: {context.get('balls', 0)}
Strikes: {context.get('strikes', 0)}
Pitch Count in Game: {context.get('pitch_count_in_game', 0)}
Batter Stand: {context.get('stand', 'R')}
Pitcher Throws: {context.get('p_throws', 'R')}
Runners on Base: 2B={context.get('on_2b', 0)}, 3B={context.get('on_3b', 0)}

Output ONLY valid JSON where keys are pitch types and values are float probabilities (0.0 to 1.0).
Required JSON keys (must include all, if a pitch is not in Base Probabilities, output 0.0):
"FF", "SL", "CH", "CU", "SI", "FC", "KC", "FS", "KN", "EP", "CS", "SC", "PO", "FO", "ST", "SV", "FA", "OT"

Ensure the probabilities sum to exactly 1.0. 
CRITICAL RULE: For this pitcher, strictly set pitches with 0.0 base probability to exactly 0.0 (e.g. KN, EP, CS, SC, OT, FA, PO, FO, ST, SV). Do not invent pitches they do not throw.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a JSON-only response bot."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            timeout=8
        )
        content = response.choices[0].message.content
        probs = json.loads(content)
        
        full_classes = ['CH', 'CS', 'CU', 'EP', 'FA', 'FC', 'FF', 'FO', 'FS', 'KC', 'KN', 'OT', 'PO', 'SC', 'SI', 'SL', 'ST', 'SV']
        final_probs = {k: 0.0 for k in full_classes}
        for k, v in probs.items():
            if k in final_probs:
                final_probs[k] = float(v)
                
        # Normalize
        total = sum(final_probs.values())
        if total > 0:
            final_probs = {k: v / total for k, v in final_probs.items()}
            return final_probs
        else:
            return None
            
    except Exception as e:
        logger.error(f"GPT-4o-mini scouting prediction failed: {e}")
        return None
