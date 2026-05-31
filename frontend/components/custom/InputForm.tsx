'use client';

import React, { useState } from 'react';
import PlayerSelect from '../shared/PlayerSelect';
import { PredictRequest, PitchType } from '../../lib/types';

interface InputFormProps {
  onSubmit: (data: PredictRequest) => void;
  onActualPitchChange: (pitch: PitchType | null) => void;
  isLoading: boolean;
  onPitcherChange?: (pitcher: { id: number; teamId: string; hand: 'R' | 'L' } | null) => void;
  onBatterChange?: (batter: { id: number; teamId: string; side: 'R' | 'L' } | null) => void;
}

const PITCH_LABELS: Record<string, string> = {
  FF: 'FF — 포심 패스트볼',
  FT: 'FT — 투심 패스트볼',
  SI: 'SI — 싱커',
  SL: 'SL — 슬라이더',
  CH: 'CH — 체인지업',
  CU: 'CU — 커브볼',
  KC: 'KC — 너클 커브',
  CB: 'CB — 커브볼',
  FC: 'FC — 커터',
  FS: 'FS — 스플리터',
  KN: 'KN — 너클볼',
};

export default function InputForm({
  onSubmit,
  onActualPitchChange,
  isLoading,
  onPitcherChange,
  onBatterChange
}: InputFormProps) {
  const [pitcher, setPitcher] = useState<{ id: number; teamId: string; hand: 'R' | 'L' } | null>(null);
  const [batter, setBatter] = useState<{ id: number; teamId: string; side: 'R' | 'L' } | null>(null);
  const [pitcherArsenal, setPitcherArsenal] = useState<PitchType[]>([]);

  React.useEffect(() => {
    onPitcherChange?.(pitcher);
  }, [pitcher, onPitcherChange]);

  React.useEffect(() => {
    onBatterChange?.(batter);
  }, [batter, onBatterChange]);

  React.useEffect(() => {
    if (!pitcher) {
      setPitcherArsenal([]);
    }
  }, [pitcher]);

  const handlePitcherSelect = async (id: number, teamId: string, hand: 'R' | 'L') => {
    setPitcher({ id, teamId, hand });
    setPitcherArsenal([]);
    setActualPitch('none');
    onActualPitchChange(null);
    try {
      const res = await fetch(`/api/pitcher-arsenal?playerId=${id}`);
      if (res.ok) {
        const data = await res.json();
        setPitcherArsenal(data.arsenal || ['FF', 'SL', 'CH']);
      } else {
        setPitcherArsenal(['FF', 'SL', 'CH']);
      }
    } catch (err) {
      console.error('Error loading pitcher arsenal:', err);
      setPitcherArsenal(['FF', 'SL', 'CH']);
    }
  };
  const [balls, setBalls] = useState<number>(0);
  const [strikes, setStrikes] = useState<number>(0);
  const [outs, setOuts] = useState<number>(0);
  const [isExtraInning, setIsExtraInning] = useState<boolean>(false);
  const [regularInning, setRegularInning] = useState<number>(1);
  const [extraInningVal, setExtraInningVal] = useState<number>(10);
  
  const [on1b, setOn1b] = useState<boolean>(false);
  const [on2b, setOn2b] = useState<boolean>(false);
  const [on3b, setOn3b] = useState<boolean>(false);
  
  const [catcher, setCatcher] = useState<{ id: number; teamId: string } | null>(null);
  const [actualPitch, setActualPitch] = useState<PitchType | 'none'>('none');

  const currentInning = isExtraInning ? extraInningVal : regularInning;

  const isValid = pitcher !== null && batter !== null && pitcher.id !== batter.id;

  const handlePredictSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!isValid || isLoading) return;

    const payload: PredictRequest = {
      pitcher: pitcher.id,
      batter: batter.id,
      balls,
      strikes,
      outs_when_up: outs,
      inning: currentInning,
      on_1b: on1b ? 1 : 0,
      on_2b: on2b ? 1 : 0,
      on_3b: on3b ? 1 : 0,
      stand: batter?.side ?? 'R', // 타자 정보에서 자동 연동
      game_pk: 0,
      game_year: 2026,
    };

    if (catcher !== null) {
      payload.fielder_2 = catcher.id;
    }

    onSubmit(payload);
  };

  const handleActualPitchChangeLocal = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value;
    if (val === 'none') {
      setActualPitch('none');
      onActualPitchChange(null);
    } else {
      const pitch = val as PitchType;
      setActualPitch(pitch);
      onActualPitchChange(pitch);
    }
  };

  return (
    <form
      onSubmit={handlePredictSubmit}
      className="flex flex-col h-full bg-gray-900 border-r border-gray-800 p-5 overflow-y-auto space-y-6 select-none scrollbar-thin scrollbar-thumb-gray-800"
    >
      <div className="flex flex-col border-b border-gray-800 pb-3">
        <h2 className="text-xl font-bold tracking-tight text-white">시뮬레이터 설정</h2>
        <span className="text-xs text-gray-400">투구 상황을 지정하여 예측 결과를 확인하세요.</span>
      </div>

      {/* Players Select */}
      <div className="space-y-4">
        <PlayerSelect
          label="투수"
          excludeId={batter?.id}
          onSelect={(id, teamId, teamName, hand) => handlePitcherSelect(id, teamId, hand)}
        />
        <PlayerSelect
          label="타자"
          excludeId={pitcher?.id}
          onSelect={(id, teamId, teamName, hand) => setBatter({ id, teamId, side: hand })}
        />
      </div>

      {/* Balls, Strikes, Outs Counts */}
      <div className="space-y-4">
        {/* Balls */}
        <div>
          <label className="block text-xs font-bold text-gray-400 uppercase tracking-wide mb-2">
            Balls
          </label>
          <div className="grid grid-cols-4 gap-2">
            {[0, 1, 2, 3].map((b) => (
              <button
                type="button"
                key={`ball-${b}`}
                onClick={() => setBalls(b)}
                className={`py-2 rounded-lg text-sm font-semibold transition duration-200 shadow-inner ${
                  balls === b
                    ? 'bg-blue-500 text-white font-bold ring-2 ring-blue-400/30'
                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                }`}
              >
                {b}
              </button>
            ))}
          </div>
        </div>

        {/* Strikes */}
        <div>
          <label className="block text-xs font-bold text-gray-400 uppercase tracking-wide mb-2">
            Strikes
          </label>
          <div className="grid grid-cols-3 gap-2">
            {[0, 1, 2].map((s) => (
              <button
                type="button"
                key={`strike-${s}`}
                onClick={() => setStrikes(s)}
                className={`py-2 rounded-lg text-sm font-semibold transition duration-200 shadow-inner ${
                  strikes === s
                    ? 'bg-red-500 text-white font-bold ring-2 ring-red-400/30'
                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                }`}
              >
                {s}
              </button>
            ))}
          </div>
        </div>

        {/* Outs */}
        <div>
          <label className="block text-xs font-bold text-gray-400 uppercase tracking-wide mb-2">
            Outs
          </label>
          <div className="grid grid-cols-3 gap-2">
            {[0, 1, 2].map((o) => (
              <button
                type="button"
                key={`out-${o}`}
                onClick={() => setOuts(o)}
                className={`py-2 rounded-lg text-sm font-semibold transition duration-200 shadow-inner ${
                  outs === o
                    ? 'bg-yellow-500 text-white font-bold ring-2 ring-yellow-400/30'
                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                }`}
              >
                {o}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Inning Count */}
      <div className="space-y-3">
        <label className="block text-xs font-bold text-gray-400 uppercase tracking-wide">
          이닝
        </label>
        <div className="grid grid-cols-5 gap-1.5">
          {[1, 2, 3, 4, 5, 6, 7, 8, 9].map((inn) => (
            <button
              type="button"
              key={`inn-${inn}`}
              onClick={() => {
                setIsExtraInning(false);
                setRegularInning(inn);
              }}
              className={`py-1.5 rounded text-xs font-bold transition duration-200 ${
                !isExtraInning && regularInning === inn
                  ? 'bg-gray-200 text-gray-900 shadow-md shadow-gray-200/10'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {inn}
            </button>
          ))}
          <button
            type="button"
            onClick={() => {
              setIsExtraInning(true);
            }}
            className={`py-1.5 rounded text-xs font-bold transition duration-200 ${
              isExtraInning
                ? 'bg-purple-600 text-white shadow-md shadow-purple-600/20'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            연장
          </button>
        </div>

        {/* Extra inning range slider */}
        {isExtraInning && (
          <div className="p-3 bg-gray-800/40 rounded-lg border border-gray-700/40 space-y-2 animate-fadeIn">
            <div className="flex items-center justify-between text-xs">
              <span className="text-gray-400">연장 이닝 설정:</span>
              <span className="font-bold text-purple-400">{extraInningVal}회초/말</span>
            </div>
            <input
              type="range"
              min={10}
              max={18}
              value={extraInningVal}
              onChange={(e) => setExtraInningVal(parseInt(e.target.value, 10))}
              className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
            />
            <div className="flex justify-between text-[10px] text-gray-500">
              <span>10회</span>
              <span>14회</span>
              <span>18회</span>
            </div>
          </div>
        )}
      </div>

      {/* Runner Diamond & Catcher */}
      <div className="flex flex-col space-y-4">
        {/* Diamond SVG */}
        <div className="flex flex-col items-center bg-gray-950/40 rounded-xl border border-gray-800 p-4">
          <label className="text-xs font-bold text-gray-400 uppercase tracking-wide mb-3 self-start">
            주자 상황
          </label>
          <div className="relative w-28 h-28 flex items-center justify-center">
            <svg width="100" height="100" viewBox="0 0 100 100" className="overflow-visible">
              {/* 배경 다이아몬드 외곽선 */}
              <polygon
                points="50,5 95,50 50,95 5,50"
                fill="none"
                stroke="#4B5563"
                strokeWidth="1"
              />

              {/* 3루 (좌측) */}
              <rect
                x="5"
                y="44"
                width="12"
                height="12"
                className="transition duration-300 animate-fadeIn"
                fill={on3b ? '#EAB308' : 'transparent'}
                stroke="white"
                strokeWidth="1.5"
                rx="1"
                style={{ cursor: 'pointer' }}
                onClick={() => setOn3b(!on3b)}
              />

              {/* 2루 (상단) */}
              <rect
                x="44"
                y="5"
                width="12"
                height="12"
                className="transition duration-300 animate-fadeIn"
                fill={on2b ? '#EAB308' : 'transparent'}
                stroke="white"
                strokeWidth="1.5"
                rx="1"
                style={{ cursor: 'pointer' }}
                onClick={() => setOn2b(!on2b)}
              />

              {/* 1루 (우측) */}
              <rect
                x="83"
                y="44"
                width="12"
                height="12"
                className="transition duration-300 animate-fadeIn"
                fill={on1b ? '#EAB308' : 'transparent'}
                stroke="white"
                strokeWidth="1.5"
                rx="1"
                style={{ cursor: 'pointer' }}
                onClick={() => setOn1b(!on1b)}
              />

              {/* 홈플레이트 (하단, 클릭 불가) */}
              <polygon
                points="50,88 56,94 50,100 44,94"
                fill="#6B7280"
              />

              {/* 베이스 레이블 */}
              <text x="50" y="23" fill="#9CA3AF" fontSize="8" fontWeight="bold" textAnchor="middle">2B</text>
              <text x="89" y="62" fill="#9CA3AF" fontSize="8" fontWeight="bold" textAnchor="middle">1B</text>
              <text x="11" y="62" fill="#9CA3AF" fontSize="8" fontWeight="bold" textAnchor="middle">3B</text>
            </svg>
          </div>
        </div>

        {/* Catcher PlayerSelect Search */}
        <PlayerSelect
          label="포수 검색 (선택)"
          onSelect={(id, teamId, teamName) => setCatcher({ id, teamId })}
        />
      </div>

      {/* Actual Pitch Dropdown */}
      <div>
        <label className="block text-xs font-bold text-gray-400 uppercase tracking-wide mb-2">
          실제 투구 (결과 검증용)
        </label>
        <select
          value={actualPitch}
          onChange={handleActualPitchChangeLocal}
          disabled={!pitcher || pitcherArsenal.length === 0}
          className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {!pitcher ? (
            <option value="none">투수를 먼저 선택하세요</option>
          ) : (
            <>
              <option value="none">선택 안 함 (미정)</option>
              {pitcherArsenal.map((pitch) => (
                <option key={pitch} value={pitch}>
                  {PITCH_LABELS[pitch] || pitch}
                </option>
              ))}
            </>
          )}
        </select>
      </div>

      {/* Submit Action */}
      <div className="pt-2">
        <button
          type="submit"
          disabled={!isValid || isLoading}
          className={`w-full py-3 px-4 rounded-xl text-sm font-bold flex items-center justify-center space-x-2 transition-all duration-300 ${
            isValid && !isLoading
              ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-600/25 active:scale-[0.98]'
              : 'bg-gray-800 text-gray-500 cursor-not-allowed border border-gray-800'
          }`}
        >
          {isLoading ? (
            <>
              <svg
                className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
              <span>분석하는 중...</span>
            </>
          ) : (
            <span>예측하기</span>
          )}
        </button>
      </div>
    </form>
  );
}
