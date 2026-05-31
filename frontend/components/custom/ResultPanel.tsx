'use client';

import React, { useState } from 'react';
import { PredictResponse, PitchType } from '../../lib/types';

interface ResultPanelProps {
  result: PredictResponse | null;
  actualPitch: PitchType | null;
}

const PITCH_BG_COLORS: Record<string, string> = {
  FF: 'bg-red-500', FT: 'bg-red-500', SI: 'bg-red-500',
  SL: 'bg-yellow-500', FC: 'bg-yellow-500',
  CH: 'bg-green-500', FS: 'bg-green-500',
  CU: 'bg-blue-500', KC: 'bg-blue-500', CB: 'bg-blue-500', KN: 'bg-blue-500'
};

export default function ResultPanel({ result, actualPitch }: ResultPanelProps) {
  const [showReasoning, setShowReasoning] = useState<boolean>(false);

  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center h-full bg-gray-900 border-l border-gray-800 p-6 text-center select-none">
        <div className="w-16 h-16 rounded-full bg-gray-800/40 border border-gray-700/50 flex items-center justify-center mb-4">
          <svg
            className="w-8 h-8 text-gray-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="1.5"
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
        </div>
        <p className="text-gray-400 font-medium text-sm leading-relaxed max-w-xs">
          투수와 타자를 선택하고<br />
          <span className="text-blue-400 font-semibold">예측하기</span>를 눌러주세요.
        </p>
      </div>
    );
  }

  const { predicted_pitch, pitch_probabilities, commentary, reasoning } = result;

  // Filter out 0% probabilities and sort in descending order
  const sortedProbabilities = Object.entries(pitch_probabilities)
    .map(([pitch, val]) => ({ pitch: pitch as PitchType, prob: val }))
    .filter((p) => p.prob > 0)
    .sort((a, b) => b.prob - a.prob);

  const topProb = pitch_probabilities[predicted_pitch] ?? 0;
  const topProbPercent = Math.round(topProb * 100);

  // Compare actual pitch against prediction
  const isMatch = actualPitch && predicted_pitch === actualPitch;

  return (
    <div className="flex flex-col h-full bg-gray-900 border-l border-gray-800 p-5 overflow-y-auto space-y-6 select-none scrollbar-thin scrollbar-thumb-gray-800">
      <div className="flex flex-col border-b border-gray-800 pb-3">
        <h2 className="text-xl font-bold tracking-tight text-white">예측 결과 분석</h2>
        <span className="text-xs text-gray-400">XGBoost 모델의 예측과 AI 분석 리포트입니다.</span>
      </div>

      {/* 1. Comparison Panel (Only if actualPitch is provided) */}
      {actualPitch && (
        <div className="bg-gray-800/40 border border-gray-700/50 rounded-xl p-4 space-y-3 shadow-inner">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-gray-400 uppercase tracking-wide">
              결과 검증 (VS 실제)
            </span>
            {isMatch ? (
              <span className="px-2.5 py-1 rounded-full text-[10px] font-bold text-white bg-green-600 shadow shadow-green-500/20 uppercase tracking-wider">
                예측 성공
              </span>
            ) : (
              <span className="px-2.5 py-1 rounded-full text-[10px] font-bold text-white bg-red-600 shadow shadow-red-500/20 uppercase tracking-wider">
                예측 불일치
              </span>
            )}
          </div>
          <div className="flex justify-around items-center py-2 bg-gray-950/40 rounded-lg border border-gray-800">
            <div className="text-center">
              <div className="text-[10px] text-gray-500 mb-0.5">모델 예측</div>
              <div className={`text-lg font-black ${PITCH_BG_COLORS[predicted_pitch] || 'text-white'} bg-clip-text`}>
                {predicted_pitch}
              </div>
            </div>
            <div className="h-6 w-px bg-gray-800" />
            <div className="text-center">
              <div className="text-[10px] text-gray-500 mb-0.5">실제 투구</div>
              <div className={`text-lg font-black ${PITCH_BG_COLORS[actualPitch] || 'text-white'} bg-clip-text`}>
                {actualPitch}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 2. Top Pick Card */}
      <div className="bg-gradient-to-br from-gray-800/80 to-gray-900/80 border border-gray-700 rounded-xl p-4 shadow-lg flex items-center justify-between">
        <div>
          <span className="text-[10px] font-bold text-blue-400 uppercase tracking-wider block mb-1">
            제 1 예측 구종
          </span>
          <h3 className="text-2xl font-black text-white leading-none">
            {predicted_pitch}{' '}
            <span className="text-sm font-semibold text-gray-400">({topProbPercent}%)</span>
          </h3>
        </div>
        <div className={`w-12 h-12 rounded-xl flex items-center justify-center font-bold text-lg text-white shadow-lg ${PITCH_BG_COLORS[predicted_pitch] || 'bg-blue-600'}`}>
          {predicted_pitch}
        </div>
      </div>

      {/* 3. Probabilities Bar Chart */}
      <div className="space-y-3">
        <span className="block text-xs font-bold text-gray-400 uppercase tracking-wide">
          구종별 예측 확률
        </span>
        <div className="space-y-2.5 bg-gray-850 p-4 rounded-xl border border-gray-800/80">
          {sortedProbabilities.map(({ pitch, prob }) => {
            const pct = Math.round(prob * 100);
            const colorClass = PITCH_BG_COLORS[pitch] || 'bg-blue-500';
            return (
              <div key={pitch} className="flex items-center space-x-3 text-xs">
                <span className="w-8 font-black text-gray-300 text-left">{pitch}</span>
                <div className="flex-1 bg-gray-850 h-2 rounded-full overflow-hidden border border-gray-800">
                  <div
                    className={`h-full ${colorClass} rounded-full transition-all duration-700 ease-out`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className="w-10 text-right font-semibold text-gray-400">{pct}%</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* 4. AI Analysis Report */}
      <div className="space-y-3">
        <span className="block text-xs font-bold text-gray-400 uppercase tracking-wide">
          AI 분석 리포트 (Claude 3.5)
        </span>
        
        {/* Commentary */}
        <div className="bg-gray-800/50 border border-gray-700/30 rounded-xl p-4 text-sm text-gray-300 leading-relaxed shadow-sm font-medium">
          {commentary}
        </div>

        {/* Reasoning Toggle */}
        {reasoning && (
          <div className="space-y-2">
            <button
              type="button"
              onClick={() => setShowReasoning(!showReasoning)}
              className="text-xs font-bold text-blue-400 hover:text-blue-300 flex items-center space-x-1 outline-none select-none transition"
            >
              <span>{showReasoning ? '근거 접기 ▲' : '근거 보기 ▼'}</span>
            </button>
            
            {showReasoning && (
              <div className="bg-gray-950/60 border border-gray-900 rounded-xl p-4 text-xs text-gray-400 leading-relaxed space-y-2 animate-fadeIn max-h-48 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-900">
                <span className="block font-bold text-gray-300 border-b border-gray-900 pb-1">
                  데이터 기반 예측 근거
                </span>
                <p className="whitespace-pre-line">{reasoning}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
