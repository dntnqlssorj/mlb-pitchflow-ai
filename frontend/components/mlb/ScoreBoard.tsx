'use client';

import React from 'react';
import { LiveSituation } from '../../lib/types';

interface ScoreBoardProps {
  situation: LiveSituation | null;
}

export default function ScoreBoard({ situation }: ScoreBoardProps) {
  if (!situation) {
    return (
      <div className="w-full h-full min-h-[300px] flex items-center justify-center bg-gray-900 border border-gray-800 rounded-2xl p-6 text-center select-none">
        <div className="space-y-2">
          <div className="text-4xl">📡</div>
          <p className="text-sm font-semibold text-gray-400">
            진행 중인 경기를 상단 드롭다운에서 선택해 주세요.
          </p>
          <p className="text-xs text-gray-500">
            실시간 이닝, 볼카운트, 3D 투구 궤적이 즉각 연동됩니다.
          </p>
        </div>
      </div>
    );
  }

  const {
    inning,
    isTopInning,
    balls,
    strikes,
    outs,
    awayScore,
    homeScore,
    currentPitcher,
    currentBatter,
    on1b,
    on2b,
    on3b,
    pitcherTeamId,
    batterTeamId,
    lastPitch,
  } = situation;

  // Format Inning notation in Korean for enhanced premium local feel
  const inningText = `${inning}회${isTopInning ? '초' : '말'}`;

  return (
    <div className="w-full bg-gray-900 border border-gray-800 rounded-2xl p-5 flex flex-col space-y-6 select-none shadow-xl text-white">
      {/* 1. Inning & Score Board Header */}
      <div className="flex flex-col space-y-3 bg-gray-950/60 rounded-xl p-4 border border-gray-800/40">
        <div className="flex items-center justify-between">
          <span className="text-xs font-extrabold tracking-wider bg-blue-600 text-white px-2 py-0.5 rounded uppercase">
            LIVE
          </span>
          <span className="text-xs font-bold text-gray-400 tracking-wide">
            {inningText}
          </span>
        </div>
        
        {/* Teams and Scores */}
        <div className="grid grid-cols-5 items-center gap-2">
          {/* Away Team */}
          <div className="col-span-2 text-center">
            <span className="text-[10px] font-bold text-gray-500 uppercase block">AWAY</span>
            <span className="text-lg font-bold block truncate" title={pitcherTeamId}>{situation.pitcherTeamId && isTopInning ? batterTeamId : pitcherTeamId}</span>
            <span className="text-3xl font-extrabold text-white mt-1 block">{awayScore}</span>
          </div>

          {/* VS Colon */}
          <div className="col-span-1 text-center text-gray-600 font-bold text-xl">:</div>

          {/* Home Team */}
          <div className="col-span-2 text-center">
            <span className="text-[10px] font-bold text-gray-500 uppercase block">HOME</span>
            <span className="text-lg font-bold block truncate" title={batterTeamId}>{situation.batterTeamId && isTopInning ? pitcherTeamId : batterTeamId}</span>
            <span className="text-3xl font-extrabold text-white mt-1 block">{homeScore}</span>
          </div>
        </div>
      </div>

      {/* 2. Count & Runner Grid */}
      <div className="grid grid-cols-2 gap-4 items-center">
        {/* Ball Counts (B-S-O circles) */}
        <div className="space-y-3 bg-gray-950/40 border border-gray-800/60 rounded-xl p-3.5 flex flex-col justify-center">
          {/* Balls */}
          <div className="flex items-center justify-between">
            <span className="text-xs font-extrabold text-blue-400 tracking-wider w-6">B</span>
            <div className="flex space-x-1.5">
              {[1, 2, 3].map((idx) => (
                <span
                  key={`ball-dot-${idx}`}
                  className={`w-3.5 h-3.5 rounded-full transition duration-300 ${
                    balls >= idx ? 'bg-blue-500 shadow-md shadow-blue-500/50' : 'bg-gray-800'
                  }`}
                />
              ))}
            </div>
          </div>

          {/* Strikes */}
          <div className="flex items-center justify-between">
            <span className="text-xs font-extrabold text-red-400 tracking-wider w-6">S</span>
            <div className="flex space-x-1.5">
              {[1, 2].map((idx) => (
                <span
                  key={`strike-dot-${idx}`}
                  className={`w-3.5 h-3.5 rounded-full transition duration-300 ${
                    strikes >= idx ? 'bg-red-500 shadow-md shadow-red-500/50' : 'bg-gray-800'
                  }`}
                />
              ))}
            </div>
          </div>

          {/* Outs */}
          <div className="flex items-center justify-between">
            <span className="text-xs font-extrabold text-yellow-400 tracking-wider w-6">O</span>
            <div className="flex space-x-1.5">
              {[1, 2].map((idx) => (
                <span
                  key={`out-dot-${idx}`}
                  className={`w-3.5 h-3.5 rounded-full transition duration-300 ${
                    outs >= idx ? 'bg-yellow-500 shadow-md shadow-yellow-500/50' : 'bg-gray-800'
                  }`}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Runner Diamond SVG */}
        <div className="flex flex-col items-center bg-gray-950/40 border border-gray-800/60 rounded-xl p-3">
          <div className="relative w-20 h-20 flex items-center justify-center">
            <svg width="70" height="70" viewBox="0 0 100 100" className="overflow-visible" style={{ pointerEvents: 'none' }}>
              <polygon
                points="50,5 95,50 50,95 5,50"
                fill="none"
                stroke="#374151"
                strokeWidth="1.5"
              />
              {/* 3B */}
              <rect
                x="5"
                y="44"
                width="12"
                height="12"
                fill={on3b ? '#EAB308' : 'transparent'}
                stroke="white"
                strokeWidth="1.5"
                rx="1"
              />
              {/* 2B */}
              <rect
                x="44"
                y="5"
                width="12"
                height="12"
                fill={on2b ? '#EAB308' : 'transparent'}
                stroke="white"
                strokeWidth="1.5"
                rx="1"
              />
              {/* 1B */}
              <rect
                x="83"
                y="44"
                width="12"
                height="12"
                fill={on1b ? '#EAB308' : 'transparent'}
                stroke="white"
                strokeWidth="1.5"
                rx="1"
              />
              {/* Home Plate */}
              <polygon points="50,88 56,94 50,100 44,94" fill="#4B5563" />
            </svg>
          </div>
        </div>
      </div>

      {/* 3. Pitcher & Batter Info Card */}
      <div className="flex flex-col space-y-3 bg-gray-950/40 border border-gray-800/50 rounded-xl p-4">
        {/* Pitcher */}
        <div className="flex justify-between items-center text-xs">
          <div className="flex flex-col">
            <span className="text-[10px] font-extrabold text-gray-500 uppercase">Pitcher ({situation.pitcherHand}HP)</span>
            <span className="font-bold text-gray-200 truncate max-w-[150px]">{currentPitcher.name}</span>
          </div>
          <span className="text-xs font-bold text-gray-400 bg-gray-900 border border-gray-800 px-2 py-1 rounded">
            {pitcherTeamId}
          </span>
        </div>

        <div className="border-t border-gray-800/50 my-1" />

        {/* Batter */}
        <div className="flex justify-between items-center text-xs">
          <div className="flex flex-col">
            <span className="text-[10px] font-extrabold text-gray-500 uppercase">Batter ({situation.batterSide}B)</span>
            <span className="font-bold text-gray-200 truncate max-w-[150px]">{currentBatter.name}</span>
          </div>
          <span className="text-xs font-bold text-gray-400 bg-gray-900 border border-gray-800 px-2 py-1 rounded">
            {batterTeamId}
          </span>
        </div>
      </div>

      {/* 4. Last Pitch Information (if any) */}
      {lastPitch && (
        <div className="bg-blue-950/20 border border-blue-900/30 rounded-xl p-3.5 text-xs text-blue-300 animate-fadeIn">
          <span className="font-bold block text-blue-400 mb-1">직전 투구 정보</span>
          <p>
            구종 코드: <strong className="text-white font-extrabold">{lastPitch.type}</strong> — {lastPitch.description}
          </p>
        </div>
      )}
    </div>
  );
}
