'use client';

import React from 'react';
import { GameInfo } from '../../lib/types';

interface GameSelectorProps {
  games: GameInfo[];
  selectedGamePk: number | null;
  onSelect: (gamePk: number) => void;
}

export default function GameSelector({
  games,
  selectedGamePk,
  onSelect,
}: GameSelectorProps) {
  const handleSelectChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value;
    if (val) {
      onSelect(Number(val));
    }
  };

  return (
    <div className="w-full flex flex-col space-y-2 select-none">
      <label className="block text-xs font-bold text-gray-400 uppercase tracking-wide">
        경기 선택 (실시간 라이브)
      </label>
      <div className="relative">
        <select
          value={selectedGamePk || ''}
          onChange={handleSelectChange}
          className="w-full bg-gray-900 border border-gray-800 rounded-xl px-4 py-3 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition duration-300 appearance-none shadow-lg cursor-pointer"
        >
          {games.length === 0 ? (
            <option value="">오늘 진행 예정인 경기가 없습니다</option>
          ) : (
            <>
              <option value="" disabled>
                진행 중인 경기를 선택하세요 ({games.length}개 발견)
              </option>
              {games.map((game) => (
                <option key={game.gamePk} value={game.gamePk}>
                  ⚾ {game.awayTeam} vs {game.homeTeam} — {game.venue} ({game.status})
                </option>
              ))}
            </>
          )}
        </select>
        <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-4 text-gray-400">
          <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
            <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/>
          </svg>
        </div>
      </div>
    </div>
  );
}
