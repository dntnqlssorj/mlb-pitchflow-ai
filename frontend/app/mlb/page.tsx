'use client';

import React, { useState, useEffect, useRef } from 'react';
import GameSelector from '../../components/mlb/GameSelector';
import ScoreBoard from '../../components/mlb/ScoreBoard';
import LiveResultPanel from '../../components/mlb/LiveResultPanel';
import LiveSceneViewer from '../../components/mlb/LiveSceneViewer';
import { GameInfo, LiveSituation, PredictResponse, PredictRequest, PitchType } from '../../lib/types';

export default function MLBLivePage() {
  const [games, setGames] = useState<GameInfo[]>([]);
  const [selectedGamePk, setSelectedGamePk] = useState<number | null>(null);
  
  const [currentSituation, setCurrentSituation] = useState<LiveSituation | null>(null);
  const [predictResult, setPredictResult] = useState<PredictResponse | null>(null);
  
  const [isGamesLoading, setIsGamesLoading] = useState<boolean>(true);
  const [isLiveLoading, setIsLiveLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Store previous situation in ref to avoid redundant predict API calls during polling
  const prevSituationRef = useRef<LiveSituation | null>(null);

  // 1. Fetch scheduled today games on mount
  useEffect(() => {
    const fetchGames = async () => {
      try {
        setIsGamesLoading(true);
        const res = await fetch('/api/mlb/today-games');
        if (!res.ok) throw new Error('오늘 경기 일정을 가져오는 데 실패했습니다.');
        const data = await res.json();
        setGames(data);
      } catch (err: any) {
        console.error('Error fetching today games:', err);
        setErrorMsg(err.message || '경기 일정 수신 중 네트워크 오류가 발생했습니다.');
      } finally {
        setIsGamesLoading(false);
      }
    };
    fetchGames();
  }, []);

  // 2. Fetch live feed & orchestrate AI prediction
  const fetchLiveFeed = async (gamePk: number, isInitial = false) => {
    if (isInitial) setIsLiveLoading(true);
    try {
      // 1) Fetch Live Situation
      const res = await fetch(`/api/mlb/live-feed?gamePk=${gamePk}`);
      if (!res.ok) throw new Error('실시간 경기 상황 정보를 수신하지 못했습니다.');
      
      const situation: LiveSituation | null = await res.json();
      setCurrentSituation(situation);

      if (!situation) {
        setPredictResult(null);
        prevSituationRef.current = null;
        return;
      }

      // Update situation cache
      prevSituationRef.current = situation;

      // 2) Fetch Latest AI Prediction pushed from n8n
      const predictRes = await fetch(`/api/mlb/prediction?gamePk=${gamePk}`);
      if (predictRes.ok) {
        const predictData: PredictResponse | null = await predictRes.json();
        setPredictResult(predictData);
      }
    } catch (err: any) {
      console.error('Error fetching live-feed or prediction:', err);
    } finally {
      if (isInitial) setIsLiveLoading(false);
    }
  };

  // 3. Polling interval (Every 30 seconds) on selected gamePk change
  useEffect(() => {
    if (!selectedGamePk) {
      setCurrentSituation(null);
      setPredictResult(null);
      prevSituationRef.current = null;
      return;
    }

    // Fire immediately on selection
    fetchLiveFeed(selectedGamePk, true);

    const intervalId = setInterval(() => {
      fetchLiveFeed(selectedGamePk, false);
    }, 30000); // 30 seconds poll

    return () => {
      clearInterval(intervalId);
    };
  }, [selectedGamePk]);

  return (
    <div className="flex flex-col h-[calc(100vh-64px)] w-full bg-gray-950 text-white overflow-hidden animate-fadeIn">
      
      {/* Top Bar: Selector + Status info */}
      <div className="h-20 bg-gray-900/60 border-b border-gray-900 flex items-center justify-between px-6 flex-shrink-0 backdrop-blur-md">
        <div className="w-1/3 max-w-lg">
          {isGamesLoading ? (
            <div className="h-10 w-full bg-gray-800 animate-pulse rounded-xl" />
          ) : (
            <GameSelector
              games={games}
              selectedGamePk={selectedGamePk}
              onSelect={setSelectedGamePk}
            />
          )}
        </div>

        {/* Global loading status or error banner */}
        <div className="flex items-center space-x-3 text-xs">
          {isLiveLoading && (
            <span className="flex items-center space-x-2 text-blue-400 font-bold bg-blue-950/40 border border-blue-900/30 px-3 py-1.5 rounded-full tracking-wider animate-pulse">
              <svg className="animate-spin h-3 w-3" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              <span>실시간 동기화 중...</span>
            </span>
          )}
          {selectedGamePk && !isLiveLoading && (
            <span className="flex items-center space-x-1.5 text-green-400 font-bold bg-green-950/40 border border-green-900/30 px-3 py-1.5 rounded-full tracking-wider">
              <span className="w-2 h-2 rounded-full bg-green-400 inline-block animate-ping" />
              <span>30초 주기 자동 갱신 활성</span>
            </span>
          )}
        </div>
      </div>

      {/* Main Grid: 3-column layout */}
      <div className="flex-1 w-full flex flex-row overflow-hidden relative">
        {errorMsg && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 z-30 bg-red-950/90 border border-red-800 rounded-xl p-3 text-xs text-red-200 backdrop-blur shadow-2xl animate-bounce">
            <span className="font-bold block mb-1 text-center">알림</span>
            <p>{errorMsg}</p>
          </div>
        )}

        {/* Left Column (30% width): ScoreBoard */}
        <div className="w-[30%] min-w-[320px] max-w-[400px] h-full p-4 flex-shrink-0 overflow-y-auto scrollbar-none border-r border-gray-900 bg-gray-950">
          <ScoreBoard situation={currentSituation} />
        </div>

        {/* Center Column (40% width / Flex-1): LiveSceneViewer */}
        <div className="flex-1 h-full p-4 relative bg-gray-950">
          <LiveSceneViewer
            predictResult={predictResult}
            pitcherHand={currentSituation?.pitcherHand}
          />
        </div>

        {/* Right Column (30% width): LiveResultPanel */}
        <div className="w-[30%] min-w-[320px] max-w-[400px] h-full p-4 flex-shrink-0 overflow-y-auto scrollbar-none border-l border-gray-900 bg-gray-950">
          <LiveResultPanel
            result={predictResult}
            actualPitch={currentSituation?.lastPitch?.type ? (currentSituation.lastPitch.type as PitchType) : null}
          />
        </div>
      </div>
    </div>
  );
}
