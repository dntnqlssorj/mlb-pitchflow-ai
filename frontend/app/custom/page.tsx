'use client';

import React, { useState } from 'react';
import InputForm from '../../components/custom/InputForm';
import SceneViewer from '../../components/custom/SceneViewer';
import ResultPanel from '../../components/custom/ResultPanel';
import { PredictRequest, PredictResponse, PitchType } from '../../lib/types';

export default function CustomPredictPage() {
  const [predictResult, setPredictResult] = useState<PredictResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [actualPitch, setActualPitch] = useState<PitchType | null>(null);
  
  // Track selected teams for active SVG styling
  const [pitcherTeamId, setPitcherTeamId] = useState<string>('');
  const [batterTeamId, setBatterTeamId] = useState<string>('');
  
  // Track selected hands/sides for character pose branching
  const [pitcherHand, setPitcherHand] = useState<'R' | 'L'>('R');
  const [batterSide, setBatterSide] = useState<'R' | 'L'>('R');
  
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleSubmit = async (data: PredictRequest) => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || '예측을 수행하는 데 실패했습니다.');
      }

      const result: PredictResponse = await response.json();
      setPredictResult(result);
    } catch (err: any) {
      console.error('Prediction submission failed:', err);
      setErrorMsg(err?.message || '네트워크 통신 오류가 발생했습니다.');
      setPredictResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-row h-[calc(100vh-64px)] w-full bg-gray-950 overflow-hidden text-white animate-fadeIn">
      {/* Left Column (1/4 width): InputForm */}
      <div className="w-1/4 min-w-[320px] max-w-[400px] h-full flex-shrink-0 relative">
        <InputForm
          onSubmit={handleSubmit}
          onActualPitchChange={setActualPitch}
          isLoading={isLoading}
          onPitcherChange={(p) => {
            setPitcherTeamId(p?.teamId || '');
            setPitcherHand(p?.hand || 'R');
          }}
          onBatterChange={(b) => {
            setBatterTeamId(b?.teamId || '');
            setBatterSide(b?.side || 'R');
          }}
        />
      </div>

      {/* Center Column (2/4 width): SceneViewer */}
      <div className="flex-1 h-full bg-gray-950 relative border-r border-l border-gray-900">
        {errorMsg && (
          <div className="absolute top-4 right-4 z-20 bg-red-950/80 border border-red-800 rounded-lg p-3 text-xs text-red-200 backdrop-blur animate-bounce shadow-lg">
            <span className="font-bold block mb-1">오류 발생</span>
            <p>{errorMsg}</p>
          </div>
        )}
        <SceneViewer
          pitcherTeamId={pitcherTeamId}
          batterTeamId={batterTeamId}
          pitcherHand={pitcherHand}
          batterSide={batterSide}
          predictResult={predictResult}
        />
      </div>

      {/* Right Column (1/4 width): ResultPanel */}
      <div className="w-1/4 min-w-[320px] max-w-[400px] h-full flex-shrink-0">
        <ResultPanel
          result={predictResult}
          actualPitch={actualPitch}
        />
      </div>
    </div>
  );
}
