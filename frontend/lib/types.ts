export type PitchType =
  | 'FF'
  | 'FT'
  | 'SI'
  | 'SL'
  | 'CH'
  | 'CU'
  | 'KC'
  | 'CB'
  | 'FC'
  | 'FS'
  | 'KN';

export interface PredictRequest {
  pitcher: number;
  batter: number;
  balls: number;
  strikes: number;
  outs_when_up: number;
  inning: number;
  on_1b: number;
  on_2b: number;
  on_3b: number;
  fielder_2?: number;
  stand: number;
  game_year?: number;
}

export interface PredictResponse {
  predicted_pitch: PitchType;
  probabilities: Record<PitchType, number>;
  commentary: string;
  reasoning: string;
  vs_actual: string | null;
}

export interface GameInfo {
  gamePk: number;
  awayTeam: string;
  homeTeam: string;
  venue: string;
  status: string;
}

export interface LiveSituation {
  inning: number;
  isTopInning: boolean;
  balls: number;
  strikes: number;
  outs: number;
  awayScore: number;
  homeScore: number;
  currentPitcher: { id: number; name: string };
  currentBatter: { id: number; name: string };
  on1b: boolean;
  on2b: boolean;
  on3b: boolean;
  pitcherTeamId: string;
  batterTeamId: string;
  pitcherHand: 'R' | 'L';
  batterSide: 'R' | 'L';
  lastPitch: { type: string; description: string } | null;
}
