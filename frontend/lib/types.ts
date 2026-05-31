export type PitchType =
  | 'FF' | 'SI' | 'SL' | 'CH' | 'FC' | 'ST' | 'CU' | 'FS'
  | 'KC' | 'SV' | 'FA' | 'EP' | 'KN' | 'CS' | 'SC' | 'FO' | 'PO'

export const PITCH_COLORS: Record<PitchType, string> = {
  FF: '#EF4444', FA: '#F97316', SI: '#F59E0B',
  FC: '#EAB308', SL: '#22C55E', ST: '#8B5CF6',
  CU: '#3B82F6', KC: '#06B6D4', CS: '#0EA5E9',
  CH: '#EC4899', FS: '#F43F5E', FO: '#84CC16',
  KN: '#A78BFA', SV: '#E11D48', EP: '#6B7280',
  SC: '#D97706', PO: '#94A3B8',
}

export const PITCH_NAMES: Record<PitchType, string> = {
  FF: '포심', FA: '패스트볼', SI: '싱커', FC: '커터',
  SL: '슬라이더', ST: '스위퍼', CU: '커브', KC: '너클커브',
  CS: '슬로우커브', CH: '체인지업', FS: '스플리터', FO: '포크볼',
  KN: '너클볼', SV: '스크류볼', EP: '이팜', SC: '스크류',
  PO: '피치아웃',
}

export interface PredictRequest {
  pitcher: number
  batter: number
  fielder_2?: number
  fielder_3?: number
  fielder_4?: number
  fielder_5?: number
  fielder_6?: number
  fielder_7?: number
  fielder_8?: number
  fielder_9?: number
  game_pk: number
  game_year: number
  balls: number
  strikes: number
  outs_when_up: number
  inning: number
  on_1b: number
  on_2b: number
  on_3b: number
  stand: string
  pitch_count_override?: number
  model_type?: string
}

export interface PredictResponse {
  model_used: string
  predicted_pitch: PitchType
  confidence: number
  pitch_probabilities: Partial<Record<PitchType, number>>
  routing?: string
  enrichment_latency_ms?: number
  commentary?: string | null
  reasoning?: string | null
  vs_actual?: string | null
}

export interface Player {
  id: number
  full_name: string
  primary_position?: string
  bat_side?: string
  pitch_hand?: string
}

export interface GameInfo {
  gamePk: number
  awayTeam: string
  homeTeam: string
  venue: string
  status: string
}

export interface LiveSituation {
  inning: number
  isTopInning: boolean
  balls: number
  strikes: number
  outs: number
  awayScore: number
  homeScore: number
  currentPitcher: { id: number; name: string }
  currentBatter: { id: number; name: string }
  on1b: boolean
  on2b: boolean
  on3b: boolean
  pitcherTeamId: string
  batterTeamId: string
  pitcherHand: 'R' | 'L'
  batterSide: 'R' | 'L'
  lastPitch: { type: string; description: string } | null
}

export interface LivePredictionEvent {
  event: string
  game_pk: number
  at_bat_number: number
  pitch_number: number
  inning: number
  count: string
  outs: number
  on_1b: number
  on_2b: number
  on_3b: number
  pitcher_id: number
  batter_id: number
  bat_side: string
  predicted_pitch: PitchType
  confidence: number
  pitch_probabilities: Partial<Record<PitchType, number>>
  model_used: string
  routing: string
  commentary?: string
  reasoning?: string
  inference_ts: string
}
