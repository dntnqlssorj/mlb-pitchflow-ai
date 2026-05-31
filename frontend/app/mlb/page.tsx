'use client'
import { useState, useEffect, useRef } from 'react'
import dynamic from 'next/dynamic'
import TabNav from '@/components/shared/TabNav'
import { GameInfo, LiveSituation, PredictResponse, PitchType, PITCH_COLORS, PITCH_NAMES } from '@/lib/types'

const LiveSceneViewer = dynamic(() => import('@/components/mlb/LiveSceneViewer'), { ssr: false })

export default function MLBLivePage() {
  const [games, setGames] = useState<GameInfo[]>([])
  const [selectedGamePk, setSelectedGamePk] = useState<number | null>(null)
  const [situation, setSituation] = useState<LiveSituation | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [loadingGames, setLoadingGames] = useState(true)
  const [loadingLive, setLoadingLive] = useState(false)
  const prevSitRef = useRef<LiveSituation | null>(null)

  // 오늘 경기 로드
  useEffect(() => {
    fetch('/api/mlb/today-games')
      .then(r => r.json())
      .then(d => setGames(d))
      .catch(() => {})
      .finally(() => setLoadingGames(false))
  }, [])

  // Live 피드 폴링
  const fetchLive = async (gamePk: number, initial = false) => {
    if (initial) setLoadingLive(true)
    try {
      const [liveRes, predRes] = await Promise.all([
        fetch(`/api/mlb/live-feed?gamePk=${gamePk}`),
        fetch(`/api/mlb/prediction?gamePk=${gamePk}`),
      ])
      if (liveRes.ok) {
        const sit: LiveSituation | null = await liveRes.json()
        setSituation(sit)
        prevSitRef.current = sit
      }
      if (predRes.ok) {
        const pred: PredictResponse | null = await predRes.json()
        setResult(pred)
      }
    } catch { /* silent */ } finally {
      if (initial) setLoadingLive(false)
    }
  }

  // 경기 선택 시 n8n에 game_pk 전달
  const handleGameSelect = async (gamePk: number) => {
    setSelectedGamePk(gamePk)
    
    // n8n 2_in_game Static Data에 active_game_pk 설정
    try {
      await fetch('http://localhost:5678/webhook/set-game', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_pk: gamePk }),
      })
    } catch (e) {
      console.log('n8n game_pk 설정 실패 (무시)', e)
    }
  }

  useEffect(() => {
    if (!selectedGamePk) { setSituation(null); setResult(null); return }
    fetchLive(selectedGamePk, true)
    const id = setInterval(() => fetchLive(selectedGamePk), 15000)
    return () => clearInterval(id)
  }, [selectedGamePk])

  const topProbs = result
    ? Object.entries(result.pitch_probabilities)
        .filter(([, p]) => (p ?? 0) > 0)
        .sort(([, a], [, b]) => (b ?? 0) - (a ?? 0))
        .slice(0, 6)
    : []

  return (
    <>
      <TabNav />
      <div style={{
        display: 'flex', height: '100vh', paddingTop: '48px',
        background: '#0d0d0d', color: '#fff', overflow: 'hidden',
      }}>

        {/* ── 좌측 — 경기 상황 ── */}
        <aside style={{
          width: '280px', minWidth: '280px',
          background: '#111', borderRight: '1px solid #2a2a2a',
          display: 'flex', flexDirection: 'column', overflowY: 'auto',
        }}>
          {/* 경기 선택 */}
          <div style={{ padding: '12px', borderBottom: '1px solid #1e1e1e' }}>
            <div style={{ fontSize: '9px', fontWeight: '700', color: '#555', letterSpacing: '0.12em', marginBottom: '6px' }}>
              SELECT GAME
            </div>
            {loadingGames ? (
              <div style={{ height: '36px', background: '#1a1a1a', borderRadius: '4px', animation: 'pulse 1.5s infinite' }} />
            ) : games.length === 0 ? (
              <div style={{ fontSize: '12px', color: '#444' }}>오늘 경기 없음</div>
            ) : (
              <select
                value={selectedGamePk ?? ''}
                onChange={e => {
                  const val = e.target.value
                  if (val) handleGameSelect(Number(val))
                  else setSelectedGamePk(null)
                }}
                style={{
                  width: '100%', padding: '8px', background: '#0d0d0d',
                  border: '1px solid #2a2a2a', color: '#fff', borderRadius: '4px', fontSize: '12px',
                }}
              >
                <option value="">-- 경기 선택 --</option>
                {games.map(g => (
                  <option key={g.gamePk} value={g.gamePk}>
                    {g.awayTeam} @ {g.homeTeam}
                  </option>
                ))}
              </select>
            )}
          </div>

          {/* 스코어보드 */}
          {situation && (
            <div style={{ padding: '12px', borderBottom: '1px solid #1e1e1e' }}>
              <ScoreBoard situation={situation} />
            </div>
          )}

          {/* 볼카운트 + 주자 */}
          {situation && (
            <div style={{ padding: '12px', borderBottom: '1px solid #1e1e1e' }}>
              <CountDisplay situation={situation} />
            </div>
          )}

          {/* 매치업 */}
          {situation && (
            <div style={{ padding: '12px' }}>
              <div style={{ fontSize: '9px', fontWeight: '700', color: '#555', letterSpacing: '0.12em', marginBottom: '8px' }}>
                MATCHUP
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <PlayerRow role="P" name={situation.currentPitcher.name} hand={situation.pitcherHand} />
                <PlayerRow role="B" name={situation.currentBatter.name} hand={situation.batterSide} />
              </div>
            </div>
          )}

          {/* 상태 */}
          {selectedGamePk && (
            <div style={{ marginTop: 'auto', padding: '10px 12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{
                width: '6px', height: '6px', borderRadius: '50%',
                background: loadingLive ? '#F59E0B' : '#22C55E',
                boxShadow: `0 0 6px ${loadingLive ? '#F59E0B' : '#22C55E'}`,
              }} />
              <span style={{ fontSize: '10px', color: '#555' }}>
                {loadingLive ? 'SYNCING...' : '15s AUTO REFRESH'}
              </span>
            </div>
          )}
        </aside>

        {/* ── 중앙 3D 뷰어 ── */}
        <main style={{ flex: 1, position: 'relative', background: '#08090f', minWidth: 0 }}>
          <LiveSceneViewer predictResult={result} pitcherHand={situation?.pitcherHand} pitcherId={situation?.currentPitcher?.id || null} year={2026} />
        </main>

        {/* ── 우측 — 예측 결과 ── */}
        <aside style={{
          width: '300px', minWidth: '300px',
          background: '#111', borderLeft: '1px solid #2a2a2a',
          overflowY: 'auto', padding: '14px',
          display: 'flex', flexDirection: 'column', gap: '12px',
        }}>
          {result ? (
            <div className="animate-fade-in">
              {/* 예측 구종 */}
              <div style={{ background: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: '6px', padding: '14px', marginBottom: '12px' }}>
                <SectionLabel>NEXT PITCH PREDICTION</SectionLabel>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: '10px' }}>
                  <div style={{
                    width: '52px', height: '52px', borderRadius: '50%', flexShrink: 0,
                    background: PITCH_COLORS[result.predicted_pitch] ?? '#666',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '13px', fontWeight: '700', color: '#fff',
                    boxShadow: `0 0 20px ${PITCH_COLORS[result.predicted_pitch] ?? '#666'}66`,
                  }}>
                    {result.predicted_pitch}
                  </div>
                  <div>
                    <div style={{ fontSize: '26px', fontFamily: "'Bebas Neue', Arial", letterSpacing: '0.05em', lineHeight: 1 }}>
                      {PITCH_NAMES[result.predicted_pitch]}
                    </div>
                    <div style={{ fontSize: '12px', color: '#9ca3af', marginTop: '3px' }}>
                      신뢰도 <span style={{ color: '#fff', fontWeight: '700' }}>{(result.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ fontSize: '10px', color: '#555', marginTop: '2px' }}>
                      {result.routing === 'per_pitcher' ? '🎯 투수 전용 모델' : '📊 글로벌 스태킹'}
                    </div>
                  </div>
                </div>
              </div>

              {/* 실제 vs 예측 */}
              {situation?.lastPitch && (
                <div style={{ background: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: '6px', padding: '14px', marginBottom: '12px' }}>
                  <SectionLabel>LAST PITCH</SectionLabel>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '8px' }}>
                    <div style={{
                      padding: '4px 10px', borderRadius: '4px',
                      background: PITCH_COLORS[situation.lastPitch.type as PitchType] ?? '#333',
                      fontSize: '13px', fontWeight: '700', color: '#fff',
                    }}>
                      {situation.lastPitch.type}
                    </div>
                    <span style={{ fontSize: '12px', color: '#6b7280' }}>{situation.lastPitch.description}</span>
                  </div>
                </div>
              )}

              {/* 확률 바 차트 */}
              <div style={{ background: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: '6px', padding: '14px', marginBottom: '12px' }}>
                <SectionLabel>PITCH PROBABILITIES</SectionLabel>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '10px' }}>
                  {topProbs.map(([pitch, prob]) => {
                    const pct = (prob ?? 0) * 100
                    const color = PITCH_COLORS[pitch as PitchType] ?? '#666'
                    const isTop = pitch === result.predicted_pitch
                    return (
                      <div key={pitch}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <div style={{ width: '7px', height: '7px', borderRadius: '50%', background: color }} />
                            <span style={{ fontSize: '12px', fontWeight: isTop ? '700' : '400', color: isTop ? '#fff' : '#9ca3af' }}>
                              {pitch}
                            </span>
                            <span style={{ fontSize: '11px', color: '#444' }}>{PITCH_NAMES[pitch as PitchType]}</span>
                          </div>
                          <span style={{ fontSize: '12px', fontWeight: '700', fontFamily: "'Bebas Neue', Arial", color: isTop ? '#fff' : '#9ca3af' }}>
                            {pct.toFixed(1)}%
                          </span>
                        </div>
                        <div style={{ height: '3px', background: '#222', borderRadius: '2px' }}>
                          <div style={{
                            height: '100%', background: color, borderRadius: '2px',
                            width: `${pct.toFixed(1)}%`, opacity: isTop ? 1 : 0.5,
                            transition: 'width 0.5s ease',
                          }} />
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* AI 해설 */}
              {result.commentary && (
                <div style={{ background: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: '6px', padding: '14px', marginBottom: '12px' }}>
                  <SectionLabel>AI ANALYSIS</SectionLabel>
                  <p style={{ fontSize: '12px', lineHeight: '1.65', color: '#d1d5db', marginTop: '8px' }}>
                    {result.commentary}
                  </p>
                </div>
              )}

              <div style={{ fontSize: '10px', color: '#333', display: 'flex', justifyContent: 'space-between' }}>
                <span>Model: {result.model_used}</span>
                {result.enrichment_latency_ms != null && <span>{result.enrichment_latency_ms.toFixed(1)}ms</span>}
              </div>
            </div>
          ) : (
            <div style={{
              display: 'flex', flexDirection: 'column', alignItems: 'center',
              justifyContent: 'center', height: '100%', color: '#222', gap: '8px',
            }}>
              <div style={{ fontSize: '32px' }}>📡</div>
              <div style={{ fontSize: '11px', letterSpacing: '0.08em' }}>WAITING FOR PREDICTION</div>
            </div>
          )}
        </aside>
      </div>
    </>
  )
}

/* ── 서브 컴포넌트 ─────────────────── */

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ fontSize: '9px', fontWeight: '700', letterSpacing: '0.12em', color: '#555' }}>
      {children}
    </div>
  )
}

function ScoreBoard({ situation: s }: { situation: LiveSituation }) {
  return (
    <div>
      <div style={{ fontSize: '9px', fontWeight: '700', color: '#555', letterSpacing: '0.12em', marginBottom: '8px' }}>
        SCORE
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '11px', color: '#9ca3af' }}>AWAY</div>
          <div style={{ fontSize: '28px', fontFamily: "'Bebas Neue', Arial" }}>{s.awayScore}</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '11px', color: '#D50032', fontWeight: '700' }}>
            {s.isTopInning ? '▲' : '▼'} {s.inning}
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '11px', color: '#9ca3af' }}>HOME</div>
          <div style={{ fontSize: '28px', fontFamily: "'Bebas Neue', Arial" }}>{s.homeScore}</div>
        </div>
      </div>
    </div>
  )
}

function CountDisplay({ situation: s }: { situation: LiveSituation }) {
  return (
    <div>
      <div style={{ fontSize: '9px', fontWeight: '700', color: '#555', letterSpacing: '0.12em', marginBottom: '8px' }}>COUNT</div>
      <div style={{ display: 'flex', gap: '12px', alignItems: 'center', marginBottom: '10px' }}>
        <div>
          <div style={{ fontSize: '9px', color: '#555' }}>B</div>
          <div style={{ display: 'flex', gap: '3px' }}>
            {[0, 1, 2, 3].map(i => (
              <div key={i} style={{
                width: '10px', height: '10px', borderRadius: '50%',
                background: i < s.balls ? '#22C55E' : '#222',
                border: '1px solid #333',
              }} />
            ))}
          </div>
        </div>
        <div>
          <div style={{ fontSize: '9px', color: '#555' }}>S</div>
          <div style={{ display: 'flex', gap: '3px' }}>
            {[0, 1, 2].map(i => (
              <div key={i} style={{
                width: '10px', height: '10px', borderRadius: '50%',
                background: i < s.strikes ? '#EF4444' : '#222',
                border: '1px solid #333',
              }} />
            ))}
          </div>
        </div>
        <div>
          <div style={{ fontSize: '9px', color: '#555' }}>O</div>
          <div style={{ display: 'flex', gap: '3px' }}>
            {[0, 1, 2].map(i => (
              <div key={i} style={{
                width: '10px', height: '10px', borderRadius: '50%',
                background: i < s.outs ? '#F59E0B' : '#222',
                border: '1px solid #333',
              }} />
            ))}
          </div>
        </div>
      </div>
      {/* 주자 미니 다이아몬드 */}
      <div style={{ position: 'relative', width: '48px', height: '48px' }}>
        {[
          { on: s.on2b, top: 0, left: '50%', ml: '-8px' },
          { on: s.on3b, top: '50%', left: 0, mt: '-8px' },
          { on: s.on1b, top: '50%', right: 0, mt: '-8px' },
        ].map((b, i) => (
          <div key={i} style={{
            position: 'absolute', width: '14px', height: '14px',
            background: b.on ? '#F59E0B' : '#1e1e1e',
            border: `1.5px solid ${b.on ? '#F59E0B' : '#333'}`,
            transform: 'rotate(45deg)',
            top: b.top, left: b.left, right: b.right,
            marginLeft: b.ml, marginTop: b.mt,
          }} />
        ))}
        <div style={{
          position: 'absolute', width: '14px', height: '14px',
          background: '#1e1e1e', border: '1.5px solid #333',
          transform: 'rotate(45deg)', bottom: 0, left: '50%', marginLeft: '-7px',
        }} />
      </div>
    </div>
  )
}

function PlayerRow({ role, name, hand }: { role: string; name: string; hand: string }) {
  const color = role === 'P' ? '#3B82F6' : '#EF4444'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <div style={{
        width: '20px', height: '20px', borderRadius: '3px',
        background: color, display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: '10px', fontWeight: '700', color: '#fff', flexShrink: 0,
      }}>{role}</div>
      <div style={{ fontSize: '12px', color: '#d1d5db', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{name}</div>
      <div style={{ fontSize: '10px', color: '#555', flexShrink: 0 }}>{hand}</div>
    </div>
  )
}
