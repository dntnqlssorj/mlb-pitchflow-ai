'use client'
import { useState } from 'react'
import dynamic from 'next/dynamic'
import TabNav from '@/components/shared/TabNav'
import PlayerSearch from '@/components/custom/PlayerSearch'
import { PredictResponse, PITCH_COLORS, PITCH_NAMES, PitchType } from '@/lib/types'

const SceneViewer = dynamic(() => import('@/components/custom/SceneViewer'), { ssr: false })

export default function CustomPage() {
  const [form, setForm] = useState({
    pitcher: '', pitcherName: '',
    batter: '', batterName: '',
    catcher: '', catcherName: '',
    balls: 0, strikes: 0, outs: 0, inning: 1,
    on_1b: 0, on_2b: 0, on_3b: 0, stand: 'R',
  })
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async () => {
    if (!form.pitcher || !form.batter) return
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('http://localhost:5678/webhook/on-demand-predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pitcher_id: Number(form.pitcher),
          batter_id: Number(form.batter),
          catcher_id: Number(form.catcher) || 0,
          balls: form.balls,
          strikes: form.strikes,
          outs: form.outs,
          inning: form.inning,
          on_1b: form.on_1b,
          on_2b: form.on_2b,
          on_3b: form.on_3b,
          stand: form.stand,
          game_pk: 0,
          actual_pitch: null,
        }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setResult(data)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '예측 실패')
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  const sortedProbs = result?.pitch_probabilities
    ? Object.entries(result.pitch_probabilities)
        .filter(([, v]) => (v ?? 0) > 0)
        .sort(([, a], [, b]) => (b ?? 0) - (a ?? 0))
        .slice(0, 8)
    : []

  return (
    <>
      <TabNav />
      <div style={{
        display: 'flex', height: '100vh', paddingTop: '48px',
        background: '#0d0d0d', color: '#fff', overflow: 'hidden',
      }}>

        {/* ── 좌측 입력 패널 ── */}
        <aside style={{
          width: '260px', minWidth: '260px',
          background: '#111', borderRight: '1px solid #2a2a2a',
          overflowY: 'auto', padding: '16px 14px',
          display: 'flex', flexDirection: 'column', gap: '14px',
        }}>
          <SectionLabel>MATCH UP</SectionLabel>

          <PlayerSearch
            label="PITCHER"
            placeholder="투수 이름 검색 (예: Ohtani)"
            value={form.pitcher}
            onChange={(id, name) => {
              setForm(f => ({ ...f, pitcher: id, pitcherName: name }))
              setResult(null)
            }}
          />
          <PlayerSearch
            label="BATTER"
            placeholder="타자 이름 검색 (예: Judge)"
            value={form.batter}
            onChange={(id, name) => setForm(f => ({ ...f, batter: id, batterName: name }))}
          />
          <PlayerSearch
            label="CATCHER (선택)"
            placeholder="포수 이름 검색"
            value={form.catcher}
            onChange={(id, name) => setForm(f => ({ ...f, catcher: id, catcherName: name }))}
          />

          {/* 좌우타 */}
          <div>
            <FieldLabel>BATTER SIDE</FieldLabel>
            <div style={{ display: 'flex', gap: '6px' }}>
              {['R', 'L'].map(s => (
                <button key={s} onClick={() => setForm(f => ({ ...f, stand: s }))} style={{
                  flex: 1, padding: '7px 0', cursor: 'pointer', borderRadius: '3px',
                  border: `1px solid ${form.stand === s ? '#D50032' : '#333'}`,
                  background: form.stand === s ? '#D50032' : 'transparent',
                  color: '#fff', fontSize: '12px', fontWeight: '700', letterSpacing: '0.05em',
                }}>
                  {s === 'R' ? '우타 R' : '좌타 L'}
                </button>
              ))}
            </div>
          </div>

          <Divider />
          <SectionLabel>COUNT</SectionLabel>

          <CountSelector label="BALLS" value={form.balls} max={3}
            onChange={v => setForm(f => ({ ...f, balls: v }))} color="#22C55E" />
          <CountSelector label="STRIKES" value={form.strikes} max={2}
            onChange={v => setForm(f => ({ ...f, strikes: v }))} color="#EF4444" />
          <CountSelector label="OUTS" value={form.outs} max={2}
            onChange={v => setForm(f => ({ ...f, outs: v }))} color="#9ca3af" />

          {/* 이닝 */}
          <div>
            <FieldLabel>INNING</FieldLabel>
            <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
              {Array.from({ length: 9 }, (_, i) => i + 1).map(inn => (
                <button key={inn} onClick={() => setForm(f => ({ ...f, inning: inn }))} style={{
                  width: '28px', height: '28px', borderRadius: '3px', cursor: 'pointer',
                  border: `1px solid ${form.inning === inn ? '#fff' : '#333'}`,
                  background: form.inning === inn ? '#333' : 'transparent',
                  color: form.inning === inn ? '#fff' : '#9ca3af',
                  fontSize: '12px', fontWeight: '600',
                }}>{inn}</button>
              ))}
              <button onClick={() => setForm(f => ({ ...f, inning: f.inning < 10 ? 10 : f.inning + 1 }))} style={{
                padding: '0 8px', height: '28px', borderRadius: '3px', cursor: 'pointer',
                border: `1px solid ${form.inning >= 10 ? '#8B5CF6' : '#333'}`,
                background: form.inning >= 10 ? '#8B5CF6' : 'transparent',
                color: '#fff', fontSize: '11px', fontWeight: '700',
              }}>
                {form.inning >= 10 ? `${form.inning}회` : '연장'}
              </button>
            </div>
          </div>

          <Divider />
          <SectionLabel>RUNNERS</SectionLabel>
          <DiamondSelector
            on1b={form.on_1b} on2b={form.on_2b} on3b={form.on_3b}
            onChange={(b1, b2, b3) => setForm(f => ({ ...f, on_1b: b1, on_2b: b2, on_3b: b3 }))}
          />

          {error && (
            <div style={{ fontSize: '11px', color: '#EF4444', padding: '6px 8px', background: '#1a0a0a', borderRadius: '4px', border: '1px solid #3a1a1a' }}>
              ⚠ {error}
            </div>
          )}

          {/* 예측 버튼 */}
          <button onClick={handleSubmit}
            disabled={loading || !form.pitcher || !form.batter}
            style={{
              width: '100%', padding: '12px 0', marginTop: '4px',
              background: loading ? '#333' : (!form.pitcher || !form.batter) ? '#1a1a1a' : '#D50032',
              color: (!form.pitcher || !form.batter) ? '#555' : '#fff',
              border: 'none', borderRadius: '4px',
              fontSize: '13px', fontWeight: '700', letterSpacing: '0.1em',
              cursor: loading || !form.pitcher || !form.batter ? 'not-allowed' : 'pointer',
              transition: 'background 0.15s',
            }}>
            {loading ? 'PREDICTING...' : 'PREDICT PITCH'}
          </button>
        </aside>

        <main style={{ flex: 1, position: 'relative', background: '#0a0a0f', minWidth: 0 }}>
          <SceneViewer result={result} pitcherId={form.pitcher ? Number(form.pitcher) : null} year={2026} />
        </main>

        {/* ── 우측 결과 패널 ── */}
        <aside style={{
          width: '300px', minWidth: '300px',
          background: '#111', borderLeft: '1px solid #2a2a2a',
          overflowY: 'auto', padding: '16px 14px',
          display: 'flex', flexDirection: 'column', gap: '12px',
        }}>
          {result ? (
            <div className="animate-fade-in">
              {/* 예측 구종 카드 */}
              <div style={{ background: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: '6px', padding: '14px', marginBottom: '12px' }}>
                <SectionLabel>PREDICTED PITCH</SectionLabel>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: '10px' }}>
                  <div style={{
                    width: '52px', height: '52px', borderRadius: '50%', flexShrink: 0,
                    background: PITCH_COLORS[result.predicted_pitch] ?? '#666',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '13px', fontWeight: '700', color: '#fff',
                    boxShadow: `0 0 16px ${PITCH_COLORS[result.predicted_pitch] ?? '#666'}55`,
                  }}>
                    {result.predicted_pitch}
                  </div>
                  <div>
                    <div style={{
                      fontSize: '26px', fontWeight: '700',
                      fontFamily: "'Bebas Neue', Arial", letterSpacing: '0.05em', lineHeight: 1,
                    }}>
                      {PITCH_NAMES[result.predicted_pitch]}
                    </div>
                    <div style={{ fontSize: '12px', color: '#9ca3af', marginTop: '3px' }}>
                      신뢰도{' '}
                      <span style={{ color: '#fff', fontWeight: '700' }}>
                        {(result.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{ fontSize: '10px', color: '#555', marginTop: '2px' }}>
                      {result.routing === 'per_pitcher' ? '🎯 투수 전용 모델' : result.routing === 'scouting_llm' ? '🔍 스카우팅 LLM' : '📊 글로벌 스태킹'}
                    </div>
                  </div>
                </div>
              </div>

              {/* 확률 바 차트 */}
              <div style={{ background: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: '6px', padding: '14px', marginBottom: '12px' }}>
                <SectionLabel>PITCH PROBABILITIES</SectionLabel>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '9px', marginTop: '10px' }}>
                  {sortedProbs.map(([pitch, prob]) => {
                    const pct = ((prob ?? 0) * 100)
                    const color = PITCH_COLORS[pitch as PitchType] ?? '#666'
                    const isTop = pitch === result.predicted_pitch
                    return (
                      <div key={pitch}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <div style={{ width: '7px', height: '7px', borderRadius: '50%', background: color, flexShrink: 0 }} />
                            <span style={{ fontSize: '12px', fontWeight: isTop ? '700' : '400', color: isTop ? '#fff' : '#9ca3af' }}>
                              {pitch}
                            </span>
                            <span style={{ fontSize: '11px', color: '#555' }}>
                              {PITCH_NAMES[pitch as PitchType]}
                            </span>
                          </div>
                          <span style={{
                            fontSize: '12px', fontWeight: '700',
                            fontFamily: "'Bebas Neue', Arial",
                            color: isTop ? '#fff' : '#9ca3af',
                          }}>
                            {pct.toFixed(1)}%
                          </span>
                        </div>
                        <div style={{ height: '3px', background: '#222', borderRadius: '2px', overflow: 'hidden' }}>
                          <div style={{
                            height: '100%', borderRadius: '2px',
                            background: color,
                            width: `${pct.toFixed(1)}%`,
                            opacity: isTop ? 1 : 0.5,
                            transition: 'width 0.5s ease',
                          }} />
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* AI 해설 (n8n 있을 때만 표시) */}
              {result.commentary && (
                <div style={{ background: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: '6px', padding: '14px', marginBottom: '12px' }}>
                  <SectionLabel>AI ANALYSIS</SectionLabel>
                  <p style={{ fontSize: '12px', lineHeight: '1.65', color: '#d1d5db', marginTop: '8px' }}>
                    {result.commentary}
                  </p>
                </div>
              )}

              {/* 모델 메타 */}
              <div style={{ fontSize: '10px', color: '#444', padding: '4px 2px', display: 'flex', justifyContent: 'space-between' }}>
                <span>Model: {result.model_used}</span>
                {result.enrichment_latency_ms != null && (
                  <span>{result.enrichment_latency_ms.toFixed(1)}ms</span>
                )}
              </div>
            </div>
          ) : (
            <div style={{
              display: 'flex', flexDirection: 'column', alignItems: 'center',
              justifyContent: 'center', height: '100%', gap: '12px', padding: '24px',
            }}>
              <div style={{ fontSize: '36px', opacity: 0.3 }}>📊</div>
              <div style={{ fontSize: '12px', fontWeight: '700', letterSpacing: '0.1em', color: '#555' }}>
                PREDICTION RESULTS
              </div>
              <div style={{ fontSize: '11px', color: '#3a3a3a', textAlign: 'center', lineHeight: '1.6' }}>
                좌측에서 투수 / 타자 ID를 입력하고<br />
                <span style={{ color: '#D50032', fontWeight: '700' }}>PREDICT PITCH</span> 버튼을 누르세요
              </div>
              <div style={{ marginTop: '8px', display: 'flex', flexDirection: 'column', gap: '6px', width: '100%' }}>
                {['PREDICTED PITCH', 'PROBABILITIES', 'AI ANALYSIS'].map(label => (
                  <div key={label} style={{
                    height: '32px', background: '#1a1a1a', borderRadius: '4px',
                    border: '1px solid #1e1e1e', display: 'flex', alignItems: 'center',
                    padding: '0 10px', gap: '8px',
                  }}>
                    <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#2a2a2a' }} />
                    <span style={{ fontSize: '10px', color: '#2a2a2a', letterSpacing: '0.08em' }}>{label}</span>
                  </div>
                ))}
              </div>
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
    <div style={{
      fontSize: '10px', fontWeight: '700', letterSpacing: '0.12em',
      color: '#6b7280', marginBottom: '0px',
      fontFamily: '-apple-system, BlinkMacSystemFont, Arial, sans-serif',
    }}>
      {children}
    </div>
  )
}

function FieldLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      fontSize: '10px', fontWeight: '700', letterSpacing: '0.1em',
      color: '#6b7280', marginBottom: '5px',
    }}>
      {children}
    </div>
  )
}

function Divider() {
  return <div style={{ height: '1px', background: '#1e1e1e' }} />
}

function InputField({ label, value, onChange, placeholder }: {
  label: string; value: string; onChange: (v: string) => void; placeholder?: string
}) {
  return (
    <div>
      <FieldLabel>{label}</FieldLabel>
      <input
        type="text" value={value} placeholder={placeholder}
        onChange={e => onChange(e.target.value)}
        style={{
          width: '100%', padding: '7px 10px',
          background: '#0d0d0d', border: '1px solid #2a2a2a',
          color: '#fff', borderRadius: '3px', fontSize: '13px',
        }}
      />
    </div>
  )
}

function CountSelector({ label, value, max, onChange, color }: {
  label: string; value: number; max: number; onChange: (v: number) => void; color: string
}) {
  return (
    <div>
      <FieldLabel>{label}</FieldLabel>
      <div style={{ display: 'flex', gap: '5px' }}>
        {Array.from({ length: max + 1 }, (_, i) => (
          <button key={i} onClick={() => onChange(i)} style={{
            flex: 1, padding: '7px 0', cursor: 'pointer', borderRadius: '3px',
            border: `1px solid ${value === i ? color : '#2a2a2a'}`,
            background: value === i ? `${color}22` : 'transparent',
            color: value === i ? color : '#9ca3af',
            fontSize: '14px', fontWeight: '700',
            transition: 'all 0.1s',
          }}>{i}</button>
        ))}
      </div>
    </div>
  )
}

function DiamondSelector({ on1b, on2b, on3b, onChange }: {
  on1b: number; on2b: number; on3b: number
  onChange: (b1: number, b2: number, b3: number) => void
}) {
  const baseStyle = (active: number, pos: 'top' | 'left' | 'right' | 'home'): React.CSSProperties => ({
    position: 'absolute',
    width: '22px', height: '22px',
    background: active && pos !== 'home' ? '#F59E0B' : '#1e1e1e',
    border: `1.5px solid ${active && pos !== 'home' ? '#F59E0B' : '#333'}`,
    transform: 'rotate(45deg)',
    cursor: pos !== 'home' ? 'pointer' : 'default',
    transition: 'all 0.15s',
    ...(pos === 'top' ? { top: 0, left: '50%', marginLeft: '-11px' } : {}),
    ...(pos === 'left' ? { top: '50%', left: 0, marginTop: '-11px' } : {}),
    ...(pos === 'right' ? { top: '50%', right: 0, marginTop: '-11px' } : {}),
    ...(pos === 'home' ? { bottom: 0, left: '50%', marginLeft: '-11px' } : {}),
  })
  return (
    <div style={{ display: 'flex', justifyContent: 'center', padding: '4px 0' }}>
      <div style={{ position: 'relative', width: '72px', height: '72px' }}>
        <button style={baseStyle(on2b, 'top')}
          onClick={() => onChange(on1b, on2b ? 0 : 1, on3b)} />
        <button style={baseStyle(on3b, 'left')}
          onClick={() => onChange(on1b, on2b, on3b ? 0 : 1)} />
        <button style={baseStyle(on1b, 'right')}
          onClick={() => onChange(on1b ? 0 : 1, on2b, on3b)} />
        <div style={baseStyle(0, 'home')} />
      </div>
    </div>
  )
}
