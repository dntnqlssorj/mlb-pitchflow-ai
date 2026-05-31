'use client'
import React, { useState } from 'react'
import { ArsenalPitch } from '@/lib/types'

interface Props {
  arsenal: ArsenalPitch[]
  predictedProbabilities: Record<string, number>
  hoveredPitch: string | null
  visiblePitches: Set<string>
  onHoverChange: (p: string | null) => void
  onVisibilityToggle: (p: string) => void
}

export default function ArsenalHUD({
  arsenal,
  predictedProbabilities,
  hoveredPitch,
  visiblePitches,
  onHoverChange,
  onVisibilityToggle,
}: Props) {
  const [collapsed, setCollapsed] = useState(false)

  if (!arsenal || arsenal.length === 0) return null

  return (
    <div
      style={{
        position: 'absolute',
        top: '12px',
        left: '12px',
        zIndex: 50,
        width: collapsed ? '220px' : '320px',
        padding: '12px 14px',
        background: 'rgba(10, 15, 30, 0.78)',
        backdropFilter: 'blur(12px) saturate(180%)',
        WebkitBackdropFilter: 'blur(12px) saturate(180%)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '8px',
        boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.45)',
        color: '#fff',
        fontFamily: 'Arial, sans-serif',
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
      }}
    >
      {/* 타이틀 헤더 */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          borderBottom: collapsed ? 'none' : '1px solid rgba(255, 255, 255, 0.1)',
          paddingBottom: '6px',
          marginBottom: collapsed ? '0' : '10px',
        }}
      >
        <span
          style={{
            fontSize: '9px',
            fontWeight: '700',
            color: '#8b949e',
            letterSpacing: '0.12em',
            textShadow: '0 1px 4px rgba(0,0,0,0.8)',
          }}
        >
          PITCH ARSENAL & VELOCITY
        </span>
        <button
          onClick={() => setCollapsed(prev => !prev)}
          style={{
            background: 'none',
            border: 'none',
            color: 'rgba(255, 255, 255, 0.6)',
            cursor: 'pointer',
            fontSize: '14px',
            lineHeight: 1,
            padding: '0 2px',
            outline: 'none',
            transition: 'color 0.2s',
          }}
          title={collapsed ? '펼치기' : '접기'}
        >
          {collapsed ? '＋' : '－'}
        </button>
      </div>

      {!collapsed && (
        <>
          {/* 리스트 테이블 헤더 */}
          <div
            style={{
              display: 'flex',
              fontSize: '9px',
              color: '#6b7280',
              fontWeight: '700',
              marginBottom: '6px',
              padding: '0 4px',
            }}
          >
            <span style={{ width: '20px' }}>VIS</span>
            <span style={{ flex: 1, paddingLeft: '8px' }}>PITCH</span>
            <span style={{ width: '48px', textAlign: 'right' }}>SPEED</span>
            <span style={{ width: '120px', paddingLeft: '16px' }}>USG% vs AI%</span>
          </div>

          {/* 구종별 행 */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {arsenal.map((pitch) => {
              const pt = pitch.pitch_type
              const isVisible = visiblePitches.has(pt)
              const isHovered = hoveredPitch === pt
              const anyHovered = hoveredPitch !== null
              const activeOpacity = isHovered ? 1.0 : anyHovered ? 0.35 : 0.9

              const speedText = pitch.avg_speed ? `${pitch.avg_speed.toFixed(0)}` : '--'

              // AI 예측 확률 스케일링 보정
              const probVal = predictedProbabilities[pt] ?? 0
              const aiPct = probVal <= 1.0 ? probVal * 100 : probVal

              return (
                <div
                  key={pt}
                  onMouseEnter={() => onHoverChange(pt)}
                  onMouseLeave={() => onHoverChange(null)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    padding: '4px',
                    borderRadius: '4px',
                    background: isHovered ? 'rgba(255, 255, 255, 0.05)' : 'transparent',
                    opacity: activeOpacity,
                    transition: 'opacity 0.15s, background 0.15s',
                    cursor: 'pointer',
                  }}
                >
                  {/* [토글버튼] */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      onVisibilityToggle(pt)
                    }}
                    style={{
                      width: '20px',
                      height: '20px',
                      background: 'transparent',
                      border: 'none',
                      color: isVisible ? '#38bdf8' : '#4b5563',
                      cursor: 'pointer',
                      fontSize: '11px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      padding: 0,
                      transition: 'color 0.15s',
                    }}
                    title={isVisible ? '궤적 숨기기' : '궤적 보이기'}
                  >
                    {isVisible ? '👁' : '❌'}
                  </button>

                  {/* [색상원] */}
                  <div
                    style={{
                      width: '8px',
                      height: '8px',
                      borderRadius: '50%',
                      background: pitch.color,
                      marginLeft: '6px',
                      flexShrink: 0,
                      boxShadow: `0 0 6px ${pitch.color}aa`,
                    }}
                  />

                  {/* [구종명] */}
                  <div
                    style={{
                      flex: 1,
                      fontSize: '11px',
                      fontWeight: '600',
                      color: '#e6edf3',
                      paddingLeft: '6px',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      textShadow: '0 1px 2px rgba(0,0,0,0.6)',
                    }}
                  >
                    {pitch.name}
                  </div>

                  {/* [avg_speed MPH] */}
                  <div
                    style={{
                      width: '48px',
                      textAlign: 'right',
                      fontSize: '11px',
                      fontFamily: 'monospace',
                      color: '#c9d1d9',
                    }}
                  >
                    {speedText}
                    <span style={{ fontSize: '8px', color: '#6b7280', marginLeft: '1px' }}>M</span>
                  </div>

                  {/* [게이지 영역] 구사율 게이지 (Historical) + AI 예측 게이지 */}
                  <div
                    style={{
                      width: '120px',
                      paddingLeft: '16px',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '3px',
                      justifyContent: 'center',
                    }}
                  >
                    {/* 1. 구사율 게이지 (Historical) */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <div
                        style={{
                          flex: 1,
                          height: '4px',
                          background: 'rgba(255,255,255,0.08)',
                          borderRadius: '2px',
                          overflow: 'hidden',
                        }}
                      >
                        <div
                          style={{
                            width: `${pitch.pct}%`,
                            height: '100%',
                            background: '#8b949e',
                            opacity: 0.45,
                            borderRadius: '2px',
                          }}
                        />
                      </div>
                      <span style={{ fontSize: '8px', color: '#8b949e', width: '22px', textAlign: 'right' }}>
                        {pitch.pct.toFixed(0)}%
                      </span>
                    </div>

                    {/* 2. AI 예측% 게이지 */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <div
                        style={{
                          flex: 1,
                          height: '4px',
                          background: 'rgba(255,255,255,0.08)',
                          borderRadius: '2px',
                          overflow: 'hidden',
                        }}
                      >
                        <div
                          style={{
                            width: `${aiPct}%`,
                            height: '100%',
                            background: pitch.color,
                            boxShadow: `0 0 4px ${pitch.color}`,
                            borderRadius: '2px',
                          }}
                        />
                      </div>
                      <span
                        style={{
                          fontSize: '8px',
                          fontWeight: aiPct > 0.01 ? '700' : '400',
                          color: aiPct > 0.01 ? pitch.color : '#444',
                          width: '22px',
                          textAlign: 'right',
                        }}
                      >
                        {aiPct > 0.01 ? `${aiPct.toFixed(0)}%` : '0%'}
                      </span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </>
      )}
    </div>
  )
}
