'use client'
import { useState, useEffect, useRef } from 'react'

interface PlayerResult {
  id: number
  fullName: string
  primaryPosition?: string
  currentTeam?: string
}

interface Props {
  label: string
  placeholder: string
  value: string
  onChange: (id: string, name: string) => void
}

export default function PlayerSearch({ label, placeholder, value, onChange }: Props) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<PlayerResult[]>([])
  const [loading, setLoading] = useState(false)
  const [open, setOpen] = useState(false)
  const [selectedName, setSelectedName] = useState('')
  const ref = useRef<HTMLDivElement>(null)

  // 외부 클릭 시 드롭다운 닫기
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  // 디바운스 검색 (300ms)
  useEffect(() => {
    if (query.length < 2) { setResults([]); setOpen(false); return }
    const timer = setTimeout(async () => {
      setLoading(true)
      try {
        const res = await fetch(
          `https://statsapi.mlb.com/api/v1/people/search?names=${encodeURIComponent(query)}&sportId=1&hydrate=currentTeam`
        )
        const data = await res.json()
        const players: PlayerResult[] = (data.people ?? []).slice(0, 8).map((p: {
          id: number
          fullName: string
          primaryPosition?: { abbreviation: string }
          currentTeam?: { abbreviation: string }
        }) => ({
          id: p.id,
          fullName: p.fullName,
          primaryPosition: p.primaryPosition?.abbreviation,
          currentTeam: p.currentTeam?.abbreviation,
        }))
        setResults(players)
        setOpen(players.length > 0)
      } catch {
        setResults([])
        setOpen(false)
      } finally {
        setLoading(false)
      }
    }, 300)
    return () => clearTimeout(timer)
  }, [query])

  const handleSelect = (player: PlayerResult) => {
    setSelectedName(player.fullName)
    setQuery(player.fullName)
    onChange(String(player.id), player.fullName)
    setOpen(false)
    setResults([])
  }

  const handleClear = () => {
    setQuery('')
    setSelectedName('')
    onChange('', '')
    setResults([])
    setOpen(false)
  }

  const isSelected = !!value

  return (
    <div ref={ref} style={{ position: 'relative' }}>
      {/* 레이블 */}
      <div style={{
        fontSize: '10px', fontWeight: '700', letterSpacing: '0.1em',
        color: '#6b7280', marginBottom: '5px',
      }}>
        {label}
      </div>

      {/* 입력 필드 */}
      <div style={{ position: 'relative' }}>
        <input
          type="text"
          value={query}
          placeholder={placeholder}
          onChange={e => {
            setQuery(e.target.value)
            if (selectedName) setSelectedName('')
          }}
          onFocus={() => results.length > 0 && setOpen(true)}
          style={{
            width: '100%',
            padding: '7px 32px 7px 10px',
            background: isSelected ? '#0d1a0d' : '#0d0d0d',
            border: `1px solid ${isSelected ? '#22C55E44' : '#2a2a2a'}`,
            color: '#fff',
            borderRadius: '3px',
            fontSize: '13px',
            outline: 'none',
            transition: 'border-color 0.15s, background 0.15s',
          }}
        />
        {/* 우측 아이콘 — 선택됨: 초록 체크 / 로딩: 점 / 없음: 숨김 */}
        <div
          onClick={isSelected ? handleClear : undefined}
          style={{
            position: 'absolute', right: '8px', top: '50%',
            transform: 'translateY(-50%)',
            fontSize: '11px', fontWeight: '700',
            cursor: isSelected ? 'pointer' : 'default',
            color: isSelected ? '#22C55E' : loading ? '#555' : 'transparent',
            userSelect: 'none',
          }}
          title={isSelected ? '지우기' : ''}
        >
          {isSelected ? '✕' : loading ? '●' : ''}
        </div>
      </div>

      {/* 선택된 ID 표시 */}
      {isSelected && (
        <div style={{ fontSize: '10px', color: '#22C55E', marginTop: '3px', opacity: 0.7 }}>
          ID: {value}
        </div>
      )}

      {/* 드롭다운 */}
      {open && results.length > 0 && (
        <div style={{
          position: 'absolute', top: 'calc(100% + 2px)', left: 0, right: 0, zIndex: 999,
          background: '#1a1a1a', border: '1px solid #333', borderRadius: '4px',
          maxHeight: '240px', overflowY: 'auto',
          boxShadow: '0 8px 32px rgba(0,0,0,0.7)',
        }}>
          {results.map((player, idx) => (
            <div
              key={player.id}
              onClick={() => handleSelect(player)}
              style={{
                padding: '9px 12px', cursor: 'pointer',
                borderBottom: idx < results.length - 1 ? '1px solid #222' : 'none',
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              }}
              onMouseEnter={e => { (e.currentTarget as HTMLDivElement).style.background = '#242424' }}
              onMouseLeave={e => { (e.currentTarget as HTMLDivElement).style.background = 'transparent' }}
            >
              <div>
                <div style={{ fontSize: '13px', fontWeight: '600', color: '#fff', lineHeight: 1.2 }}>
                  {player.fullName}
                </div>
                <div style={{ fontSize: '10px', color: '#6b7280', marginTop: '3px' }}>
                  {[player.primaryPosition, player.currentTeam ?? 'FA', `#${player.id}`]
                    .filter(Boolean).join(' · ')}
                </div>
              </div>
              <div style={{
                fontSize: '10px', color: '#333', fontFamily: "'Bebas Neue', Arial",
                letterSpacing: '0.05em', flexShrink: 0, paddingLeft: '8px',
              }}>
                {player.id}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
