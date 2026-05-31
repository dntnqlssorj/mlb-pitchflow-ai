'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

export default function TabNav() {
  const path = usePathname()

  return (
    <header style={{
      position: 'fixed', top: 0, left: 0, right: 0, zIndex: 100,
      background: '#0d0d0d', borderBottom: '1px solid #333',
      display: 'flex', alignItems: 'center', height: '48px',
      padding: '0 16px', gap: '0',
    }}>
      {/* 로고 */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '6px',
        fontFamily: "'Bebas Neue', 'Arial Narrow', Arial, sans-serif",
        fontSize: '20px', letterSpacing: '0.1em', color: '#fff',
        paddingRight: '24px',
      }}>
        <span style={{ color: '#D50032' }}>MLB</span>
        <span>PitchFlow AI</span>
      </div>

      {/* 구분선 */}
      <div style={{ width: '1px', height: '24px', background: '#333', marginRight: '24px' }} />

      {/* 탭 */}
      <nav style={{ display: 'flex', gap: '0', height: '48px' }}>
        {[
          { href: '/mlb', label: 'LIVE' },
          { href: '/custom', label: 'CUSTOM' },
        ].map(({ href, label }) => {
          const active = path.startsWith(href)
          return (
            <Link key={href} href={href} style={{
              padding: '0 18px',
              height: '48px',
              display: 'flex', alignItems: 'center',
              fontSize: '12px', fontWeight: '700',
              letterSpacing: '0.1em',
              color: active ? '#fff' : '#9ca3af',
              borderBottom: active ? '2px solid #D50032' : '2px solid transparent',
              textDecoration: 'none',
              transition: 'color 0.15s',
            }}>
              {label}
            </Link>
          )
        })}
      </nav>

      {/* 우측 상태 표시 */}
      <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <div style={{
          width: '6px', height: '6px', borderRadius: '50%',
          background: '#22C55E', boxShadow: '0 0 6px #22C55E',
        }} />
        <span style={{ fontSize: '11px', color: '#6b7280', letterSpacing: '0.05em' }}>
          API ONLINE
        </span>
      </div>
    </header>
  )
}
