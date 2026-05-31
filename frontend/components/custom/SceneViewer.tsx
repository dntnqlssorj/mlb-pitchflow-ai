'use client'
import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { PredictResponse, PITCH_COLORS, PitchType } from '@/lib/types'

// 구종별 궤적 제어점 [release, mid, home]
const PITCH_CURVES: Record<string, [number, number, number][]> = {
  FF: [[0, 1.8, -18], [0,    1.5,  -9], [0,    0.90, 0]],
  FA: [[0, 1.8, -18], [0,    1.5,  -9], [0,    0.88, 0]],
  SI: [[0, 1.8, -18], [-0.1, 1.3,  -9], [-0.2, 0.70, 0]],
  FC: [[0, 1.8, -18], [0.1,  1.5,  -9], [0.2,  0.85, 0]],
  SL: [[0, 1.8, -18], [0.2,  1.4,  -9], [0.4,  0.70, 0]],
  ST: [[0, 1.8, -18], [0.35, 1.4,  -9], [0.55, 0.75, 0]],
  CU: [[0, 1.8, -18], [0,    2.0,  -9], [0,    0.50, 0]],
  KC: [[0, 1.8, -18], [0,    1.9,  -9], [0,    0.55, 0]],
  CS: [[0, 1.8, -18], [0.1,  1.8,  -9], [0.15, 0.65, 0]],
  CH: [[0, 1.8, -18], [0,    1.3,  -9], [0,    0.60, 0]],
  FS: [[0, 1.8, -18], [0,    1.2,  -9], [0,    0.50, 0]],
  FO: [[0, 1.8, -18], [0.05, 1.4,  -9], [0.10, 0.75, 0]],
  KN: [[0.1, 1.8, -18], [-0.1, 1.5, -9], [0.05, 0.80, 0]],
  SV: [[0, 1.8, -18], [-0.2, 1.5,  -9], [-0.35, 0.90, 0]],
  EP: [[0, 1.8, -18], [0,    1.3,  -9], [0,    0.70, 0]],
  SC: [[0, 1.8, -18], [-0.15, 1.4, -9], [-0.2, 0.80, 0]],
  PO: [[0, 1.8, -18], [0,    1.6,  -9], [0.3,  0.90, 0]],
}

interface Props {
  result: PredictResponse | null
  pitcherHand?: 'R' | 'L'
}

export default function SceneViewer({ result, pitcherHand = 'R' }: Props) {
  const mountRef = useRef<HTMLDivElement>(null)
  const trajGroupRef = useRef<THREE.Group | null>(null)

  // ── Scene Setup (마운트 시 1회) ──────────────────────────
  useEffect(() => {
    if (!mountRef.current) return
    const container = mountRef.current
    const w = container.clientWidth
    const h = container.clientHeight

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(w, h)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.shadowMap.enabled = true
    renderer.outputColorSpace = THREE.SRGBColorSpace
    container.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x08090f)
    scene.fog = new THREE.FogExp2(0x08090f, 0.022)

    // 카메라 — 포수 시점
    const camera = new THREE.PerspectiveCamera(52, w / h, 0.1, 500)
    camera.position.set(0, 1.2, 9.5)
    camera.lookAt(0, 1.0, 0)

    // 조명
    scene.add(new THREE.AmbientLight(0xffffff, 1.8))
    const dir1 = new THREE.DirectionalLight(0xffffff, 2.5)
    dir1.position.set(4, 10, 6)
    dir1.castShadow = true
    scene.add(dir1)
    const dir2 = new THREE.DirectionalLight(0x8888ff, 1.0)
    dir2.position.set(-4, 4, 2)
    scene.add(dir2)

    // 그라운드
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(40, 40),
      new THREE.MeshStandardMaterial({ color: 0x1a2e1a, roughness: 0.9 })
    )
    ground.rotation.x = -Math.PI / 2
    ground.receiveShadow = true
    scene.add(ground)

    // 투수 마운드
    const mound = new THREE.Mesh(
      new THREE.CylinderGeometry(1.8, 1.8, 0.25, 32),
      new THREE.MeshStandardMaterial({ color: 0x6b4c2a, roughness: 0.95 })
    )
    mound.position.set(0, 0.12, -9)
    scene.add(mound)

    // 투수판 (rubber)
    const rubber = new THREE.Mesh(
      new THREE.BoxGeometry(0.5, 0.05, 0.15),
      new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5 })
    )
    rubber.position.set(0, 0.26, -8.8)
    scene.add(rubber)

    // 홈플레이트
    const hp = new THREE.Mesh(
      new THREE.BoxGeometry(0.5, 0.02, 0.4),
      new THREE.MeshStandardMaterial({ color: 0xe8e8e8, roughness: 0.4 })
    )
    hp.position.set(0, 0.01, 0.6)
    scene.add(hp)

    // 스트라이크존 (반투명 + 와이어)
    const szGeo = new THREE.BoxGeometry(0.43, 0.56, 0.02)
    const szMesh = new THREE.Mesh(szGeo, new THREE.MeshStandardMaterial({
      color: 0x4444ff, transparent: true, opacity: 0.15,
    }))
    szMesh.position.set(0, 0.93, 0.4)
    scene.add(szMesh)

    const szEdge = new THREE.LineSegments(
      new THREE.EdgesGeometry(szGeo),
      new THREE.LineBasicMaterial({ color: 0xffff00, opacity: 0.9, transparent: true, linewidth: 2 })
    )
    szEdge.position.set(0, 0.93, 0.4)
    scene.add(szEdge)

    // 베이스라인 점선
    const linePoints = [
      new THREE.Vector3(0, 0.02, -9),
      new THREE.Vector3(0, 0.02, 0.6),
    ]
    const lineMat = new THREE.LineDashedMaterial({ color: 0x333333, dashSize: 0.3, gapSize: 0.2 })
    const lineGeo = new THREE.BufferGeometry().setFromPoints(linePoints)
    const centerLine = new THREE.Line(lineGeo, lineMat)
    centerLine.computeLineDistances()
    scene.add(centerLine)

    // 궤적 그룹
    const trajGroup = new THREE.Group()
    scene.add(trajGroup)
    trajGroupRef.current = trajGroup

    // 리사이즈
    const onResize = () => {
      const nw = container.clientWidth, nh = container.clientHeight
      camera.aspect = nw / nh
      camera.updateProjectionMatrix()
      renderer.setSize(nw, nh)
    }
    window.addEventListener('resize', onResize)

    // 렌더 루프
    let raf: number
    const animate = () => {
      raf = requestAnimationFrame(animate)
      renderer.render(scene, camera)
    }
    animate()

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', onResize)
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  // ── 궤적 업데이트 ─────────────────────────────────────
  useEffect(() => {
    const grp = trajGroupRef.current
    if (!grp) return

    // 기존 궤적 제거
    while (grp.children.length > 0) {
      const c = grp.children[0]
      if (c instanceof THREE.Mesh) {
        c.geometry.dispose()
        if (Array.isArray(c.material)) c.material.forEach(m => m.dispose())
        else c.material.dispose()
      }
      grp.remove(c)
    }

    if (!result?.pitch_probabilities) return

    const sorted = Object.entries(result.pitch_probabilities)
      .filter(([, p]) => (p ?? 0) > 0.01)
      .sort(([, a], [, b]) => (b ?? 0) - (a ?? 0))
      .slice(0, 5)

    const opacities = [1.0, 0.65, 0.4, 0.25, 0.15]

    sorted.forEach(([pitch, prob], idx) => {
      const ctrl = PITCH_CURVES[pitch]
      if (!ctrl) return

      const isLeft = pitcherHand === 'L'
      const pts = ctrl.map(([x, y, z], i) => {
        const ratio = (z - (-18)) / (0 - (-18))
        const tz = -9 + ratio * (0.5 - (-9))
        if (i === 0) return new THREE.Vector3(isLeft ? -0.22 : 0.22, 1.45, tz)
        return new THREE.Vector3(x, y, tz)
      })

      const curve = new THREE.CatmullRomCurve3(pts)
      const isTop = idx === 0
      const radius = isTop ? 0.026 : 0.016
      const tubeGeo = new THREE.TubeGeometry(curve, 36, radius, 8, false)

      const color = PITCH_COLORS[pitch as PitchType] ?? '#888888'
      const colorHex = parseInt(color.replace('#', ''), 16)

      const tubeMat = new THREE.MeshStandardMaterial({
        color: colorHex,
        transparent: true,
        opacity: opacities[idx] ?? 0.15,
        roughness: 0.2,
        metalness: 0.1,
        emissive: colorHex,
        emissiveIntensity: isTop ? 0.3 : 0.1,
      })
      grp.add(new THREE.Mesh(tubeGeo, tubeMat))

      // Top-1 궤적에 야구공 애니메이션
      if (isTop) {
        const ballGeo = new THREE.SphereGeometry(0.042, 16, 16)
        const ballMat = new THREE.MeshStandardMaterial({
          color: 0xfefefe, roughness: 0.8, metalness: 0.0,
        })
        const ball = new THREE.Mesh(ballGeo, ballMat)
        ball.position.copy(pts[Math.floor(pts.length / 2)] ?? pts[1])
        grp.add(ball)

        // 구종 레이블 (스프라이트 텍스트 대신 간단 구)
        const labelGeo = new THREE.SphereGeometry(0.015, 8, 8)
        const labelMat = new THREE.MeshStandardMaterial({ color: colorHex, emissive: colorHex, emissiveIntensity: 1 })
        const label = new THREE.Mesh(labelGeo, labelMat)
        label.position.copy(pts[0])
        grp.add(label)
      }
    })
  }, [result, pitcherHand])

  // 범례 (확률 기준 동적 생성)
  const legendItems = result?.pitch_probabilities
    ? Object.entries(result.pitch_probabilities)
        .filter(([, p]) => (p ?? 0) > 0.01)
        .sort(([, a], [, b]) => (b ?? 0) - (a ?? 0))
        .slice(0, 5)
    : []

  return (
    <div ref={mountRef} style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden' }}>
      {/* 좌상단 범례 */}
      {legendItems.length > 0 && (
        <div style={{
          position: 'absolute', top: '12px', left: '12px', zIndex: 10,
          background: 'rgba(8,9,15,0.85)', border: '1px solid #1e1e1e',
          borderRadius: '6px', padding: '10px 12px',
          display: 'flex', flexDirection: 'column', gap: '6px',
          backdropFilter: 'blur(8px)',
        }}>
          <div style={{ fontSize: '9px', fontWeight: '700', color: '#555', letterSpacing: '0.12em', marginBottom: '2px' }}>
            TRAJECTORY
          </div>
          {legendItems.map(([pitch, prob]) => (
            <div key={pitch} style={{ display: 'flex', alignItems: 'center', gap: '7px' }}>
              <div style={{
                width: '8px', height: '8px', borderRadius: '50%',
                background: PITCH_COLORS[pitch as PitchType] ?? '#888',
                flexShrink: 0,
              }} />
              <span style={{ fontSize: '11px', color: '#ccc', fontWeight: pitch === result?.predicted_pitch ? '700' : '400' }}>
                {pitch}
              </span>
              <span style={{ fontSize: '11px', color: '#666', marginLeft: 'auto', paddingLeft: '12px' }}>
                {((prob ?? 0) * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      )}

      {/* 우상단 — 카메라 시점 안내 */}
      <div style={{
        position: 'absolute', top: '12px', right: '12px', zIndex: 10,
        fontSize: '9px', color: '#333', letterSpacing: '0.1em',
      }}>
        CATCHER VIEW
      </div>
    </div>
  )
}
