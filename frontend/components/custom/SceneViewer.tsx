'use client'
import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { PredictResponse, PITCH_COLORS, PitchType, ArsenalPitch } from '@/lib/types'
import ArsenalHUD from './ArsenalHUD'
import { PITCH_CURVES as LIB_PITCH_CURVES } from '@/lib/pitchCurves'

// 스트라이크 존 규격 상수
const ZONE_WIDTH  = 0.432   // m (17인치)
const ZONE_HEIGHT = 0.610   // m (평균 sz_top - sz_bot)
const ZONE_CENTER_X = 0
const ZONE_CENTER_Y = 1.0   // m (지면 기준 존 중심 높이)
const ZONE_CENTER_Z = LIB_PITCH_CURVES['FF']?.[2]?.z ?? 0.0   // m (홈플레이트 위치 - pitchCurves 실제값 직접 참조)

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
  pitcherId?: number | null
  year?: number
}

export default function SceneViewer({ result, pitcherHand = 'R', pitcherId = null, year = 2026 }: Props) {
  const mountRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const trajGroupRef = useRef<THREE.Group | null>(null)

  // 레퍼토리 및 상호작용 상태
  const [arsenal, setArsenal] = useState<ArsenalPitch[]>([])
  const [hoveredPitch, setHoveredPitch] = useState<string | null>(null)
  const [visiblePitches, setVisiblePitches] = useState<Set<string>>(new Set())

  // ── 레퍼토리 API 연동 ──────────────────────────
  useEffect(() => {
    if (!pitcherId) {
      setArsenal([])
      setVisiblePitches(new Set())
      return
    }
    fetch(`http://localhost:8000/api/pitcher-arsenal?pitcherId=${pitcherId}&year=${year}`)
      .then(res => {
        if (!res.ok) throw new Error("Arsenal not found")
        return res.json()
      })
      .then(data => {
        const list: ArsenalPitch[] = data.arsenal || []
        setArsenal(list)
        setVisiblePitches(new Set(list.map(p => p.pitch_type)))
      })
      .catch(() => {
        setArsenal([])
        setVisiblePitches(new Set())
      })
  }, [pitcherId, year])

  // ── Scene Setup (마운트 시 1회) ──────────────────────────
  useEffect(() => {
    if (!mountRef.current || !canvasRef.current) return
    const container = mountRef.current
    const canvas = canvasRef.current
    const w = container.clientWidth
    const h = container.clientHeight

    const renderer = new THREE.WebGLRenderer({
      canvas: canvas,
      antialias: true,
      alpha: true
    })
    renderer.setSize(w, h)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.shadowMap.enabled = true
    renderer.outputColorSpace = THREE.SRGBColorSpace

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x08090f)
    scene.fog = new THREE.FogExp2(0x08090f, 0.022)

    // (A) 카메라 재조정 (Catcher POV - 홈플레이트 뒤에서 마운트 방향 조망)
    const camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 500)
    if (ZONE_CENTER_Z > 0) {
      camera.position.set(0, 1.2, -2.0)
      camera.lookAt(0, 1.0, 10)
    } else {
      camera.position.set(0, 1.2, 2.0)
      camera.lookAt(0, 1.0, -10)
    }

    // 기존 OrbitControls 설정 및 target 변경
    const { OrbitControls } = require('three/examples/jsm/controls/OrbitControls.js')
    const controls = new OrbitControls(camera, renderer.domElement)
    if (ZONE_CENTER_Z > 0) {
      controls.target.set(0, 1.0, 10)
      camera.position.set(0, 1.2, -2.0)
    } else {
      controls.target.set(0, 1.0, -10)
      camera.position.set(0, 1.2, 2.0)
    }
    controls.update()

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
    mound.position.set(0, 0.12, -9.0) // 투수 플레이트 쪽으로 (z = -9.0)
    scene.add(mound)

    // (B) 스트라이크 존 9분할 그리드 (대칭 3등분 재계산)
    const szGroup = new THREE.Group()

    // 뒷배경 반투명 판
    const bgGeo = new THREE.PlaneGeometry(ZONE_WIDTH, ZONE_HEIGHT)
    const bgMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.07,
      side: THREE.DoubleSide
    })
    const bgMesh = new THREE.Mesh(bgGeo, bgMat)
    bgMesh.position.set(ZONE_CENTER_X, ZONE_CENTER_Y, ZONE_CENTER_Z) // 홈플레이트 평면 z = 0.0
    szGroup.add(bgMesh)

    // 9분할 그리드 계산식
    const x1 = ZONE_CENTER_X - ZONE_WIDTH / 6     // -0.072
    const x2 = ZONE_CENTER_X + ZONE_WIDTH / 6     // +0.072
    const y_bottom = ZONE_CENTER_Y - ZONE_HEIGHT / 2  // 0.695
    const y_top    = ZONE_CENTER_Y + ZONE_HEIGHT / 2  // 1.305
    const y1 = ZONE_CENTER_Y - ZONE_HEIGHT / 6     // 0.898
    const y2 = ZONE_CENTER_Y + ZONE_HEIGHT / 6     // 1.102

    // 그리드 구분선 (내부 격자선)
    const lineMat = new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.35
    })
    const gridPoints = [
      // 수직선 1 (x = x1)
      new THREE.Vector3(x1, y_bottom, ZONE_CENTER_Z),
      new THREE.Vector3(x1, y_top, ZONE_CENTER_Z),
      // 수직선 2 (x = x2)
      new THREE.Vector3(x2, y_bottom, ZONE_CENTER_Z),
      new THREE.Vector3(x2, y_top, ZONE_CENTER_Z),
      // 수평선 1 (y = y1)
      new THREE.Vector3(-ZONE_WIDTH / 2, y1, ZONE_CENTER_Z),
      new THREE.Vector3(ZONE_WIDTH / 2, y1, ZONE_CENTER_Z),
      // 수평선 2 (y = y2)
      new THREE.Vector3(-ZONE_WIDTH / 2, y2, ZONE_CENTER_Z),
      new THREE.Vector3(ZONE_WIDTH / 2, y2, ZONE_CENTER_Z)
    ]
    const gridGeo = new THREE.BufferGeometry().setFromPoints(gridPoints)
    const gridLines = new THREE.LineSegments(gridGeo, lineMat)
    szGroup.add(gridLines)

    // 외곽 테두리 (opacity 0.6)
    const borderMat = new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.6
    })
    const borderPoints = [
      // 하단
      new THREE.Vector3(-ZONE_WIDTH / 2, y_bottom, ZONE_CENTER_Z),
      new THREE.Vector3(ZONE_WIDTH / 2, y_bottom, ZONE_CENTER_Z),
      // 상단
      new THREE.Vector3(-ZONE_WIDTH / 2, y_top, ZONE_CENTER_Z),
      new THREE.Vector3(ZONE_WIDTH / 2, y_top, ZONE_CENTER_Z),
      // 좌측
      new THREE.Vector3(-ZONE_WIDTH / 2, y_bottom, ZONE_CENTER_Z),
      new THREE.Vector3(-ZONE_WIDTH / 2, y_top, ZONE_CENTER_Z),
      // 우측
      new THREE.Vector3(ZONE_WIDTH / 2, y_bottom, ZONE_CENTER_Z),
      new THREE.Vector3(ZONE_WIDTH / 2, y_top, ZONE_CENTER_Z)
    ]
    const borderGeo = new THREE.BufferGeometry().setFromPoints(borderPoints)
    const borderLines = new THREE.LineSegments(borderGeo, borderMat)
    szGroup.add(borderLines)

    scene.add(szGroup)

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
      controls.update()
      renderer.render(scene, camera)
    }
    animate()

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', onResize)
      renderer.dispose()
    }
  }, [])

  // ── 궤적 & 착탄 지점 렌더링 업데이트 (arsenal, hoveredPitch, visiblePitches 기반) ───────────────────────
  useEffect(() => {
    const grp = trajGroupRef.current
    if (!grp) return

    // 기존 자식 요소 일괄 정리
    while (grp.children.length > 0) {
      const c = grp.children[0]
      if (c instanceof THREE.Group) {
        c.children.forEach(mesh => {
          if (mesh instanceof THREE.Mesh) {
            mesh.geometry.dispose()
            if (Array.isArray(mesh.material)) mesh.material.forEach(m => m.dispose())
            else mesh.material.dispose()
          }
        })
      } else if (c instanceof THREE.Mesh) {
        c.geometry.dispose()
        if (Array.isArray(c.material)) c.material.forEach(m => m.dispose())
        else c.material.dispose()
      }
      grp.remove(c)
    }

    if (!arsenal || arsenal.length === 0) return

    const anyHovered = hoveredPitch !== null
    const predictedPitch = result?.predicted_pitch
    const hasPrediction = predictedPitch !== undefined && predictedPitch !== null

    arsenal.forEach((pitch, idx) => {
      const pt = pitch.pitch_type
      const ctrl = PITCH_CURVES[pt] || PITCH_CURVES['FF']
      if (!ctrl) return

      // 가시성 필터
      const isVisible = visiblePitches.has(pt)

      // 호버 및 예측 매칭 조건
      const isHovered = hoveredPitch === pt
      const isPredicted = predictedPitch === pt
      
      // 1. tubularRadius 설정
      let radius = idx === 0 ? 0.055 : 0.04
      if (isHovered) {
        radius = 0.07
      }

      // 2. 머티리얼 속성 분기 (호버 상호작용 우선, 예측 강조 차선 적용)
      let tubeOpacity = 1.0
      let tubeEmissiveIntensity = 0.4
      let sphereOpacity = 1.0
      let sphereEmissiveIntensity = 0.25
      let sphereScale = 1.0

      if (anyHovered) {
        if (isHovered) {
          tubeOpacity = 1.0
          tubeEmissiveIntensity = 1.2
          sphereOpacity = 1.0
          sphereEmissiveIntensity = 0.85
          sphereScale = 1.4
        } else {
          tubeOpacity = 0.15
          tubeEmissiveIntensity = 0.1
          sphereOpacity = 0.2
          sphereEmissiveIntensity = 0.1
          sphereScale = 1.0
        }
      } else if (hasPrediction) {
        if (isPredicted) {
          tubeOpacity = 1.0
          tubeEmissiveIntensity = 1.2
          sphereOpacity = 1.0
          sphereEmissiveIntensity = 0.85
          sphereScale = 1.4
        } else {
          tubeOpacity = 0.3
          tubeEmissiveIntensity = 0.1
          sphereOpacity = 0.3
          sphereEmissiveIntensity = 0.1
          sphereScale = 1.0
        }
      }

      const colorHex = parseInt(pitch.color.replace('#', ''), 16)

      // (C) 착탄 SphereGeometry 렌더링 (옵션 A: 크림색 리얼 야구공 + 구종별 실밥 링 2개)
      // 착탄 야구공은 예측 완료 여부(result)에 상관없이 arsenal이 존재하면 항상 상시 렌더링함!
      const hasAvg = pitch.avg_plate_x !== null && pitch.avg_plate_z !== null
      if (hasAvg) {
        const ballGroup = new THREE.Group()
        
        // ft -> m 변환 적용 (z는 ZONE_CENTER_Z 고정)
        const sx = pitch.avg_plate_x! * 0.3048
        const sy = pitch.avg_plate_z! * 0.3048
        const sz = ZONE_CENTER_Z
        ballGroup.position.set(sx, sy, sz)
        ballGroup.scale.set(sphereScale, sphereScale, sphereScale)
        ballGroup.visible = isVisible

        // A-1. 야구공 가죽 바디 (크림색 바탕 + 구종 광원 방사)
        const sphereGeo = new THREE.SphereGeometry(0.037, 16, 16)
        const sphereMat = new THREE.MeshStandardMaterial({
          color: 0xF5F0E8,           // 크림색 야구공 본체
          roughness: 0.85,           // 가죽 가공 거칠기
          metalness: 0.0,
          transparent: true,
          opacity: sphereOpacity,
          emissive: colorHex,        // 구종별 오라 방출
          emissiveIntensity: sphereEmissiveIntensity
        })
        const sphereMesh = new THREE.Mesh(sphereGeo, sphereMat)
        ballGroup.add(sphereMesh)

        // A-2. 리얼 가죽 실밥 느낌을 내는 구종 컬러의 얇은 링 2개
        const seamMat = new THREE.MeshStandardMaterial({
          color: colorHex,
          roughness: 0.5,
          transparent: true,
          opacity: sphereOpacity * 0.9,
          emissive: colorHex,
          emissiveIntensity: sphereEmissiveIntensity * 1.2
        })

        const ringGeo1 = new THREE.TorusGeometry(0.037, 0.003, 6, 24)
        const ring1 = new THREE.Mesh(ringGeo1, seamMat)
        ring1.rotation.set(Math.PI / 6, Math.PI / 6, 0)
        ballGroup.add(ring1)

        const ringGeo2 = new THREE.TorusGeometry(0.037, 0.003, 6, 24)
        const ring2 = new THREE.Mesh(ringGeo2, seamMat)
        ring2.rotation.set(-Math.PI / 6, -Math.PI / 6, 0)
        ballGroup.add(ring2)

        grp.add(ballGroup)
      }

      // (D) TubeGeometry 3번째 제어점 동적 교체 (ZONE_CENTER_Z 수렴)
      const isLeft = pitcherHand === 'L'
      const startPt = new THREE.Vector3(isLeft ? -0.22 : 0.22, 1.45, -9.0)
      const midPt = new THREE.Vector3(ctrl[1][0], ctrl[1][1], -4.5) // -9.0 의 정확한 절반 비율로 수정
      
      const endPt = hasAvg
        ? new THREE.Vector3(pitch.avg_plate_x! * 0.3048, pitch.avg_plate_z! * 0.3048, ZONE_CENTER_Z)
        : new THREE.Vector3(ctrl[2][0], ctrl[2][1], ZONE_CENTER_Z) // fallback

      const curve = new THREE.CatmullRomCurve3([startPt, midPt, endPt])
      const tubeGeo = new THREE.TubeGeometry(curve, 32, radius, 8, false)

      const tubeMat = new THREE.MeshStandardMaterial({
        color: colorHex,
        transparent: true,
        opacity: tubeOpacity,
        roughness: 0.2,
        metalness: 0.1,
        emissive: colorHex,
        emissiveIntensity: tubeEmissiveIntensity,
      })

      const tubeMesh = new THREE.Mesh(tubeGeo, tubeMat)
      tubeMesh.visible = isVisible
      grp.add(tubeMesh)
    })
  }, [arsenal, hoveredPitch, visiblePitches, pitcherHand, result])

  const predictedProbabilities = result?.pitch_probabilities || {}

  const handleVisibilityToggle = (pitchType: string) => {
    setVisiblePitches(prev => {
      const next = new Set(prev)
      if (next.has(pitchType)) {
        next.delete(pitchType)
      } else {
        next.add(pitchType)
      }
      return next
    })
  }

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden' }}>
      {/* 3D WebGL 렌더링 캔버스 홀더 */}
      <div ref={mountRef} style={{ width: '100%', height: '100%', position: 'absolute', inset: 0, zIndex: 0 }}>
        <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
      </div>

      {/* arsenal 데이터 미로딩 시 안내 오버레이 텍스트 및 기본 야구공(이모지) 조건부 렌더링 */}
      {arsenal.length === 0 && (
        <div style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'rgba(255, 255, 255, 0.85)',
          pointerEvents: 'none',
          gap: '8px',
          textShadow: '0 1px 4px rgba(0,0,0,0.9)',
          zIndex: 5,
        }}>
          <div style={{ fontSize: '40px' }}>⚾</div>
          <div style={{
            fontSize: '13px',
            letterSpacing: '0.1em',
            fontFamily: "'Bebas Neue', Arial",
            color: 'rgba(255, 255, 255, 0.85)',
          }}>
            PITCH TRAJECTORY VIEWER
          </div>
          <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.45)' }}>
            투수 / 타자 ID 입력 후 PREDICT PITCH
          </div>
        </div>
      )}

      {/* (TASK 6) ArsenalHUD 내부 절대 레이아웃 오버레이 장착 */}
      {arsenal.length > 0 && (
        <div style={{
          position: 'absolute',
          top: '12px',
          left: '12px',
          zIndex: 10,
          pointerEvents: 'auto'
        }}>
          <ArsenalHUD
            arsenal={arsenal}
            predictedProbabilities={predictedProbabilities as Record<string, number>}
            hoveredPitch={hoveredPitch}
            visiblePitches={visiblePitches}
            onHoverChange={setHoveredPitch}
            onVisibilityToggle={handleVisibilityToggle}
          />
        </div>
      )}

      {/* 우상단 — 카메라 시점 안내 오버레이 레이블 개선 */}
      <div style={{
        position: 'absolute',
        top: '12px',
        right: '12px',
        zIndex: 10,
        color: '#ffffff',
        fontSize: '11px',
        letterSpacing: '0.08em',
        fontWeight: 'bold',
        textShadow: '0 1px 6px rgba(0,0,0,1.0)',
        background: 'rgba(0,0,0,0.45)',
        padding: '3px 8px',
        borderRadius: '4px',
        border: '1px solid rgba(255,255,255,0.06)'
      }}>
        CATCHER POV (9-GRID ACTIVE)
      </div>
    </div>
  )
}
