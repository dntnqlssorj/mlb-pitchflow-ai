'use client';

import React, { useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';
import * as THREE from 'three';
import { PredictResponse } from '../../lib/types';
import { PITCH_CURVES, PITCH_COLORS } from '../../lib/pitchCurves';

interface SceneViewerProps {
  pitcherTeamId: string;
  batterTeamId: string;
  pitcherHand: 'R' | 'L';
  batterSide: 'R' | 'L';
  predictResult: PredictResponse | null;
}

const teamColors: Record<string, string> = {
  NYY: '#003087', BOS: '#BD3039', LAD: '#005A9C',
  SF: '#FD5A1E', CHC: '#0E3386', HOU: '#002D62',
  ATL: '#CE1141', NYM: '#002D72', PHI: '#E81828',
  SD: '#2F241D', SEA: '#0C2C56', TOR: '#134A8E',
  MIN: '#002B5C', CLE: '#00385D', TEX: '#003278',
  TB: '#092C5C', BAL: '#DF4601', DET: '#0C2340',
  KC: '#004687', CWS: '#27251F', MIL: '#12284B',
  STL: '#C41E3A', CIN: '#C6011F', PIT: '#27251F',
  MIA: '#00A3E0', WSH: '#AB0003', COL: '#33006F',
  ARI: '#A71930', LAA: '#003263', OAK: '#003831'
};

// 1. 투수 stick figure 생성 함수
function createPitcherFigure(color: string, pitcherHand: 'R' | 'L'): THREE.Group {
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({ color: new THREE.Color(color), roughness: 0.5 });
  const gloveMaterial = new THREE.MeshStandardMaterial({ color: 0x8B4513, roughness: 0.8 });

  // 머리
  const headGeo = new THREE.SphereGeometry(0.12, 16, 16);
  const head = new THREE.Mesh(headGeo, material);
  head.position.set(0, 1.7, 0);
  head.castShadow = true;
  group.add(head);

  // 몸통
  const torsoGeo = new THREE.CylinderGeometry(0.06, 0.06, 0.5, 8);
  const torso = new THREE.Mesh(torsoGeo, material);
  torso.position.set(0, 1.3, 0);
  torso.castShadow = true;
  group.add(torso);

  // 어깨 (가로선)
  const shouldersGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.5, 8);
  const shoulders = new THREE.Mesh(shouldersGeo, material);
  shoulders.rotation.z = Math.PI / 2;
  shoulders.position.set(0, 1.5, 0);
  shoulders.castShadow = true;
  group.add(shoulders);

  const isLeft = pitcherHand === 'L';

  // 투구 팔 (우투 기준 — 우측 위로)
  const pitchArmGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.4, 8);
  const pitchArm = new THREE.Mesh(pitchArmGeo, material);
  pitchArm.rotation.z = isLeft ? Math.PI / 4 : -Math.PI / 4;
  pitchArm.position.set(isLeft ? -0.22 : 0.22, 1.42, 0);
  pitchArm.castShadow = true;
  group.add(pitchArm);

  // 글러브 팔 (좌측 앞으로)
  const gloveArmGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.35, 8);
  const gloveArm = new THREE.Mesh(gloveArmGeo, material);
  gloveArm.rotation.z = isLeft ? -Math.PI / 3 : Math.PI / 3;
  gloveArm.position.set(isLeft ? 0.18 : -0.18, 1.45, 0);
  gloveArm.castShadow = true;
  group.add(gloveArm);

  // 글러브
  const gloveGeo = new THREE.SphereGeometry(0.07, 8, 8);
  const glove = new THREE.Mesh(gloveGeo, gloveMaterial);
  glove.position.set(isLeft ? 0.32 : -0.32, 1.6, 0);
  glove.castShadow = true;
  group.add(glove);

  // 허리
  const hipsGeo = new THREE.CylinderGeometry(0.05, 0.05, 0.3, 8);
  const hips = new THREE.Mesh(hipsGeo, material);
  hips.position.set(0, 0.95, 0);
  hips.castShadow = true;
  group.add(hips);

  // 왼다리
  const leftLegGeo = new THREE.CylinderGeometry(0.05, 0.05, 0.55, 8);
  const leftLeg = new THREE.Mesh(leftLegGeo, material);
  leftLeg.rotation.z = Math.PI / 12;
  leftLeg.position.set(-0.1, 0.6, 0);
  leftLeg.castShadow = true;
  group.add(leftLeg);

  // 오른다리 (들린 발)
  const rightLegGeo = new THREE.CylinderGeometry(0.05, 0.05, 0.45, 8);
  const rightLeg = new THREE.Mesh(rightLegGeo, material);
  rightLeg.rotation.z = -Math.PI / 4;
  rightLeg.rotation.x = Math.PI / 6;
  rightLeg.position.set(0.15, 0.7, 0.1);
  rightLeg.castShadow = true;
  group.add(rightLeg);

  return group;
}

// 2. 타자 stick figure 생성 함수
function createBatterFigure(color: string, side: 'R' | 'L'): THREE.Group {
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({ color: new THREE.Color(color), roughness: 0.5 });
  const helmetMaterial = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.9 });
  const batMaterial = new THREE.MeshStandardMaterial({ color: 0x8B4513, roughness: 0.7 });

  // 머리
  const headGeo = new THREE.SphereGeometry(0.13, 16, 16);
  const head = new THREE.Mesh(headGeo, material);
  head.position.set(0, 1.7, 0);
  head.castShadow = true;
  group.add(head);

  // 헬멧 (납작한 반구)
  const helmetGeo = new THREE.SphereGeometry(0.15, 16, 8);
  const helmet = new THREE.Mesh(helmetGeo, helmetMaterial);
  helmet.scale.y = 0.7;
  helmet.position.set(0, 1.78, 0);
  helmet.castShadow = true;
  group.add(helmet);

  // 몸통
  const torsoGeo = new THREE.CylinderGeometry(0.08, 0.07, 0.5, 8);
  const torso = new THREE.Mesh(torsoGeo, material);
  torso.position.set(0, 1.3, 0);
  torso.castShadow = true;
  group.add(torso);

  // 어깨
  const shouldersGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.55, 8);
  const shoulders = new THREE.Mesh(shouldersGeo, material);
  shoulders.rotation.z = Math.PI / 2;
  shoulders.position.set(0, 1.52, 0);
  shoulders.castShadow = true;
  group.add(shoulders);

  // 양팔 (배트 잡는 자세 — 우타 기준)
  // 왼팔
  const leftArmGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.4, 8);
  const leftArm = new THREE.Mesh(leftArmGeo, material);
  leftArm.rotation.z = Math.PI / 5;
  leftArm.rotation.x = -Math.PI / 6;
  leftArm.position.set(-0.22, 1.42, -0.1);
  leftArm.castShadow = true;
  group.add(leftArm);

  // 오른팔
  const rightArmGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.38, 8);
  const rightArm = new THREE.Mesh(rightArmGeo, material);
  rightArm.rotation.z = Math.PI / 4;
  rightArm.position.set(0.2, 1.48, 0);
  rightArm.castShadow = true;
  group.add(rightArm);

  // 배트
  const batGeo = new THREE.CylinderGeometry(0.025, 0.015, 0.8, 8);
  const bat = new THREE.Mesh(batGeo, batMaterial);
  bat.rotation.z = Math.PI / 5;
  bat.rotation.x = Math.PI / 8;
  bat.position.set(-0.05, 1.85, -0.2);
  bat.castShadow = true;
  group.add(bat);

  // 하체
  const hipsGeo = new THREE.CylinderGeometry(0.06, 0.06, 0.3, 8);
  const hips = new THREE.Mesh(hipsGeo, material);
  hips.position.set(0, 0.95, 0);
  hips.castShadow = true;
  group.add(hips);

  // 왼다리
  const leftLegGeo = new THREE.CylinderGeometry(0.05, 0.05, 0.55, 8);
  const leftLeg = new THREE.Mesh(leftLegGeo, material);
  leftLeg.rotation.z = Math.PI / 10;
  leftLeg.position.set(-0.12, 0.62, 0);
  leftLeg.castShadow = true;
  group.add(leftLeg);

  // 오른다리
  const rightLegGeo = new THREE.CylinderGeometry(0.05, 0.05, 0.55, 8);
  const rightLeg = new THREE.Mesh(rightLegGeo, material);
  rightLeg.rotation.z = -Math.PI / 10;
  rightLeg.position.set(0.12, 0.62, 0);
  rightLeg.castShadow = true;
  group.add(rightLeg);

  // 좌타일 경우 전체 그룹 Y축 180도 회전
  if (side === 'L') {
    group.rotation.y = Math.PI;
  }

  return group;
}

function SceneViewerInner({
  pitcherTeamId,
  batterTeamId,
  pitcherHand,
  batterSide,
  predictResult,
}: SceneViewerProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // THREE scene references for figures
  const sceneRef = useRef<THREE.Scene | null>(null);
  const trajectoriesGroupRef = useRef<THREE.Group | null>(null);
  
  const pitcherModelRef = useRef<THREE.Group | null>(null);
  const batterModelRef = useRef<THREE.Group | null>(null);

  // Resolve team colors (default to '#555566' if empty or missing)
  const pitcherColor = teamColors[pitcherTeamId] || '#555566';
  const batterColor = teamColors[batterTeamId] || '#555566';

  // 1. Initial Scene Setup
  useEffect(() => {
    if (!mountRef.current || !canvasRef.current) return;

    const container = mountRef.current;
    const canvas = canvasRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Create Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0f1d);
    scene.fog = new THREE.FogExp2(0x0a0f1d, 0.025);
    sceneRef.current = scene;

    // Calibrated Camera: position (0, 2.0, 6), lookAt (0, 1.0, -3), fov: 45
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 2.0, 6);
    camera.lookAt(0, 1.0, -3);

    // Create WebGL Renderer drawing to ref canvas
    const renderer = new THREE.WebGLRenderer({
      canvas: canvas,
      antialias: true,
      alpha: true,
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    // Reinforced Lights (Ambient: 2.0, Dir1: 2.5, Dir2: 1.5)
    const ambientLight = new THREE.AmbientLight(0xffffff, 2.0);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 2.5);
    dirLight1.position.set(3, 8, 5);
    dirLight1.castShadow = true;
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xffffff, 1.5);
    dirLight2.position.set(-3, 5, 3);
    scene.add(dirLight2);

    // Create Ground (Grass Field, 30x30)
    const groundGeo = new THREE.PlaneGeometry(30, 30);
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0x1a3a1a,
      roughness: 0.8,
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = 0;
    ground.receiveShadow = true;
    scene.add(ground);

    // Create Mound Base Plate (Cylinder Z=-8, Y=0.1)
    const moundGeo = new THREE.CylinderGeometry(1.5, 1.5, 0.2, 32);
    const moundMat = new THREE.MeshStandardMaterial({ color: 0x5c3d1a, roughness: 0.9 });
    const mound = new THREE.Mesh(moundGeo, moundMat);
    mound.position.set(0, 0.1, -8);
    scene.add(mound);

    // Home Plate (BoxGeometry Z=0.5)
    const hpGeo = new THREE.BoxGeometry(0.5, 0.02, 0.35);
    const hpMat = new THREE.MeshStandardMaterial({ color: 0xe0e0e0, roughness: 0.5 });
    const homePlate = new THREE.Mesh(hpGeo, hpMat);
    homePlate.position.set(0, 0.01, 0.5);
    scene.add(homePlate);

    // Strike Zone Box outline (BoxGeometry Z=0.3)
    const szGeo = new THREE.BoxGeometry(0.45, 0.6, 0.02);
    const szMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.15,
    });
    const strikeZone = new THREE.Mesh(szGeo, szMat);
    strikeZone.position.set(0, 0.95, 0.3);
    scene.add(strikeZone);

    // Strike Zone Outer Boundary Line
    const edges = new THREE.EdgesGeometry(szGeo);
    const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 });
    const strikeZoneOutline = new THREE.LineSegments(edges, lineMat);
    strikeZoneOutline.position.set(0, 0.95, 0.3);
    scene.add(strikeZoneOutline);

    // Group for Trajectory Tubes
    const trajectoriesGroup = new THREE.Group();
    scene.add(trajectoriesGroup);
    trajectoriesGroupRef.current = trajectoriesGroup;

    // Handle Window Resizing
    const handleResize = () => {
      if (!container || !canvas) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    // Animation Loop
    let animationFrameId: number;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    // Clean Up
    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
      renderer.dispose();
    };
  }, []);

  // 2. Load and manage stick figures
  useEffect(() => {
    // Clean up previous figures
    if (pitcherModelRef.current) {
      sceneRef.current?.remove(pitcherModelRef.current);
      pitcherModelRef.current = null;
    }
    if (batterModelRef.current) {
      sceneRef.current?.remove(batterModelRef.current);
      batterModelRef.current = null;
    }

    // 1. Create and Position Pitcher Figure
    const pitcherGroup = createPitcherFigure(pitcherColor, pitcherHand);
    pitcherGroup.position.set(0, 0, -8);
    pitcherGroup.rotation.y = Math.PI;
    sceneRef.current?.add(pitcherGroup);
    pitcherModelRef.current = pitcherGroup;

    // 2. Create and Position Batter Figure
    const batterGroup = createBatterFigure(batterColor, batterSide);
    const batterX = batterSide === 'R' ? 0.4 : -0.4;
    batterGroup.position.set(batterX, 0, 0.5);
    sceneRef.current?.add(batterGroup);
    batterModelRef.current = batterGroup;
  }, [pitcherHand, pitcherTeamId, batterSide, batterTeamId, pitcherColor, batterColor]);

  // 3. React to Prediction Results Changes (Draw Trajectory curves)
  useEffect(() => {
    const trajectoriesGroup = trajectoriesGroupRef.current;
    if (!trajectoriesGroup) return;

    // Clear existing trajectories
    while (trajectoriesGroup.children.length > 0) {
      const child = trajectoriesGroup.children[0];
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose();
        if (Array.isArray(child.material)) {
          child.material.forEach((mat) => mat.dispose());
        } else {
          child.material.dispose();
        }
      }
      trajectoriesGroup.remove(child);
    }

    if (!predictResult || !predictResult.probabilities) return;

    // Extract top 3 pitch types sorted by probability
    const sortedPitches = Object.entries(predictResult.probabilities)
      .filter(([_, prob]) => prob > 0)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3);

    // Opacity based on probability rank: 1st: 1.0, 2nd: 0.6, 3rd: 0.3
    const opacities = [1.0, 0.6, 0.3];

    sortedPitches.forEach(([pitch, prob], index) => {
      const points = PITCH_CURVES[pitch];
      if (!points) return;

      // Map control points from the default z range [-18, 0] to release point Z=-8 and landing Z=0.3
      const adjustedPoints = points.map((p) => {
        const ratio = (p.z - (-18)) / (0 - (-18)); // 0 to 1
        const targetZ = -8 + ratio * (0.3 - (-8));
        
        // Match release height Y=1.42 (pitching hand height), released from pitcher's arm
        if (p.z === -18) {
          const isLeft = pitcherHand === 'L';
          return new THREE.Vector3(isLeft ? -0.22 : 0.22, 1.42, targetZ);
        }
        return new THREE.Vector3(p.x, p.y, targetZ);
      });

      // Generate Tube Geometry along spline curve
      const curve = new THREE.CatmullRomCurve3(adjustedPoints);
      const tubeGeo = new THREE.TubeGeometry(curve, 32, 0.022, 8, false);

      const colorHex = PITCH_COLORS[pitch] ?? 0xffffff;
      const opacityVal = opacities[index] ?? 0.3;

      const tubeMat = new THREE.MeshStandardMaterial({
        color: colorHex,
        transparent: true,
        opacity: opacityVal,
        roughness: 0.3,
        metalness: 0.1,
      });

      const tubeMesh = new THREE.Mesh(tubeGeo, tubeMat);
      tubeMesh.castShadow = true;
      trajectoriesGroup.add(tubeMesh);

      // Add a small baseball representation flying on the top-1 predicted path
      if (index === 0) {
        const ballGeo = new THREE.SphereGeometry(0.045, 16, 16);
        const ballMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9 });
        const ballMesh = new THREE.Mesh(ballGeo, ballMat);
        
        // Position ball at midpoint along the path
        ballMesh.position.copy(adjustedPoints[1]);
        trajectoriesGroup.add(ballMesh);
      }
    });
  }, [predictResult, pitcherHand]);

  return (
    <div ref={mountRef} className="relative w-full h-full overflow-hidden select-none bg-[#0a0f1d]">
      
      {/* Legend Map overlay */}
      <div className="absolute top-4 left-4 z-20 bg-gray-900/80 border border-gray-800 rounded-lg p-3 text-xs text-gray-300 space-y-2 backdrop-blur">
        <span className="font-bold text-white block pb-1 border-b border-gray-800">구종 범례</span>
        <div className="flex items-center space-x-2">
          <span className="w-2.5 h-2.5 rounded-full bg-red-500 inline-block" />
          <span>패스트볼 (FF/FT/SI)</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="w-2.5 h-2.5 rounded-full bg-yellow-500 inline-block" />
          <span>슬라이더/컷터 (SL/FC)</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="w-2.5 h-2.5 rounded-full bg-green-500 inline-block" />
          <span>체인지업/싱커 (CH/FS)</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="w-2.5 h-2.5 rounded-full bg-blue-500 inline-block" />
          <span>커브/너클볼 (CU/KC/CB/KN)</span>
        </div>
      </div>

      {/* THREE dynamic WebGL Canvas */}
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full block z-0" />
    </div>
  );
}

// Disable SSR for WebGL / canvas integration
export default dynamic(() => Promise.resolve(SceneViewerInner), { ssr: false });
