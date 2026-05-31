'use client';

import React, { useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';
import * as THREE from 'three';
import { PredictResponse } from '../../lib/types';
import { PITCH_CURVES, PITCH_COLORS } from '../../lib/pitchCurves';

interface LiveSceneViewerProps {
  predictResult: PredictResponse | null;
  pitcherHand?: 'R' | 'L';
}

function LiveSceneViewerInner({
  predictResult,
  pitcherHand = 'R',
}: LiveSceneViewerProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const sceneRef = useRef<THREE.Scene | null>(null);
  const trajectoriesGroupRef = useRef<THREE.Group | null>(null);

  // 1. Scene Setup
  useEffect(() => {
    if (!mountRef.current || !canvasRef.current) return;

    const container = mountRef.current;
    const canvas = canvasRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0f1d);
    scene.fog = new THREE.FogExp2(0x0a0f1d, 0.025);
    sceneRef.current = scene;

    // Calibrated Camera
    const camera = new THREE.PerspectiveCamera(52, width / height, 0.1, 1000);
    camera.position.set(0, 1.2, 9.5);
    camera.lookAt(0, 1.0, 0);

    const renderer = new THREE.WebGLRenderer({
      canvas: canvas,
      antialias: true,
      alpha: true,
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 2.0);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 2.5);
    dirLight1.position.set(3, 8, 5);
    dirLight1.castShadow = true;
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xffffff, 1.5);
    dirLight2.position.set(-3, 5, 3);
    scene.add(dirLight2);

    // Grass Ground
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

    // Pitcher Mound Plate
    const moundGeo = new THREE.CylinderGeometry(1.5, 1.5, 0.2, 32);
    const moundMat = new THREE.MeshStandardMaterial({ color: 0x5c3d1a, roughness: 0.9 });
    const mound = new THREE.Mesh(moundGeo, moundMat);
    mound.position.set(0, 0.1, -8);
    scene.add(mound);

    // Home Plate
    const hpGeo = new THREE.BoxGeometry(0.5, 0.02, 0.35);
    const hpMat = new THREE.MeshStandardMaterial({ color: 0xe0e0e0, roughness: 0.5 });
    const homePlate = new THREE.Mesh(hpGeo, hpMat);
    homePlate.position.set(0, 0.01, 0.5);
    scene.add(homePlate);

    // Strike Zone Box
    const szGeo = new THREE.BoxGeometry(0.45, 0.6, 0.02);
    const szMat = new THREE.MeshStandardMaterial({
      color: 0x4444ff,
      transparent: true,
      opacity: 0.15,
    });
    const strikeZone = new THREE.Mesh(szGeo, szMat);
    strikeZone.position.set(0, 0.95, 0.3);
    scene.add(strikeZone);

    // Strike Zone Outline
    const edges = new THREE.EdgesGeometry(szGeo);
    const lineMat = new THREE.LineBasicMaterial({ color: 0xffff00, opacity: 0.9, transparent: true, linewidth: 2 });
    const strikeZoneOutline = new THREE.LineSegments(edges, lineMat);
    strikeZoneOutline.position.set(0, 0.95, 0.3);
    scene.add(strikeZoneOutline);

    // Group for Trajectories
    const trajectoriesGroup = new THREE.Group();
    scene.add(trajectoriesGroup);
    trajectoriesGroupRef.current = trajectoriesGroup;

    // Resize Handler
    const handleResize = () => {
      if (!container || !canvas) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    // Loop
    let animationFrameId: number;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
      renderer.dispose();
    };
  }, []);

  // 2. Predict Trajectory drawing
  useEffect(() => {
    const trajectoriesGroup = trajectoriesGroupRef.current;
    if (!trajectoriesGroup) return;

    // Clear previous curves
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

    if (!predictResult || !predictResult.pitch_probabilities) return;

    const sortedPitches = Object.entries(predictResult.pitch_probabilities)
      .filter(([_, prob]) => prob > 0)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3);

    const opacities = [1.0, 0.6, 0.3];

    sortedPitches.forEach(([pitch, prob], index) => {
      const points = PITCH_CURVES[pitch];
      if (!points) return;

      const adjustedPoints = points.map((p) => {
        const ratio = (p.z - (-18)) / (0 - (-18));
        const targetZ = -8 + ratio * (0.3 - (-8));
        
        // Pitching release point matching pitcher stand height
        if (p.z === -18) {
          const isLeft = pitcherHand === 'L';
          return new THREE.Vector3(isLeft ? -0.22 : 0.22, 1.42, targetZ);
        }
        return new THREE.Vector3(p.x, p.y, targetZ);
      });

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

      // Flying baseball visualization on top-1 path
      if (index === 0) {
        const ballGeo = new THREE.SphereGeometry(0.045, 16, 16);
        const ballMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9 });
        const ballMesh = new THREE.Mesh(ballGeo, ballMat);
        ballMesh.position.copy(adjustedPoints[1]);
        trajectoriesGroup.add(ballMesh);
      }
    });
  }, [predictResult, pitcherHand]);

  return (
    <div ref={mountRef} className="relative w-full h-full overflow-hidden select-none bg-[#0a0f1d] rounded-2xl border border-gray-800">
      {/* 구종 범례 — 17개 구종 색상 기준 */}
      <div className="absolute top-4 left-4 z-20 bg-gray-900/80 border border-gray-800 rounded-lg p-3 text-xs text-gray-300 space-y-2 backdrop-blur shadow-lg">
        <span className="font-bold text-white block pb-1 border-b border-gray-800">구종 범례</span>
        {[
          { color: '#EF4444', label: '포심 (FF/FA)' },
          { color: '#F59E0B', label: '싱커 (SI)' },
          { color: '#EAB308', label: '커터 (FC)' },
          { color: '#22C55E', label: '슬라이더 (SL)' },
          { color: '#8B5CF6', label: '스위퍼 (ST)' },
          { color: '#3B82F6', label: '커브 (CU/KC/CS)' },
          { color: '#EC4899', label: '체인지업 (CH)' },
          { color: '#F43F5E', label: '스플리터 (FS/FO)' },
        ].map(({ color, label }) => (
          <div key={label} className="flex items-center space-x-2">
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: color, display: 'inline-block', flexShrink: 0 }} />
            <span>{label}</span>
          </div>
        ))}
      </div>

      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full block z-0" />
    </div>
  );
}

export default dynamic(() => Promise.resolve(LiveSceneViewerInner), { ssr: false });
