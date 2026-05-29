import { Vector3 } from 'three';

export const PITCH_CURVES: Record<string, Vector3[]> = {
  FF: [
    new Vector3(0, 1.8, -18),
    new Vector3(0, 1.5, -9),
    new Vector3(0, 0.9, 0)
  ],
  FT: [
    new Vector3(0, 1.8, -18),
    new Vector3(-0.1, 1.4, -9),
    new Vector3(-0.2, 0.8, 0)
  ],
  SI: [
    new Vector3(0, 1.8, -18),
    new Vector3(-0.1, 1.3, -9),
    new Vector3(-0.2, 0.7, 0)
  ],
  SL: [
    new Vector3(0, 1.8, -18),
    new Vector3(0.2, 1.4, -9),
    new Vector3(0.4, 0.7, 0)
  ],
  FC: [
    new Vector3(0, 1.8, -18),
    new Vector3(0.1, 1.5, -9),
    new Vector3(0.2, 0.85, 0)
  ],
  CH: [
    new Vector3(0, 1.8, -18),
    new Vector3(0, 1.3, -9),
    new Vector3(0, 0.6, 0)
  ],
  FS: [
    new Vector3(0, 1.8, -18),
    new Vector3(0, 1.2, -9),
    new Vector3(0, 0.5, 0)
  ],
  CU: [
    new Vector3(0, 1.8, -18),
    new Vector3(0, 2.0, -9),
    new Vector3(0, 0.5, 0)
  ],
  KC: [
    new Vector3(0, 1.8, -18),
    new Vector3(0, 1.9, -9),
    new Vector3(0, 0.55, 0)
  ],
  CB: [
    new Vector3(0, 1.8, -18),
    new Vector3(0, 2.1, -9),
    new Vector3(0, 0.45, 0)
  ],
  KN: [
    new Vector3(0.1, 1.8, -18),
    new Vector3(-0.1, 1.5, -9),
    new Vector3(0.05, 0.8, 0)
  ]
};

export const PITCH_COLORS: Record<string, number> = {
  FF: 0xEF4444, FT: 0xEF4444, SI: 0xEF4444,
  SL: 0xEAB308, FC: 0xEAB308,
  CH: 0x22C55E, FS: 0x22C55E,
  CU: 0x3B82F6, KC: 0x3B82F6, CB: 0x3B82F6, KN: 0x3B82F6
};
