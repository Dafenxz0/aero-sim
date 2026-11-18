import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  PRESETS,
  parseNacaCode,
  naca4SurfaceAt,
  pointInNaca4,
  generateObstacleField,
  integrateSurfaceForces,
  computeAeroTelemetry,
  decodeTelemetryBuffer,
  maccormackStep1D,
  jacobiPoissonStep2D,
} from '../src/physics.js';
import * as shaders from '../src/shaders.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

test('parseNacaCode parses 4-digit codes and rejects malformed inputs', () => {
  const n2412 = parseNacaCode('2412');
  assert.ok(Math.abs(n2412.m - 0.02) < 1e-12);
  assert.ok(Math.abs(n2412.p - 0.4) < 1e-12);
  assert.ok(Math.abs(n2412.t - 0.12) < 1e-12);

  assert.throws(() => parseNacaCode('241'), /Invalid 4-digit NACA/);
  assert.throws(() => parseNacaCode('abcd'), /Invalid 4-digit NACA/);
});

test('NACA 0012 is symmetric and NACA 2412 has positive camber', () => {
  const p0012 = parseNacaCode('0012');
  const s0 = naca4SurfaceAt(0.3, p0012);
  assert.ok(Math.abs(s0.yu + s0.yl) < 1e-12, 'NACA 0012 upper/lower must be symmetric');
  assert.ok(s0.yu > 0 && s0.yl < 0);

  const te = naca4SurfaceAt(1.0, p0012);
  assert.ok(Math.abs(te.yu - te.yl) < 1e-6, 'Trailing edge should close cleanly');

  const p2412 = parseNacaCode('2412');
  const s2 = naca4SurfaceAt(0.4, p2412);
  assert.ok(s2.yc > 0.019 && s2.yc < 0.021, 'Max camber at p=0.4 should equal m=0.02');
  assert.ok(pointInNaca4(0.3, 0.02, p2412));
  assert.ok(!pointInNaca4(0.3, 0.25, p2412));
});

test('All obstacle presets generate valid non-empty masks and positive AoA pitches nose up', () => {
  const W = 160;
  const H = 80;
  for (const id of Object.keys(PRESETS)) {
    const { mask, refLengthCells } = generateObstacleField(W, H, id, 8);
    assert.equal(mask.length, W * H);
    let solid = 0;
    for (let i = 0; i < mask.length; i++) {
      if (mask[i] > 0.5) solid++;
    }
    assert.ok(solid > 25, `Preset ${id} should produce solid cells (got ${solid})`);
    assert.ok(refLengthCells > 4, `Preset ${id} should report positive refLengthCells`);
  }

  // Positive AoA (+15 deg) should place leading edge higher (+y) than trailing edge
  const { mask } = generateObstacleField(W, H, 'naca0012', 15);
  let leftSumY = 0;
  let leftCount = 0;
  let rightSumY = 0;
  let rightCount = 0;
  const midX = Math.floor(W * 0.42);
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      if (mask[j * W + i] > 0.5) {
        if (i < midX) {
          leftSumY += j;
          leftCount++;
        } else {
          rightSumY += j;
          rightCount++;
        }
      }
    }
  }
  assert.ok(leftCount > 0 && rightCount > 0);
  assert.ok(
    leftSumY / leftCount > rightSumY / rightCount,
    'Positive AoA must pitch leading edge (+y) above trailing edge',
  );
});

test('Pressure integration yields zero force in uniform pressure and positive Cl/Cd under differential pressure', () => {
  const W = 16;
  const H = 16;
  const obs = new Float32Array(W * H);
  // 4x4 square block in center
  for (let j = 6; j < 10; j++) {
    for (let i = 6; i < 10; i++) {
      obs[j * W + i] = 1;
    }
  }

  // Uniform pressure -> zero net force
  const uniformP = new Float32Array(W * H).fill(5.0);
  const fUniform = integrateSurfaceForces(uniformP, obs, W, H);
  assert.equal(fUniform.fx, 0);
  assert.equal(fUniform.fy, 0);
  assert.equal(fUniform.obstacleCells, 16);

  // Higher pressure on upstream (i=5) and bottom (j=5) faces -> positive Drag and positive Lift
  const diffP = new Float32Array(W * H);
  for (let j = 6; j < 10; j++) diffP[j * W + 5] = 10; // front stagnation
  for (let i = 6; i < 10; i++) diffP[5 * W + i] = 8; // high pressure beneath
  for (let i = 6; i < 10; i++) diffP[10 * W + i] = -4; // suction above

  const fDiff = integrateSurfaceForces(diffP, obs, W, H);
  assert.equal(fDiff.fx, 40);
  assert.equal(fDiff.fy, 48);

  const telem = computeAeroTelemetry({
    fx: fDiff.fx,
    fy: fDiff.fy,
    uInf: 120,
    dt: 1 / 120,
    refLengthCells: 4,
    viscosity: 0.0005,
  });
  assert.ok(telem.cd > 0);
  assert.ok(telem.cl > 0);
  assert.ok(telem.ldRatio > 1.0);
  assert.ok(telem.reynolds > 1000);
});

test('decodeTelemetryBuffer unpacks 16-bit pressure and obstacle mask accurately', () => {
  const W = 4;
  const H = 2;
  const pScale = 300;
  const buf = new Uint8Array(W * H * 4);

  // Cell 0: p = +150 -> norm = 0.75 -> raw16 = round(0.75 * 65535) = 49151
  const raw0 = Math.round(0.75 * 65535);
  buf[0] = Math.floor(raw0 / 256);
  buf[1] = raw0 % 256;
  buf[2] = 255;
  buf[3] = 255;

  // Cell 1: p = -75 -> norm = 0.375
  const raw1 = Math.round(0.375 * 65535);
  buf[4] = Math.floor(raw1 / 256);
  buf[5] = raw1 % 256;
  buf[6] = 0;
  buf[7] = 255;

  const { pressure, obstacle } = decodeTelemetryBuffer(buf, W, H, pScale);
  assert.ok(Math.abs(pressure[0] - 150) < 0.02);
  assert.equal(obstacle[0], 1);
  assert.ok(Math.abs(pressure[1] - -75) < 0.02);
  assert.equal(obstacle[1], 0);
});

test('MacCormack 1D advection shifts pulse accurately within monotonic bounds', () => {
  const n = 64;
  let field = new Float32Array(n);
  const vel = new Float32Array(n).fill(1.0);
  for (let i = 0; i < n; i++) {
    const d = i - 16;
    field[i] = Math.exp(-(d * d) / 18);
  }
  const origPeak = Math.max(...field);
  for (let step = 0; step < 10; step++) {
    field = maccormackStep1D(field, vel, 0.5, 1.0);
  }
  const peakIdx = field.indexOf(Math.max(...field));
  assert.ok(Math.abs(peakIdx - 21) <= 1, `Expected pulse peak near 21, got ${peakIdx}`);
  assert.ok(Math.max(...field) <= origPeak + 1e-6, 'Clamped MacCormack must not overshoot');
  assert.ok(Math.min(...field) >= -1e-6, 'Clamped MacCormack must not undershoot');
});

test('Jacobi Poisson 2D solver reduces residual and keeps solid cells zeroed', () => {
  const W = 24;
  const H = 24;
  const p0 = new Float32Array(W * H);
  const div = new Float32Array(W * H);
  const obs = new Float32Array(W * H);
  obs[12 * W + 12] = 1;
  div[12 * W + 11] = -2.0; // compression upstream of obstacle

  const pSolved = jacobiPoissonStep2D(p0, div, obs, W, H, 60);
  assert.equal(pSolved[12 * W + 12], 0, 'Obstacle cell pressure stays zero');
  assert.ok(pSolved[12 * W + 11] > 0.4, 'Compression upstream induces positive stagnation pressure');
  assert.ok(pSolved[12 * W + 11] > pSolved[12 * W + 8], 'Pressure decays away from stagnation');
});

test('All GLSL shaders have WebGL2 header, balanced braces, and no placeholders', () => {
  const entries = Object.entries(shaders);
  assert.ok(entries.length >= 15);
  for (const [name, src] of entries) {
    assert.equal(typeof src, 'string');
    assert.ok(src.startsWith('#version 300 es'), `${name} must start with #version 300 es`);
    assert.ok(!src.includes('TODO') && !src.includes('undefined'), `${name} has forbidden text`);
    const open = (src.match(/\{/g) || []).length;
    const close = (src.match(/\}/g) || []).length;
    assert.equal(open, close, `${name} has unbalanced braces (${open} vs ${close})`);
  }
});

test('index.html uses relative ./ paths, includes #error-banner, and exposes 5 vis modes', () => {
  const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
  assert.ok(html.includes('href="./styles.css"'));
  assert.ok(html.includes('src="./src/main.js"'));
  assert.ok(html.includes('id="error-banner"'));
  assert.ok(!/(?:href|src)="\/(?!\/)/.test(html), 'Must not use root-absolute / asset paths');
  for (let m = 0; m < 5; m++) {
    assert.ok(html.includes(`data-vis="${m}"`), `Missing visualization mode ${m}`);
  }
});
