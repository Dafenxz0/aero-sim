/**
 * Pure aerodynamic geometry, obstacle SDF generation, pressure integration,
 * and numerical verification helpers shared by the WebGL2 wind tunnel and tests.
 */

export const PRESETS = Object.freeze({
  naca2412: { id: 'naca2412', label: 'NACA 2412 (Cambered)', code: '2412', supportsAoA: true },
  naca0012: { id: 'naca0012', label: 'NACA 0012 (Symmetric)', code: '0012', supportsAoA: true },
  cylinder: { id: 'cylinder', label: 'Cylinder (Vortex Street)', supportsAoA: false },
  wedge: { id: 'wedge', label: 'Bluff Car / Wedge', supportsAoA: true },
  venturi: { id: 'venturi', label: 'Double Slit / Venturi', supportsAoA: false },
});

/**
 * Parse a 4-digit NACA airfoil designation (e.g. "2412", "0012", "4415").
 * @param {string} code
 * @returns {{ m: number, p: number, t: number }}
 */
export function parseNacaCode(code) {
  const clean = String(code).trim();
  if (!/^\d{4}$/.test(clean)) {
    throw new Error(`Invalid 4-digit NACA code: "${code}"`);
  }
  return {
    m: Number(clean[0]) / 100,
    p: Number(clean[1]) / 10,
    t: Number(clean.slice(2, 4)) / 100,
  };
}

/**
 * Compute upper and lower surface coordinates (x, y) for a NACA 4-digit airfoil
 * at normalized chord coordinate xc in [0, 1].
 * Uses closed trailing edge coefficient (-0.1036) so trailing edge meets cleanly.
 * @param {number} xc
 * @param {{ m: number, p: number, t: number }} params
 */
export function naca4SurfaceAt(xc, { m, p, t }) {
  const x = Math.max(0, Math.min(1, xc));
  const yt =
    5 *
    t *
    (0.2969 * Math.sqrt(x) -
      0.126 * x -
      0.3516 * x * x +
      0.2843 * x * x * x -
      0.1036 * x * x * x * x);

  let yc = 0;
  let dyc = 0;
  if (m > 0 && p > 0 && p < 1) {
    if (x < p) {
      yc = (m / (p * p)) * (2 * p * x - x * x);
      dyc = ((2 * m) / (p * p)) * (p - x);
    } else {
      yc = (m / ((1 - p) * (1 - p))) * (1 - 2 * p + 2 * p * x - x * x);
      dyc = ((2 * m) / ((1 - p) * (1 - p))) * (p - x);
    }
  }

  const theta = Math.atan(dyc);
  const sinT = Math.sin(theta);
  const cosT = Math.cos(theta);

  return {
    xu: x - yt * sinT,
    yu: yc + yt * cosT,
    xl: x + yt * sinT,
    yl: yc - yt * cosT,
    yc,
    yt,
  };
}

/**
 * Check whether normalized local airfoil coordinate (lx, ly) lies inside the NACA profile.
 * Leading edge is at lx = 0, trailing edge at lx = 1.
 */
export function pointInNaca4(lx, ly, params) {
  if (lx < 0 || lx > 1) return false;
  const s = naca4SurfaceAt(lx, params);
  return ly >= s.yl && ly <= s.yu;
}

/**
 * Check whether normalized point (px, py) lies inside a bluff wedge / automotive fastback silhouette.
 * Coordinates centered around (0, 0) within [-0.5, 0.5] chord.
 */
export function pointInBluffWedge(lx, ly) {
  // lx in [0, 1], ly in [-0.25, 0.25]
  if (lx < 0 || lx > 1) return false;
  const bottom = -0.11;
  let top;
  if (lx < 0.22) {
    // Sloped nose hood
    const u = lx / 0.22;
    top = -0.02 + 0.14 * Math.sqrt(u);
  } else if (lx < 0.45) {
    // Windshield ramp
    const u = (lx - 0.22) / 0.23;
    top = 0.12 + 0.10 * u;
  } else if (lx < 0.72) {
    // Cabin roof
    top = 0.22;
  } else {
    // Fastback / Kammback rear
    const u = (lx - 0.72) / 0.28;
    top = 0.22 - 0.13 * u;
  }
  // Slight rear diffuser cut
  const effBottom = lx > 0.85 ? bottom + ((lx - 0.85) / 0.15) * 0.04 : bottom;
  return ly >= effBottom && ly <= top;
}

/**
 * Generate a Float32Array obstacle mask (0 = fluid, 1 = solid) of size width * height.
 * Tunnel coordinates: x in [0, aspect], y in [0, 1], where aspect = width / height.
 * Positive AoA pitches the nose UP (leading edge up, trailing edge down) in standard aerodynamics.
 *
 * @param {number} width
 * @param {number} height
 * @param {string} presetId
 * @param {number} aoaDeg Angle of attack in degrees
 * @returns {{ mask: Float32Array, refLengthCells: number }}
 */
export function generateObstacleField(width, height, presetId, aoaDeg = 8) {
  const mask = new Float32Array(width * height);
  const aspect = width / height;
  const cx = aspect * 0.34;
  const cy = 0.5;
  const chord = 0.62; // in tunnel-height units
  // Standard aerodynamic sign: +AoA raises leading edge (+y) relative to trailing edge (-y)
  const rad = (aoaDeg * Math.PI) / 180;
  const cosA = Math.cos(rad);
  const sinA = Math.sin(rad);

  const nacaParams =
    presetId === 'naca2412'
      ? parseNacaCode('2412')
      : presetId === 'naca0012'
        ? parseNacaCode('0012')
        : null;

  let refLengthCells = chord * height;

  for (let j = 0; j < height; j++) {
    const y = (j + 0.5) / height;
    for (let i = 0; i < width; i++) {
      const x = ((i + 0.5) / width) * aspect;
      const idx = j * width + i;

      if (presetId === 'cylinder') {
        const radius = 0.11;
        const dx = x - cx;
        const dy = y - cy;
        if (dx * dx + dy * dy <= radius * radius) {
          mask[idx] = 1;
        }
        refLengthCells = 2 * radius * height;
      } else if (presetId === 'venturi') {
        // Converging-diverging Venturi nozzle + double-slit baffle option
        const wallXMin = aspect * 0.24;
        const wallXMax = aspect * 0.58;
        if (x >= wallXMin && x <= wallXMax) {
          const u = (x - wallXMin) / (wallXMax - wallXMin); // 0..1
          const throat = 0.17 * Math.sin(Math.PI * u) ** 2;
          // Top and bottom converging nozzle blocks
          if (y > 1 - 0.12 - throat || y < 0.12 + throat) {
            mask[idx] = 1;
          }
        }
        // Central splitter / double-slit bluff bar near throat
        const slitX = aspect * 0.41;
        if (Math.abs(x - slitX) < 0.028 && Math.abs(y - 0.5) < 0.085) {
          mask[idx] = 1;
        }
        refLengthCells = 0.34 * height;
      } else {
        // Rotate point around quarter-chord pivot (cx, cy) into body frame
        const dx = x - cx;
        const dy = y - cy;
        // Body frame: nose at lx = 0 (-0.25*chord), tail at lx = 1 (+0.75*chord)
        // Rotating body clockwise by rad raises leading edge (+y at lx<0.25) when +AoA
        const bx = dx * cosA - dy * sinA;
        const by = dx * sinA + dy * cosA;
        const lx = bx / chord + 0.25;
        const ly = by / chord;

        if (nacaParams) {
          if (pointInNaca4(lx, ly, nacaParams)) {
            mask[idx] = 1;
          }
        } else if (presetId === 'wedge') {
          if (pointInBluffWedge(lx, ly)) {
            mask[idx] = 1;
          }
        }
      }
    }
  }

  return { mask, refLengthCells: Math.max(4, refLengthCells) };
}

/**
 * Integrate pressure forces around solid obstacle cells on a 2D grid.
 * Sign convention:
 *  - Fluid flows +x; +Fx is Drag (downstream), +Fy is Lift (upward, +y).
 *  - If left neighbor (i-1, j) is fluid, pressure pushes +x onto solid.
 *  - If bottom neighbor (i, j-1) is fluid, pressure pushes +y onto solid.
 *
 * @param {Float32Array} pressureGrid Pressure values (units: cells/sec as solved by projection)
 * @param {Float32Array} obstacleMask 0 = fluid, >=0.5 = solid
 * @param {number} width
 * @param {number} height
 * @returns {{ fx: number, fy: number, surfaceCells: number, obstacleCells: number }}
 */
export function integrateSurfaceForces(pressureGrid, obstacleMask, width, height) {
  let fx = 0;
  let fy = 0;
  let surfaceCells = 0;
  let obstacleCells = 0;

  // Exclude outer 1-cell border so tunnel walls aren't counted as obstacle
  for (let j = 1; j < height - 1; j++) {
    const row = j * width;
    for (let i = 1; i < width - 1; i++) {
      const idx = row + i;
      if (obstacleMask[idx] < 0.5) continue;
      obstacleCells++;

      let onSurface = false;
      if (obstacleMask[idx - 1] < 0.5) {
        fx += pressureGrid[idx - 1];
        onSurface = true;
      }
      if (obstacleMask[idx + 1] < 0.5) {
        fx -= pressureGrid[idx + 1];
        onSurface = true;
      }
      if (obstacleMask[idx - width] < 0.5) {
        fy += pressureGrid[idx - width];
        onSurface = true;
      }
      if (obstacleMask[idx + width] < 0.5) {
        fy -= pressureGrid[idx + width];
        onSurface = true;
      }
      if (onSurface) surfaceCells++;
    }
  }

  return { fx, fy, surfaceCells, obstacleCells };
}

/**
 * Convert integrated raw solver pressure force into aerodynamic coefficients
 * (Cl, Cd, L/D, Reynolds number).
 *
 * @param {object} opts
 * @param {number} opts.fx Raw summed pressure force x
 * @param {number} opts.fy Raw summed pressure force y
 * @param {number} opts.uInf Inflow speed in grid-cells/sec
 * @param {number} opts.dt Substep dt in seconds
 * @param {number} opts.refLengthCells Characteristic chord/diameter in grid cells
 * @param {number} opts.viscosity Kinematic viscosity scale (0.0001..0.02 in normalized units)
 */
export function computeAeroTelemetry({ fx, fy, uInf, dt, refLengthCells, viscosity }) {
  const safeU = Math.max(1e-3, uInf);
  const safeDt = Math.max(1e-5, dt);
  const safeL = Math.max(1, refLengthCells);
  // Physical pressure p_phys / rho = p_sim / dt; dynamic pressure q = 0.5 * U^2
  const dynForce = 0.5 * safeU * safeU * safeL;
  const physFx = fx / safeDt;
  const physFy = fy / safeDt;

  const cd = physFx / dynForce;
  const cl = physFy / dynForce;
  const ldRatio = Math.abs(cd) > 1e-4 ? cl / cd : 0;

  // Map grid inflow to physical wind-tunnel speed (120 cells/s -> 15.0 m/s) and chord (0.62 * H -> 0.5 m)
  const uMeters = (safeU / 120) * 15.0;
  const chordMeters = (safeL / 138) * 0.5;
  const nuPhysical = Math.max(1.5e-5, 1.5e-5 + viscosity * 8e-3);
  const reynolds = (uMeters * chordMeters) / nuPhysical;

  return {
    cl,
    cd,
    ldRatio,
    reynolds,
    uMeters,
    fx: physFx,
    fy: physFy,
  };
}

/**
 * Decode 16-bit packed RGBA8 telemetry buffer (2 bytes per pressure value + 1 byte obstacle mask)
 * into Float32Arrays ready for `integrateSurfaceForces`.
 */
export function decodeTelemetryBuffer(rgbaBytes, width, height, pScale) {
  const n = width * height;
  const pressure = new Float32Array(n);
  const obstacle = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    const raw16 = rgbaBytes[o] * 256 + rgbaBytes[o + 1];
    const norm = raw16 / 65535;
    pressure[i] = (norm * 2 - 1) * pScale;
    obstacle[i] = rgbaBytes[o + 2] >= 128 ? 1 : 0;
  }
  return { pressure, obstacle };
}

/**
 * CPU reference 1D MacCormack advection step (for unit tests verifying 2nd-order accuracy & clamping).
 */
export function maccormackStep1D(field, velocity, dt, dx = 1) {
  const n = field.length;
  const pred = new Float32Array(n);
  const corr = new Float32Array(n);
  const out = new Float32Array(n);

  const sampleLinear = (arr, x) => {
    const cx = Math.max(0, Math.min(n - 1, x));
    const i0 = Math.floor(cx);
    const i1 = Math.min(n - 1, i0 + 1);
    const f = cx - i0;
    return arr[i0] * (1 - f) + arr[i1] * f;
  };

  for (let i = 0; i < n; i++) {
    pred[i] = sampleLinear(field, i - (velocity[i] * dt) / dx);
  }
  for (let i = 0; i < n; i++) {
    corr[i] = sampleLinear(pred, i + (velocity[i] * dt) / dx);
  }
  for (let i = 0; i < n; i++) {
    const backX = Math.max(0, Math.min(n - 1, i - (velocity[i] * dt) / dx));
    const i0 = Math.floor(backX);
    const i1 = Math.min(n - 1, i0 + 1);
    const lo = Math.min(field[i0], field[i1]);
    const hi = Math.max(field[i0], field[i1]);
    const mc = pred[i] + 0.5 * (field[i] - corr[i]);
    out[i] = Math.max(lo, Math.min(hi, mc));
  }
  return out;
}

/**
 * CPU reference 2D Jacobi Poisson solver step with solid-obstacle Neumann BC (for unit tests).
 */
export function jacobiPoissonStep2D(pressure, divergence, obstacle, width, height, iterations = 40) {
  let src = new Float32Array(pressure);
  let dst = new Float32Array(pressure.length);

  for (let it = 0; it < iterations; it++) {
    for (let j = 1; j < height - 1; j++) {
      for (let i = 1; i < width - 1; i++) {
        const idx = j * width + i;
        if (obstacle[idx] >= 0.5) {
          dst[idx] = 0;
          continue;
        }
        const pC = src[idx];
        const pL = obstacle[idx - 1] >= 0.5 ? pC : src[idx - 1];
        const pR = obstacle[idx + 1] >= 0.5 ? pC : src[idx + 1];
        const pB = obstacle[idx - width] >= 0.5 ? pC : src[idx - width];
        const pT = obstacle[idx + width] >= 0.5 ? pC : src[idx + width];
        dst[idx] = 0.25 * (pL + pR + pB + pT - divergence[idx]);
      }
    }
    const tmp = src;
    src = dst;
    dst = tmp;
  }
  return src;
}
