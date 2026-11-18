/**
 * WebGL2 GLSL ES 3.00 shaders for incompressible Navier-Stokes wind tunnel,
 * MacCormack advection, vorticity confinement, Jacobi pressure solver,
 * solid obstacle boundary enforcement, GPGPU tracer particles, and telemetry encoding.
 */

export const BASE_VS = `#version 300 es
precision highp float;
in vec2 aPosition;
out vec2 vUv;

void main() {
  vUv = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

const BILINEAR_HELPER = `
vec4 sampleBilinear(sampler2D tex, vec2 uv, vec2 texSize) {
  vec2 st = uv * texSize - 0.5;
  vec2 i = floor(st);
  vec2 f = fract(st);
  vec2 inv = 1.0 / texSize;
  vec2 uv00 = (i + 0.5) * inv;
  vec2 uv10 = (i + vec2(1.5, 0.5)) * inv;
  vec2 uv01 = (i + vec2(0.5, 1.5)) * inv;
  vec2 uv11 = (i + vec2(1.5, 1.5)) * inv;
  vec4 a = texture(tex, clamp(uv00, 0.5 * inv, 1.0 - 0.5 * inv));
  vec4 b = texture(tex, clamp(uv10, 0.5 * inv, 1.0 - 0.5 * inv));
  vec4 c = texture(tex, clamp(uv01, 0.5 * inv, 1.0 - 0.5 * inv));
  vec4 d = texture(tex, clamp(uv11, 0.5 * inv, 1.0 - 0.5 * inv));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
`;

export const SEMI_LAGRANGIAN_FS = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform sampler2D uObstacles;
uniform vec2 uTexelSize;
uniform vec2 uGridSize;
uniform float uDt;
uniform float uDirection; // +1.0 for forward predictor, -1.0 for backward corrector

${BILINEAR_HELPER}

void main() {
  if (texture(uObstacles, vUv).r > 0.5) {
    outColor = vec4(0.0);
    return;
  }
  vec2 vel = texture(uVelocity, vUv).xy;
  vec2 coord = vUv - uDirection * uDt * vel * uTexelSize;
  outColor = sampleBilinear(uSource, coord, uGridSize);
}
`;

export const MACCORMACK_FS = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uVelocity;
uniform sampler2D uSource;     // phi^n
uniform sampler2D uPredictor;  // phi_hat^(n+1)
uniform sampler2D uObstacles;
uniform vec2 uTexelSize;
uniform vec2 uGridSize;
uniform float uDt;
uniform float uDissipation;

${BILINEAR_HELPER}

void main() {
  if (texture(uObstacles, vUv).r > 0.5) {
    outColor = vec4(0.0);
    return;
  }

  vec2 vel = texture(uVelocity, vUv).xy;
  vec2 backUv = vUv - uDt * vel * uTexelSize;
  vec2 fwdUv  = vUv + uDt * vel * uTexelSize;

  vec4 phiN   = texture(uSource, vUv);
  vec4 phiHat = texture(uPredictor, vUv);
  vec4 phiRev = sampleBilinear(uPredictor, fwdUv, uGridSize);

  // 4-tap neighborhood around departure point for monotonicity clamping
  vec2 st = backUv * uGridSize - 0.5;
  vec2 i = floor(st);
  vec2 inv = uTexelSize;
  vec2 uv00 = clamp((i + vec2(0.5, 0.5)) * inv, 0.5 * inv, 1.0 - 0.5 * inv);
  vec2 uv10 = clamp((i + vec2(1.5, 0.5)) * inv, 0.5 * inv, 1.0 - 0.5 * inv);
  vec2 uv01 = clamp((i + vec2(0.5, 1.5)) * inv, 0.5 * inv, 1.0 - 0.5 * inv);
  vec2 uv11 = clamp((i + vec2(1.5, 1.5)) * inv, 0.5 * inv, 1.0 - 0.5 * inv);

  float obsNear =
    texture(uObstacles, uv00).r +
    texture(uObstacles, uv10).r +
    texture(uObstacles, uv01).r +
    texture(uObstacles, uv11).r;

  vec4 result;
  if (obsNear > 0.1) {
    // Fallback to semi-Lagrangian near solid walls to prevent ringing
    result = phiHat;
  } else {
    vec4 n00 = texture(uSource, uv00);
    vec4 n10 = texture(uSource, uv10);
    vec4 n01 = texture(uSource, uv01);
    vec4 n11 = texture(uSource, uv11);
    vec4 minVal = min(min(n00, n10), min(n01, n11));
    vec4 maxVal = max(max(n00, n10), max(n01, n11));
    vec4 mc = phiHat + 0.5 * (phiN - phiRev);
    result = clamp(mc, minVal, maxVal);
  }

  outColor = result * uDissipation;
}
`;

export const CURL_FS = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uVelocity;
uniform sampler2D uObstacles;
uniform vec2 uTexelSize;

void main() {
  if (texture(uObstacles, vUv).r > 0.5) {
    outColor = vec4(0.0);
    return;
  }
  float L = texture(uVelocity, vUv - vec2(uTexelSize.x, 0.0)).y;
  float R = texture(uVelocity, vUv + vec2(uTexelSize.x, 0.0)).y;
  float B = texture(uVelocity, vUv - vec2(0.0, uTexelSize.y)).x;
  float T = texture(uVelocity, vUv + vec2(0.0, uTexelSize.y)).x;
  float vorticity = 0.5 * ((R - L) - (T - B));
  outColor = vec4(vorticity, 0.0, 0.0, 1.0);
}
`;

export const VORTICITY_CONFINEMENT_FS = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform sampler2D uObstacles;
uniform vec2 uTexelSize;
uniform float uCurlStrength;
uniform float uDt;

void main() {
  vec2 vel = texture(uVelocity, vUv).xy;
  if (texture(uObstacles, vUv).r > 0.5) {
    outColor = vec4(0.0);
    return;
  }

  float L = abs(texture(uCurl, vUv - vec2(uTexelSize.x, 0.0)).x);
  float R = abs(texture(uCurl, vUv + vec2(uTexelSize.x, 0.0)).x);
  float B = abs(texture(uCurl, vUv - vec2(0.0, uTexelSize.y)).x);
  float T = abs(texture(uCurl, vUv + vec2(0.0, uTexelSize.y)).x);
  float C = texture(uCurl, vUv).x;

  vec2 grad = 0.5 * vec2(R - L, T - B);
  float len = length(grad) + 1e-5;
  vec2 N = grad / len;
  // Confinement force f = eps * (N x omega)
  vec2 force = uCurlStrength * vec2(N.y, -N.x) * C;
  outColor = vec4(vel + force * uDt, 0.0, 1.0);
}
`;

export const VISCOSITY_JACOBI_FS = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uVelocity;
uniform sampler2D uObstacles;
uniform vec2 uTexelSize;
uniform float uViscosity;
uniform float uDt;

void main() {
  if (texture(uObstacles, vUv).r > 0.5) {
    outColor = vec4(0.0);
    return;
  }
  vec2 C = texture(uVelocity, vUv).xy;
  vec2 L = texture(uVelocity, vUv - vec2(uTexelSize.x, 0.0)).xy;
  vec2 R = texture(uVelocity, vUv + vec2(uTexelSize.x, 0.0)).xy;
  vec2 B = texture(uVelocity, vUv - vec2(0.0, uTexelSize.y)).xy;
  vec2 T = texture(uVelocity, vUv + vec2(0.0, uTexelSize.y)).xy;

  float alpha = max(0.0, uViscosity * uDt * 400.0);
  vec2 diffused = (C + alpha * (L + R + B + T)) / (1.0 + 4.0 * alpha);
  outColor = vec4(diffused, 0.0, 1.0);
}
`;

export const DIVERGENCE_FS = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uVelocity;
uniform sampler2D uObstacles;
uniform vec2 uTexelSize;

void main() {
  if (texture(uObstacles, vUv).r > 0.5) {
    outColor = vec4(0.0);
    return;
  }

  vec2 C = texture(uVelocity, vUv).xy;
  vec2 uvL = vUv - vec2(uTexelSize.x, 0.0);
  vec2 uvR = vUv + vec2(uTexelSize.x, 0.0);
  vec2 uvB = vUv - vec2(0.0, uTexelSize.y);
  vec2 uvT = vUv + vec2(0.0, uTexelSize.y);

  float L = texture(uObstacles, uvL).r > 0.5 ? -C.x : texture(uVelocity, uvL).x;
  float R = texture(uObstacles, uvR).r > 0.5 ? -C.x : texture(uVelocity, uvR).x;
  float B = texture(uObstacles, uvB).r > 0.5 ? -C.y : texture(uVelocity, uvB).y;
  float T = texture(uObstacles, uvT).r > 0.5 ? -C.y : texture(uVelocity, uvT).y;

  // Tunnel top/bottom impermeable walls
  if (vUv.y - uTexelSize.y < 0.0) B = -C.y;
  if (vUv.y + uTexelSize.y > 1.0) T = -C.y;

  float div = 0.5 * ((R - L) + (T - B));
  outColor = vec4(div, 0.0, 0.0, 1.0);
}
`;

export const JACOBI_PRESSURE_FS = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uPressure;
uniform sampler2D uDivergence;
uniform sampler2D uObstacles;
uniform vec2 uTexelSize;

void main() {
  if (texture(uObstacles, vUv).r > 0.5) {
    outColor = vec4(0.0);
    return;
  }
  // Right outflow Dirichlet p = 0 so pressure drains cleanly at tunnel exit
  if (vUv.x > 1.0 - 1.5 * uTexelSize.x) {
    outColor = vec4(0.0);
    return;
  }

  float pC = texture(uPressure, vUv).x;
  vec2 uvL = vUv - vec2(uTexelSize.x, 0.0);
  vec2 uvR = vUv + vec2(uTexelSize.x, 0.0);
  vec2 uvB = vUv - vec2(0.0, uTexelSize.y);
  vec2 uvT = vUv + vec2(0.0, uTexelSize.y);

  // Neumann boundary condition dp/dn = 0 at solid obstacles and tunnel top/bottom/inlet
  float pL = (vUv.x - uTexelSize.x < 0.0 || texture(uObstacles, uvL).r > 0.5) ? pC : texture(uPressure, uvL).x;
  float pR = (texture(uObstacles, uvR).r > 0.5) ? pC : texture(uPressure, uvR).x;
  float pB = (vUv.y - uTexelSize.y < 0.0 || texture(uObstacles, uvB).r > 0.5) ? pC : texture(uPressure, uvB).x;
  float pT = (vUv.y + uTexelSize.y > 1.0 || texture(uObstacles, uvT).r > 0.5) ? pC : texture(uPressure, uvT).x;

  float div = texture(uDivergence, vUv).x;
  float pNext = 0.25 * (pL + pR + pB + pT - div);
  outColor = vec4(pNext, 0.0, 0.0, 1.0);
}
`;

export const GRADIENT_SUBTRACT_FS = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uPressure;
uniform sampler2D uVelocity;
uniform sampler2D uObstacles;
uniform vec2 uTexelSize;
uniform float uInflowSpeed;
uniform int uNoSlip;

void main() {
  if (texture(uObstacles, vUv).r > 0.5) {
    outColor = vec4(0.0);
    return;
  }

  // Enforce steady wind-tunnel inflow at left boundary strip
  if (vUv.x < 2.5 * uTexelSize.x) {
    outColor = vec4(uInflowSpeed, 0.0, 0.0, 1.0);
    return;
  }

  float pC = texture(uPressure, vUv).x;
  vec2 uvL = vUv - vec2(uTexelSize.x, 0.0);
  vec2 uvR = vUv + vec2(uTexelSize.x, 0.0);
  vec2 uvB = vUv - vec2(0.0, uTexelSize.y);
  vec2 uvT = vUv + vec2(0.0, uTexelSize.y);

  float obsL = texture(uObstacles, uvL).r;
  float obsR = texture(uObstacles, uvR).r;
  float obsB = texture(uObstacles, uvB).r;
  float obsT = texture(uObstacles, uvT).r;

  float pL = obsL > 0.5 ? pC : texture(uPressure, uvL).x;
  float pR = obsR > 0.5 ? pC : texture(uPressure, uvR).x;
  float pB = (vUv.y - uTexelSize.y < 0.0 || obsB > 0.5) ? pC : texture(uPressure, uvB).x;
  float pT = (vUv.y + uTexelSize.y > 1.0 || obsT > 0.5) ? pC : texture(uPressure, uvT).x;

  vec2 vel = texture(uVelocity, vUv).xy;
  vel -= 0.5 * vec2(pR - pL, pT - pB);

  // Zero-gradient outflow on right exit + no-backflow stabilizer
  if (vUv.x > 1.0 - 2.0 * uTexelSize.x) {
    vel = texture(uVelocity, vUv - vec2(uTexelSize.x, 0.0)).xy;
    vel.x = max(vel.x, 0.0);
  }

  // Impermeable top/bottom tunnel walls
  if (vUv.y < 1.5 * uTexelSize.y || vUv.y > 1.0 - 1.5 * uTexelSize.y) {
    vel.y = 0.0;
  }

  // Boundary layer shear for no-slip solid obstacles
  if (uNoSlip == 1) {
    float adjSolid = step(0.5, obsL) + step(0.5, obsR) + step(0.5, obsB) + step(0.5, obsT);
    if (adjSolid > 0.0) {
      vel *= 0.86;
    }
  }

  outColor = vec4(vel, 0.0, 1.0);
}
`;

export const WIND_DYE_INFLOW_FS = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uSource;
uniform sampler2D uObstacles;
uniform vec2 uTexelSize;
uniform float uSmokeRakes; // 1.0 on, 0.0 off
uniform float uTime;

// Interactive jet/dye splat
uniform int uSplatActive;
uniform vec2 uSplatPos;
uniform vec2 uSplatForce;
uniform vec3 uSplatColor;
uniform float uSplatRadius;
uniform int uIsVelocity; // 1 = velocity field, 0 = dye field

void main() {
  if (texture(uObstacles, vUv).r > 0.5) {
    outColor = vec4(0.0);
    return;
  }

  vec4 val = texture(uSource, vUv);

  if (uIsVelocity == 0 && uSmokeRakes > 0.5 && vUv.x < 3.0 * uTexelSize.x) {
    // 11 horizontal smoke rakes at tunnel inlet
    float band = fract(vUv.y * 11.0);
    float jet = smoothstep(0.32, 0.14, abs(band - 0.5));
    float centerHighlight = smoothstep(0.055, 0.0, abs(vUv.y - 0.5));
    vec3 rakeCol = mix(vec3(0.20, 0.72, 0.96), vec3(0.98, 0.58, 0.22), centerHighlight);
    val.rgb = max(val.rgb, rakeCol * jet * 0.95);
  }

  if (uSplatActive == 1) {
    vec2 diff = vUv - uSplatPos;
    diff.x *= (uTexelSize.y / uTexelSize.x); // isotropic brush radius
    float d2 = dot(diff, diff);
    float falloff = exp(-d2 / max(1e-6, uSplatRadius * uSplatRadius));
    if (uIsVelocity == 1) {
      val.xy += uSplatForce * falloff;
    } else {
      val.rgb = clamp(val.rgb + uSplatColor * falloff, 0.0, 2.5);
    }
  }

  outColor = val;
}
`;

export const OBSTACLE_BRUSH_FS = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uObstacles;
uniform vec2 uTexelSize;
uniform vec2 uBrushPos;
uniform float uBrushRadius;
uniform float uBrushValue; // 1.0 = draw solid, 0.0 = erase

void main() {
  float obs = texture(uObstacles, vUv).r;
  vec2 diff = vUv - uBrushPos;
  diff.x *= (uTexelSize.y / uTexelSize.x);
  if (length(diff) <= uBrushRadius) {
    obs = uBrushValue;
  }
  outColor = vec4(obs, 0.0, 0.0, 1.0);
}
`;

export const PARTICLE_ADVECT_FS = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uParticles; // xy = uv pos, z = age (0..1), w = speed
uniform sampler2D uVelocity;
uniform sampler2D uObstacles;
uniform vec2 uTexelSize;
uniform vec2 uGridSize;
uniform float uDt;
uniform float uTime;

${BILINEAR_HELPER}

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

void main() {
  vec4 p = texture(uParticles, vUv);
  vec2 pos = p.xy;
  float age = p.z + uDt * 0.22;

  vec2 vel = sampleBilinear(uVelocity, pos, uGridSize).xy;
  vec2 nextPos = pos + uDt * vel * uTexelSize * 1.15;
  float obs = texture(uObstacles, clamp(nextPos, 0.0, 1.0)).r;

  bool outOfBounds = nextPos.x > 0.995 || nextPos.x < 0.0 || nextPos.y < 0.01 || nextPos.y > 0.99;
  if (outOfBounds || obs > 0.5 || age > 1.0) {
    float r1 = hash12(vUv * 137.0 + uTime * 1.37);
    float r2 = hash12(vUv * 293.0 + uTime * 2.11);
    nextPos = vec2(0.005 + r1 * 0.04, 0.04 + r2 * 0.92);
    age = r1 * 0.15;
  }

  outColor = vec4(nextPos, age, length(vel));
}
`;

export const PARTICLE_RENDER_VS = `#version 300 es
precision highp float;
uniform sampler2D uParticles;
uniform int uTexWidth;
uniform float uInflowSpeed;

out float vSpeedNorm;
out float vAlpha;

void main() {
  int idx = gl_VertexID;
  int x = idx % uTexWidth;
  int y = idx / uTexWidth;
  vec4 p = texelFetch(uParticles, ivec2(x, y), 0);

  vec2 clip = p.xy * 2.0 - 1.0;
  gl_Position = vec4(clip, 0.0, 1.0);
  gl_PointSize = 1.9;
  vSpeedNorm = clamp(p.w / max(1.0, uInflowSpeed * 1.65), 0.0, 1.0);
  vAlpha = sin(clamp(p.z, 0.0, 1.0) * 3.14159265);
}
`;

export const PARTICLE_RENDER_FS = `#version 300 es
precision highp float;
in float vSpeedNorm;
in float vAlpha;
out vec4 outColor;

void main() {
  vec2 c = gl_PointCoord - 0.5;
  float r2 = dot(c, c);
  if (r2 > 0.25) discard;
  vec3 slowCol = vec3(0.24, 0.78, 0.98);
  vec3 fastCol = vec3(0.99, 0.86, 0.36);
  vec3 col = mix(slowCol, fastCol, vSpeedNorm);
  outColor = vec4(col, vAlpha * 0.85);
}
`;

export const TRAIL_FADE_FS = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uTrail;
uniform float uFade;

void main() {
  vec4 t = texture(uTrail, vUv);
  outColor = vec4(t.rgb * uFade, t.a * uFade);
}
`;

export const TELEMETRY_ENCODE_FS = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uPressure;
uniform sampler2D uObstacles;
uniform float uPressureScale;

void main() {
  float p = texture(uPressure, vUv).x;
  float obs = texture(uObstacles, vUv).r;
  float norm = clamp((p / max(1e-4, uPressureScale)) * 0.5 + 0.5, 0.0, 1.0);
  float packed16 = floor(norm * 65535.0 + 0.5);
  float hi = floor(packed16 / 256.0) / 255.0;
  float lo = mod(packed16, 256.0) / 255.0;
  outColor = vec4(hi, lo, obs > 0.5 ? 1.0 : 0.0, 1.0);
}
`;

export const DISPLAY_FS = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uDye;
uniform sampler2D uVelocity;
uniform sampler2D uPressure;
uniform sampler2D uCurl;
uniform sampler2D uObstacles;
uniform sampler2D uTrails;
uniform vec2 uTexelSize;
uniform float uInflowSpeed;
uniform int uVisMode; // 0=Dye, 1=Vorticity, 2=Velocity, 3=Pressure, 4=Streamlines

vec3 colormapVelocity(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 c0 = vec3(0.04, 0.07, 0.14);
  vec3 c1 = vec3(0.06, 0.36, 0.56);
  vec3 c2 = vec3(0.11, 0.72, 0.62);
  vec3 c3 = vec3(0.95, 0.76, 0.22);
  vec3 c4 = vec3(0.99, 0.32, 0.16);
  if (t < 0.25) return mix(c0, c1, t / 0.25);
  if (t < 0.55) return mix(c1, c2, (t - 0.25) / 0.30);
  if (t < 0.80) return mix(c2, c3, (t - 0.55) / 0.25);
  return mix(c3, c4, (t - 0.80) / 0.20);
}

vec3 colormapDiverging(float v) {
  // v in [-1, 1] -> cyan/blue (-) -> dark slate (0) -> orange/red (+)
  float s = clamp(v, -1.0, 1.0);
  vec3 negFar = vec3(0.12, 0.68, 0.98);
  vec3 negMid = vec3(0.07, 0.28, 0.58);
  vec3 zero   = vec3(0.06, 0.08, 0.12);
  vec3 posMid = vec3(0.72, 0.24, 0.10);
  vec3 posFar = vec3(1.00, 0.62, 0.18);
  if (s < -0.45) return mix(negMid, negFar, (-s - 0.45) / 0.55);
  if (s < 0.0)   return mix(zero, negMid, -s / 0.45);
  if (s < 0.45)  return mix(zero, posMid, s / 0.45);
  return mix(posMid, posFar, (s - 0.45) / 0.55);
}

void main() {
  float obs = texture(uObstacles, vUv).r;
  vec2 vel = texture(uVelocity, vUv).xy;
  float speed = length(vel);
  float uRef = max(1.0, uInflowSpeed);

  vec3 col = vec3(0.04, 0.06, 0.09);

  if (uVisMode == 0) {
    // Smoke Dye mode
    vec3 dye = texture(uDye, vUv).rgb;
    float bgVel = clamp(speed / (uRef * 2.0), 0.0, 1.0) * 0.07;
    col = vec3(0.035, 0.05, 0.075) + vec3(0.06, 0.12, 0.20) * bgVel + dye;
  } else if (uVisMode == 1) {
    // Vorticity (blue/red curl)
    float curl = texture(uCurl, vUv).x;
    col = colormapDiverging(curl / max(8.0, uRef * 0.35));
  } else if (uVisMode == 2) {
    // Velocity Magnitude
    col = colormapVelocity(speed / (uRef * 1.75));
  } else if (uVisMode == 3) {
    // Pressure Field
    float p = texture(uPressure, vUv).x;
    col = colormapDiverging(p / max(8.0, uRef * 0.85));
  } else {
    // Streamlines / Tracer Particles
    vec3 base = colormapVelocity(speed / (uRef * 1.9)) * 0.35;
    vec3 trail = texture(uTrails, vUv).rgb;
    col = base + trail;
  }

  // Subtle calibration grid lines
  vec2 gridUv = vUv * vec2(24.0, 12.0);
  vec2 gridDist = abs(fract(gridUv - 0.5) - 0.5) / fwidth(gridUv);
  float gridLine = 1.0 - min(min(gridDist.x, gridDist.y), 1.0);
  col = mix(col, col + vec3(0.06, 0.08, 0.11), gridLine * 0.35);

  // Crisp obstacle silhouette + metallic edge highlight
  if (obs > 0.5) {
    float nObs =
      texture(uObstacles, vUv - vec2(uTexelSize.x, 0.0)).r +
      texture(uObstacles, vUv + vec2(uTexelSize.x, 0.0)).r +
      texture(uObstacles, vUv - vec2(0.0, uTexelSize.y)).r +
      texture(uObstacles, vUv + vec2(0.0, uTexelSize.y)).r;
    bool isEdge = nObs < 3.9;
    vec3 bodyCol = vec3(0.14, 0.17, 0.22);
    vec3 edgeCol = vec3(0.68, 0.78, 0.90);
    col = isEdge ? edgeCol : bodyCol;
  }

  outColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`;
