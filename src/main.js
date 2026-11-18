import { PRESETS } from './physics.js';
import { WindTunnelCFD } from './webgl-cfd.js';

const VIS_LEGENDS = [
  {
    label: 'Smoke Rake & Dye Transport',
    gradient: 'linear-gradient(90deg, #09111c, #33b8f5 55%, #fa9438)',
  },
  {
    label: 'Vorticity ω (CW Blue ↔ CCW Amber)',
    gradient: 'linear-gradient(90deg, #1fadfa, #124794 28%, #0f141e 50%, #b83d1a 72%, #ff9e2e)',
  },
  {
    label: 'Velocity Magnitude |u| / U∞',
    gradient: 'linear-gradient(90deg, #0a1224, #0f5c8f 28%, #1cb89e 56%, #f2c238 80%, #fc5229)',
  },
  {
    label: 'Static Pressure (Suction Blue ↔ Stagnation Red)',
    gradient: 'linear-gradient(90deg, #1fadfa, #124794 28%, #0f141e 50%, #b83d1a 72%, #ff9e2e)',
  },
  {
    label: 'GPU Tracer Streamlines (8,192 Particles)',
    gradient: 'linear-gradient(90deg, #0a1224, #3dc7fa 55%, #fcdb5c)',
  },
];

const state = {
  running: true,
  visMode: 0,
  preset: 'naca2412',
  aoaDeg: 8,
  inflowSpeed: 125,
  vorticity: 14,
  jacobiIterations: 36,
  viscosity: 0.0005,
  noSlip: true,
  smokeRakes: true,
  tool: 'dye',
  brushRadius: 0.032,
  clHistory: [],
  cdHistory: [],
};

const MAX_HISTORY = 160;

const canvas = document.getElementById('cfd-canvas');
const errorBanner = document.getElementById('error-banner');
const errorMessage = document.getElementById('error-message');
const sparkCanvas = document.getElementById('sparkline-canvas');
const sparkCtx = sparkCanvas.getContext('2d');

let cfd = null;

try {
  cfd = new WindTunnelCFD(canvas);
} catch (err) {
  errorMessage.textContent = err instanceof Error ? err.message : String(err);
  errorBanner.classList.remove('hidden');
}

// DOM bindings
const btnPlay = document.getElementById('btn-play');
const btnStep = document.getElementById('btn-step');
const btnReset = document.getElementById('btn-reset');
const btnClearWalls = document.getElementById('btn-clear-walls');
const presetSelect = document.getElementById('preset-select');
const presetTag = document.getElementById('preset-tag');
const aoaSlider = document.getElementById('aoa-slider');
const aoaVal = document.getElementById('aoa-val');
const speedSlider = document.getElementById('speed-slider');
const speedVal = document.getElementById('speed-val');
const curlSlider = document.getElementById('curl-slider');
const curlVal = document.getElementById('curl-val');
const jacobiSlider = document.getElementById('jacobi-slider');
const jacobiVal = document.getElementById('jacobi-val');
const viscSlider = document.getElementById('visc-slider');
const viscVal = document.getElementById('visc-val');
const noSlipToggle = document.getElementById('noslip-toggle');
const rakesToggle = document.getElementById('rakes-toggle');
const brushSlider = document.getElementById('brush-slider');
const brushVal = document.getElementById('brush-val');
const fpsReadout = document.getElementById('fps-readout');
const valCl = document.getElementById('val-cl');
const valCd = document.getElementById('val-cd');
const valLd = document.getElementById('val-ld');
const valRe = document.getElementById('val-re');
const legendLabel = document.getElementById('vis-legend-label');
const legendScale = document.getElementById('legend-scale');

function updateReadouts() {
  const presetInfo = PRESETS[state.preset];
  presetTag.textContent = presetInfo ? presetInfo.label.split(' (')[0] : state.preset;
  aoaSlider.disabled = presetInfo ? !presetInfo.supportsAoA : false;
  aoaVal.textContent = aoaSlider.disabled
    ? 'N/A'
    : `${state.aoaDeg > 0 ? '+' : ''}${state.aoaDeg}°`;
  const uMeters = (state.inflowSpeed / 120) * 15.0;
  speedVal.textContent = `${uMeters.toFixed(1)} m/s`;
  curlVal.textContent = state.vorticity.toFixed(1);
  jacobiVal.textContent = String(state.jacobiIterations);
  viscVal.textContent = state.viscosity.toFixed(4);
  brushVal.textContent = state.brushRadius.toFixed(3);
}

function setVisMode(mode) {
  state.visMode = mode;
  document.querySelectorAll('.vis-btn').forEach((btn) => {
    btn.classList.toggle('active', Number(btn.dataset.vis) === mode);
  });
  const leg = VIS_LEGENDS[mode] || VIS_LEGENDS[0];
  legendLabel.textContent = leg.label;
  legendScale.style.background = leg.gradient;
}

document.querySelectorAll('.vis-btn').forEach((btn) => {
  btn.addEventListener('click', () => setVisMode(Number(btn.dataset.vis)));
});

document.querySelectorAll('.tool-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    state.tool = btn.dataset.tool;
    document.querySelectorAll('.tool-btn').forEach((b) => {
      b.classList.toggle('active', b === btn);
    });
  });
});

btnPlay.addEventListener('click', () => {
  state.running = !state.running;
  btnPlay.textContent = state.running ? 'Pause' : 'Resume';
  btnPlay.classList.toggle('primary', state.running);
});

btnStep.addEventListener('click', () => {
  if (!cfd) return;
  if (state.running) {
    state.running = false;
    btnPlay.textContent = 'Resume';
    btnPlay.classList.remove('primary');
  }
  cfd.step({ ...state, dt: 1 / 60 });
  cfd.render(state.visMode, state.inflowSpeed);
  updateTelemetry();
});

btnReset.addEventListener('click', () => {
  if (!cfd) return;
  state.clHistory.length = 0;
  state.cdHistory.length = 0;
  cfd.resetFlow(state.inflowSpeed);
});

btnClearWalls.addEventListener('click', () => {
  if (!cfd) return;
  cfd.clearObstacles();
});

presetSelect.addEventListener('change', () => {
  state.preset = presetSelect.value;
  updateReadouts();
  if (cfd) {
    cfd.setPreset(state.preset, state.aoaDeg);
    cfd.resetFlow(state.inflowSpeed);
    state.clHistory.length = 0;
    state.cdHistory.length = 0;
  }
});

aoaSlider.addEventListener('input', () => {
  state.aoaDeg = Number(aoaSlider.value);
  updateReadouts();
  if (cfd) {
    cfd.setPreset(state.preset, state.aoaDeg);
  }
});

speedSlider.addEventListener('input', () => {
  state.inflowSpeed = Number(speedSlider.value);
  updateReadouts();
});

curlSlider.addEventListener('input', () => {
  state.vorticity = Number(curlSlider.value);
  updateReadouts();
});

jacobiSlider.addEventListener('input', () => {
  state.jacobiIterations = Number(jacobiSlider.value);
  updateReadouts();
});

viscSlider.addEventListener('input', () => {
  state.viscosity = Number(viscSlider.value);
  updateReadouts();
});

noSlipToggle.addEventListener('change', () => {
  state.noSlip = noSlipToggle.checked;
});

rakesToggle.addEventListener('change', () => {
  state.smokeRakes = rakesToggle.checked;
});

brushSlider.addEventListener('input', () => {
  state.brushRadius = Number(brushSlider.value);
  updateReadouts();
});

// Pointer interaction on wind tunnel canvas
let pointerDown = false;
let lastPointer = null;

function pointerToUV(e) {
  const rect = canvas.getBoundingClientRect();
  const u = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  const v = Math.max(0, Math.min(1, 1 - (e.clientY - rect.top) / rect.height));
  return { u, v };
}

function handlePointerAction(curr) {
  if (!cfd) return;
  if (state.tool === 'draw' || state.tool === 'erase') {
    cfd.applyObstacleBrush(curr.u, curr.v, state.brushRadius, state.tool === 'draw');
  } else {
    const dx = lastPointer ? (curr.u - lastPointer.u) * 2600 : 80;
    const dy = lastPointer ? (curr.v - lastPointer.v) * 1300 : 0;
    const hueShift = (cfd.time * 0.6) % 1;
    const color =
      hueShift < 0.5 ? [0.22, 0.78, 1.0] : [1.0, 0.55, 0.18];
    cfd.queueSplat(curr.u, curr.v, dx, dy, color, state.brushRadius);
  }
}

canvas.addEventListener('pointerdown', (e) => {
  pointerDown = true;
  canvas.setPointerCapture(e.pointerId);
  const uv = pointerToUV(e);
  handlePointerAction(uv);
  lastPointer = uv;
});

canvas.addEventListener('pointermove', (e) => {
  if (!pointerDown) return;
  const uv = pointerToUV(e);
  handlePointerAction(uv);
  lastPointer = uv;
});

function releasePointer() {
  pointerDown = false;
  lastPointer = null;
}

canvas.addEventListener('pointerup', releasePointer);
canvas.addEventListener('pointercancel', releasePointer);

function drawSparkline() {
  const w = sparkCanvas.width;
  const h = sparkCanvas.height;
  sparkCtx.clearRect(0, 0, w, h);

  sparkCtx.fillStyle = '#0c1017';
  sparkCtx.fillRect(0, 0, w, h);

  const clArr = state.clHistory;
  const cdArr = state.cdHistory;
  if (clArr.length < 2) return;

  let minVal = -0.2;
  let maxVal = 0.8;
  for (let i = 0; i < clArr.length; i++) {
    if (clArr[i] < minVal) minVal = clArr[i];
    if (clArr[i] > maxVal) maxVal = clArr[i];
    if (cdArr[i] < minVal) minVal = cdArr[i];
    if (cdArr[i] > maxVal) maxVal = cdArr[i];
  }
  const pad = Math.max(0.15, (maxVal - minVal) * 0.12);
  minVal -= pad;
  maxVal += pad;
  const span = Math.max(1e-3, maxVal - minVal);
  const toY = (val) => h - 4 - ((val - minVal) / span) * (h - 8);

  // Zero reference line
  const zeroY = toY(0);
  if (zeroY >= 0 && zeroY <= h) {
    sparkCtx.strokeStyle = '#243044';
    sparkCtx.lineWidth = 1;
    sparkCtx.beginPath();
    sparkCtx.moveTo(0, zeroY);
    sparkCtx.lineTo(w, zeroY);
    sparkCtx.stroke();
  }

  const plotSeries = (arr, color) => {
    sparkCtx.strokeStyle = color;
    sparkCtx.lineWidth = 1.6;
    sparkCtx.beginPath();
    for (let i = 0; i < arr.length; i++) {
      const x = (i / (MAX_HISTORY - 1)) * w;
      const y = toY(arr[i]);
      if (i === 0) sparkCtx.moveTo(x, y);
      else sparkCtx.lineTo(x, y);
    }
    sparkCtx.stroke();
  };

  plotSeries(cdArr, '#f59e0b');
  plotSeries(clArr, '#38bdf8');
}

function formatSci(val) {
  if (!Number.isFinite(val) || val <= 0) return '0.0e0';
  const exp = Math.floor(Math.log10(val));
  const mant = val / 10 ** exp;
  return `${mant.toFixed(1)}e${exp}`;
}

function updateTelemetry() {
  if (!cfd) return;
  const t = cfd.readTelemetry({ ...state, dt: 1 / 60 });
  valCl.textContent = t.cl.toFixed(3);
  valCd.textContent = t.cd.toFixed(3);
  valLd.textContent = t.ldRatio.toFixed(2);
  valRe.textContent = formatSci(t.reynolds);

  state.clHistory.push(t.cl);
  state.cdHistory.push(t.cd);
  if (state.clHistory.length > MAX_HISTORY) {
    state.clHistory.shift();
    state.cdHistory.shift();
  }
  drawSparkline();
}

updateReadouts();
setVisMode(0);

let frameCount = 0;
let lastFpsTime = performance.now();
let telemCounter = 0;

function tick(now) {
  if (cfd) {
    if (state.running) {
      cfd.step({ ...state, dt: 1 / 60 });
      telemCounter++;
      if (telemCounter % 5 === 0) {
        updateTelemetry();
      }
    }
    cfd.render(state.visMode, state.inflowSpeed);

    frameCount++;
    if (now - lastFpsTime >= 500) {
      const fps = (frameCount * 1000) / (now - lastFpsTime);
      fpsReadout.textContent = `FPS: ${fps.toFixed(0)}`;
      frameCount = 0;
      lastFpsTime = now;
    }
  }
  requestAnimationFrame(tick);
}

requestAnimationFrame(tick);
