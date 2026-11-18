import {
  BASE_VS,
  SEMI_LAGRANGIAN_FS,
  MACCORMACK_FS,
  CURL_FS,
  VORTICITY_CONFINEMENT_FS,
  VISCOSITY_JACOBI_FS,
  DIVERGENCE_FS,
  JACOBI_PRESSURE_FS,
  GRADIENT_SUBTRACT_FS,
  WIND_DYE_INFLOW_FS,
  OBSTACLE_BRUSH_FS,
  PARTICLE_ADVECT_FS,
  PARTICLE_RENDER_VS,
  PARTICLE_RENDER_FS,
  TRAIL_FADE_FS,
  TELEMETRY_ENCODE_FS,
  DISPLAY_FS,
} from './shaders.js';
import {
  generateObstacleField,
  decodeTelemetryBuffer,
  integrateSurfaceForces,
  computeAeroTelemetry,
} from './physics.js';

export const SIM_WIDTH = 448;
export const SIM_HEIGHT = 224;
export const TELEM_WIDTH = 128;
export const TELEM_HEIGHT = 64;
export const PARTICLE_TEX_W = 128;
export const PARTICLE_TEX_H = 64; // 8,192 GPU tracer particles
export const TRAIL_WIDTH = 896;
export const TRAIL_HEIGHT = 448;

export class WebGL2CapabilityError extends Error {
  constructor(message) {
    super(message);
    this.name = 'WebGL2CapabilityError';
  }
}

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader) || 'Unknown shader compile error';
    gl.deleteShader(shader);
    throw new Error(`Shader compilation failed:\n${info}`);
  }
  return shader;
}

function createProgram(gl, vsSource, fsSource) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSource);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSource);
  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program) || 'Unknown program link error';
    gl.deleteProgram(program);
    throw new Error(`Program link failed:\n${info}`);
  }

  const uniforms = {};
  const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < count; i++) {
    const info = gl.getActiveUniform(program, i);
    if (info) {
      uniforms[info.name] = gl.getUniformLocation(program, info.name);
    }
  }
  return { program, uniforms };
}

function createFBO(gl, width, height, internalFormat, format, type, filter = gl.NEAREST) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, null);

  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    gl.deleteTexture(tex);
    gl.deleteFramebuffer(fbo);
    throw new WebGL2CapabilityError(
      `Framebuffer incomplete for format ${internalFormat} (status 0x${status.toString(16)}).`,
    );
  }

  return {
    width,
    height,
    tex,
    fbo,
    attach(unit) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      return unit;
    },
  };
}

function createPingPong(gl, width, height, internalFormat, format, type, filter = gl.NEAREST) {
  let read = createFBO(gl, width, height, internalFormat, format, type, filter);
  let write = createFBO(gl, width, height, internalFormat, format, type, filter);
  return {
    get read() {
      return read;
    },
    get write() {
      return write;
    },
    swap() {
      const tmp = read;
      read = write;
      write = tmp;
    },
  };
}

export class WindTunnelCFD {
  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      powerPreference: 'high-performance',
    });

    if (!gl) {
      throw new WebGL2CapabilityError(
        'WebGL2 is unavailable in this browser or device. Enable hardware acceleration or use a WebGL2-capable browser.',
      );
    }

    const extFloat = gl.getExtension('EXT_color_buffer_float');
    const extHalfFloat = gl.getExtension('EXT_color_buffer_half_float');
    if (!extFloat && !extHalfFloat) {
      throw new WebGL2CapabilityError(
        'Required WebGL2 floating-point render target extension (EXT_color_buffer_float or EXT_color_buffer_half_float) is unsupported on this GPU.',
      );
    }

    this.gl = gl;
    this.simWidth = SIM_WIDTH;
    this.simHeight = SIM_HEIGHT;
    this.texelSize = [1 / SIM_WIDTH, 1 / SIM_HEIGHT];
    this.refLengthCells = 0.62 * SIM_HEIGHT;
    this.time = 0;
    this.splatQueue = [];
    this.telemBytes = new Uint8Array(TELEM_WIDTH * TELEM_HEIGHT * 4);

    this._initQuad();
    this._initPrograms();
    this._initFBOs();
    this._seedParticles();
    this.setPreset('naca2412', 8);
    this.resetFlow(125);
  }

  _initQuad() {
    const gl = this.gl;
    this.quadVAO = gl.createVertexArray();
    gl.bindVertexArray(this.quadVAO);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    this.particleVAO = gl.createVertexArray();
    gl.bindVertexArray(null);
  }

  _initPrograms() {
    const gl = this.gl;
    this.prog = {
      semiLag: createProgram(gl, BASE_VS, SEMI_LAGRANGIAN_FS),
      macCormack: createProgram(gl, BASE_VS, MACCORMACK_FS),
      curl: createProgram(gl, BASE_VS, CURL_FS),
      vorticity: createProgram(gl, BASE_VS, VORTICITY_CONFINEMENT_FS),
      viscosity: createProgram(gl, BASE_VS, VISCOSITY_JACOBI_FS),
      divergence: createProgram(gl, BASE_VS, DIVERGENCE_FS),
      jacobi: createProgram(gl, BASE_VS, JACOBI_PRESSURE_FS),
      gradSub: createProgram(gl, BASE_VS, GRADIENT_SUBTRACT_FS),
      inflow: createProgram(gl, BASE_VS, WIND_DYE_INFLOW_FS),
      obsBrush: createProgram(gl, BASE_VS, OBSTACLE_BRUSH_FS),
      particleAdvect: createProgram(gl, BASE_VS, PARTICLE_ADVECT_FS),
      particleRender: createProgram(gl, PARTICLE_RENDER_VS, PARTICLE_RENDER_FS),
      trailFade: createProgram(gl, BASE_VS, TRAIL_FADE_FS),
      telemEncode: createProgram(gl, BASE_VS, TELEMETRY_ENCODE_FS),
      display: createProgram(gl, BASE_VS, DISPLAY_FS),
    };
  }

  _initFBOs() {
    const gl = this.gl;
    const W = SIM_WIDTH;
    const H = SIM_HEIGHT;

    this.velocity = createPingPong(gl, W, H, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.NEAREST);
    this.pressure = createPingPong(gl, W, H, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.NEAREST);
    this.dye = createPingPong(gl, W, H, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.NEAREST);
    this.curl = createFBO(gl, W, H, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.NEAREST);
    this.divergence = createFBO(gl, W, H, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.NEAREST);
    this.predictor = createFBO(gl, W, H, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.NEAREST);
    this.obstacles = createPingPong(gl, W, H, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.NEAREST);

    this.particles = createPingPong(
      gl,
      PARTICLE_TEX_W,
      PARTICLE_TEX_H,
      gl.RGBA16F,
      gl.RGBA,
      gl.HALF_FLOAT,
      gl.NEAREST,
    );
    this.trails = createPingPong(
      gl,
      TRAIL_WIDTH,
      TRAIL_HEIGHT,
      gl.RGBA8,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      gl.LINEAR,
    );
    this.telemetry = createFBO(
      gl,
      TELEM_WIDTH,
      TELEM_HEIGHT,
      gl.RGBA8,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      gl.NEAREST,
    );
  }

  _seedParticles() {
    const gl = this.gl;
    const count = PARTICLE_TEX_W * PARTICLE_TEX_H;
    // Pack into Float32 -> upload to RGBA16F via texImage2D(RGBA16F, RGBA, FLOAT)
    const data = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
      data[i * 4 + 0] = Math.random();
      data[i * 4 + 1] = 0.04 + Math.random() * 0.92;
      data[i * 4 + 2] = Math.random();
      data[i * 4 + 3] = 0;
    }
    for (const target of [this.particles.read, this.particles.write]) {
      gl.bindTexture(gl.TEXTURE_2D, target.tex);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA16F,
        PARTICLE_TEX_W,
        PARTICLE_TEX_H,
        0,
        gl.RGBA,
        gl.FLOAT,
        data,
      );
    }
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  _blit(targetFBO, progObj, setUniforms) {
    const gl = this.gl;
    gl.useProgram(progObj.program);
    setUniforms(progObj.uniforms);
    gl.bindVertexArray(this.quadVAO);
    if (targetFBO) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, targetFBO.fbo);
      gl.viewport(0, 0, targetFBO.width, targetFBO.height);
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    }
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  setPreset(presetId, aoaDeg) {
    const gl = this.gl;
    const { mask, refLengthCells } = generateObstacleField(SIM_WIDTH, SIM_HEIGHT, presetId, aoaDeg);
    this.refLengthCells = refLengthCells;

    const rgba = new Uint8Array(SIM_WIDTH * SIM_HEIGHT * 4);
    for (let i = 0; i < mask.length; i++) {
      rgba[i * 4] = mask[i] >= 0.5 ? 255 : 0;
      rgba[i * 4 + 3] = 255;
    }
    for (const target of [this.obstacles.read, this.obstacles.write]) {
      gl.bindTexture(gl.TEXTURE_2D, target.tex);
      gl.texSubImage2D(
        gl.TEXTURE_2D,
        0,
        0,
        0,
        SIM_WIDTH,
        SIM_HEIGHT,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        rgba,
      );
    }
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  clearObstacles() {
    const gl = this.gl;
    for (const target of [this.obstacles.read, this.obstacles.write]) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  applyObstacleBrush(u, v, radius, isDraw) {
    this._blit(this.obstacles.write, this.prog.obsBrush, (uni) => {
      glUniform1i(this.gl, uni.uObstacles, this.obstacles.read.attach(0));
      this.gl.uniform2fv(uni.uTexelSize, this.texelSize);
      this.gl.uniform2f(uni.uBrushPos, u, v);
      this.gl.uniform1f(uni.uBrushRadius, radius);
      this.gl.uniform1f(uni.uBrushValue, isDraw ? 1.0 : 0.0);
    });
    this.obstacles.swap();
  }

  queueSplat(u, v, forceX, forceY, color, radius = 0.032) {
    this.splatQueue.push({ u, v, forceX, forceY, color, radius });
  }

  resetFlow(inflowSpeed) {
    const gl = this.gl;
    // Seed uniform inflow velocity field
    const velData = new Float32Array(SIM_WIDTH * SIM_HEIGHT * 4);
    for (let i = 0; i < SIM_WIDTH * SIM_HEIGHT; i++) {
      velData[i * 4] = inflowSpeed;
    }
    for (const target of [this.velocity.read, this.velocity.write]) {
      gl.bindTexture(gl.TEXTURE_2D, target.tex);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA16F,
        SIM_WIDTH,
        SIM_HEIGHT,
        0,
        gl.RGBA,
        gl.FLOAT,
        velData,
      );
    }
    gl.bindTexture(gl.TEXTURE_2D, null);

    const clearList = [
      this.pressure.read,
      this.pressure.write,
      this.dye.read,
      this.dye.write,
      this.curl,
      this.divergence,
      this.trails.read,
      this.trails.write,
    ];
    for (const fbo of clearList) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.fbo);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this._seedParticles();
  }

  /**
   * Advance the WebGL2 Navier-Stokes solver by one frame (2 substeps for CFL stability).
   */
  step(params) {
    const gl = this.gl;
    const substeps = 2;
    const dt = (params.dt ?? 1 / 60) / substeps;
    const {
      inflowSpeed,
      vorticity,
      jacobiIterations,
      viscosity,
      noSlip,
      smokeRakes,
      visMode,
    } = params;

    for (let s = 0; s < substeps; s++) {
      this.time += dt;
      const splat = s === 0 ? this.splatQueue.shift() || null : null;

      // 1. Inflow smoke rakes + user jet/dye splats
      this._applyInflowAndSplat(this.dye, 0, smokeRakes, splat);
      if (splat) {
        this._applyInflowAndSplat(this.velocity, 1, smokeRakes, splat);
      }

      // 2. MacCormack velocity advection
      this._advectMacCormack(this.velocity, dt, 1.0);

      // 3. Optional viscous diffusion
      if (viscosity > 0.0005) {
        for (let k = 0; k < 2; k++) {
          this._blit(this.velocity.write, this.prog.viscosity, (uni) => {
            glUniform1i(gl, uni.uVelocity, this.velocity.read.attach(0));
            glUniform1i(gl, uni.uObstacles, this.obstacles.read.attach(1));
            gl.uniform2fv(uni.uTexelSize, this.texelSize);
            gl.uniform1f(uni.uViscosity, viscosity);
            gl.uniform1f(uni.uDt, dt);
          });
          this.velocity.swap();
        }
      }

      // 4. Curl & Vorticity Confinement
      this._blit(this.curl, this.prog.curl, (uni) => {
        glUniform1i(gl, uni.uVelocity, this.velocity.read.attach(0));
        glUniform1i(gl, uni.uObstacles, this.obstacles.read.attach(1));
        gl.uniform2fv(uni.uTexelSize, this.texelSize);
      });

      if (vorticity > 0.01) {
        this._blit(this.velocity.write, this.prog.vorticity, (uni) => {
          glUniform1i(gl, uni.uVelocity, this.velocity.read.attach(0));
          glUniform1i(gl, uni.uCurl, this.curl.attach(1));
          glUniform1i(gl, uni.uObstacles, this.obstacles.read.attach(2));
          gl.uniform2fv(uni.uTexelSize, this.texelSize);
          gl.uniform1f(uni.uCurlStrength, vorticity);
          gl.uniform1f(uni.uDt, dt);
        });
        this.velocity.swap();
      }

      // 5. Divergence of intermediate velocity
      this._blit(this.divergence, this.prog.divergence, (uni) => {
        glUniform1i(gl, uni.uVelocity, this.velocity.read.attach(0));
        glUniform1i(gl, uni.uObstacles, this.obstacles.read.attach(1));
        gl.uniform2fv(uni.uTexelSize, this.texelSize);
      });

      // 6. Jacobi Pressure Poisson iterations (warm-started from previous frame pressure)
      for (let it = 0; it < jacobiIterations; it++) {
        this._blit(this.pressure.write, this.prog.jacobi, (uni) => {
          glUniform1i(gl, uni.uPressure, this.pressure.read.attach(0));
          glUniform1i(gl, uni.uDivergence, this.divergence.attach(1));
          glUniform1i(gl, uni.uObstacles, this.obstacles.read.attach(2));
          gl.uniform2fv(uni.uTexelSize, this.texelSize);
        });
        this.pressure.swap();
      }

      // 7. Gradient projection + inflow/outflow/obstacle BC
      this._blit(this.velocity.write, this.prog.gradSub, (uni) => {
        glUniform1i(gl, uni.uPressure, this.pressure.read.attach(0));
        glUniform1i(gl, uni.uVelocity, this.velocity.read.attach(1));
        glUniform1i(gl, uni.uObstacles, this.obstacles.read.attach(2));
        gl.uniform2fv(uni.uTexelSize, this.texelSize);
        gl.uniform1f(uni.uInflowSpeed, inflowSpeed);
        gl.uniform1i(uni.uNoSlip, noSlip ? 1 : 0);
      });
      this.velocity.swap();

      // 8. MacCormack dye advection
      this._advectMacCormack(this.dye, dt, 0.996);

      // 9. GPU Tracer Particle advection & trail accumulation
      if (visMode === 4) {
        this._stepParticles(dt, inflowSpeed);
      }
    }
  }

  _applyInflowAndSplat(pingPong, isVelocity, smokeRakes, splat) {
    const gl = this.gl;
    this._blit(pingPong.write, this.prog.inflow, (uni) => {
      glUniform1i(gl, uni.uSource, pingPong.read.attach(0));
      glUniform1i(gl, uni.uObstacles, this.obstacles.read.attach(1));
      gl.uniform2fv(uni.uTexelSize, this.texelSize);
      gl.uniform1f(uni.uSmokeRakes, smokeRakes ? 1.0 : 0.0);
      gl.uniform1f(uni.uTime, this.time);
      gl.uniform1i(uni.uIsVelocity, isVelocity);
      if (splat) {
        gl.uniform1i(uni.uSplatActive, 1);
        gl.uniform2f(uni.uSplatPos, splat.u, splat.v);
        gl.uniform2f(uni.uSplatForce, splat.forceX, splat.forceY);
        gl.uniform3fv(uni.uSplatColor, splat.color);
        gl.uniform1f(uni.uSplatRadius, splat.radius);
      } else {
        gl.uniform1i(uni.uSplatActive, 0);
      }
    });
    pingPong.swap();
  }

  _advectMacCormack(pingPong, dt, dissipation) {
    const gl = this.gl;
    // Predictor
    this._blit(this.predictor, this.prog.semiLag, (uni) => {
      glUniform1i(gl, uni.uVelocity, this.velocity.read.attach(0));
      glUniform1i(gl, uni.uSource, pingPong.read.attach(1));
      glUniform1i(gl, uni.uObstacles, this.obstacles.read.attach(2));
      gl.uniform2fv(uni.uTexelSize, this.texelSize);
      gl.uniform2f(uni.uGridSize, SIM_WIDTH, SIM_HEIGHT);
      gl.uniform1f(uni.uDt, dt);
      gl.uniform1f(uni.uDirection, 1.0);
    });
    // Corrector + 4-tap clamp
    this._blit(pingPong.write, this.prog.macCormack, (uni) => {
      glUniform1i(gl, uni.uVelocity, this.velocity.read.attach(0));
      glUniform1i(gl, uni.uSource, pingPong.read.attach(1));
      glUniform1i(gl, uni.uPredictor, this.predictor.attach(2));
      glUniform1i(gl, uni.uObstacles, this.obstacles.read.attach(3));
      gl.uniform2fv(uni.uTexelSize, this.texelSize);
      gl.uniform2f(uni.uGridSize, SIM_WIDTH, SIM_HEIGHT);
      gl.uniform1f(uni.uDt, dt);
      gl.uniform1f(uni.uDissipation, dissipation);
    });
    pingPong.swap();
  }

  _stepParticles(dt, inflowSpeed) {
    const gl = this.gl;
    this._blit(this.particles.write, this.prog.particleAdvect, (uni) => {
      glUniform1i(gl, uni.uParticles, this.particles.read.attach(0));
      glUniform1i(gl, uni.uVelocity, this.velocity.read.attach(1));
      glUniform1i(gl, uni.uObstacles, this.obstacles.read.attach(2));
      gl.uniform2fv(uni.uTexelSize, this.texelSize);
      gl.uniform2f(uni.uGridSize, SIM_WIDTH, SIM_HEIGHT);
      gl.uniform1f(uni.uDt, dt);
      gl.uniform1f(uni.uTime, this.time);
    });
    this.particles.swap();

    // Fade previous trail buffer
    this._blit(this.trails.write, this.prog.trailFade, (uni) => {
      glUniform1i(gl, uni.uTrail, this.trails.read.attach(0));
      gl.uniform1f(uni.uFade, 0.90);
    });

    // Blend new particle positions onto trails.write
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    gl.useProgram(this.prog.particleRender.program);
    const pUni = this.prog.particleRender.uniforms;
    glUniform1i(gl, pUni.uParticles, this.particles.read.attach(0));
    gl.uniform1i(pUni.uTexWidth, PARTICLE_TEX_W);
    gl.uniform1f(pUni.uInflowSpeed, inflowSpeed);
    gl.bindVertexArray(this.particleVAO);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.trails.write.fbo);
    gl.viewport(0, 0, TRAIL_WIDTH, TRAIL_HEIGHT);
    gl.drawArrays(gl.POINTS, 0, PARTICLE_TEX_W * PARTICLE_TEX_H);
    gl.disable(gl.BLEND);
    gl.bindVertexArray(null);

    this.trails.swap();
  }

  /**
   * Read downsampled pressure + obstacle mask from GPU and integrate Cl, Cd, L/D, Re.
   */
  readTelemetry(params) {
    const gl = this.gl;
    const pScale = Math.max(20.0, params.inflowSpeed * 3.0);

    this._blit(this.telemetry, this.prog.telemEncode, (uni) => {
      glUniform1i(gl, uni.uPressure, this.pressure.read.attach(0));
      glUniform1i(gl, uni.uObstacles, this.obstacles.read.attach(1));
      gl.uniform1f(uni.uPressureScale, pScale);
    });

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.telemetry.fbo);
    gl.readPixels(0, 0, TELEM_WIDTH, TELEM_HEIGHT, gl.RGBA, gl.UNSIGNED_BYTE, this.telemBytes);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    const { pressure, obstacle } = decodeTelemetryBuffer(
      this.telemBytes,
      TELEM_WIDTH,
      TELEM_HEIGHT,
      pScale,
    );
    const forces = integrateSurfaceForces(pressure, obstacle, TELEM_WIDTH, TELEM_HEIGHT);
    const refLengthTelem = this.refLengthCells * (TELEM_WIDTH / SIM_WIDTH);
    const substepDt = (params.dt ?? 1 / 60) / 2;

    return {
      ...computeAeroTelemetry({
        fx: forces.fx,
        fy: forces.fy,
        uInf: params.inflowSpeed,
        dt: substepDt,
        refLengthCells: refLengthTelem,
        viscosity: params.viscosity,
      }),
      surfaceCells: forces.surfaceCells,
      obstacleCells: forces.obstacleCells,
    };
  }

  render(visMode, inflowSpeed) {
    const gl = this.gl;
    this._blit(null, this.prog.display, (uni) => {
      glUniform1i(gl, uni.uDye, this.dye.read.attach(0));
      glUniform1i(gl, uni.uVelocity, this.velocity.read.attach(1));
      glUniform1i(gl, uni.uPressure, this.pressure.read.attach(2));
      glUniform1i(gl, uni.uCurl, this.curl.attach(3));
      glUniform1i(gl, uni.uObstacles, this.obstacles.read.attach(4));
      glUniform1i(gl, uni.uTrails, this.trails.read.attach(5));
      gl.uniform2fv(uni.uTexelSize, this.texelSize);
      gl.uniform1f(uni.uInflowSpeed, inflowSpeed);
      gl.uniform1i(uni.uVisMode, visMode);
    });
  }
}

function glUniform1i(gl, loc, val) {
  if (loc !== null && loc !== undefined) {
    gl.uniform1i(loc, val);
  }
}
