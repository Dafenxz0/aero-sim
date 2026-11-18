# AERO-SIM — Interactive WebGL2 Navier-Stokes Wind Tunnel

[![CI & Deploy Pages](https://github.com/Dafenxz0/aero-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/Dafenxz0/aero-sim/actions/workflows/ci.yml)

**Live Demo:** [https://dafenxz0.github.io/aero-sim/](https://dafenxz0.github.io/aero-sim/)

An interactive, real-time 2D incompressible Navier-Stokes computational fluid dynamics (CFD) wind tunnel running on WebGL2 `RGBA16F` ping-pong framebuffers, featuring NACA airfoils, live aerodynamic telemetry ($C_L$, $C_D$, $L/D$, $Re$), vorticity confinement, MacCormack advection, and 8,192 GPGPU tracer particles.

## Physics & Numerical Solver

The solver advances the 2D incompressible Navier-Stokes equations on a $448 \times 224$ `RGBA16F` grid (`EXT_color_buffer_float` / `EXT_color_buffer_half_float` verified at startup with `#error-banner` fallback):

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}_{\text{vort}}, \qquad \nabla \cdot \mathbf{u} = 0$$

### Solver Pipeline (2 substeps/frame)
1. **Inlet & Jet Injection:** Steady wind-tunnel inflow $(U_\infty, 0)$, 11 horizontal smoke rakes, and interactive pointer impulse/dye splats.
2. **MacCormack Advection:** Predictor-corrector 2nd-order advection ($\phi^{n+1} = \hat{\phi}^{n+1} + \frac{1}{2}(\phi^n - \hat{\phi}^n)$) clamped to the 4-texel neighborhood around the departure point, falling back to semi-Lagrangian near solid walls.
3. **Viscous Diffusion:** Optional implicit Jacobi diffusion controlled by kinematic viscosity $\nu$.
4. **Vorticity Confinement:** Computes scalar curl $\omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$ and injects $\mathbf{f}_{\text{vort}} = \epsilon (\mathbf{N} \times \omega)$ where $\mathbf{N} = \nabla|\omega| / (|\nabla|\omega|| + \varepsilon)$ to preserve shed vortices.
5. **Pressure Poisson & Projection:** Solves $\nabla^2 p = \nabla \cdot \mathbf{u}$ via warm-started Jacobi iterations with Neumann obstacle boundary conditions ($\partial p/\partial n = 0$) and Dirichlet outlet ($p = 0$), followed by $\mathbf{u} \leftarrow \mathbf{u} - \nabla p$ and no-slip/free-slip wall enforcement.
6. **Surface Pressure Integration:** Reads a 16-bit packed `RGBA8` pressure/obstacle field (`128×64`) to integrate live surface forces:
   $$C_L = \frac{F_y}{\frac{1}{2}\rho U_\infty^2 L_{\text{ref}}}, \qquad C_D = \frac{F_x}{\frac{1}{2}\rho U_\infty^2 L_{\text{ref}}}, \qquad Re = \frac{U_\infty L_{\text{ref}}}{\nu}$$

### NACA 4-Digit Airfoil Geometry
Thickness distribution (closed trailing edge) and camber line:
$$y_t(x) = 5t\left(0.2969\sqrt{x} - 0.1260x - 0.3516x^2 + 0.2843x^3 - 0.1036x^4\right)$$

## Features & Controls
- **5 Presets:** NACA 2412 (Cambered), NACA 0012 (Symmetric), Cylinder (Von Kármán vortex street), Bluff Car / Wedge, Double Slit / Venturi.
- **Live Aerodynamics:** Angle of Attack (`-25°` to `+25°`), Tunnel Velocity ($U_\infty$), Vorticity Confinement, Jacobi Iterations, Viscosity, No-Slip/Free-Slip toggle, and live dual-channel $C_L / C_D$ sparkline.
- **5 Visualization Modes:** Smoke Dye, Vorticity (blue/amber curl), Velocity Magnitude, Pressure Field, and 8,192 GPU Tracer Streamlines.
- **Interactive Brushes:** Inject Dye/Velocity Jet, Draw Solid Wall, Erase Wall.

## Local Development & Testing

```bash
# Syntax check & unit tests (Node.js >= 20)
node --check src/physics.js src/shaders.js src/webgl-cfd.js src/main.js
node --test

# Serve locally (ES modules require HTTP)
npx serve .
```

## License

[MIT](./LICENSE)
