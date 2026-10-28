# Numerical contract

The oscillator state uses radians, seconds, unit mass, and unit gravity unless
per-site parameters say otherwise. Periodic-ring dynamics are

`theta''[i] = -sin(theta[i]) + kappa*(theta[i-1] - 2*theta[i] + theta[i+1]) - gamma*theta'[i] + u[i]`.

The conservative and externally forced portion uses kick-drift-kick Velocity
Verlet. Linear damping is integrated by exact half-step exponential operators
around that update. Velocity Verlet is symplectic only when damping is zero and
forcing derives from a time-independent potential. It bounds energy error; it
does not conserve the exact Hamiltonian at every discrete step.

For the default `dt=0.005`, conservative tests require maximum relative total
energy error below `1e-4` over 20,000 steps. Scalar CPU replay is bit-exact.
Accelerated `f32` backends require activation, goodness, and weight absolute
error below `2e-5` for one tick and `2e-4` over 1,000 ticks.

Energy is kinetic, on-site pendulum potential, nearest-neighbor coupling
potential, and total. Damped or externally driven runs are not expected to
conserve total energy.

## Low-Bitwidth Quantization Tolerances (Phase 6)

When operating under quantized neuromorphic execution modes (`QuantizedSramTileSimulator`):

- **FP8 (E4M3 / E5M2)**: Single-tick activation relative divergence must remain below `1.5e-2` (1.5%) compared to the 32-bit reference trajectory, with no divergence in contrastive goodness sign.
- **INT4 Logarithmic**: Dynamic homeostatic scale factors bound activation clipping to $< 0.5\%$ of dynamic range.

## Lyapunov Stability Verification Envelope (Phase 7)

For closed-loop physical plants evaluated via `LyapunovStabilityCertifier`:

- Trajectories are certified as provably stable when the estimated maximal Lyapunov exponent $\lambda \le 0.05$ and the energy drift ratio $\Delta E / E_0 < 10.0$ under transient shock perturbations.
- Damped linear regimes require $\lambda < 0$, guaranteeing exponential asymptotic convergence to equilibrium.
