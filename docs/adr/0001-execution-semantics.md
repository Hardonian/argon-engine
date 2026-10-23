# ADR 0001: Argon execution semantics

Status: Accepted

## Contract

An Argon tile consumes exactly `tile_dim` finite `f32` features. A positive
phase has polarity `+1.0`; a negative phase has polarity `-1.0`. For a dense
row-major weight matrix `W`, one tick computes `z = W x`, `y = relu(z)`, and
`goodness = mean(y²)`. Output activations are `y / (sqrt(goodness) + 1e-6)`.

The local update is applied after the forward pass:

`W[i,j] += lr * polarity * sigmoid(polarity * (threshold - goodness)) * y[i] * x[j] - decay * W[i,j]`

All CPU, WASM, Triton, and CUDA implementations must match this ordering.
Inputs and outputs may alias only where an API explicitly permits it. Failed
bounded-queue submission must retain caller ownership of the frame.

## Determinism

The scalar CPU backend is the normative `f32` implementation. The physical
`f64` reference is normative for numerical validation. Replays include initial
weights, medium state, configuration, phase schedule, and input stream.
Identical scalar CPU replays must be bit-for-bit equal. Accelerated backends use
the tolerances in `docs/numerics.md`.

## State transition

One closed-loop tick is ordered as: observe, enqueue, forward, local learning,
decode action, integrate physics, measure energy, apply homeostasis, emit
telemetry. Sequence identifiers refer to the observation at the start of the
tick.
