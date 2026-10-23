import { describe, it } from "node:test";
import assert from "node:assert/strict";

export type PhasePolarity = 1 | -1;

export interface ArgonHarnessConfig {
  readonly tileDim: number;
  readonly learningRate: number;
  readonly goodnessThreshold: number;
  readonly weightDecay: number;
}

export interface ArgonTickResult {
  readonly sequenceId: number;
  readonly goodness: number;
  readonly normalizedActivations: Float32Array;
}

function readFloat32At(buffer: Float32Array, index: number): number {
  const val = buffer[index];
  if (val === undefined) {
    throw new RangeError(`Float32Array index out of bounds: ${String(index)} (len=${String(buffer.length)})`);
  }
  return val;
}

function writeFloat32At(buffer: Float32Array, index: number, value: number): void {
  if (index < 0 || index >= buffer.length) {
    throw new RangeError(`Float32Array write index out of bounds: ${String(index)} (len=${String(buffer.length)})`);
  }
  buffer[index] = value;
}

export function validateHarnessConfig(config: unknown): config is ArgonHarnessConfig {
  if (typeof config !== "object" || config === null) {
    return false;
  }
  const candidate = config as Record<string, unknown>;
  const tileDim = candidate["tileDim"];
  const learningRate = candidate["learningRate"];
  const goodnessThreshold = candidate["goodnessThreshold"];
  const weightDecay = candidate["weightDecay"];

  return (
    typeof tileDim === "number" &&
    Number.isInteger(tileDim) &&
    tileDim > 0 &&
    tileDim % 4 === 0 &&
    typeof learningRate === "number" &&
    Number.isFinite(learningRate) &&
    learningRate > 0 &&
    learningRate <= 1.0 &&
    typeof goodnessThreshold === "number" &&
    Number.isFinite(goodnessThreshold) &&
    goodnessThreshold > 0 &&
    typeof weightDecay === "number" &&
    Number.isFinite(weightDecay) &&
    weightDecay >= 0 &&
    weightDecay < 1.0
  );
}

export class ArgonOrchestrator {
  private readonly config: ArgonHarnessConfig;
  private readonly weights: Float32Array;
  private readonly scratchActivations: Float32Array;
  private sequenceCounter: number = 0;

  public constructor(config: ArgonHarnessConfig) {
    if (!validateHarnessConfig(config)) {
      throw new Error("Invalid ArgonHarnessConfig supplied to ArgonOrchestrator.");
    }
    this.config = config;
    const totalWeights = config.tileDim * config.tileDim;
    this.weights = new Float32Array(totalWeights);
    this.scratchActivations = new Float32Array(config.tileDim);

    for (let i = 0; i < config.tileDim; i += 1) {
      for (let j = 0; j < config.tileDim; j += 1) {
        const idx = i * config.tileDim + j;
        writeFloat32At(this.weights, idx, i === j ? 0.45 : 0.02);
      }
    }
  }

  public step(sensoryInput: Float32Array, polarity: PhasePolarity): ArgonTickResult {
    const dim = this.config.tileDim;
    if (sensoryInput.length !== dim) {
      throw new Error(`Dimension mismatch: expected ${String(dim)}, received ${String(sensoryInput.length)}`);
    }

    let sumSquared = 0.0;
    for (let i = 0; i < dim; i += 1) {
      let acc = 0.0;
      const rowOffset = i * dim;
      for (let j = 0; j < dim; j += 1) {
        const w = readFloat32At(this.weights, rowOffset + j);
        const x = readFloat32At(sensoryInput, j);
        acc += w * x;
      }
      const relu = acc > 0.0 ? acc : 0.0;
      writeFloat32At(this.scratchActivations, i, relu);
      sumSquared += relu * relu;
    }

    const goodness = sumSquared / dim;
    const margin = polarity * (this.config.goodnessThreshold - goodness);
    const modulation = 1.0 / (1.0 + Math.exp(-margin));
    const effectiveLr = this.config.learningRate * polarity * modulation;
    const invRms = 1.0 / (Math.sqrt(goodness) + 1e-6);

    const output = new Float32Array(dim);
    for (let i = 0; i < dim; i += 1) {
      const y = readFloat32At(this.scratchActivations, i);
      const rowOffset = i * dim;
      const postDrive = effectiveLr * y;

      for (let j = 0; j < dim; j += 1) {
        const idx = rowOffset + j;
        const currentW = readFloat32At(this.weights, idx);
        const x = readFloat32At(sensoryInput, j);
        const updatedW = currentW + postDrive * x - this.config.weightDecay * currentW;
        writeFloat32At(this.weights, idx, updatedW);
      }
      writeFloat32At(output, i, y * invRms);
    }

    this.sequenceCounter += 1;
    return {
      sequenceId: this.sequenceCounter,
      goodness,
      normalizedActivations: output,
    };
  }
}

void describe("ArgonOrchestrator strict harness verification", () => {
  void it("increases goodness monotonically under positive phase sensory stream", () => {
    const engine = new ArgonOrchestrator({
      tileDim: 8,
      learningRate: 0.05,
      goodnessThreshold: 3.0,
      weightDecay: 0.001,
    });

    const input = new Float32Array([0.9, 0.4, 0.7, 0.3, 0.8, 0.5, 0.6, 0.2]);
    const firstTick = engine.step(input, 1);
    let lastTick = firstTick;
    for (let step = 0; step < 8; step += 1) {
      lastTick = engine.step(input, 1);
    }

    assert.ok(
      lastTick.goodness > firstTick.goodness,
      `Expected goodness to rise: ${String(firstTick.goodness)} -> ${String(lastTick.goodness)}`
    );
    assert.strictEqual(lastTick.sequenceId, 9);
  });
});
