import { WasmArgonOrchestrator } from "../dist/index.js";

let engine;
let output;

self.onmessage = ({ data }) => {
  if (data.type === "initialize") {
    const exports = data.instance.exports;
    engine = new WasmArgonOrchestrator(exports, data.config);
    output = new Float32Array(data.config.tileDim);
    self.postMessage({ type: "ready" });
    return;
  }
  if (data.type === "step" && engine !== undefined && output !== undefined) {
    const result = engine.stepInto(new Float32Array(data.features), data.polarity, output);
    const dimension = output.length;
    self.postMessage(
      {
        type: "tick",
        sequenceId: result.sequenceId,
        goodness: result.goodness,
        activations: output,
      },
      [output.buffer],
    );
    output = new Float32Array(dimension);
  }
};
