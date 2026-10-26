#include <cassert>
#include <iostream>
#include <vector>

#include "argon.hpp"

int main() {
  // Initialize a 2-layer, 8-dimensional Argon engine with RAII ownership
  auto engine_res = argon::Engine::Create(2, 8, 0.02f, 1.0f, 0.0005f, 8);
  if (!engine_res.Ok()) {
    std::cerr << "Failed to create Argon engine: " << engine_res.ErrorMessage() << "\n";
    return 1;
  }
  auto engine = std::move(engine_res.Value());
  assert(engine.TileDim() == 8);
  assert(engine.NumLayers() == 2);

  // 1. Zero-copy step using std::span
  const std::vector<float> input = {0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.2f, 0.3f, 0.4f};
  std::vector<float> output(8, 0.0f);

  auto step_res = engine.Step(1, 1.0f, input, output);
  assert(step_res.Ok());
  assert(step_res->sequence_id == 1);
  assert(step_res->total_ticks == 1);
  assert(step_res->mean_goodness > 0.0f);

  // 2. Weight checksum
  auto checksum_res = engine.WeightChecksum();
  assert(checksum_res.Ok());
  uint64_t initial_checksum = checksum_res.Value();
  assert(initial_checksum != 0);

  // 3. Export weights
  auto export_res = engine.ExportWeights();
  assert(export_res.Ok());
  assert(export_res->size() == 2 * 8 * 8);

  // 4. Homeostatic scaling
  auto scale_res = engine.ScaleWeights(2.0f);
  assert(scale_res.Ok());
  assert(engine.WeightChecksum().Value() != initial_checksum);

  // 5. Restore weights and verify bit-exact checksum
  auto restore_res = engine.RestoreWeights(*export_res);
  assert(restore_res.Ok());
  assert(engine.WeightChecksum().Value() == initial_checksum);

  std::cout << "All C++20 Google-style RAII smoke tests passed successfully.\n";
  return 0;
}
