#include "argon.h"

int main(void) {
  ArgonEngineHandle *engine = 0;
  if (argon_engine_abi_version() != ARGON_ABI_VERSION) return 1;
  if (argon_engine_create(2, 8, 0.02f, 1.0f, 0.0005f, 8, &engine) !=
      ARGON_STATUS_OK) return 2;
  const float input[8] = {0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.2f, 0.3f, 0.4f};
  float output[8] = {0};
  ArgonTickTelemetry telemetry = {0};
  if (argon_engine_step_zero_copy(engine, 7, 1.0f, input, 8, output, 8,
                                  &telemetry) != ARGON_STATUS_OK) return 3;
  if (telemetry.sequence_id != 7 || telemetry.total_ticks != 1) return 4;
  return argon_engine_destroy(engine) == ARGON_STATUS_OK ? 0 : 5;
}
