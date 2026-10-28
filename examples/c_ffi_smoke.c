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

  uint64_t initial_checksum = 0;
  if (argon_engine_weight_checksum(engine, &initial_checksum) != ARGON_STATUS_OK) return 5;
  if (initial_checksum == 0) return 6;

  float weights_buf[128] = {0};
  size_t written = 0;
  if (argon_engine_export_weights(engine, weights_buf, 128, &written) != ARGON_STATUS_OK) return 7;
  if (written != 128) return 8;

  if (argon_engine_scale_weights(engine, 2.0f) != ARGON_STATUS_OK) return 9;

  if (argon_engine_restore_weights(engine, weights_buf, written) != ARGON_STATUS_OK) return 10;
  uint64_t restored_checksum = 0;
  if (argon_engine_weight_checksum(engine, &restored_checksum) != ARGON_STATUS_OK) return 11;
  if (restored_checksum != initial_checksum) return 12;

  uint64_t attestation = 0;
  if (argon_engine_attestation_hash(engine, &attestation) != ARGON_STATUS_OK) return 13;
  if (attestation == 0) return 14;

  if (argon_engine_verify_integrity(engine) != ARGON_STATUS_OK) return 15;

  return argon_engine_destroy(engine) == ARGON_STATUS_OK ? 0 : 16;
}
