#ifndef ARGON_ENGINE_H
#define ARGON_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ARGON_ABI_VERSION 1u

typedef struct ArgonEngineHandle ArgonEngineHandle;

typedef enum ArgonStatus {
  ARGON_STATUS_OK = 0,
  ARGON_STATUS_NULL_POINTER = -1,
  ARGON_STATUS_INVALID_CONFIG = -2,
  ARGON_STATUS_DIMENSION_MISMATCH = -3,
  ARGON_STATUS_RING_BUFFER = -4,
  ARGON_STATUS_INTERNAL_PANIC = -99
} ArgonStatus;

typedef struct ArgonTickTelemetry {
  uint64_t sequence_id;
  float mean_goodness;
  float output_l2_norm;
  uint64_t total_ticks;
} ArgonTickTelemetry;

uint32_t argon_engine_abi_version(void);
const char *argon_status_message(ArgonStatus status);
ArgonStatus argon_engine_create(size_t num_layers, size_t tile_dim,
                                float learning_rate,
                                float goodness_threshold, float weight_decay,
                                size_t ring_capacity,
                                ArgonEngineHandle **out_handle);
ArgonStatus argon_engine_step_zero_copy(
    ArgonEngineHandle *handle, uint64_t sequence_id, float phase_polarity,
    const float *input_ptr, size_t input_len, float *output_ptr,
    size_t output_len, ArgonTickTelemetry *out_telemetry);
ArgonStatus argon_engine_weight_checksum(const ArgonEngineHandle *handle,
                                         uint64_t *out_checksum);
ArgonStatus argon_engine_scale_weights(ArgonEngineHandle *handle,
                                       float target_norm);
ArgonStatus argon_engine_export_weights(const ArgonEngineHandle *handle,
                                        float *out_weights, size_t max_len,
                                        size_t *out_written);
ArgonStatus argon_engine_restore_weights(ArgonEngineHandle *handle,
                                         const float *in_weights, size_t len);
ArgonStatus argon_engine_attestation_hash(const ArgonEngineHandle *handle,
                                          uint64_t *out_hash);
ArgonStatus argon_engine_verify_integrity(const ArgonEngineHandle *handle);
ArgonStatus argon_engine_destroy(ArgonEngineHandle *handle);

#ifdef __cplusplus
}
#endif

#endif
