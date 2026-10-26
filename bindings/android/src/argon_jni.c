#include <jni.h>
#include <android/log.h>
#include "argon.h"

#define LOG_TAG "ArgonEngineJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

JNIEXPORT jlong JNICALL
Java_ai_argon_ArgonEngine_nativeCreate(
    JNIEnv *env,
    jclass clazz,
    jlong num_layers,
    jlong tile_dim,
    jfloat learning_rate,
    jfloat goodness_threshold,
    jfloat weight_decay,
    jlong ring_capacity) {
  ArgonEngineHandle *handle = NULL;
  ArgonStatus status = argon_engine_create(
      (size_t)num_layers, (size_t)tile_dim, learning_rate,
      goodness_threshold, weight_decay, (size_t)ring_capacity, &handle);

  if (status != ARGON_STATUS_OK) {
    LOGE("argon_engine_create failed: %s", argon_status_message(status));
    return 0;
  }
  return (jlong)handle;
}

JNIEXPORT jint JNICALL
Java_ai_argon_ArgonEngine_nativeStep(
    JNIEnv *env,
    jclass clazz,
    jlong handle_ptr,
    jlong sequence_id,
    jfloat polarity,
    jfloatArray in_features,
    jfloatArray out_activations) {
  ArgonEngineHandle *handle = (ArgonEngineHandle *)handle_ptr;
  if (!handle) return ARGON_STATUS_NULL_POINTER;

  jsize in_len = (*env)->GetArrayLength(env, in_features);
  jsize out_len = (*env)->GetArrayLength(env, out_activations);

  jfloat *in_ptr = (*env)->GetFloatArrayElements(env, in_features, NULL);
  jfloat *out_ptr = (*env)->GetFloatArrayElements(env, out_activations, NULL);

  ArgonTickTelemetry telemetry = {0};
  ArgonStatus status = argon_engine_step_zero_copy(
      handle, (uint64_t)sequence_id, polarity, in_ptr, (size_t)in_len,
      out_ptr, (size_t)out_len, &telemetry);

  (*env)->ReleaseFloatArrayElements(env, in_features, in_ptr, JNI_ABORT);
  (*env)->ReleaseFloatArrayElements(env, out_activations, out_ptr, 0);

  return (jint)status;
}

JNIEXPORT void JNICALL
Java_ai_argon_ArgonEngine_nativeDestroy(
    JNIEnv *env,
    jclass clazz,
    jlong handle_ptr) {
  ArgonEngineHandle *handle = (ArgonEngineHandle *)handle_ptr;
  if (handle) {
    argon_engine_destroy(handle);
  }
}
