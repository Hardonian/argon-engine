// Copyright (c) Argon Engine Authors. All rights reserved.
// Modern C++20 RAII abstraction complying with the Google C++ Style Guide.

#ifndef ARGON_ENGINE_HPP_
#define ARGON_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__has_include)
#  if __has_include(<span>) && (defined(__cpp_lib_span) || __cplusplus >= 202002L)
#    include <span>
#  endif
#  if __has_include("argon.h")
#    include "argon.h"
#  elif __has_include("../include/argon.h")
#    include "../include/argon.h"
#  else
#    include <argon.h>
#  endif
#else
#  include "argon.h"
#endif

namespace argon {

#if defined(__cpp_lib_span) || (defined(__cplusplus) && __cplusplus >= 202002L && __has_include(<span>))
template <typename T, size_t Extent = std::dynamic_extent>
using span = std::span<T, Extent>;
#else
inline constexpr size_t dynamic_extent = static_cast<size_t>(-1);

template <typename T, size_t Extent = dynamic_extent>
class span {
 public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using size_type = size_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using iterator = T*;
  using const_iterator = const T*;

  constexpr span() noexcept : data_(nullptr), size_(0) {}
  constexpr span(pointer ptr, size_type count) noexcept : data_(ptr), size_(count) {}
  constexpr span(pointer first, pointer last) noexcept : data_(first), size_(static_cast<size_type>(last - first)) {}

  template <size_t N>
  constexpr span(element_type (&arr)[N]) noexcept : data_(arr), size_(N) {}

  template <typename Container,
            typename = std::enable_if_t<!std::is_same_v<std::decay_t<Container>, span> &&
                                        std::is_convertible_v<decltype(std::declval<Container>().data()), pointer>>>
  constexpr span(Container& cont) noexcept : data_(cont.data()), size_(cont.size()) {}

  template <typename Container,
            typename = std::enable_if_t<!std::is_same_v<std::decay_t<Container>, span> &&
                                        std::is_convertible_v<decltype(std::declval<const Container>().data()), pointer>>>
  constexpr span(const Container& cont) noexcept : data_(cont.data()), size_(cont.size()) {}

  constexpr pointer data() const noexcept { return data_; }
  constexpr size_type size() const noexcept { return size_; }
  constexpr size_type size_bytes() const noexcept { return size_ * sizeof(element_type); }
  [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

  constexpr reference operator[](size_type idx) const noexcept { return data_[idx]; }
  constexpr reference front() const noexcept { return data_[0]; }
  constexpr reference back() const noexcept { return data_[size_ - 1]; }

  constexpr iterator begin() const noexcept { return data_; }
  constexpr iterator end() const noexcept { return data_ + size_; }
  constexpr const_iterator cbegin() const noexcept { return data_; }
  constexpr const_iterator cend() const noexcept { return data_ + size_; }

 private:
  pointer data_;
  size_type size_;
};
#endif

/// Status wrapper representing C-ABI operational outcomes.
class Status {
 public:
  constexpr Status() noexcept : code_(ARGON_STATUS_OK) {}
  constexpr Status(ArgonStatus code) noexcept : code_(code) {}

  constexpr bool Ok() const noexcept { return code_ == ARGON_STATUS_OK; }
  constexpr ArgonStatus Code() const noexcept { return code_; }
  const char* Message() const noexcept { return argon_status_message(code_); }

 private:
  ArgonStatus code_;
};

/// absl::StatusOr-style result holding either an active value or an error status.
template <typename T>
class Result {
 public:
  Result(T value) : value_(std::move(value)), status_(ARGON_STATUS_OK) {}
  Result(Status status) : status_(status) {}
  Result(ArgonStatus code) : status_(code) {}

  bool Ok() const noexcept { return status_.Ok(); }
  Status GetStatus() const noexcept { return status_; }
  const char* ErrorMessage() const noexcept { return status_.Message(); }

  T& Value() & { return *value_; }
  const T& Value() const& { return *value_; }
  T&& Value() && { return std::move(*value_); }

  T* operator->() { return &(*value_); }
  const T* operator->() const { return &(*value_); }
  T& operator*() { return *value_; }
  const T& operator*() const { return *value_; }

 private:
  std::optional<T> value_;
  Status status_;
};

// Specialization for void results
template <>
class Result<void> {
 public:
  Result() noexcept : status_(ARGON_STATUS_OK) {}
  Result(Status status) noexcept : status_(status) {}
  Result(ArgonStatus code) noexcept : status_(code) {}

  bool Ok() const noexcept { return status_.Ok(); }
  Status GetStatus() const noexcept { return status_; }
  const char* ErrorMessage() const noexcept { return status_.Message(); }

 private:
  Status status_;
};

/// High-performance RAII wrapper around native ArgonEngineHandle.
class Engine {
 public:
  /// Creates a new persistent engine coordinator.
  static Result<Engine> Create(size_t num_layers, size_t tile_dim,
                               float learning_rate = 0.02f,
                               float goodness_threshold = 1.0f,
                               float weight_decay = 0.0005f,
                               size_t ring_capacity = 8) {
    if (argon_engine_abi_version() != ARGON_ABI_VERSION) {
      return Result<Engine>(ARGON_STATUS_INVALID_CONFIG);
    }
    ArgonEngineHandle* handle = nullptr;
    ArgonStatus status = argon_engine_create(
        num_layers, tile_dim, learning_rate, goodness_threshold, weight_decay,
        ring_capacity, &handle);
    if (status != ARGON_STATUS_OK || !handle) {
      return Result<Engine>(status);
    }
    return Result<Engine>(Engine(handle, num_layers, tile_dim));
  }

  ~Engine() {
    if (handle_) {
      argon_engine_destroy(handle_);
      handle_ = nullptr;
    }
  }

  // Non-copyable for unique ownership
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Movable
  Engine(Engine&& other) noexcept
      : handle_(other.handle_),
        num_layers_(other.num_layers_),
        tile_dim_(other.tile_dim_) {
    other.handle_ = nullptr;
  }

  Engine& operator=(Engine&& other) noexcept {
    if (this != &other) {
      if (handle_) {
        argon_engine_destroy(handle_);
      }
      handle_ = other.handle_;
      num_layers_ = other.num_layers_;
      tile_dim_ = other.tile_dim_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  /// Executes one synchronous zero-copy forward and plasticity step.
  Result<ArgonTickTelemetry> Step(uint64_t sequence_id, float phase_polarity,
                                 span<const float> input,
                                 span<float> output) {
    if (!handle_) return Result<ArgonTickTelemetry>(ARGON_STATUS_NULL_POINTER);
    if (input.size() != tile_dim_ || output.size() != tile_dim_) {
      return Result<ArgonTickTelemetry>(ARGON_STATUS_DIMENSION_MISMATCH);
    }

    ArgonTickTelemetry telemetry = {};
    ArgonStatus status = argon_engine_step_zero_copy(
        handle_, sequence_id, phase_polarity, input.data(), input.size(),
        output.data(), output.size(), &telemetry);

    if (status != ARGON_STATUS_OK) {
      return Result<ArgonTickTelemetry>(status);
    }
    return Result<ArgonTickTelemetry>(telemetry);
  }

  /// Calculates the deterministic bitwise checksum of all resident synaptic weights.
  Result<uint64_t> WeightChecksum() const {
    if (!handle_) return Result<uint64_t>(ARGON_STATUS_NULL_POINTER);
    uint64_t checksum = 0;
    ArgonStatus status = argon_engine_weight_checksum(handle_, &checksum);
    if (status != ARGON_STATUS_OK) {
      return Result<uint64_t>(status);
    }
    return Result<uint64_t>(checksum);
  }

  /// Rescales synaptic weights according to local homeostatic target norm.
  Result<void> ScaleWeights(float target_norm) {
    if (!handle_) return Result<void>(ARGON_STATUS_NULL_POINTER);
    ArgonStatus status = argon_engine_scale_weights(handle_, target_norm);
    if (status != ARGON_STATUS_OK) {
      return Result<void>(status);
    }
    return Result<void>();
  }

  /// Exports all resident synaptic weights into an allocated vector.
  Result<std::vector<float>> ExportWeights() const {
    if (!handle_) return Result<std::vector<float>>(ARGON_STATUS_NULL_POINTER);
    const size_t total_weights = num_layers_ * tile_dim_ * tile_dim_;
    std::vector<float> buffer(total_weights);
    size_t written = 0;
    ArgonStatus status = argon_engine_export_weights(
        handle_, buffer.data(), buffer.size(), &written);
    if (status != ARGON_STATUS_OK) {
      return Result<std::vector<float>>(status);
    }
    buffer.resize(written);
    return Result<std::vector<float>>(std::move(buffer));
  }

  /// Restores synaptic weights from caller-provided memory.
  Result<void> RestoreWeights(span<const float> weights) {
    if (!handle_) return Result<void>(ARGON_STATUS_NULL_POINTER);
    ArgonStatus status =
        argon_engine_restore_weights(handle_, weights.data(), weights.size());
    if (status != ARGON_STATUS_OK) {
      return Result<void>(status);
    }
    return Result<void>();
  }

  /// Computes the cryptographic attestation hash of topology and weights (Milestone M7.2).
  Result<uint64_t> AttestationHash() const {
    if (!handle_) return Result<uint64_t>(ARGON_STATUS_NULL_POINTER);
    uint64_t hash = 0;
    ArgonStatus status = argon_engine_attestation_hash(handle_, &hash);
    if (status != ARGON_STATUS_OK) {
      return Result<uint64_t>(status);
    }
    return Result<uint64_t>(hash);
  }

  /// Verifies internal synaptic weight bounds and integrity invariants (Milestone M7.2).
  Result<void> VerifyIntegrity() const {
    if (!handle_) return Result<void>(ARGON_STATUS_NULL_POINTER);
    ArgonStatus status = argon_engine_verify_integrity(handle_);
    if (status != ARGON_STATUS_OK) {
      return Result<void>(status);
    }
    return Result<void>();
  }

  size_t NumLayers() const noexcept { return num_layers_; }
  size_t TileDim() const noexcept { return tile_dim_; }
  bool IsValid() const noexcept { return handle_ != nullptr; }

 private:
  Engine(ArgonEngineHandle* handle, size_t num_layers, size_t tile_dim) noexcept
      : handle_(handle), num_layers_(num_layers), tile_dim_(tile_dim) {}

  ArgonEngineHandle* handle_ = nullptr;
  size_t num_layers_ = 0;
  size_t tile_dim_ = 0;
};

}  // namespace argon

#endif  // ARGON_ENGINE_HPP_
