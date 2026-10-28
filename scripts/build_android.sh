#!/usr/bin/env bash
# ==============================================================================
# Argon Engine — Android NDK Multi-ABI Cross-Compilation & AAR Packager
# ==============================================================================
set -euo pipefail

TARGET_ABIS=("arm64-v8a" "armeabi-v7a" "x86_64")
OUTPUT_DIR="bindings/android/jniLibs"

log_info() { printf "\033[1;34m[ARGON-ANDROID]\033[0m %s\n" "$1"; }
log_ok()   { printf "\033[1;32m[ARGON-OK]\033[0m      %s\n" "$1"; }
log_warn() { printf "\033[1;33m[ARGON-WARN]\033[0m    %s\n" "$1"; }

mkdir -p "${OUTPUT_DIR}"

if ! command -v cargo-ndk >/dev/null 2>&1; then
  log_warn "cargo-ndk is not installed. To build Android native libraries, run:"
  log_warn "  cargo install cargo-ndk"
  log_warn "and ensure ANDROID_NDK_HOME is exported."
  exit 0
fi

for abi in "${TARGET_ABIS[@]}"; do
  log_info "Compiling argon-ffi for ABI: ${abi}..."
  cargo ndk -t "${abi}" -o "${OUTPUT_DIR}" build --release -p argon-ffi
  log_ok "Built ${OUTPUT_DIR}/${abi}/libargon_ffi.so"
done

log_ok "Android multi-ABI packaging complete in ${OUTPUT_DIR}."
