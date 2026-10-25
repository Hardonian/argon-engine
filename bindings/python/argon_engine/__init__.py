"""Argon Engine native zero-copy bindings for Python and NumPy."""

import ctypes
import os
import platform
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

ARGON_ABI_VERSION = 1


class ArgonStatus:
    OK = 0
    NULL_POINTER = -1
    INVALID_CONFIG = -2
    DIMENSION_MISMATCH = -3
    RING_BUFFER = -4
    INTERNAL_PANIC = -99


class ArgonTickTelemetry(ctypes.Structure):
    _fields_ = [
        ("sequence_id", ctypes.c_uint64),
        ("mean_goodness", ctypes.c_float),
        ("output_l2_norm", ctypes.c_float),
        ("total_ticks", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class TickTelemetry:
    sequence_id: int
    mean_goodness: float
    output_l2_norm: float
    total_ticks: int


def _find_library() -> str:
    env_path = os.environ.get("ARGON_LIB_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    system = platform.system()
    if system == "Windows":
        lib_name = "argon_ffi.dll"
    elif system == "Darwin":
        lib_name = "libargon_ffi.dylib"
    else:
        lib_name = "libargon_ffi.so"

    # Search common target directories relative to the root of the workspace
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    search_dirs = [
        os.path.join(base_dir, "target", "release"),
        os.path.join(base_dir, "target", "debug"),
        os.path.join(base_dir, "target", "debug", "deps"),
        os.path.join(base_dir, "target", "release", "deps"),
    ]
    for directory in search_dirs:
        candidate = os.path.join(directory, lib_name)
        if os.path.exists(candidate):
            return candidate

    return lib_name


_LIB = None


def _get_lib():
    global _LIB
    if _LIB is not None:
        return _LIB

    lib_path = _find_library()
    try:
        lib = ctypes.CDLL(lib_path)
    except OSError as e:
        raise RuntimeError(
            f"Failed to load Argon FFI library at '{lib_path}'. "
            "Set ARGON_LIB_PATH or build argon-ffi with `cargo build -p argon-ffi`."
        ) from e

    lib.argon_engine_abi_version.restype = ctypes.c_uint32
    lib.argon_engine_abi_version.argtypes = []

    lib.argon_status_message.restype = ctypes.c_char_p
    lib.argon_status_message.argtypes = [ctypes.c_int]

    lib.argon_engine_create.restype = ctypes.c_int
    lib.argon_engine_create.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
    ]

    lib.argon_engine_step_zero_copy.restype = ctypes.c_int
    lib.argon_engine_step_zero_copy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ArgonTickTelemetry),
    ]

    lib.argon_engine_weight_checksum.restype = ctypes.c_int
    lib.argon_engine_weight_checksum.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
    ]

    lib.argon_engine_scale_weights.restype = ctypes.c_int
    lib.argon_engine_scale_weights.argtypes = [
        ctypes.c_void_p,
        ctypes.c_float,
    ]

    lib.argon_engine_export_weights.restype = ctypes.c_int
    lib.argon_engine_export_weights.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]

    lib.argon_engine_restore_weights.restype = ctypes.c_int
    lib.argon_engine_restore_weights.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]

    lib.argon_engine_destroy.restype = ctypes.c_int
    lib.argon_engine_destroy.argtypes = [ctypes.c_void_p]

    _LIB = lib
    return _LIB


class ArgonEngine:
    """Argon Engine coordinator interface with zero-copy NumPy execution."""

    def __init__(
        self,
        num_layers: int = 2,
        tile_dim: int = 8,
        learning_rate: float = 0.02,
        goodness_threshold: float = 1.0,
        weight_decay: float = 0.0005,
        ring_capacity: int = 8,
    ):
        self.num_layers = num_layers
        self.tile_dim = tile_dim
        self.lib = _get_lib()

        abi_version = self.lib.argon_engine_abi_version()
        if abi_version != ARGON_ABI_VERSION:
            raise RuntimeError(
                f"ABI version mismatch: expected {ARGON_ABI_VERSION}, got {abi_version}"
            )

        self._handle = ctypes.c_void_p()
        status = self.lib.argon_engine_create(
            ctypes.c_size_t(num_layers),
            ctypes.c_size_t(tile_dim),
            ctypes.c_float(learning_rate),
            ctypes.c_float(goodness_threshold),
            ctypes.c_float(weight_decay),
            ctypes.c_size_t(ring_capacity),
            ctypes.byref(self._handle),
        )
        if status != ArgonStatus.OK:
            msg = self.lib.argon_status_message(status).decode("utf-8")
            raise RuntimeError(f"argon_engine_create failed: {msg} (code={status})")

        self._sequence_id = 0

    def step(
        self,
        sensory_input: np.ndarray,
        polarity: float = 1.0,
        output: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, TickTelemetry]:
        """Executes one zero-copy forward-forward / Hebbian tick.

        If `output` is provided, the result is written directly into it with zero allocation.
        """
        if self._handle.value is None:
            raise RuntimeError("ArgonEngine instance has been closed")

        if not isinstance(sensory_input, np.ndarray):
            sensory_input = np.asarray(sensory_input, dtype=np.float32)
        elif sensory_input.dtype != np.float32 or not sensory_input.flags.c_contiguous:
            sensory_input = np.ascontiguousarray(sensory_input, dtype=np.float32)

        if sensory_input.shape != (self.tile_dim,):
            raise ValueError(
                f"sensory_input shape mismatch: expected ({self.tile_dim},), got {sensory_input.shape}"
            )

        if output is None:
            output = np.empty(self.tile_dim, dtype=np.float32)
        else:
            if output.dtype != np.float32 or not output.flags.c_contiguous:
                raise ValueError("output array must be a contiguous float32 numpy array")
            if output.shape != (self.tile_dim,):
                raise ValueError(
                    f"output shape mismatch: expected ({self.tile_dim},), got {output.shape}"
                )

        c_telemetry = ArgonTickTelemetry()
        input_ptr = sensory_input.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self._sequence_id += 1
        status = self.lib.argon_engine_step_zero_copy(
            self._handle,
            ctypes.c_uint64(self._sequence_id),
            ctypes.c_float(polarity),
            input_ptr,
            ctypes.c_size_t(sensory_input.size),
            output_ptr,
            ctypes.c_size_t(output.size),
            ctypes.byref(c_telemetry),
        )
        if status != ArgonStatus.OK:
            msg = self.lib.argon_status_message(status).decode("utf-8")
            raise RuntimeError(f"argon_engine_step_zero_copy failed: {msg} (code={status})")

        telemetry = TickTelemetry(
            sequence_id=c_telemetry.sequence_id,
            mean_goodness=float(c_telemetry.mean_goodness),
            output_l2_norm=float(c_telemetry.output_l2_norm),
            total_ticks=c_telemetry.total_ticks,
        )
        return output, telemetry

    def weight_checksum(self) -> int:
        """Returns the bitwise checksum of all synaptic weights resident in the engine."""
        if self._handle.value is None:
            raise RuntimeError("ArgonEngine instance has been closed")
        out_checksum = ctypes.c_uint64(0)
        status = self.lib.argon_engine_weight_checksum(self._handle, ctypes.byref(out_checksum))
        if status != ArgonStatus.OK:
            msg = self.lib.argon_status_message(status).decode("utf-8")
            raise RuntimeError(f"argon_engine_weight_checksum failed: {msg} (code={status})")
        return int(out_checksum.value)

    def scale_weights(self, target_norm: float) -> None:
        """Rescales all resident layer weights towards the target homeostatic norm."""
        if self._handle.value is None:
            raise RuntimeError("ArgonEngine instance has been closed")
        status = self.lib.argon_engine_scale_weights(self._handle, ctypes.c_float(target_norm))
        if status != ArgonStatus.OK:
            msg = self.lib.argon_status_message(status).decode("utf-8")
            raise RuntimeError(f"argon_engine_scale_weights failed: {msg} (code={status})")

    def export_weights(self) -> np.ndarray:
        """Exports all layer weights into a contiguous 1D float32 NumPy array."""
        if self._handle.value is None:
            raise RuntimeError("ArgonEngine instance has been closed")
        max_len = self.num_layers * self.tile_dim * self.tile_dim
        buf = np.empty(max_len, dtype=np.float32)
        out_written = ctypes.c_size_t(0)
        buf_ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        status = self.lib.argon_engine_export_weights(
            self._handle, buf_ptr, ctypes.c_size_t(max_len), ctypes.byref(out_written)
        )
        if status != ArgonStatus.OK:
            msg = self.lib.argon_status_message(status).decode("utf-8")
            raise RuntimeError(f"argon_engine_export_weights failed: {msg} (code={status})")
        return buf[: out_written.value].copy()

    def restore_weights(self, weights: np.ndarray) -> None:
        """Restores all synaptic weights from a contiguous float32 NumPy array."""
        if self._handle.value is None:
            raise RuntimeError("ArgonEngine instance has been closed")
        if not isinstance(weights, np.ndarray):
            weights = np.asarray(weights, dtype=np.float32)
        elif weights.dtype != np.float32 or not weights.flags.c_contiguous:
            weights = np.ascontiguousarray(weights, dtype=np.float32)
        weights_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        status = self.lib.argon_engine_restore_weights(
            self._handle, weights_ptr, ctypes.c_size_t(weights.size)
        )
        if status != ArgonStatus.OK:
            msg = self.lib.argon_status_message(status).decode("utf-8")
            raise RuntimeError(f"argon_engine_restore_weights failed: {msg} (code={status})")

    def close(self) -> None:
        """Releases the underlying native coordinator."""
        if getattr(self, "_handle", None) is not None and self._handle.value is not None:
            self.lib.argon_engine_destroy(self._handle)
            self._handle = ctypes.c_void_p(None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
