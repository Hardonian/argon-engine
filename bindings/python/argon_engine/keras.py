"""Google Keras 3 Multi-Backend Layer for Argon Engine.

Implements Forward-Forward neural plasticity as a first-class Keras 3 layer,
compatible across JAX, TensorFlow, and PyTorch backends.
"""

from typing import Any, Dict, Optional, Tuple

try:
    import keras  # type: ignore
    from keras import ops  # type: ignore
    HAS_KERAS = True
except ImportError:
    try:
        import tensorflow.keras as keras  # type: ignore
        from tensorflow.keras import backend as ops  # type: ignore
        HAS_KERAS = True
    except ImportError:
        import numpy as ops
        HAS_KERAS = False
        keras = None


if HAS_KERAS:
    @keras.saving.register_keras_serializable(package="argon")
    class ArgonKerasLayer(keras.layers.Layer):
        """Keras 3 Layer implementing an Argon Engine Forward-Forward SRAM tile.

        Attributes:
            units: Dimensionality of the SRAM tile (default 8).
            learning_rate: Base plasticity rate (default 0.02).
            goodness_threshold: Target homeostatic goodness threshold (default 1.0).
            weight_decay: L2 synaptic weight decay factor (default 0.0005).
        """

        def __init__(
            self,
            units: int = 8,
            learning_rate: float = 0.02,
            goodness_threshold: float = 1.0,
            weight_decay: float = 0.0005,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.units = int(units)
            self.learning_rate = float(learning_rate)
            self.goodness_threshold = float(goodness_threshold)
            self.weight_decay = float(weight_decay)
            self.last_goodness = None

        def build(self, input_shape):
            dim = int(input_shape[-1])
            if dim != self.units:
                raise ValueError(
                    f"Input feature dimension {dim} must match layer units {self.units} "
                    f"for square SRAM tile Forward-Forward processing."
                )

            # Initialize with canonical diagonal 0.5 weights
            initial_val = ops.cast(ops.eye(self.units) * 0.5, dtype=self.dtype)
            self.kernel = self.add_weight(
                name="kernel",
                shape=(self.units, self.units),
                initializer=lambda shape, dtype: initial_val,
                trainable=True,
            )
            super().build(input_shape)

        def call(
            self,
            inputs,
            polarity: Optional[float] = None,
            training: Optional[bool] = None,
        ):
            # 1. Forward linear drive + ReLU activation
            accum = ops.matmul(inputs, self.kernel)
            y = ops.relu(accum)

            # 2. Mean squared goodness per sample
            sum_squared = ops.sum(ops.square(y), axis=-1, keepdims=True)
            goodness = sum_squared / float(self.units)
            self.last_goodness = goodness

            # 3. RMS normalization
            inv_rms = 1.0 / (ops.sqrt(goodness) + 1e-6)
            normalized_output = y * inv_rms

            # 4. In-flight contrastive plasticity update if training and polarity provided
            if training and polarity is not None and hasattr(self.kernel, "assign_add"):
                margin = polarity * (self.goodness_threshold - goodness)
                modulation = ops.sigmoid(margin)
                effective_lr = self.learning_rate * polarity * modulation
                post_drive = effective_lr * y

                # Batch outer-product update
                # inputs: (B, D), post_drive: (B, D) -> delta_w: (D, D)
                delta_w = ops.matmul(ops.transpose(inputs), post_drive) / float(ops.shape(inputs)[0]) - self.weight_decay * self.kernel
                self.kernel.assign_add(delta_w)

            return normalized_output

        def compute_output_shape(self, input_shape):
            return input_shape

        def get_config(self) -> Dict[str, Any]:
            config = super().get_config()
            config.update({
                "units": self.units,
                "learning_rate": self.learning_rate,
                "goodness_threshold": self.goodness_threshold,
                "weight_decay": self.weight_decay,
            })
            return config

else:
    class ArgonKerasLayer:
        """Standalone fallback implementation when Keras 3 is not installed.

        Implements the identical API, weight structure, and serialization
        conventions using pure NumPy for evaluation in lightweight runtime environments.
        """

        def __init__(
            self,
            units: int = 8,
            learning_rate: float = 0.02,
            goodness_threshold: float = 1.0,
            weight_decay: float = 0.0005,
            **kwargs,
        ):
            self.units = int(units)
            self.learning_rate = float(learning_rate)
            self.goodness_threshold = float(goodness_threshold)
            self.weight_decay = float(weight_decay)
            self.kernel = ops.eye(self.units, dtype=ops.float32) * 0.5
            self.last_goodness = None

        def build(self, input_shape):
            pass

        def __call__(
            self,
            inputs,
            polarity: Optional[float] = None,
            training: Optional[bool] = None,
        ):
            inputs_arr = ops.asarray(inputs, dtype=ops.float32)
            accum = ops.matmul(inputs_arr, self.kernel)
            y = ops.maximum(accum, 0.0)

            sum_squared = ops.sum(y * y, axis=-1, keepdims=True)
            goodness = sum_squared / float(self.units)
            self.last_goodness = goodness

            inv_rms = 1.0 / (ops.sqrt(goodness) + 1e-6)
            normalized_output = y * inv_rms

            if training and polarity is not None:
                margin = polarity * (self.goodness_threshold - goodness)
                modulation = 1.0 / (1.0 + ops.exp(-margin))
                effective_lr = self.learning_rate * polarity * modulation
                post_drive = effective_lr * y

                if inputs_arr.ndim == 1:
                    delta_w = ops.outer(inputs_arr, post_drive) - self.weight_decay * self.kernel
                else:
                    batch_size = float(inputs_arr.shape[0])
                    delta_w = ops.matmul(inputs_arr.T, post_drive) / batch_size - self.weight_decay * self.kernel
                
                self.kernel = self.kernel + delta_w

            return normalized_output

        def compute_output_shape(self, input_shape):
            return input_shape

        def get_config(self) -> Dict[str, Any]:
            return {
                "units": self.units,
                "learning_rate": self.learning_rate,
                "goodness_threshold": self.goodness_threshold,
                "weight_decay": self.weight_decay,
            }
