# Argon Engine: Android NDK Integration

Argon Engine compiles to a lightweight, zero-dependency C-ABI shared library (`libargon_ffi.so`) that can be embedded directly into Android native applications, robotics controllers, and edge sensor pipelines on devices such as Google Pixel (Tensor G3/G4).

## 1. Cross-compiling for Android

Use `cargo-ndk` to cross-compile `argon-ffi` for your target Android ABI (`arm64-v8a` recommended):

```bash
cargo ndk -t arm64-v8a -o bindings/android/jniLibs build --release -p argon-ffi
```

## 2. CMake Integration in Android Studio

Include `bindings/android/CMakeLists.txt` in your app's `CMakeLists.txt`:

```cmake
add_subdirectory(path/to/argon-engine/bindings/android argon_android)
target_link_libraries(your_native_lib PRIVATE argon_jni)
```

## 3. Kotlin / Java API Example

```kotlin
package ai.argon

class ArgonEngine(
    numLayers: Long = 2,
    tileDim: Long = 8,
    learningRate: Float = 0.02f,
    goodnessThreshold: Float = 1.0f,
    weightDecay: Float = 0.0005f
) : AutoCloseable {
    private var handle: Long = nativeCreate(numLayers, tileDim, learningRate, goodnessThreshold, weightDecay, 8)

    fun step(sequenceId: Long, polarity: Float, input: FloatArray, output: FloatArray): Int {
        return nativeStep(handle, sequenceId, polarity, input, output)
    }

    override fun close() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
    }

    private external fun nativeCreate(layers: Long, dim: Long, lr: Float, thresh: Float, decay: Float, ringCap: Long): Long
    private external fun nativeStep(handle: Long, seq: Long, polarity: Float, input: FloatArray, output: FloatArray): Int
    private external fun nativeDestroy(handle: Long)

    companion object {
        init {
            System.loadLibrary("argon_jni")
        }
    }
}
```
