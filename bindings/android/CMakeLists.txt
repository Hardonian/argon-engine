cmake_minimum_required(VERSION 3.22.1)

project(argon_android_ndk LANGUAGES C CXX)

# Root of the Argon Engine repository
set(ARGON_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")

# Include C-ABI header
include_directories("${ARGON_ROOT}/include")

# Locate prebuilt libargon_ffi.so for target Android ABI (arm64-v8a, armeabi-v7a, x86_64)
set(ARGON_LIB_DIR "${CMAKE_CURRENT_LIST_DIR}/jniLibs/${ANDROID_ABI}")

add_library(argon_ffi SHARED IMPORTED)
set_target_properties(argon_ffi PROPERTIES
    IMPORTED_LOCATION "${ARGON_LIB_DIR}/libargon_ffi.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ARGON_ROOT}/include"
)

# Optional JNI wrapper library linking Argon directly into Android app
add_library(argon_jni SHARED
    "${CMAKE_CURRENT_LIST_DIR}/src/argon_jni.c"
)

target_link_libraries(argon_jni
    PRIVATE
    argon_ffi
    log
)
