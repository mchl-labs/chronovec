cmake_minimum_required(VERSION 3.18)
project(chronovec LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# std::thread (the MVCC reader registry and elsewhere) needs an explicit
# pthread link on some toolchains -- this was previously relying on the
# platform default C++ runtime pulling it in implicitly, which happened to
# hold on the toolchains this had been built with so far but failed outright
# ("undefined reference to `pthread_create'") on a manylinux_2_28/GCC 14
# toolchain the first time a wheel build ran on one.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(chronovec SHARED bindings/rust/chronovec-sys/native/src/chronovec.cpp bindings/rust/chronovec-sys/native/src/distance.cpp)
target_include_directories(chronovec PUBLIC bindings/rust/chronovec-sys/native/include)
# PUBLIC, not PRIVATE: the smoke-test executables below use std::thread
# directly too and link against this target, not just the library's own
# translation units -- they need Threads::Threads to propagate to them.
target_link_libraries(chronovec PUBLIC Threads::Threads)
if(MSVC)
  target_compile_options(chronovec PRIVATE /O2 /W4)
else()
  target_compile_options(chronovec PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

# AVX2 is an opt-in build target. The distance kernels guard their AVX2 paths
# with #ifdef __AVX2__, but a library compiled with -mavx2 can still fault on
# an older x86 host. Portable wheels and default source builds must therefore
# use the scalar fallback unless the deploy target is known to support AVX2.
# ARM uses NEON via __aarch64__ guards in distance.cpp.
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm|ARM|aarch64|ARM64")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-mavx2 CHRONOVEC_COMPILER_SUPPORTS_AVX2)
  option(CHRONOVEC_ENABLE_AVX2
    "Enable AVX2 SIMD kernels for x86-64 distance functions (target must support AVX2)" OFF)
  if(CHRONOVEC_ENABLE_AVX2)
    target_compile_options(chronovec PRIVATE -mavx2)
    message(STATUS "chronovec: AVX2 distance kernels enabled")
  endif()
endif()

# Install the shared library next to the Python package so ctypes can find it
# whether installed via `pip install .` (scikit-build-core) or a manual cmake
# build followed by `cmake --install`.
install(TARGETS chronovec
    LIBRARY DESTINATION chronovec
    RUNTIME DESTINATION chronovec
)

# Batched insert assigns a block of rows to pages with one matmul over the page
# centroids. That kernel is the bulk-ingest bottleneck, and on Apple silicon
# Accelerate reaches the AMX unit, which a scalar loop cannot approach. BLAS is
# optional: without it the same routine runs a blocked scalar fallback, so a
# dependency-free build still works and only gives up bulk throughput.
option(CHRONOVEC_USE_BLAS "Use a BLAS sgemm for batched insert assignment" ON)
if(CHRONOVEC_USE_BLAS)
  if(APPLE)
    find_library(CHRONOVEC_ACCELERATE Accelerate)
    if(CHRONOVEC_ACCELERATE)
      target_link_libraries(chronovec PRIVATE ${CHRONOVEC_ACCELERATE})
      target_compile_definitions(chronovec PRIVATE CHRONOVEC_BLAS=1 CHRONOVEC_BLAS_ACCELERATE=1)
      message(STATUS "chronovec: batched assignment uses Accelerate")
    endif()
  else()
    find_package(BLAS)
    if(BLAS_FOUND)
      find_path(CHRONOVEC_CBLAS_INCLUDE cblas.h)
      if(CHRONOVEC_CBLAS_INCLUDE)
        target_include_directories(chronovec PRIVATE ${CHRONOVEC_CBLAS_INCLUDE})
        target_link_libraries(chronovec PRIVATE ${BLAS_LIBRARIES})
        target_compile_definitions(chronovec PRIVATE CHRONOVEC_BLAS=1)
        message(STATUS "chronovec: batched assignment uses ${BLAS_LIBRARIES}")
      endif()
    endif()
  endif()
endif()

option(CHRONOVEC_SANITIZE_ADDRESS "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(CHRONOVEC_SANITIZE_THREAD "Build with ThreadSanitizer" OFF)
option(CHRONOVEC_BUILD_FUZZER "Build the libFuzzer C API harness (Clang only)" OFF)
if(CHRONOVEC_SANITIZE_ADDRESS AND CHRONOVEC_SANITIZE_THREAD)
  message(FATAL_ERROR "Address/UB sanitizers and ThreadSanitizer require separate builds")
endif()
if(CHRONOVEC_SANITIZE_ADDRESS)
  # PUBLIC, not PRIVATE: executables that merely link against chronovec (the
  # smoke tests below) need the same -fsanitize flags at their own compile
  # and link steps, or the linker can't find the sanitizer runtime symbols
  # chronovec.so was built expecting.
  target_compile_options(chronovec PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(chronovec PUBLIC -fsanitize=address,undefined)
elseif(CHRONOVEC_SANITIZE_THREAD)
  target_compile_options(chronovec PUBLIC -fsanitize=thread -fno-omit-frame-pointer)
  target_link_options(chronovec PUBLIC -fsanitize=thread)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(chronovec_c_api_smoke native/tests/c_api_smoke.c)
  target_link_libraries(chronovec_c_api_smoke PRIVATE chronovec)
  # Force the C++ driver for the final link: this target's own source is C,
  # but it links a C++ shared library. Under ASan/UBSan (vptr) that library's
  # object code calls into C++-only sanitizer runtime symbols the plain C
  # driver doesn't pull in, producing link-time "undefined reference to
  # __ubsan_vptr_type_cache" on Linux.
  set_target_properties(chronovec_c_api_smoke PROPERTIES LINKER_LANGUAGE CXX)
  add_test(NAME chronovec_c_api_smoke COMMAND chronovec_c_api_smoke)
  add_executable(chronovec_concurrent_smoke native/tests/concurrent_smoke.cpp)
  target_link_libraries(chronovec_concurrent_smoke PRIVATE chronovec)
  add_test(NAME chronovec_concurrent_smoke COMMAND chronovec_concurrent_smoke)
  # The batch path with parallel apply lanes, which the older smoke test
  # predates. Run this one under ThreadSanitizer.
  add_executable(chronovec_concurrent_batch_smoke
                 native/tests/concurrent_batch_smoke.cpp)
  target_link_libraries(chronovec_concurrent_batch_smoke PRIVATE chronovec)
  add_test(NAME chronovec_concurrent_batch_smoke
           COMMAND chronovec_concurrent_batch_smoke)
endif()

if(CHRONOVEC_BUILD_FUZZER)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "CHRONOVEC_BUILD_FUZZER requires Clang/libFuzzer")
  endif()
  add_executable(chronovec_fuzz native/tests/fuzz_c_api.cpp)
  target_link_libraries(chronovec_fuzz PRIVATE chronovec)
  target_compile_options(chronovec_fuzz PRIVATE -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer)
  target_link_options(chronovec_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
endif()

option(CHRONOVEC_BUILD_SQLITE "Build the SQLite loadable extension" ON)
if(CHRONOVEC_BUILD_SQLITE)
  find_package(SQLite3)
  if(SQLite3_FOUND)
    if(APPLE)
      find_path(CHRONOVEC_SQLITE_EXTENSION_INCLUDE sqlite3ext.h
        HINTS /opt/homebrew/opt/sqlite/include /usr/local/opt/sqlite/include
        NO_DEFAULT_PATH)
      if(NOT CHRONOVEC_SQLITE_EXTENSION_INCLUDE)
        message(FATAL_ERROR "Apple's SDK disables loadable SQLite extensions; install Homebrew sqlite or disable CHRONOVEC_BUILD_SQLITE")
      endif()
    else()
      set(CHRONOVEC_SQLITE_EXTENSION_INCLUDE ${SQLite3_INCLUDE_DIRS})
    endif()
    add_library(chronovec_sqlite MODULE native/sqlite/chronovec_sqlite.cpp native/sqlite/chronovec_vtab.cpp)
    target_include_directories(chronovec_sqlite BEFORE PRIVATE ${CHRONOVEC_SQLITE_EXTENSION_INCLUDE} bindings/rust/chronovec-sys/native/include)
    target_compile_definitions(chronovec_sqlite PRIVATE CHRONOVEC_SQLITE_LOADABLE=1)
    target_link_libraries(chronovec_sqlite PRIVATE chronovec)
    set_target_properties(chronovec_sqlite PROPERTIES PREFIX "" OUTPUT_NAME "chronovec_sqlite")
    if(APPLE)
      set_target_properties(chronovec_sqlite PROPERTIES BUILD_RPATH "@loader_path")
    elseif(UNIX)
      set_target_properties(chronovec_sqlite PROPERTIES BUILD_RPATH "$ORIGIN")
    endif()

    # Static registration target for embedded SQLite/dqlite servers. Unlike
    # the loadable module, this resolves SQLite calls against the host library.
    add_library(chronovec_sqlite_static STATIC native/sqlite/chronovec_vtab.cpp)
    target_include_directories(chronovec_sqlite_static PRIVATE bindings/rust/chronovec-sys/native/include ${SQLite3_INCLUDE_DIRS})
    target_compile_definitions(chronovec_sqlite_static PRIVATE SQLITE_CORE=1)
    if(TARGET SQLite3::SQLite3)
      target_link_libraries(chronovec_sqlite_static PUBLIC chronovec SQLite3::SQLite3)
    elseif(TARGET SQLite::SQLite3)
      target_link_libraries(chronovec_sqlite_static PUBLIC chronovec SQLite::SQLite3)
    else()
      target_link_libraries(chronovec_sqlite_static PUBLIC chronovec ${SQLite3_LIBRARIES})
    endif()
  endif()
endif()
