mkdir build
cd build

# Configure.
cmake -DCMAKE_TOOLCHAIN_FILE=${SRC_DIR}/cmake/toolchains/clang-cxx20-libcxx.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -GNinja \
      ${SRC_DIR};

# Build.
cmake --build .
