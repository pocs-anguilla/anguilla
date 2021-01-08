mkdir build
cd build

# Configure.
cmake -DCMAKE_TOOLCHAIN_FILE=${SRC_DIR}/clang-cxx20-toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -GNinja \
      ${SRC_DIR};

# Build.
cmake --build .
