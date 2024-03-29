mkdir build
cd build

# Configure.
if [ -z ${OSX_ARCH+x} ];
then
cmake -DBUILD_EXAMPLES=OFF \
      -DBUILD_DOCUMENTATION=OFF \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=${SRC_DIR}/toolchains/cxx11.cmake \
      ${SRC_DIR}/shark;
else
cmake -DBUILD_EXAMPLES=OFF \
      -DBUILD_DOCUMENTATION=OFF \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=${SRC_DIR}/toolchains/clang-cxx11-libstdcxx.cmake \
      ${SRC_DIR}/shark;
fi

# Build.
make -j ${CPU_COUNT}

# Test.
make -j ${CPU_COUNT} test || true
