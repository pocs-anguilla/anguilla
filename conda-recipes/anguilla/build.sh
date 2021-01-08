export CMAKE_TOOLCHAIN_FILE=${SRC_DIR}/cmake/toolchains/clang-cxx20-libcxx.cmake
export CMAKE_GENERATOR=Ninja
python -m pip install . -vvv
