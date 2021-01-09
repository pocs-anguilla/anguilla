export CMAKE_GENERATOR=Ninja
export CMAKE_INSTALL_PREFIX=${PREFIX}
export CMAKE_PREFIX_PATH=${PREFIX}
export CONDA_BUILD=1

python -m pip install ${SRC_DIR} -vvv
