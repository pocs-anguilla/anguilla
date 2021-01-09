export CMAKE_GENERATOR=Ninja
export CMAKE_INSTALL_PREFIX=${PREFIX}

python -m pip install ${SRC_DIR} -vvv
