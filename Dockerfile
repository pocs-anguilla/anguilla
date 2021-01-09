FROM condaforge/miniforge3

RUN conda update -y --all \
    && conda install conda-build conda-verify

COPY environment.yml .
RUN conda env create -f environment.yml

SHELL ["/bin/bash", "-c"]

RUN . /opt/conda/etc/profile.d/conda.sh \
    && conda activate development \
    && cd /tmp \
    && git clone -b 4.1 --depth 1 https://github.com/Shark-ML/Shark.git \
    && git clone --depth 1 https://github.com/ruslo/polly.git \
    && mkdir build \
    && cd build \
    && cmake -DBUILD_EXAMPLES=OFF \
    -DBUILD_DOCUMENTATION=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=14 \
    -DCMAKE_TOOLCHAIN_FILE=../polly/clang-libstdcxx.cmake \
    ../Shark \
    && make -j $(nproc) \
    && make -j $(nproc) test \
    && make install \
    && cd .. \
    && rm -rf /tmp/*

RUN . /opt/conda/etc/profile.d/conda.sh \
    && conda activate development \
    && cd /tmp \
    && git clone -b v2.4 --depth 1 https://github.com/numbbo/coco.git \
    && cd coco \
    && python do.py run-python \
    && python do.py install-postprocessing install-user \
    && cd .. \
    && rm -rf /tmp/*
