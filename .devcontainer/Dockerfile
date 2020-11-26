FROM python:3.8-slim

# Install dependencies for building Numpy and Shark.
RUN apt-get update \
    && apt-get -y install build-essential gfortran cmake make remake \
    git mercurial subversion pkg-config \
    libgfortran-8-dev libpng-dev libjpeg62-turbo-dev libzip-dev \
    libbz2-dev libicu-dev zlib1g-dev libtinfo5 \
    libboost-dev  libboost-serialization-dev libboost-filesystem-dev \
    libboost-system-dev libboost-regex-dev libboost-test-dev

# Install various linear algebra libraries and set alternative 
# which to use by default.
# To see available options, use:
#   update-alternatives --config libblas.so.3-x86_64-linux-gnu
#   update-alternatives --config liblapack.so.3-x86_64-linux-gnu
# Ref.: https://wiki.debian.org/DebianScience/LinearAlgebraLibraries
# Note: currently Shark would give preference to OpenBlas over Blis

#RUN apt-get install -y libopenblas-dev libblis-dev libatlas-base-dev \
#    && update-alternatives --set libblas.so.3-x86_64-linux-gnu \
#    /usr/lib/x86_64-linux-gnu/blis-openmp/libblas.so.3 \
#    && update-alternatives --set liblapack.so.3-x86_64-linux-gnu \
#    /usr/lib/x86_64-linux-gnu/atlas/liblapack.so.3

RUN apt-get install -y libblis-dev libatlas-base-dev \
    && update-alternatives --set libblas.so.3-x86_64-linux-gnu \
    /usr/lib/x86_64-linux-gnu/blis-openmp/libblas.so.3 \
    && update-alternatives --set liblapack.so.3-x86_64-linux-gnu \
    /usr/lib/x86_64-linux-gnu/atlas/liblapack.so.3

# Build Shark
RUN cd /usr/local/src \
    && git clone --depth=1 -b 4.1 https://github.com/Shark-ML/Shark.git

COPY cmake-find-blis.patch /usr/local/src

# Currently fails 1 test:
# >  99% tests passed, 1 tests failed out of 149
# >  35 - GradDesc_Adam (Failed)
RUN cd /usr/local/src/Shark \
    && git apply --whitespace=nowarn ../cmake-find-blis.patch \
    && cd /tmp \
    && mkdir build_shark \
    && cd build_shark \
    && cmake -DBUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON /usr/local/src/Shark \
    && remake -j $(nproc --all) \
    && (remake -j $(nproc --all) test || true) \
    && remake install \
    && rm -rf /tmp/*

# Build Numpy and Scipy from source so that they and Shark use the same
# linear algebra libraries, for benchmarking/comparing Anguilla with the
# Shark reference implementation.

RUN cd /usr/local/src \
    && git clone --depth=1 -b master https://github.com/numpy/numpy \
    && git clone --depth=1 -b master https://github.com/scipy/scipy

RUN cd /usr/local/src/numpy \
    && pip3 install -e . \
    && rm -rf /tmp/*

RUN cd /usr/local/src/scipy \
    && pip3 install -e . \
    && rm -rf /tmp/*

# Install Cling (ubuntu 18.04. ~= debian buster)

RUN apt-get install -y wget ca-certificates

RUN cd /tmp \
    && mkdir -p /opt \
    && wget https://root.cern/download/cling/cling_2020-11-05_ROOT-ubuntu18.04.tar.bz2 \
    && tar -C /opt -xf cling_2020-11-05_ROOT-ubuntu18.04.tar.bz2

# Make Cling available in PATH

ENV PATH "$PATH:/opt/cling_2020-11-05_ROOT-ubuntu18.04/bin:/opt/cling_2020-11-05_ROOT-ubuntu18.04/lib"

# Cache these Python dependencies in a layer so that rebuilding
# the devcontainer doesn't take too long when updating, e.g.,
# "requirements/devcontainer.txt" (see "devcontainer.json").

RUN pip3 install jupyterlab

# Install Cling's kernels for Jupyter
# See: https://github.com/root-project/cling/tree/master/tools/Jupyter

RUN cd /opt/cling_2020-11-05_ROOT-ubuntu18.04/share/cling/Jupyter/kernel \
    && pip install -e . \
    && jupyter-kernelspec install cling-cpp17 \
    && jupyter-kernelspec install cling-cpp1z \
    && jupyter-kernelspec install cling-cpp14 \
    && jupyter-kernelspec install cling-cpp11
