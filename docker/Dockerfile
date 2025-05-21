FROM ghcr.io/fenics/dolfinx/dolfinx:v0.9.0

WORKDIR /tmp

# This argument should be the same as what-ever the python version of the dol
ARG PYTHON_VERSION=3.12


RUN python3 -m pip install  -U pip setuptools

# # # Install h5py https://github.com/h5py/h5py/issues/2222
# # RUN python3 -m pip install --no-cache-dir --no-binary=h5py git+https://github.com/h5py/h5py.git
# # RUN python3 -m pip install  meshio 

RUN apt-get -qq update && \
    apt-get -y install vim liblapacke liblapacke-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# The following is taken from DOLFINx. This shouldn't be needed in a newer (>0.9.X) DOLFINx version
ARG PYVISTA_VERSION=0.44.1

# pyvista dependencies from apt
RUN apt-get -qq update && \
    apt-get -y install libgl1-mesa-dev xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install pyvista from PyPI. pyvisa depends on (py)vtk), but vtk wheels are not
# available on pypi for linux/arm64, so we use a custom build wheel.
# matplotlib improves plotting quality with better color maps and
# properly rendering colorbars.
# trame is the preferred backend for pyvista.
RUN dpkgArch="$(dpkg --print-architecture)"; \
    case "$dpkgArch" in arm64) \
    pip install --no-cache-dir https://github.com/finsberg/vtk-aarch64/releases/download/vtk-9.3.0-cp312/vtk-9.3.0.dev0-cp312-cp312-linux_aarch64.whl ;; \
    esac; \
    python3 -m pip install matplotlib; \
    python3 -m pip install --no-cache-dir pyvista[trame]==${PYVISTA_VERSION} && \
    python3 -m pip cache purge

# Copy GUGaR source dir
COPY . qugar

# Install in Release mode
RUN  cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local/qugar \
    -DBUILD_TESTING=OFF \
    -Dqugar_DEVELOPER_MODE=OFF \
    -Dqugar_BUILD_DOC=OFF \
    -Dqugar_BUILD_CATCH2=ON \
    -Dqugar_WITH_DEMOS=OFF \
    -B build-release qugar/cpp/ && \
    ninja install -j4 -C build-release && \
    python3 -m pip install -v --no-build-isolation --check-build-dependencies --no-dependencies --no-cache-dir ./qugar/python


WORKDIR /root
