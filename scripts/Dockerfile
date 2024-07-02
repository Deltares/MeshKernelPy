# Container for compiling MeshKernelPy
# By using an old linux operating system, dynamic libraries should be compatible with most linux os

FROM quay.io/pypa/manylinux_2_28_x86_64:2024-07-01-8dac23b

WORKDIR /root

COPY . .

# versions of deps (must be set pripor to running install_deps.sh)
ENV DEVTOOLSET=gcc-toolset-12
ENV CMAKE_VERSION="3.26.4"
ENV BOOST_VERSION="1.83.0"
ENV MINICONDA3_VERSION="py310_23.11.0-2"
ENV PYTHON_VERSION="3.10"

# install the dependencies
RUN ./install_deps.sh

# Append/define necessary paths/vars
ENV PATH="/opt/rh/${DEVTOOLSET}/root/usr/bin:$PATH"
ENV PATH="/opt/conda/bin:$PATH"
ENV Boost_INCLUDE_DIR=/opt/${BOOST_VERSION}/include
ARG GNU_COMPILER_DIRECTORY=/opt/rh/${DEVTOOLSET}/root/usr/bin
ENV CC=${GNU_COMPILER_DIRECTORY}/gcc
ENV CXX=${GNU_COMPILER_DIRECTORY}/g++

# build the wheel
CMD ["bash", "./scripts/build_wheel.sh"]
