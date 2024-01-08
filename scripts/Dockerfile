# Container for compiling libraries using centos7. 
# By using an old linux operating system, dynamic libraries should be compatible with most linux os

FROM quay.io/pypa/manylinux2014_x86_64:2023-11-20-745eb52

WORKDIR /root

COPY . .

# versions of deps (must be set pripor to running install_deps.sh)
ENV DEVTOOLSET=devtoolset-11
ENV CMAKE_VERSION="3.23.1"
ENV BOOST_VERSION="1.81.0"
ENV MINICONDA3_VERSION="py38_4.10.3"
ENV PYTHON_VERSION="3.8"

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
