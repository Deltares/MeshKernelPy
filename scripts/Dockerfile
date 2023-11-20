# Container for compiling libraries using centos7. 
# By using an old linux operating system, dynamic libraries should be compatible with most linux os

FROM quay.io/pypa/manylinux2014_x86_64

WORKDIR /root

ADD . .

ENV DEVTOOLSET=devtoolset-11

RUN ./compile_deps.sh

ENV PATH="/opt/rh/${DEVTOOLSET}/root/usr/bin:$PATH"
ENV PATH="/opt/conda/bin:$PATH"
ENV Boost_INCLUDE_DIR=/opt/boost_1_81_0/include
ARG GNU_COMPILER_DIRECTORY=/opt/rh/${DEVTOOLSET}/root/usr/bin
ENV CC=${GNU_COMPILER_DIRECTORY}/gcc
ENV CXX=${GNU_COMPILER_DIRECTORY}/g++

CMD ["bash", "./scripts/build_deps.sh"]
