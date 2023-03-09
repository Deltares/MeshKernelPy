# Container for compiling libraries using centos7. 
# By using an old linux operating system, dynamic libraries should be compatible with most linux os

FROM quay.io/pypa/manylinux2014_x86_64
WORKDIR /root

ADD . .

# add gcc compiler
RUN ./compile_deps.sh

ENV GNU_COMPILER_DIRECTORY=/opt/rh/devtoolset-9/root/usr/bin
ENV Boost_INCLUDE_DIR=/opt/boost_1_81_0/include
ENV PATH="/opt/rh/devtoolset-9/root/usr/bin:$PATH"
ENV CC=${GNU_COMPILER_DIRECTORY}/gcc
ENV CXX=${GNU_COMPILER_DIRECTORY}/g++

CMD ["bash", "./scripts/build_deps.sh"]
