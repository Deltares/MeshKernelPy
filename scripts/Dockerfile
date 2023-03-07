# Container for compiling libraries using centos7. 
# By using an old linux operating system, dynamic libraries should be compatible with most linux os

FROM quay.io/pypa/manylinux2014_x86_64
WORKDIR /root

ADD . .

# add gcc compiler
RUN ./compile_deps.sh

ENV Boost_INCLUDE_DIR=/opt/boost_1_81_0/include
ENV Boost_LIBRARYDIR=/opt/boost_1_81_0/lib
ENV PATH="/opt/rh/devtoolset-9/root/usr/bin:$PATH"
ENV CC=/opt/rh/devtoolset-9/root/usr/bin/gcc 
ENV CXX=/opt/rh/devtoolset-9/root/usr/bin/g++

CMD ["bash", "./scripts/build_deps.sh"]
