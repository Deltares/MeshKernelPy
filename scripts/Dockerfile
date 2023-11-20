# Container for compiling libraries using centos7. 
# By using an old linux operating system, dynamic libraries should be compatible with most linux os

FROM quay.io/pypa/manylinux2014_x86_64

WORKDIR /root

ADD . .

RUN ./compile_deps.sh

CMD ["bash", "./scripts/build_deps.sh"]
