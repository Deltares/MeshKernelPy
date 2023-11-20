# Container for compiling libraries using centos7. 
# By using an old linux operating system, dynamic libraries should be compatible with most linux os

FROM quay.io/pypa/manylinux2014_x86_64

FROM continuumio/miniconda

WORKDIR /root

ADD . .

ENV DOCKER_CONDA_ENV="docker_conda_env"

RUN conda create -y --force -n ${DOCKER_CONDA_ENV} python=3.9 pip

RUN ./compile_deps.sh

CMD ["bash", "./scripts/build_deps.sh"]
