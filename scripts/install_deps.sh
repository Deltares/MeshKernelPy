#!/bin/sh

error() {
  # store last exit code before invoking any other command
  local EXIT_CODE=$?
  # print error message
  echo $(basename "$0"): $1
  exit $EXIT_CODE
}

# add development tools
yum -y install git make wget which bzip2 netcdf centos-release-scl
yum -y install ${DEVTOOLSET}
scl enable ${DEVTOOLSET} bash

# enter root
(
  cd /root  || error "Could not chnage the directory to root"

  # install CMake
  CMAKE_SCRIPT=cmake-${CMAKE_VERSION}-linux-x86_64.sh
  wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_SCRIPT} \
    || error "[cmake] ${CMAKE_SCRIPT} downlaod failed"
  CMAKE_INSTALL_PREFIX=/opt/cmake
  mkdir ${CMAKE_INSTALL_PREFIX} || error "[cmake] Creation of ${CMAKE_INSTALL_PREFIX} failed"
  chmod +x ${CMAKE_SCRIPT} || error "[cmake] Changing the permissions of ${CMAKE_SCRIPT} failed"
  bash ${CMAKE_SCRIPT} --skip-license --prefix=${CMAKE_INSTALL_PREFIX} || error "[cmake] Installation failed"

  # install boost
  BOOST_LIB=boost_${BOOST_VERSION//./_}
  wget https://boostorg.jfrog.io/artifactory/main/release/${BOOST_VERSION}/source/${BOOST_LIB}.tar.gz \
    || error "[boost] ${BOOST_LIB}.tar.gz download failed"
  export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/usr/local/lib64:/usr/lib64:$LD_LIBRARY_PATH
  tar -xzf ${BOOST_LIB}.tar.gz || error "[boost] ${BOOST_LIB}.tar.gz extraction failed"
  cd ${BOOST_LIB}
  ./bootstrap.sh --with-libraries=filesystem,system || error "[boost] bootstrap failed"
  ./b2 -j4 cxxflags="-fPIC" runtime-link=static variant=release link=static --prefix=/opt/boost_${BOOST_VERSION} install \
    || error "[boost] Installation failed"

  # install miniconda
  MINICONDA3_SCRIPT=Miniconda3-${MINICONDA3_VERSION}-Linux-x86_64.sh
  wget https://repo.continuum.io/miniconda/${MINICONDA3_SCRIPT} \
    || error "[miniconda] ${MINICONDA3_SCRIPT} downlaod failed"
  chmod +x ${MINICONDA3_SCRIPT} || error "[miniconda] Changing the permissions of ${MINICONDA3_SCRIPT} failed"
  bash ${MINICONDA3_SCRIPT} -b -p /opt/conda || error '[miniconda] Installation failed'
)