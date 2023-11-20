
# add development tools
yum -y install git make wget which bzip2 netcdf centos-release-scl
yum -y install ${DEVTOOLSET}
scl enable ${DEVTOOLSET} bash

# enter root
cd /root

# install CMake
CMAKE_VERSION='3.23.1'
wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh \
  || (echo 'compile_deps.sh: wget cmake failed'  && exit 1)  
chmod +x cmake-${CMAKE_VERSION}-linux-x86_64.sh
mkdir /opt/cmake
./cmake-${CMAKE_VERSION}-linux-x86_64.sh --skip-license --prefix=/opt/cmake \
  || (echo 'compile_deps.sh: install cmake failed'  && exit 1)

# install boost
BOOST_VERSION='1_81_0'
BOOST_VERSION_DOT='1.81.0'
wget https://boostorg.jfrog.io/artifactory/main/release/${BOOST_VERSION_DOT}/source/boost_${BOOST_VERSION}.tar.gz \
|| (echo 'compile_deps.sh: wget boost failed'  && exit 1)
export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/usr/local/lib64:/usr/lib64:$LD_LIBRARY_PATH
tar -xzf boost_${BOOST_VERSION}.tar.gz
cd boost_${BOOST_VERSION}
BOOST_INSTALL_PREFIX=/opt/boost_${BOOST_VERSION}
./bootstrap.sh --with-libraries=filesystem,system \
  || (echo 'compile_deps.sh: bootstrap boost failed' && exit 1)
./b2 -j4 cxxflags="-fPIC" runtime-link=static variant=release link=static --prefix=${BOOST_INSTALL_PREFIX} install \
  || (echo 'compile_deps.sh: install boost failed' && exit 1)

# install miniconda
MINICONDA_VERSION='py38_4.10.3'
wget https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh \
  || (echo 'compile_deps.sh: wget miniconda failed'  && exit 1)
chmod +x Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh
bash Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -b -p /opt/conda \
  || (echo 'compile_deps.sh: install miniconda failed'  && exit 1)

# leave root
cd ..