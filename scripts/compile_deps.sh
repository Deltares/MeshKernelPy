
set -x

# add development tools
yum -y install git make wget which bzip2 netcdf centos-release-scl
DEVTOOLSET=devtoolset-11
yum -y install ${DEVTOOLSET}
scl enable ${DEVTOOLSET} bash
export PATH="${PATH}:/opt/rh/${DEVTOOLSET}/root/usr/bin"
export CC=/opt/rh/${DEVTOOLSET}/root/usr/bin/gcc 
export CXX=/opt/rh/${DEVTOOLSET}/root/usr/bin/g++

# enter root
cd /root

# install CMake
CMAKE_VERSION='3.23.1'
wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh || exit 1
chmod +x cmake-${CMAKE_VERSION}-linux-x86_64.sh
mkdir /opt/cmake
./cmake-${CMAKE_VERSION}-linux-x86_64.sh --skip-license --prefix=/opt/cmake || exit 1

# install boost
BOOST_VERSION='1_81_0'
BOOST_VERSION_DOT='1.81.0'
wget https://boostorg.jfrog.io/artifactory/main/release/${BOOST_VERSION_DOT}/source/boost_${BOOST_VERSION}.tar.gz || exit 1
export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/usr/local/lib64:/usr/lib64:$LD_LIBRARY_PATH
tar -xzf boost_${BOOST_VERSION}.tar.gz
cd boost_${BOOST_VERSION}
BOOST_INSTALL_PREFIX=/opt/boost_${BOOST_VERSION}
./bootstrap.sh --with-libraries=filesystem,system || exit 1
./b2 -j4 cxxflags="-fPIC" runtime-link=static variant=release link=static --prefix=${BOOST_INSTALL_PREFIX} install
export Boost_INCLUDE_DIR=${BOOST_INSTALL_PREFIX}/include

# leave root
cd ..