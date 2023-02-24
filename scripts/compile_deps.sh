# add development tools
yum -y install git make wget which bzip2 centos-release-scl
yum -y install devtoolset-9
scl enable devtoolset-9 bash
export PATH="/opt/rh/devtoolset-9/root/usr/bin:$PATH"
export CC=/opt/rh/devtoolset-9/root/usr/bin/gcc 
export CXX=/opt/rh/devtoolset-9/root/usr/bin/g++

# add cmake
cd /root
export CMAKE_VERSION='3.23.1'
wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh
chmod +x cmake-${CMAKE_VERSION}-linux-x86_64.sh
mkdir /opt/cmake
./cmake-${CMAKE_VERSION}-linux-x86_64.sh --skip-license --prefix=/opt/cmake

# add boost
export BOOST_VERSION='1_81_0'
export BOOST_VERSION_DOT='1.81.0'
wget https://boostorg.jfrog.io/artifactory/main/release/${BOOST_VERSION_DOT}/source/boost_${BOOST_VERSION}.tar.gz
export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/usr/local/lib64:/usr/lib64:$LD_LIBRARY_PATH
tar -xzf boost_${BOOST_VERSION}.tar.gz
cd boost_${BOOST_VERSION}
./bootstrap.sh --with-libraries=filesystem,system
./b2 -j4 cxxflags="-fPIC" runtime-link=static variant=release link=static --prefix=/opt/${BOOST_VERSION} install
cd ..
rm boost_${BOOST_VERSION}.tar.gz
