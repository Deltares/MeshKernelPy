#!/bin/sh

DOCKER_CONDA_ENV="docker_conda_env"

# create conda env and activate it
conda create -y --force -n "${DOCKER_CONDA_ENV}" python="${PYTHON_VERSION}" pip
. activate "${DOCKER_CONDA_ENV}"

remove_conda_env() {
  # deactivate and remove conda env
  conda deactivate
  conda env remove -n "${DOCKER_CONDA_ENV}"
}

error() {
  # store last exit code before invoking any other command
  local EXIT_CODE="$?"
  # print error message
  echo "$(basename "$0")": "$1"
  remove_conda_env
  exit $EXIT_CODE
}

# clean up residual data from previous run
rm -rf ./build ./*.egg-info #./dist

python -m pip install \
  setuptools \
  wheel \
  auditwheel \
  numpy \
  matplotlib \
  || error "pip install failed"

python setup.py build_ext || error "[setup] building C/C++ extension modules failed"
python setup.py sdist || error "[setup] Creation of source distribution failed"
python setup.py bdist_wheel || error "[setup] Building the wheel failed"

(
  cd dist || error "Could not change the directory to dist"
  list=()
  for file in *linux_x86_64.whl; do
    list+=("$file")
  done
  auditwheel show "${list[0]}"
  auditwheel repair "${list[0]}"
)

cp ./dist/wheelhouse/*.whl .

remove_conda_env