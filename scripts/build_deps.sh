DOCKER_CONDA_ENV="docker_conda_env"

echo $PATH
# create conda env
conda create -y --force -n ${DOCKER_CONDA_ENV} python=3.8 pip

# and activate it
source activate ${DOCKER_CONDA_ENV}

python --version

# clean up residual data deom previous run
rm -rf build *.egg-info #dist

python -m pip install \
  setuptools \
  wheel \
  auditwheel \
  numpy \
  matplotlib \
  type_enforced

python setup.py build_ext || (echo 'build_deps.sh: setup.py build_ext failed' && exit 1)
python setup.py sdist bdist_wheel || (echo 'build_deps.sh: setup.py sdist bdist_wheel failed' && exit 1)

cd dist/
list=()
for file in *linux_x86_64.whl; do
    list+=("$file")
done
auditwheel show ${list[0]}
auditwheel repair ${list[0]}
cd ..
cp ./dist/wheelhouse/*.whl .

conda deactivate
conda env remove -n ${DOCKER_CONDA_ENV} --all
