set -x

echo ${DOCKER_CONDA_ENV}

source activate ${DOCKER_CONDA_ENV}

python --version

rm -rf build *.egg-info dist

python -m pip install setuptools wheel auditwheel
python -m pip install numpy matplotlib type_enforced

python setup.py build_ext || exit 1
python setup.py sdist bdist_wheel || exit 1

cd dist/
list=()
for file in *linux_x86_64.whl; do
    list+=("$file")
done
auditwheel show ${list[0]}
auditwheel repair ${list[0]}
cd ..
cp ./dist/wheelhouse/*.whl .

conda remove -n ${DOCKER_CONDA_ENV} --all
