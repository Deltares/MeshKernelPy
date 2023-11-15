# Build wheels
PYBIN=/opt/python/cp38-cp38/bin/
rm -rf build *.egg-info
${PYBIN}/python3 -m pip install numpy matplotlib type_enforced || echo 'build_deps.sh: pip failed' && exit 1
${PYBIN}/python3 setup.py build_ext || echo 'build_deps.sh: setup.py build_ext failed' && exit 1
${PYBIN}/python3 setup.py sdist bdist_wheel || echo 'build_deps.sh: setup.py sdist bdist_wheel failed' && exit 1
cd dist/
list=()
for file in *linux_x86_64.whl; do
    list+=("$file")
done
auditwheel show ${list[0]}
auditwheel repair ${list[0]}
cd ..
cp ./dist/wheelhouse/*.whl .