# build linux wheels
PYBIN=/opt/python/cp38-cp38/bin/
${PYBIN}/python3 setup.py bdist_wheel
cd dist/
list=()
for file in *linux_x86_64.whl; do
    list+=("$file")
done
auditwheel show ${list[0]}
auditwheel repair ${list[0]}