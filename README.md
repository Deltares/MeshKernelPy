# MeshKernelPy

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Deltares_MeshKernelPy&metric=alert_status)](https://sonarcloud.io/dashboard?id=Deltares_MeshKernelPy)
[![PyPI version](https://badge.fury.io/py/meshkernel.svg)](https://badge.fury.io/py/meshkernel)

`MeshKernelPy` is a library for creating and editing meshes.
It supports 1D and 2D unstructured meshes.
Support for curvilinear meshes is planned.
The underlying C++ library `MeshKernel` can be found [here](https://github.com/Deltares/MeshKernel).

# Installation

## Windows

The library can be installed from PyPI by executing

```bash
pip install meshkernel
```

If you encounter any issues importing the pip wheels on Windows, you may need to install the [Visual C++ Redistributable for Visual Studio 2019](https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

## Linux

Currently, we only offer wheels specific to Deltares' CentOS machines.
We plan to release a manylinux wheel at PyPI in the future. 

# Examples

## Creating a triangular mesh inside a polygon

In this example a mesh is created by discretizing the polygon perimeter with the desired edge length.

![](https://raw.githubusercontent.com/Deltares/MeshKernelPy/main/docs/images/TriangularMeshInPolygon.jpg)

## Mesh orthogonalization

Finite volume staggered flow solvers require the mesh to be as much orthogonal as possible. 
MeshKernel provides an algorithm to adapt the mesh and achieve a good balance between mesh orthogonality and smoothness.

![](https://raw.githubusercontent.com/Deltares/MeshKernelPy/main/docs/images/MeshOrthogonalization.jpg)

## Mesh refinement

A mesh can be refined in areas based on samples or polygon selections. 

![](https://raw.githubusercontent.com/Deltares/MeshKernelPy/main/docs/images/GridRefinement.jpg)

# License

`MeshKernelPy` uses the MIT license.
However, the wheels on PyPI bundle the LGPL licensed [MeshKernel](https://github.com/Deltares/MeshKernel).
Please make sure that this fits your needs before depending on it.


# Contributing

In order to install `MeshKernelPy` locally, please execute the following line inside your virtual environment

```bash
pip install -e ".[tests, lint, docs]"
```

Then add a compiled `MeshKernelApi.dll` into your `src` folder.

Also make sure that your editor is configured to format the code with [`black`](https://black.readthedocs.io/en/stable/) and [`isort`](https://pycqa.github.io/isort/).
When modifying `Jupyter` notebooks, the [`jupyterlab-code-formatter`](https://jupyterlab-code-formatter.readthedocs.io/en/latest/installation.html) can be used.

# Building wheels

To deploy Linux wheels to PyPI, we provide a Docker image that is based on manylinux2014_x86_64. 
This image includes cmake and boost, which are necessary for compiling the native MeshKernel library (written in C++). 
To build the Docker image, please follow these steps:

```powershell
cd scripts
docker build --progress=plain . -t build_linux_library
cd ..
```

Once the Docker image has been built, start the container in interactive mode and mount the current folder to the /root folder of the container using the following command:

```powershell
docker run -v %cd%:/root --rm -ti build_linux_library bash 
```

In the terminal of the container, run the following commands to build deployable Linux wheels:

```bash
PYBIN=/opt/python/cp38-cp38/bin/
${PYBIN}/python3 setup.py bdist_wheel
cd dist/
auditwheel show meshkernel-2.0.2-py3-none-linux_x86_64.whl
auditwheel repair meshkernel-2.0.2-py3-none-linux_x86_64.whl
```


