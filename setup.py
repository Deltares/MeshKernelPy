import codecs
import os.path
import platform
import sys
from pathlib import Path

from setuptools import Distribution, find_namespace_packages, setup

# edit author dictionary as necessary
author_dict = {
    "Julian Hofer": "julian.hofer@deltares.nl",
    "Prisca van der Sluis": "prisca.vandersluis@deltares.nl",
}
__author__ = ", ".join(author_dict.keys())
__author_email__ = ", ".join(s for _, s in author_dict.items())


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError("Unable to find version string.")


class BinaryDistribution(Distribution):
    def has_ext_modules(foo):
        return True


def get_meshkernel_name():
    if platform.system() == "Windows":
        return "MeshKernelApi.dll"
    elif platform.system() == "Linux":
        return "libMeshKernelApi.so"
    raise OSError("Unsupported operating system")


long_description = read("README.md")

setup(
    name="meshkernel",
    description="`meshkernel` is a library which can be used to manipulate meshes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=__author__,
    author_email=__author_email__,
    url="https://github.com/Deltares/MeshKernelPy",
    license="MIT",
    platforms="Windows, Linux",
    install_requires=["numpy"],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "flake8",
            "black==21.4b1",
            "isort",
        ]
    },
    python_requires=">=3.8",
    packages=find_namespace_packages(include=("meshkernel")),
    package_data={
        "meshkernel": [get_meshkernel_name()],
    },
    distclass=BinaryDistribution,
    version=get_version("meshkernel/__init__.py"),
    classifiers=["Topic :: Scientific/Engineering :: Mathematics"],
)
