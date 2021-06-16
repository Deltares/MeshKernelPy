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


def read(rel_path: str) -> str:
    """Used to read a text file

    Args:
        rel_path (str): Relative path to the file

    Returns:
        str: File content
    """
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    """Get the version string

    Args:
        rel_path (str): Relative path to the file

    Raises:
        RuntimeError: Raised if the version string could not be found

    Returns:
        str: The version string
    """
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError("Unable to find version string.")


def get_meshkernel_name() -> str:
    """Get the filename of the MeshKernel library

    Raises:
        OSError: If the operating system is not supported

    Returns:
        str: Filename of the MeshKernel library
    """
    if platform.system() == "Windows":
        return "MeshKernelApi.dll"
    elif platform.system() == "Linux":
        return "libMeshKernelApi.so"
    raise OSError("Unsupported operating system")


try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        """Class describing our wheel.
        Basically it says that it is not a pure Python package,
        but it also does not contain any Python source and
        therefore works for all Python versions
        """

        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            # Mark us as not a pure python package
            self.root_is_pure = False

        def get_tag(self):
            python, abi, plat = _bdist_wheel.get_tag(self)
            # We don't contain any python source
            python, abi = "py3", "none"
            return python, abi, plat


except ImportError:
    bdist_wheel = None


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
        "tests": ["pytest", "pytest-cov", "nbval", "matplotlib"],
        "lint": [
            "flake8",
            "black==21.4b1",
            "isort",
        ],
        "docs": ["sphinx", "sphinx_book_theme", "myst_nb"],
    },
    python_requires=">=3.8",
    packages=["meshkernel"],
    package_data={
        "meshkernel": [get_meshkernel_name()],
    },
    cmdclass={"bdist_wheel": bdist_wheel},
    version=get_version("meshkernel/version.py"),
    classifiers=["Topic :: Scientific/Engineering :: Mathematics"],
)
