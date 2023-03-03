import codecs
import os
import pathlib
import platform
import shutil

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as build_ext_orig

author_dict = {
    "Julian Hofer": "julian.hofer@deltares.nl",
    "Prisca van der Sluis": "prisca.vandersluis@deltares.nl",
    "Luca Carniato": "luca.carniato@deltares.nl",
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


def get_library_name() -> str:
    """Get the filename of the MeshKernel library

    Raises:
        OSError: If the operating system is not supported

    Returns:
        str: Filename of the MeshKernel library
    """
    system = platform.system()
    if system == "Windows":
        return "MeshKernelApi.dll"
    elif system == "Linux":
        return "libMeshKernelApi.so"
    elif system == "Darwin":
        return "libMeshKernelApi.dylib"
    else:
        raise OSError(f"Unsupported operating system: {system}")


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


class CMakeExtension(Extension):
    """Class for building a native cmake extension (C++)"""

    def __init__(self, repository):
        """Constructor of CMakeExtension class

        Args:
            repository (str): The git repository of the extension to build
        """

        name = repository.split("/")[-1]
        super().__init__(name, sources=[])
        self.repository = repository


class build_ext(build_ext_orig):
    """Class for building an  extension using cmake"""

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):

        cwd = str(pathlib.Path().absolute())
        build_temp = pathlib.Path(self.build_temp)

        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        os.chdir(str(build_temp))
        if not os.path.isdir(ext.name):
            self.spawn(["git", "clone", ext.repository])

        os.chdir(ext.name)

        if not self.dry_run:

            library_name = get_library_name()
            system = platform.system()
            if system == "Linux":
                self.spawn(
                    [
                        "cmake",
                        "-S",
                        ".",
                        "-B",
                        "build",
                        "-DCMAKE_BUILD_TYPE=Release",
                        "-DADD_UNIT_TESTS_PROJECTS=OFF",
                    ]
                )
                self.spawn(["cmake", "--build", "build", "--config", "Release", "-j4"])
                meshkernel_path = os.path.join(
                    *[
                        pathlib.Path().absolute(),
                        "build",
                        "src",
                        "MeshKernelApi",
                        library_name,
                    ]
                )
                self.spawn(["strip", "--strip-unneeded", str(meshkernel_path)])
            if system == "Windows":
                self.spawn(
                    [
                        "cmake",
                        "-S",
                        ".",
                        "-B",
                        "build",
                        "-G",
                        "Visual Studio 16 2019",
                        "-DCMAKE_BUILD_TYPE=Release",
                        "-DADD_UNIT_TESTS_PROJECTS=OFF",
                    ]
                )
                self.spawn(["cmake", "--build", "build", "--config", "Release", "-j4"])
                meshkernel_path = os.path.join(
                    *[
                        pathlib.Path().absolute(),
                        "build",
                        "src",
                        "MeshKernelApi",
                        "Release",
                        library_name,
                    ]
                )

            shutil.copyfile(
                meshkernel_path, os.path.join(*[cwd, "meshkernel", library_name])
            )

        os.chdir(cwd)


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
            "black",
            "isort",
        ],
        "docs": ["sphinx", "sphinx_book_theme", "myst_nb"],
    },
    python_requires=">=3.8",
    package_data={
        "meshkernel": [get_library_name()],
    },
    packages=find_packages(),
    ext_modules=[CMakeExtension("https://github.com/Deltares/MeshKernel")],
    cmdclass={"bdist_wheel": bdist_wheel, "build_ext": build_ext},
    version=get_version("meshkernel/version.py"),
    classifiers=["Topic :: Scientific/Engineering :: Mathematics"],
)
