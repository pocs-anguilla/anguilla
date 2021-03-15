#!/usr/bin/env python

# Adapted from the reference example for Pybind11 + Scikit-build:
# https://github.com/pybind/scikit_build_example/blob/master/setup.py

import os
from skbuild import setup

# Temporary work-around for building PyPI package
# with the BLAS compatible library BLIS.
# TODO: figure out if it is possible to use same BLAS as Numpy.
def compute_install_requires():
    deps = [
        "numpy >= 1.20",
    ]
    if os.getenv("CONDA_BUILD", None) is None:
        deps.append("blis >= 0.7.4")
    return deps


def main():
    author = "Anguilla Development Team"
    version = "0.0.19"
    with open("README.rst", encoding="utf-8") as f:
        long_description = f.read()

    setup(
        name="anguilla",
        author=author,
        version=version,
        description="An implementation of MO-CMA-ES.",
        long_description=long_description,
        long_description_content_type="text/x-rst",
        packages=[
            "anguilla",
            "anguilla.archive",
            "anguilla.dominance",
            "anguilla.ds",
            "anguilla.fitness",
            "anguilla.fitness.benchmark",
            "anguilla.hypervolume",
            "anguilla.optimizers",
        ],
        package_dir={".": "anguilla"},
        cmake_install_dir="anguilla",
        cmake_languages=["CXX"],
        cmake_minimum_required_version="3.18",
        cmake_args=[
            "-DCMAKE_CXX_STANDARD=17",
            "-DCMAKE_CXX_EXTENSIONS=OFF",
        ],
        python_requires=">3.7",
        install_requires=compute_install_requires(),
        extras_require={
            ':python_version <= "3.7"': [
                "typing_extensions",
            ],
        },
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
            "Programming Language :: C++",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3 :: Only",
            "Topic :: Scientific/Engineering",
            "Typing :: Typed",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
        ],
    )


if __name__ == "__main__":
    main()
