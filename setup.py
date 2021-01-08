#!/usr/bin/env python

# Adapted from the reference example for Pybind11 + Scikit-build:
# https://github.com/pybind/scikit_build_example/blob/master/setup.py

from skbuild import setup


def main():
    author = "Anguilla Development Team"
    version = "0.0.1"
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
            "anguilla.optimizers",
            "anguilla.hypervolume",
            "anguilla.fitness",
        ],
        package_dir={".": "anguilla"},
        cmake_install_dir="anguilla",
        cmake_languages=["CXX"],
        cmake_minimum_required_version="3.15",
        install_requires=[
            "numpy",
        ],
        extras_require={
            ':python_version == "3.6"': [
                "dataclasses",
            ],
        },
    )


if __name__ == "__main__":
    main()
