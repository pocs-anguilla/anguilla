#!/usr/bin/env python

# Adapted from the reference example for Pybind11 + Scikit-build:
# https://github.com/pybind/scikit_build_example/blob/master/setup.py

from skbuild import setup


def main():
    author = "Anguilla Development Team"
    version = "0.0.13"
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
            "anguilla.optimizers",
            "anguilla.fitness",
            "anguilla.fitness.benchmark",
            "anguilla.hypervolume",
        ],
        package_dir={".": "anguilla"},
        cmake_install_dir="anguilla",
        cmake_languages=["CXX"],
        cmake_minimum_required_version="3.18",
        cmake_args=["-DCMAKE_CXX_STANDARD=17", "-DCMAKE_CXX_EXTENSIONS=OFF"],
        install_requires=[
            "numpy >= 1.19.4",
        ],
        extras_require={
            ':python_version == "3.6"': [
                "dataclasses",
            ],
            ':python_version <= "3.7"': [
                "typing_extensions",
            ],
        },
    )


if __name__ == "__main__":
    main()
