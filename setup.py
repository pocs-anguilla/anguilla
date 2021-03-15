#!/usr/bin/env python

# Adapted from the reference example for Pybind11 + Scikit-build:
# https://github.com/pybind/scikit_build_example/blob/master/setup.py

from skbuild import setup


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
        install_requires=[
            "numpy >= 1.20",
        ],
        extras_require={
            ':python_version <= "3.7"': [
                "typing_extensions",
            ],
        },
        classifiers=[
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: C++",
            "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
            "Development Status :: 3 - Alpha",
        ],
    )


if __name__ == "__main__":
    main()
