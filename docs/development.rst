.. _development:

*****************
Development Guide
*****************

.. contents:: :local:

Environment
###########
A `dev container <https://code.visualstudio.com/docs/remote/containers>`_ definition for VS Code is provided for convenience.
This requires having `Docker <https://www.docker.com/>`_ installed in addition to VS Code.

In VS Code, you can open a Bash terminal that runs in the dev container. Some useful commands are shown below.

To initialize the Git submodules (for external C++ libraries such as Boost.Intrusive):

.. code-block:: bash

  make update_submodules

To run Jupyter Lab using the devcontainer:

.. code-block:: bash

  make jupyter

  # Or with the senvironment environment variables necessary for debugging
  make jupyter_debug

The Jupyter setup supports both Python and C++ (through `cling <https://github.com/root-project/cling>`_). This allows you to run Shark from a notebook!

To build and view the documentation:

.. code-block:: bash

  make -C docs view

To build a local copy of the C++ extension modules:

.. code-block:: bash

  make cxx_extension

  # Compiles version suitable for debugging
  make cxx_extension_debug

To run the test suite:

.. code-block:: bash

  make test

  # Or with the senvironment environment variables necessary for debugging
  make test_debug

The debugging version of the C++ extensions are dynamically linked to `ASan/UBSan <https://github.com/google/sanitizers>`_.

Supported Python versions
#########################
The implementation supports Python 3.6 to 3.9, so any new features can be
used as long there are backports available.

Versioning
##########

* We use semantic versioning.
* All releases should contain a changelog and be tagged in Git.

When bumping the version number the following files should be updated (relative paths to the root of the project are shown):

* `setup.py`
* `CMakeLists.txt`
* `anguilla/__init__.py`
* `conda-recipes/anguilla/meta.yaml`
* `sonar-project.properties`

File structure
##############

The files in the project are organized as shown in the following table.

Conventions
###########

Programming
***********

* Fix any `PEP-8 <https://www.python.org/dev/peps/pep-0008/>`_ and `PEP-257 <https://www.python.org/dev/peps/pep-0257/>`_ 
  issues raised by the linter.
* Fix any `PEP-484 <https://www.python.org/dev/peps/pep-0484/>`_ and `PEP-526 <https://www.python.org/dev/peps/pep-0526/>`_ 
  issues raised by the linter whenever possible.
* Use explicit type casts (e.g. ``2. * float(4)`` over ``2. * 4``).
* Use the standard library functions from the ``math`` module for scalars. Use Numpy otherwise or if the function is not provided
  by the ``math`` module.

Documentation
*************

We follow the `Numpy documentation <https://numpydoc.readthedocs.io/en/latest/format.html>`_ conventions.

.. |docs| image:: https://readthedocs.org/projects/anguilla/badge/?version=latest
          :target: https://anguilla.readthedocs.io/en/latest/?badge=latest
          :alt: Documentation status

Continuous Integration
######################

We use `GitHub Actions <https://docs.github.com/en/actions/learn-github-actions>`_.
You find the workflows in the `.github` directory.

The following actions are used:

* https://github.com/actions/checkout
* https://github.com/actions/upload-artifact
* https://github.com/actions/download-artifact
* https://github.com/actions/setup-python
* https://github.com/conda-incubator/setup-miniconda
* https://github.com/joerick/cibuildwheel
* https://github.com/pypa/gh-action-pypi-publish

Binary wheels are built using `cibuildwheel <https://cibuildwheel.readthedocs.io/en/stable/>`_.

Tests
*****

|conda| |wheels| |codecov|

Tests are run against the built Conda packages / binary wheels.
In the case of Windows, tests can't run when using Python 3.8+ due to a change in how DLLs are loaded (which affects C extensions).

.. |codecov| image:: https://codecov.io/gh/pocs-anguilla/anguilla/branch/develop/graph/badge.svg?token=Z29R3QIDY6
             :target: https://codecov.io/gh/pocs-anguilla/anguilla
             :alt: Coverage Status

.. |conda| image:: https://github.com/pocs-anguilla/anguilla/workflows/Conda/badge.svg?branch=develop
           :target: https://github.com/pocs-anguilla/anguilla
           :alt: Conda build

.. |wheels| image:: https://github.com/pocs-anguilla/anguilla/workflows/Wheels/badge.svg?branch=develop
           :target: https://github.com/pocs-anguilla/anguilla
           :alt: Wheels build

Static Analysis
***************

|deepsource| |deepcode| |sonarcloud|

The CI pipeline includes 3 static analysis services, namely, `DeepSource <https://deepsource.io/>`_, `DeepCode <https://www.deepcode.ai/>`_ and `SonarCloud <https://sonarcloud.io/>`_;
which provide automated code reviews. We thank these bots (and their creators) for pointing out bugs, 
code smells and other ways to improve the code base.

.. |deepsource| image:: https://deepsource.io/gh/pocs-anguilla/anguilla.svg/?label=active+issues&show_trend=true&token=CZElZ2ZetdLdyxuEWD6Y7NYo
                :target: https://deepsource.io/gh/pocs-anguilla/anguilla/?ref=repository-badge
                :alt: Static analysis status (deepsource)

.. |deepcode|   image:: https://www.deepcode.ai/api/gh/badge?key=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJwbGF0Zm9ybTEiOiJnaCIsIm93bmVyMSI6InBvY3MtYW5ndWlsbGEiLCJyZXBvMSI6ImFuZ3VpbGxhIiwiaW5jbHVkZUxpbnQiOmZhbHNlLCJhdXRob3JJZCI6MjUzNDIsImlhdCI6MTYwNjQwMjExN30.PAYMuKXLpi3tBoJQufB62gBHtODZ7HZrhFpnJ1lcmu8
                :target: https://www.deepcode.ai/app/gh/pocs-anguilla/anguilla/_/dashboard?utm_content=gh%2Fpocs-anguilla%2Fanguilla

.. |sonarcloud| image:: https://sonarcloud.io/images/project_badges/sonarcloud-black.svg
                :height: 20
                :width: 85
                :target: https://sonarcloud.io/dashboard?id=pocs-anguilla_anguilla
                :alt: sonarcloud badge

Other
*****

We also use:

* `Codecov <https://about.codecov.io/>`_
* `Dependabot <https://dependabot.com>`_

Secrets Inventory
*****************

The repository defines the following secrets:

* ANACONDA_TOKEN
* CODECOV_TOKEN
* PYPI_TEST_TOKEN
* PYPI_TOKEN
* SONAR_TOKEN
