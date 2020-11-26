.. _development:

*****************
Development Guide
*****************

Environment
###########
A `dev container <https://code.visualstudio.com/docs/remote/containers>`_ definition for VS Code is provided for convenience.
This requires having `Docker <https://www.docker.com/>`_ installed in addition to VS Code.

In VS Code, you can open a Bash terminal that runs in the dev container. Some useful commands are shown below.

To run Jupyter Lab using the devcontainer:

.. code-block:: bash

  jupyter-lab --allow-root

The Jupyter setup supports both Python and C++ (through `cling <https://github.com/root-project/cling>`_). This allows you to run Shark from a notebook!

To build and view the documentation:

.. code-block:: bash

  make -C docs view

To run the test suite:

.. code-block:: bash

  python -m unittest

Supported Python versions
#########################
The implementation supports Python 3.6 to 3.9, so any new features can be
used as long there are backports available.

Versioning
##########

* We use semantic versioning.
* All releases should contain a proper changelog and be tagged in Git.

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

* We follow the `Numpy documentation <https://numpydoc.readthedocs.io/en/latest/format.html>`_ conventions.

Continuous integration
######################

Pending.
