.. Anguilla documentation master file

********************
Welcome to Anguilla!
********************

The main goal of this project is to provide a MO-CMA-ES implementation in 
Python 3.

.. warning::
   🚧 Work in progress: this repository is a draft under development. 🚧

.. Note::

   The project is currently being developed as part of a POCS.

   It is based on the reference implementations from 
   `The Shark Machine Learning Library <https://www.shark-ml.org/>`_ and
   `PyCMA <https://github.com/CMA-ES/pycma>`_.
   This prototype draws inspiration from the reference projects, the cited literature and the discussions being held in those projects.

   So far, we've implemented a prototype of CMA-ES while we get acquainted with the concepts required to implement MO-CMA-ES.
   The current state of the project showcases some minimal working functionality of the different components, such as 
   implementation, documentation, tests, and continuous integration setup.

   The name is an internal project name. The reference implementation 
   for MO-CMA-ES is `The Shark Machine Learning Library
   <https://www.shark-ml.org/>`_ and this implementation
   is in Python 3, so with eels being snake-like fish, `Anguilla <https://en.wikipedia.org/wiki/European_eel>`_ sounded like a good fit (as Eel is already used by another project).

Installing
##########

.. code-block:: bash

   # (currently does not work)

   pip install anguilla

.. code-block:: bash

   # (currently does not work)

   conda install --channel=conda-forge anguilla

A short example
###############

.. code-block:: python

   # TODO: Showcase a MO-CMA-ES example here

Please check out the :ref:`Quickstart Guide <quickstart>` for more detailed examples.

More usage information of currently available implementations can be found in the ":ref:`modindex`" page.

Indices and tables
##################

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :maxdepth: 3
   :caption: Contents:
   :hidden:

   quickstart
   development
   notes
   references
