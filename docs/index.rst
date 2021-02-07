.. Anguilla documentation master file

********************
Welcome to Anguilla!
********************

The main goal of this project is to provide a MO-CMA-ES implementation in 
Python 3.

Installing
##########

.. code-block:: bash

   pip install -i https://test.pypi.org/simple/ anguilla

.. code-block:: bash

   # (currently does not work)

   conda install -c Anguilla anguilla

A short example
###############

.. code-block:: python

   optimizer = MOCMA(parent_points,
                     parent_fitness,
                     n_offspring=n_offspring,
                     success_notion=success_notion,
                     max_evaluations=50000,
                     rng=rng,
                  )
   result = optimizer.fmin(fn)

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
