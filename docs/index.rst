.. Anguilla documentation master file

********************
Welcome to Anguilla!
********************

The main goal of this project is to provide a MO-CMA-ES implementation in 
Python 3.

🚧 work in progress 🚧

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

   from anguilla.optimizers.cma import CMA, StrategyParameters, StoppingConditions
   from anguilla.fitness.benchmark import Sphere

   # TODO: Showcase a MO-CMA-ES example

   # Note: Interfaces subject to changes (WIP).

    parameters = StrategyParameters(n=n)
    conditions = StoppingConditions(n=n,
                                    population_size=parameters.population_size,
                                    ftarget=1e-14,
                                    )
    fitness_function = Sphere()
    initial_point = fitness_function.propose_initial_point(n)
    optimizer = CMA(initial_point=initial_point,
                    initial_sigma=0.3,
                    strategy_parameters=parameters,
                    stopping_conditions=conditions)
    while not optimizer.stop():
        candidate_solutions = optimizer.ask()
        # TODO: Will fix so that transposition is not required below
        function_values = fitness_function(candidate_solutions.T)
        ranked_indices = optimizer.rank(candidate_solutions, function_values)
        optimizer.tell(candidate_solutions, ranked_indices)
        if (optimizer._fevals % 20) == 0:
            display(optimizer._fevals, optimizer._best_value, optimizer._best_solution)

More usage information of currently available implementations can be found in :py:mod:`anguilla.optimizers.cma.CMA`,
:py:mod:`anguilla.optimizers.cma.StrategyParameters`, :py:mod:`anguilla.optimizers.cma.StoppingConditions` and
:py:mod:`anguilla.fitness.benchmark`.

Please check out the :ref:`Quickstart Guide <quickstart>` for more detailed examples.

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
   references
