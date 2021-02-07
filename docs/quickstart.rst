.. _quickstart:

Quickstart Guide
================

.. contents:: :local:


Introduction
------------

In this tutorial, we provide an introduction to setting up an instance of the MO-CMA-ES optimizer implementation
available in the library and how to optimize a function using the two available interfaces.

Table of Contents
-----------------

* Basic setup
* Instantiating a benchmark function
* Instantiating the MOCMA optimizer
* The `fmin` interface
* The `ask` and `tell` interface
* Using a custom function
* References

Basic setup
-----------

To install the library, we can use either `PyPI` or `Conda`:

.. code-block:: bash

    # Note this is an alternative to the official repository.
    # It will be uploaded to the official PyPI repository.
    pip install -i https://test.pypi.org/simple/ anguilla 

We then import the required libraries.

.. code-block:: python

    import numpy as np

    from anguilla.fitness import benchmark
    from anguilla.optimizers.mocma import MOCMA

We set up a seed for reproducibility of the results.

.. code-block:: python

    seed = 54545
    rng = np.random.default_rng(seed)

Instantiating a benchmark function
----------------------------------

To run the optimizer we need a function to minimize. You probably want to optimize your own custom function. Here we will use a benchmark function from the implementations available in the library. However, we will also cover how to use the optimizer with custom functions.

The library provides implementations of functions from various benchmarks used in real-valued multi-objective optimization (MOO).

For this tutorial we will be using the DTLZ1$: \mathbb{R}^{30} \mapsto \mathbb{R}^3$ benchmark function.

.. code-block:: python

    n_dimensions = 30
    n_objectives = 3
    fn = benchmark.DTLZ1(n_dimensions, n_objectives, rng=rng)

Instantiating the MOCMA optimizer
---------------------------------

Parents
-------

The optimizer infers the number of *parents*, *dimensions* and *objectives*
from the initial `parent_points` and `parent_fitness` arrays we pass to its constructor 
at instantiation.

Note that we use `n_parents` to refer to $\mu$.

.. code-block:: python

    n_parents = 100
    parent_points = fn.random_points(n_parents)
    parent_fitness = fn(parent_points)

Offspring
---------

We can omit providing a value for `n_offspring` when instantiating the optimizer.
If `n_parents = 100`, then we would've be constructing an instance of $(100+100)$-MOCMA-ES 
If we'd like to use the steady-state variant, i.e $(100+1)-MOCMA-ES
Then we need to set `n_offspring = 1`.

Note that we use `n_offspring` to refer to $\lambda$.

.. code-block:: python

    n_offspring = 100

Notion of success
-----------------

The implementation supports both notions of success defined in the literature [1] [2].
By default, if none is provided the optimizer uses the population-based notion of success.

.. code-block:: python

    success_notion = "population"

Creating the instance
---------------------

We are ready to instantiate the MOCMA optimizer.
Note that we need to specify at least one stopping condition,
in this case we restrict the number of function evaluations by
setting `max_evaluations=50000`. Other stopping conditions are
`max_generations` and `target_indicator_value` (which requires providing a reference point).

.. code-block python

    optimizer = MOCMA(parent_points,
                    parent_fitness,
                    n_offspring=n_offspring,
                    success_notion=success_notion,
                    max_evaluations=50000,
                    rng=rng,
                    )

To pass a reference point, we would've done the following:

.. code-block:: python

    our_reference_point = None
    optimizer.indicator.reference = our_reference_point

The `fmin` interface
--------------------

Now, to optimize we can simply call the `fmin` method on the optimizer instance:

.. code-block:: python

    result = optimizer.fmin(fn)

The optimizer returns an instance of `OptimizerResult`, which contains both the solution and the stopping conditions that were triggered.

We can access the approximated Pareto set and front by:

.. code-block:: python

    pareto_set = result.solution.points
    pareto_front = result.solution.fitness


We can inspect which condition(s) triggered the stop by:

.. code-block:: python

    result.stopping_conditions
