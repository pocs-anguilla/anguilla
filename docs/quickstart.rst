.. _quickstart:

Quickstart Guide
================

.. contents:: :local:


Introduction
------------

In this tutorial, we provide an introduction to setting up an instance of the MO-CMA-ES optimizer implementation
available in the library and how to optimize a function using the two available interfaces.

**In this guide:**

* Basic setup
* Instantiating a benchmark function
* Instantiating the MOCMA optimizer
* The ``fmin`` interface
* The ``ask`` and ``tell`` interface
* Using a custom function
* References

Basic setup
-----------

To install the library, we can use either ``PyPI`` or ``Conda``:

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
#######

The optimizer infers the number of *parents*, *dimensions* and *objectives*
from the initial ``parent_points`` and ``parent_fitness`` arrays we pass to its constructor 
at instantiation.

Note that we use ``n_parents`` to refer to $\mu$.

.. code-block:: python

    n_parents = 100
    parent_points = fn.random_points(n_parents)
    parent_fitness = fn(parent_points)

Offspring
#########

We can omit providing a value for ``n_offspring`` when instantiating the optimizer.
If ``n_parents = 100``, then we would've be constructing an instance of $(100+100)$-MOCMA-ES 
If we'd like to use the steady-state variant, i.e $(100+1)-MOCMA-ES
Then we need to set ``n_offspring = 1``.

Note that we use ``n_offspring`` to refer to $\lambda$.

.. code-block:: python

    n_offspring = 100

Notion of success
#################

The implementation supports both notions of success defined in the literature [1] [2].
By default, if none is provided the optimizer uses the population-based notion of success.

.. code-block:: python

    success_notion = "population"

Creating the instance
---------------------

We are ready to instantiate the MOCMA optimizer.
Note that we need to specify at least one stopping condition,
in this case we restrict the number of function evaluations by
setting ``max_evaluations=50000``. Other stopping conditions are
``max_generations`` and ``target_indicator_value`` (which requires providing a reference point).

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

The ``fmin`` interface
----------------------

Now, to optimize we can simply call the ``fmin`` method on the optimizer instance:

.. code-block:: python

    result = optimizer.fmin(fn)

The optimizer returns an instance of ``OptimizerResult``, which contains both the solution and the stopping conditions that were triggered.

We can access the approximated Pareto set and front by:

.. code-block:: python

    pareto_set = result.solution.points
    pareto_front = result.solution.fitness


We can inspect which condition(s) triggered the stop by:

.. code-block:: python

    result.stopping_conditions

The ``ask`` and ``tell`` interface
----------------------------------

An alternative interface, called the ask-and-tell interface, allows for decoupled
execution between calls to the optimizer and the function.
The ``fmin`` implementation uses the ask-and-tell interface under the hood.

The ``ask`` method generates new search points (offspring) by using mutation
operators on the parent population.

The ``tell`` method is used to inform the optimizer about the fitness of these
new points. It then performs environmental selection to produce the new parent population.

When could we need to use this interface? Say, for example, that we want to checkpoint solutions every ``5000`` function evaluations:

.. code-block:: python

    optimizer = MOCMA(parent_points,
                    parent_fitness,
                    n_offspring=n_offspring,
                    success_notion=success_notion,
                    max_evaluations=5000,#50000,
                    rng=rng,
                    )

    while not optimizer.stop.triggered:
        points = optimizer.ask()
        if fn.has_constraints:
            fitness, penalized_fitness = fn.evaluate_with_penalty(points)
            optimizer.tell(fitness, penalized_fitness)
        else:
            fitness = fn(points)
            optimizer.tell(fitness)
        if optimizer.evaluation_count % 5000 == 0:
            # Access current Pareto front approximation
            # and save it to a file
            np.savetxt(f"./output/fitness-{optimizer.evaluation_count}.csv",
                    optimizer.best.fitness,
                    delimiter=","
                    )

Note that we treated constrained functions a bit differently in the above example.
This is because we should penalize search points that violate any constraints.
To inform the optimizer about this, we can pass the penalized fitness.

In the example above we've could also done:

.. code-block:: python

    fitness = fn.evaluate_with_penalty(points)
    optimizer.tell(*fitness)


In addition, some functions are noisy. Usually, these functions are evaluated multiple times 
and an average of their fitness is reported to the optimizer. Naturally, this consumes more function evaluations
from the budget we may've defined as a stopping condition. To inform the optimizer about this, we could do:

.. code-block:: python

    for i in range(n_repetitions):
        fitness[i] = noisy_function(point)
    optimizer.tell(np.average(fitness, axis=0), evaluation_count=n_evaluations) 

Note, however, that you don't need to use the ask-and-tell interface to optimize noisy functions. You can abstract the noisy evaluation in a custom function and use ``fmin`` interface. In the next section, we will see how to define a custom function that the optimizer can minimize.

After the optimizer finishes the optimization run, we can get the solution by:

.. code-block:: python

    solution = optimizer.best

Using a custom function
-----------------------

WIP
