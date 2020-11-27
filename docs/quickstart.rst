.. _quickstart:

Quickstart Guide
================

.. contents:: :local:

A brief overview
----------------

Variants of CMA-ES differ in the evolution strategy they use, in how they peform adaptation, 
and in wether they work with single or multi-objective functions.
They are distinguished by prefixes that denote these differences.

In the following, a short introduction to variants of CMA-ES is provided to give some background on 
what is currently possible with the implementations provided in this project.
Also, pointers to sources in which to get more in-depth information are provided.

Evolution strategy
******************

**Background**

An evolution strategy (ES) defines, among other things, how *selection* and *recombination* are performed.
The symbols :math:`\lambda` and :math:`\mu` denote the population size and the parent population size, respectively.

*Selection* defines how the new parent population is chosen.
The new :math:`\mu` parents can be selected among the :math:`\lambda` offspring or among both the :math:`\mu` parents plus their :math:`\lambda` offspring.
The former case is called non-elitist (or comma) selection and the latter elitist (or plus) selection :cite:`2002:es-comprehensive-intro`.

*Recombination* defines how the information of the parents is combined to create each of the new :math:`\lambda` offspring.
The number of parents that are mixed to generate a new individual is denoted by :math:`\rho` and called the mixing number. In CMA-ES, :math:`\rho = \mu` :cite:`2001:cdsa-es`.
The recombination of the :math:`\rho` parents can be *dominant* or *intermediate* :cite:`2002:es-comprehensive-intro`.
In CMA-ES, the recombination is *intermediate* :cite:`2001:cdsa-es`.
Furthermore, *intermediate* (I) and *weighted intermediate* (W) are used to refer to using *equal* weights or any other weights in general, respectively.

In practical terms, using individuals represented by :math:`x \in \mathbb{R}^n`,
it means that a weighted sum of the parents is used to define each of the :math:`x_i` coordinates of the new individual :cite:`2016:cma-es-tutorial`.

**Notation**

The differences explained above are summarized by prefixing with :math:`(\mu / \rho_{\{I,W\}} \overset{+}{,} \lambda)`. 
In the context of CMA-ES, since :math:`\rho = \mu`, it is restricted to :math:`(\mu / \mu_{\{I,W\}} \overset{+}{,} \lambda)`,
which can be shortened to :math:`(\mu_{\{I,W\}} \overset{+}{,} \lambda)` :cite:`2001:cdsa-es` [#f3]_.

**Implementation**

The current implementation only supports non-elitist selection and any weighted recombination of :math:`\mu` weights. 

Some examples are shown below. All parameters that are not provided with a custom value are initialized according to :cite:`2016:cma-es-tutorial`.

.. code-block:: python

  from anguilla.optimizers.cma import StrategyParameters, RecombinationType

  # Defines mu, weights and other parameters automatically
  # Uses RecombinationType.SUPERLINEAR as default
  parameters = StrategyParameters(population_size=5)

  # Defines mu, weights, and other parameters automatically
  parameters = StrategyParameters(population_size=5, recombination_type=RecombinationType.EQUAL)

  # Defines population_size, mu, and other parameters automatically
  parameters = StrategyParameters(weights=np.array([...]))

See :py:mod:`anguilla.optimizers.cma.StrategyParameters` and :py:mod:`anguilla.optimizers.cma.RecombinationType` [#f1]_.

Original vs. Hybrid
*******************

**Background**

When introduced by :cite:`2001:cdsa-es`, only the rank-one update was used in the covariance matrix adaptation. This is known as the *original* CMA-ES.
Later, the rank-:math:`\mu` update was introduced. This variant is called *hybrid* CMA-ES :cite:`2006:active-cma-es`.
The parameters :math:`c_1` and :math:`c_\mu` are used as weights (:math:`c_1 + c_\mu = 1`) to define how much each update contributes to the adaptation.

**Notation**

The variants are denoted as *Original-CMA-ES* and *Hybrid-CMA-ES*. 
Currently, *CMA-ES* implicitly denotes the hybrid approach.

**Implementation**

Pending.

See :py:mod:`anguilla.optimizers.cma.StrategyParameters` [#f1]_.

Passive vs. Active
******************

**Background**

Pending.

**Notation**

Pending.

**Implementation**

See :py:mod:`anguilla.optimizers.cma.StrategyParameters` [#f1]_.

Single-objective vs. Multi-objective
************************************

**Background**

Pending.

**Notation**

Variants that work with multi-objective functions are prefixed with *MO*.
Furthermore, the current state of the art is a variant with unbounded population, denoted as *U-MO-CMA-ES* :cite:`2016:mo-cma-es`.

**Implementation**

A basic [#f2]_ implementation for single-objective CMA-ES is implemented as presented in :cite:`2016:cma-es-tutorial` and :cite:`2013:oo-optimizers`, following the reference implementations from :cite:`2008:shark` and :cite:`2019:pycma`.

An implementation for MO-CMA-ES is the main objective of this project, as presented in  :cite:`2007:mo-cma-es`, :cite:`2010:mo-cma-es` and
:cite:`2016:mo-cma-es` following the reference implementation from :cite:`2008:shark`. Currently, a work in progress.

See :py:mod:`anguilla.optimizers.cma.CMA` [#f1]_.

.. [#f1] Subject to more comprehensive testing of the implementation and variations on the provided interfaces.

.. [#f2] For a full-featured implementation of CMA-ES, `check <https://github.com/CMA-ES/pycma>`_ PyCMA :cite:`2019:pycma`.

.. [#f3] The most recent notation incorporates the concept of age which is not mentioned here, please refer to :cite:`2015:es-overview`. 

Detailed examples
-----------------

Pending
