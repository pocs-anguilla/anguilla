.. _notes:

**********
Some notes
**********

.. Note::
    In this page we are keeping some notes about the theory
    to help us order ideas as we develop the MO-CMA-ES implementations.

Evolution strategies
####################

An evolution strategy (ES) optimizes an objective function by evolving a population of individuals (candidate solutions) by means of genetic operators.
These algorithms are iterative, an at each generation (iteration) the population is evolved.
A run of the algorithm is called an evolution run :cite:`2002:es-comprehensive-intro`.

Search and solution spaces
##########################

The goal of these algorithms is to optimize an objective function :math:`f : X \mapsto Y`
where :math:`X` is called the search space and :math:`Y` the solution space.

+--------------+----------------------+----------------------+-------------------------------+
| Variant      | Search space         | Solution space       | Application                   |
+==============+======================+======================+===============================+
| CMA-ES       | :math:`\mathbb{R}^n` | :math:`\mathbb{R}`   | Real-valued                   |
|              |                      |                      | single-objective optimization |
+--------------+----------------------+----------------------+-------------------------------+
| MO-CMA-ES    | :math:`\mathbb{R}^n` | :math:`\mathbb{R}^m` | Real-valued                   |
|              |                      |                      | multi-objective optimization  |
+--------------+----------------------+----------------------+-------------------------------+
| UP-MO-CMA-ES | :math:`\mathbb{R}^n` | :math:`\mathbb{R}^2` | Real-valued                   |
|              |                      |                      | bi-objective optimization     |
+--------------+----------------------+----------------------+-------------------------------+

When the solution space is multidimensional, the notation :math:`f: x \mapsto (f_1(x), \cdots, f_m(x))` is preferred to explicitly denote
the multiple objective functions, where :math:`f_{1 \cdots m} : \mathbb{R}^n \mapsto \mathbb{R}`.

Individuals
###########

Each individual (also referred to as candidate solution) is modeled as :math:`a_k = [x_k, s_k, f(x_k)]`,
where :math:`k` is an index identifying the individual inside the population :cite:`2002:es-comprehensive-intro`.
The table below explains the makeup of an individual.

+--------------------------------------+----------------+
| Name                                 | Symbol         |
+======================================+================+
| Search point (or                     | :math:`x_k`    |
| object parameter set)                |                |
+--------------------------------------+----------------+
| Internal (or evolvable / endogenous) | :math:`s_k`    |
| strategy parameters                  |                |
+--------------------------------------+----------------+
| Fitness (objective function value)   | :math:`f(x_k)` |
+--------------------------------------+----------------+

Genetic operators
#################

Genetic operators can affect the search point or the internal parameters associated with an individual.

The operators consist of mutation, selection and recombination.
Selection can include mating selection and environmental selection.

[To be continued]

Fitness
#######

Some objective functions can have box constraints, for example. The mutation operator can produce search points outside their feasible area.
In order to account for this, the penalized fitness of the search point can be used as described in :cite:`2007:mo-cma-es` and shown below:

:math:`f_m^\text{penalty}(x) = f_m(feasible(x)) + \alpha || x - feasible(x) ||_2^2`

Where :math:`\alpha > 0` is the penalty parameter and :math:`feasible(x)` denotes the closest feasible point to :math:`x`.

The unpenalized fitness uses the same definition but with :math:`\alpha = 0`.

Other types of constraints would need to be handled in a similar way.

Furthermore, some objective functions can be noisy. Their fitness is estimated using the average of multiple evaluations :cite:`2008:shark`.

Ranking
#######

The selection operator requires an ordering of the individuals.
When working with single objective functions this corresponds to the ordering of their fitness.
With multi-objective functions a two-level ordering is used. First, individuals are ordered through non-dominated sorting.
Second, individuals with the same first level order are sorted according to their hypervolume contribution :cite:`2007:mo-cma-es`.

[To be continued]

Self-adaptation
###############
Adaptation refers to the control (tuning) of the internal parameters with the goal of 
increasing the likelihood of successful steps (e.g. increased success rate and convergence rate).
When the adaptation uses genetic operators it is called self-adaptation :cite:`2015:es-overview` :cite:`2002:es-comprehensive-intro`.

If internal parameters are not controlled (e.g. using a constant mutation strength) the algorithm performs similar to a random search :cite:`2015:es-overview`.
Early evolution strategies introduced rules to control the step size based on heuristics.
The most famous being the :math:`1/5`-rule, which increases or decreases the step size depending on the success rate (e.g. how many offspring are better than their parents [#f3]_).

[To be continued]


Internal strategy parameters
############################
Some internal parameters are used to adapt other internal parameters and others affect the statistical properties of genetic operators (particularly the mutation operator) :cite:`2002:es-comprehensive-intro`.
Also, some are initialized with the value of an external parameter (see next section).

:math:`(1+\lambda)`-CMA-ES, MO-CMA-ES, UP-MO-CMA-ES
---------------------------------------------------

The table below [#f1]_ collects the internal parameters for the variants featured in :cite:`2007:mo-cma-es`, :cite:`2010:mo-cma-es`, :cite:`2016:mo-cma-es`.

+-------------------------------+-----------------------------+---------------------------------+-------------------------------------+------------------------------+
| Parameter                     | Symbol                      | Domain                          | Initial value                       | Affects                      |
+===============================+=============================+=================================+=====================================+==============================+
| Step size (mutation strength) | :math:`\sigma`              | :math:`\mathbb{R}_+`            | :math:`\sigma_0`                    | Mutation                     |
+-------------------------------+-----------------------------+---------------------------------+-------------------------------------+------------------------------+
| Smoothed success probability  | :math:`\bar{p}_\text{succ}` | :math:`[0,1]`                   | :math:`p_\text{succ}^\text{target}` | Adaptation of :math:`\sigma` |
+-------------------------------+-----------------------------+---------------------------------+-------------------------------------+------------------------------+
| Covariance matrix             | :math:`C`                   | :math:`\mathbb{R}^{n \times n}` | :math:`I`                           | Mutation                     |
+-------------------------------+-----------------------------+---------------------------------+-------------------------------------+------------------------------+
| Evolution path                | :math:`p_c`                 | :math:`\mathbb{R}^n`            | :math:`0`                           | Adaptation of :math:`C`      |
+-------------------------------+-----------------------------+---------------------------------+-------------------------------------+------------------------------+

:math:`(\mu, \lambda)`-CMA-ES
-----------------------------

The table below [#f1]_ collects the internal parameters for the variant shown in :cite:`2016:cma-es-tutorial`.

+-------------------------------+------------------+---------------------------------+------------------+------------------------------+
| Parameter                     | Symbol           | Domain                          | Initial value    | Affects                      |
+===============================+==================+=================================+==================+==============================+
| Parental mean                 | :math:`m`        | :math:`\mathbb{R}^n`            | :math:`x_0`      | Mutation                     |
+-------------------------------+------------------+---------------------------------+------------------+------------------------------+
| Step size (mutation strength) | :math:`\sigma`   | :math:`\mathbb{R}_+`            | :math:`\sigma_0` | Mutation                     |
+-------------------------------+------------------+---------------------------------+------------------+------------------------------+
| Evolution path                | :math:`p_\sigma` | :math:`\mathbb{R}^n`            | :math:`0`        | Adaptation of :math:`\sigma` |
+-------------------------------+------------------+---------------------------------+------------------+------------------------------+
| Covariance matrix             | :math:`C`        | :math:`\mathbb{R}^{n \times n}` | :math:`I`        | Mutation                     |
+-------------------------------+------------------+---------------------------------+------------------+------------------------------+
| Evolution path                | :math:`p_c`      | :math:`\mathbb{R}^n`            | :math:`0`        | Adaptation of :math:`C`      |
+-------------------------------+------------------+---------------------------------+------------------+------------------------------+

External strategy parameters
############################

An evolution strategy has external (or exogenous) parameters which remain constant during the execution of the algorithm :cite:`2002:es-comprehensive-intro`.
The external parameters affect genetic operators and the adaptation of internal parameters.

The default values found in the literature were tuned (through experimentation) to sensible defaults so that they work well out-of-the-box with a group of functions with certain characteristics.
Other types of functions might require specific tuning.

The recommended default values can be expressed in terms of other external parameters (e.g. the population size) and/or meta-parameters (e.g. the dimensionality of the search space).

:math:`(1+\lambda)`-CMA-ES, MO-CMA-ES, UP-MO-CMA-ES
---------------------------------------------------

The table below [#f1]_ summarizes external parameters for the variants featured in :cite:`2007:mo-cma-es`, :cite:`2010:mo-cma-es`, :cite:`2016:mo-cma-es`.

+-----------------------------------------------+-------------------------------------+-----------------------------+-------------------------------------------+---------------------------------------------------------------------------------------------+
|                      Name                     |                Symbol               |            Domain           |                  Affects                  |                                        Default value                                        |
|                                               |                                     |                             |                                           +------------------------------+---------------------+----------------------------------------+
|                                               |                                     |                             |                                           |  :math:`(1+\lambda)`-CMA-ES  |      MO-CMA-ES      |              UP-MO-CMA-ES              |
+===============================================+=====================================+=============================+===========================================+==============================+=====================+========================================+
|               Number of parents               |      :math:`\lambda_\text{MO}`      | :math:`\lambda_{MO} \geq 1` |                 Selection                 |        Not applicable        |  Problem dependent  |  Not applicable (unbounded population) |
+-----------------------------------------------+-------------------------------------+-----------------------------+                                           +------------------------------+---------------------+----------------------------------------+
|        Number of offspring per parent         |           :math:`\lambda`           |    :math:`\lambda \geq 1`   |                                           |                                           1 [#f2]_                                          |
+-----------------------------------------------+-------------------------------------+-----------------------------+-------------------------------------------+---------------------------------------------------------------------------------------------+
|               Initial step size               |           :math:`\sigma_0`          |     :math:`\mathbb{R}_+`    |      Initialization of :math:`\sigma`     |                                      Problem dependent                                      |
+-----------------------------------------------+-------------------------------------+-----------------------------+-------------------------------------------+---------------------------------------------------------------------------------------------+
|               Step size damping               |              :math:`d`              |       :math:`d \geq 1`      |        Adaptation of :math:`\sigma`       |                             :math:`d = 1 + \frac{n}{2 \lambda}`                             |
+-----------------------------------------------+-------------------------------------+-----------------------------+-------------------------------------------+---------------------------------------------------------------------------------------------+
|           Target success probability          | :math:`p_\text{succ}^\text{target}` |        :math:`[0,1]`        |        Adaptation of :math:`\sigma`       |                            :math:`\frac{1}{5 + \sqrt{\lambda}/2}`                           |
+-----------------------------------------------+-------------------------------------+-----------------------------+-------------------------------------------+---------------------------------------------------------------------------------------------+
|        Success rate averaging parameter       |             :math:`c_p`             |        :math:`[0,1]`        | Adaptation of :math:`\bar{p}_\text{succ}` | :math:`\frac{p_\text{succ}^\text{target} \lambda}{2 + p_\text{succ}^\text{target} \lambda}` |
+-----------------------------------------------+-------------------------------------+-----------------------------+-------------------------------------------+---------------------------------------------------------------------------------------------+
|        Smoothed success rate threshold        |       :math:`p_\text{thresh}`       |        :math:`[0,1]`        |          Adaptation of :math:`C`          |                                             0.44                                            |
+-----------------------------------------------+-------------------------------------+-----------------------------+-------------------------------------------+----------------------------------------------------+----------------------------------------+
|          Evolution path learning rate         |             :math:`c_c`             |        :math:`[0,1]`        |         Adaptation of :math:`p_c`         |                :math:`\frac{2}{n+2}`               |             Not applicable             |
+-----------------------------------------------+-------------------------------------+-----------------------------+-------------------------------------------+----------------------------------------------------+----------------------------------------+
|        Covariance matrix learning rate        |         :math:`c_\text{cov}`        |        :math:`[0,1]`        |          Adaptation of :math:`C`          |              :math:`\frac{2}{n^2 + 6}`             |      :math:`\frac{2}{n^{2.1} + 3}`     |
+-----------------------------------------------+-------------------------------------+-----------------------------+-------------------------------------------+----------------------------------------------------+----------------------------------------+
|           Extreme point probability           |       :math:`p_\text{extreme}`      |        :math:`[0,1]`        |                 Selection                 |                   Not applicable                   |               :math:`1/5`              |
+-----------------------------------------------+-------------------------------------+-----------------------------+                                           |                                                    +----------------------------------------+
|      Selected point convergence threshold     |      :math:`\sigma_\text{min}`      |     :math:`\mathbb{R}_+`    |                                           |                                                    |            :math:`10^{-15}`            |
+-----------------------------------------------+-------------------------------------+-----------------------------+                                           |                                                    +----------------------------------------+
|       Interior point probability weight       |            :math:`\alpha`           |     :math:`\mathbb{R}_+`    |                                           |                                                    |                :math:`3`               |
+-----------------------------------------------+-------------------------------------+-----------------------------+-------------------------------------------+                                                    +----------------------------------------+
| Covariance matrix recombination learning rate |             :math:`c_r`             |        :math:`[0,1]`        |          Adaptation of :math:`C`          |                                                    |         :math:`c_\text{cov}/2`         |
+-----------------------------------------------+-------------------------------------+-----------------------------+-------------------------------------------+----------------------------------------------------+----------------------------------------+

:math:`(\mu, \lambda)`-CMA-ES
-----------------------------

The table below [#f1]_ collects the external parameters for the variant shown in :cite:`2016:cma-es-tutorial`.

+---------------------------------------+--------------------------------+---------------------------------+------------------------------------------------------------------------------------------------------------------------+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| Parameter                             | Symbol                         | Domain                          | Additional constraints                                                                                                 | Affects                          | Default value                                                                                                                                       |
+=======================================+================================+=================================+========================================================================================================================+==================================+=====================================================================================================================================================+
| Number of parents                     | :math:`\mu`                    | :math:`1 \leq \mu \leq \lambda` |                                                                                                                        | Selection                        | :math:`|\{ w_i > 0 \}| = \lfloor \lambda / 2 \rfloor`                                                                                               |
+---------------------------------------+--------------------------------+---------------------------------+------------------------------------------------------------------------------------------------------------------------+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| Number of offspring                   | :math:`\lambda`                | :math:`\lambda \geq 2`          |                                                                                                                        | Selection                        | :math:`4 + \lfloor 3 \ln n \rfloor`                                                                                                                 |
+---------------------------------------+--------------------------------+---------------------------------+------------------------------------------------------------------------------------------------------------------------+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| Parent mixing number                  | :math:`\rho`                   | :math:`1 \leq \rho \leq \mu`    | :math:`\rho = \mu`                                                                                                     | Recombination                    | :math:`\mu`                                                                                                                                         |
+---------------------------------------+--------------------------------+---------------------------------+------------------------------------------------------------------------------------------------------------------------+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| Initial step size                     | :math:`\sigma_0`               | :math:`\mathbb{R}_+`            |                                                                                                                        | Initialization of :math:`\sigma` | Problem dependent                                                                                                                                   |
+---------------------------------------+--------------------------------+---------------------------------+------------------------------------------------------------------------------------------------------------------------+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| Step size damping                     | :math:`d_\sigma`               | :math:`d \geq 1`                |                                                                                                                        | Adaptation of :math:`\sigma`     | :math:`1 + 2 \max \left( 0, \sqrt{\frac{\mu_\text{eff}-1}{n+1}} - 1 \right) + c_\sigma`                                                             |
+---------------------------------------+--------------------------------+---------------------------------+------------------------------------------------------------------------------------------------------------------------+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| Evolution path learning rate          | :math:`c_\sigma`               | :math:`[0,1]`                   |                                                                                                                        | Adaptation of :math:`\sigma`     | :math:`\frac{\mu_\text{eff} + 2}{n + \mu_\text{eff} + 5}`                                                                                           |
+---------------------------------------+--------------------------------+---------------------------------+------------------------------------------------------------------------------------------------------------------------+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| Recombination weights [#f4]_          | :math:`w_1, \cdots, w_\lambda` | :math:`|w_i| \in [0,1]`         | :math:`w_1 \geq \cdots \geq  w_\lambda`, :math:`w_1, \cdots, w_{\mu-1} > 0`, :math:`w_\mu, \cdots, w_{\lambda} \leq 0` | Recombination                    | See Eq. (53)                                                                                                                                        |
+---------------------------------------+--------------------------------+---------------------------------+------------------------------------------------------------------------------------------------------------------------+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| Parental mean learning rate           | :math:`c_m`                    | :math:`[0,1]`                   |                                                                                                                        | Adaptation of :math:`m`          | 1                                                                                                                                                   |
+---------------------------------------+--------------------------------+---------------------------------+------------------------------------------------------------------------------------------------------------------------+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| Evolution path learning rate          | :math:`c_c`                    | :math:`[0,1]`                   |                                                                                                                        | Adaptation of :math:`C`          | :math:`\frac{4 + \mu_\text{eff}/n}{n + 4 + 2 \mu_\text{eff}/n}`                                                                                     |
+---------------------------------------+--------------------------------+---------------------------------+------------------------------------------------------------------------------------------------------------------------+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| Rank-one update learning rate         | :math:`c_1`                    | :math:`[0,1]`                   | :math:`c_1 \leq 1 - c_\mu`                                                                                             | Adaptation of :math:`C`          | :math:`\frac{\alpha_\text{cov}}{(n+1.3)^2 + \mu_\text{eff}}`                                                                                        |
+---------------------------------------+--------------------------------+---------------------------------+------------------------------------------------------------------------------------------------------------------------+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| Rank-:math:`\mu` learning rate [#f5]_ | :math:`c_\mu`                  | :math:`[0,1]`                   | :math:`c_\mu \leq 1 - c_1`                                                                                             | Adaptation of :math:`C`          | :math:`\min \left( 1 - c_1, \alpha_\text{cov} \frac{\mu_\text{eff} - 2 + 1 / \mu_\text{eff}}{(n+2)^2 + \alpha_\text{cov} \mu_\text{eff}/2} \right)` |
+---------------------------------------+--------------------------------+---------------------------------+------------------------------------------------------------------------------------------------------------------------+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+

.. [#f1] The usage of adaptation in the tables is generally used to refer to control or tuning and does not exclude self-adapation.
.. [#f2] When :math:`\lambda = 1` it is known as steady-state :cite:`2015:es-overview`.
.. [#f3] There are two defined notions of success: population-based or individual-based. Please see :cite:`2010:mo-cma-es`.
.. [#f4] When :math:`w_\mu, \cdots, w_\lambda = 0` this corresponds to Passive-CMA-ES and when :math:`w_\mu, \cdots, w_\lambda < 0` to Active-CMA-ES. Please see :cite:`2006:active-cma-es`.
.. [#f5] When :math:`c_\mu = 0` this corresponds to Original-CMA-ES as introduced in :cite:`2001:cdsa-es` (i.e. no rank-:math:`\mu` update) and otherwise it is known as Hybrid-CMA-ES :cite:`2006:active-cma-es`. Currently, CMA-ES implicitly denotes Hybrid-CMA-ES.
