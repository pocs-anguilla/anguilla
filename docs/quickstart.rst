.. _quickstart:

Quickstart Guide
================

A brief overview
----------------

This section provides a short introduction to variants of CMA-ES (non-exhaustive), 
informs which are currently implemented by this project and provides pointers for more information.

.. list-table:: Summary of CMA-ES variants
    :widths: 5 5 5 5 5 
    :header-rows: 1

    * - Variant
      - Synonyms or related notation
      - Implemented
      - Application
      - References
    * - CMA-ES
      - Original-CMA-ES, Non-elistist CMA-ES, :math:`(\mu, \sigma)`-CMA-ES
      - Yes [#f1]_ [#f2]_
      - Single-objective
      - :cite:`2016:cma-es-tutorial`
    * - Active-CMA-ES
      - Non-elistist Active-CMA-ES, :math:`(\mu, \sigma)`-Active-CMA-ES, aCMA-ES
      - WIP
      - Single-objective
      - :cite:`2006:active-cma-es` :cite:`2016:cma-es-tutorial`
    * - Elitist CMA-ES
      - :math:`(\mu + \sigma)`-CMA-ES
      - WIP
      - Single-objective
      - :cite:`2007:mo-cma-es`
    * - MO-CMA-ES
      -  :math:`(\mu + \sigma)`-MO-CMA-ES
      - WIP
      - Multi-objective
      - :cite:`2007:mo-cma-es` :cite:`2010:mo-cma-es`
    * - U-MO-CMA-ES
      - Unbounded population MO-CMA-ES (state-of-the-art)
      - WIP
      - Multi-objective
      - :cite:`2016:mo-cma-es`

.. [#f1] Subject to more comprehensive testing of the implementation and variations on the provided interfaces.

.. [#f2] For a full-featured implementation of CMA-ES, `check <https://github.com/CMA-ES/pycma>`_ PyCMA :cite:`2019:pycma`.

Detailed examples
-----------------

Non-elistist CMA-ES
********************

Pending.

Non-elistist Active-CMA-ES
**************************

Pending.

MO-CMA-ES
**********

Pending.

