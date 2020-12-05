.. _references:

**********
References
**********

This page lists all the references cited and consulted for this project.

Publications
*************

.. bibliography:: references.bib

Other
*****

* `Shark Pull Request 235 <https://github.com/Shark-ML/Shark/pull/235>`_:
  in this PR, users egomeh and Ulfgard show how to initialize the negative
  weights for active CMA for linear and equal recombination types.
  We implement the same initialization.
* `PyCMA Pull Request 154 <https://github.com/CMA-ES/pycma/pull/154>`_: in
  this PR, user ARF1 suggests the use of Python data classes for implementing
  the stopping conditions results structure. We follow this idea of using
  data classes to implement the strategy parameters structure as well.
* `M. Moitzi's Binary Tree Package <https://github.com/mozman/bintrees>`_:
  from this codebase, we took the ABCTree's `floor_item` method and adapted
  it, to have a more concise implementation of the `lower_bound_by_key`
  method used in the RBTree class.
