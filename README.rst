========
Anguilla
========
|conda| |wheels| |docs| |deepsource| |deepcode| |sonarcloud|

.. |conda| image:: https://github.com/pocs-anguilla/anguilla/workflows/Conda/badge.svg?branch=develop
           :target: https://github.com/pocs-anguilla/anguilla
           :alt: Conda build

.. |wheels| image:: https://github.com/pocs-anguilla/anguilla/workflows/Wheels/badge.svg?branch=develop
           :target: https://github.com/pocs-anguilla/anguilla
           :alt: Wheels build

.. |docs| image:: https://readthedocs.org/projects/anguilla/badge/?version=latest
          :target: https://anguilla.readthedocs.io/en/latest/?badge=latest
          :alt: Documentation status

.. |deepsource| image:: https://deepsource.io/gh/pocs-anguilla/anguilla.svg/?label=active+issues&show_trend=true&token=CZElZ2ZetdLdyxuEWD6Y7NYo
                :target: https://deepsource.io/gh/pocs-anguilla/anguilla/?ref=repository-badge
                :alt: Static analysis status (DeepSource)
.. |deepcode|   image:: https://www.deepcode.ai/api/gh/badge?key=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJwbGF0Zm9ybTEiOiJnaCIsIm93bmVyMSI6InBvY3MtYW5ndWlsbGEiLCJyZXBvMSI6ImFuZ3VpbGxhIiwiaW5jbHVkZUxpbnQiOmZhbHNlLCJhdXRob3JJZCI6MjUzNDIsImlhdCI6MTYwNjQwMjExN30.PAYMuKXLpi3tBoJQufB62gBHtODZ7HZrhFpnJ1lcmu8
                :target: https://www.deepcode.ai/app/gh/pocs-anguilla/anguilla/_/dashboard?utm_content=gh%2Fpocs-anguilla%2Fanguilla
                :alt: Static analysis status (DeepCode)
.. |sonarcloud| image:: https://sonarcloud.io/images/project_badges/sonarcloud-black.svg
                :height: 20
                :target: https://sonarcloud.io/dashboard?id=pocs-anguilla_anguilla
                :alt: Static analysis badge (SonarCloud)

Anguilla is a Python 3 implementation of (UP-)MO-CMA-ES.
It is based on the reference implementations from 
`The Shark Machine Learning Library <https://www.shark-ml.org/>`_ and
`PyCMA <https://github.com/CMA-ES/pycma>`_.

For more details, please see the `documentation <https://anguilla.readthedocs.io/en/latest/>`_
and the following example notebooks:

* See `MO-CMA-ES Walk-through or <notebooks/experiments/mocma/walkthrough.ipynb>`_ |mo-cma-es-colab|
* See `UP-MO-CMA-ES Walk-through or <notebooks/experiments/upmocma/walkthrough.ipynb>`_ |up-mo-cma-es-colab|

.. |mo-cma-es-colab| image:: https://colab.research.google.com/assets/colab-badge.svg
                     :target: https://colab.research.google.com/github/pocs-anguilla/anguilla/blob/develop/notebooks/experiments/mocma/walkthrough.ipynb
                     :alt: Open In Colab

.. |up-mo-cma-es-colab| image:: https://colab.research.google.com/assets/colab-badge.svg
                        :target: https://colab.research.google.com/github/pocs-anguilla/anguilla/blob/develop/notebooks/experiments/upmocma/walkthrough.ipynb
                        :alt: Open In Colab

Install
=======

PyPI
----

.. code-block:: console
   pip install -i https://test.pypi.org/simple/ anguilla

Conda
-----

Temporarily only (will move to Conda-Forge):

.. code-block:: console
   conda install -c anguilla anguilla

References
==========

See `REFERENCES <REFERENCES>`_.

License
=======

See `LICENSE <LICENSE>`_.
