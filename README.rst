========
Anguilla
========
|conda| |wheels| |docs| |coveralls| |deepsource| |deepcode| |sonarcloud|

.. |coveralls| image:: https://coveralls.io/repos/github/pocs-anguilla/anguilla/badge.svg?branch=develop
               :target: https://coveralls.io/github/pocs-anguilla/anguilla?branch=develop
               :alt: Coverage Status

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

For more details, please see the `documentation <https://anguilla.readthedocs.io/en/latest/>`_.

Install
=======

PyPI
----

Currently available in PyPI testing index:

.. code-block:: bash

   pip install -i https://test.pypi.org/simple/ anguilla

Conda
-----

Only temporarily in Anaconda.org (will move to Conda-Forge):

.. code-block:: bash

   conda install -c Anguilla anguilla

References
==========

See `REFERENCES <REFERENCES>`_.

License
=======

See `LICENSE <LICENSE>`_.
