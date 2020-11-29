"""This module contains implementations related to the CMA-ES algorithm
for single-objective real-valued optimization."""
from __future__ import annotations

import math
import typing
import enum
import dataclasses
import numpy as np

import anguilla.util
from anguilla.optimizers.base import AbstractOptimizer


class RecombinationType(enum.Enum):
    r"""Recombination types available to initialize the weights.

    Notes
    -----
    .. table:: Summary of recombination types

        +--------------+--------------------------+-------------+
        | Type         | Symbol                   | Use         |
        |              |                          |             |
        +--------------+--------------------------+-------------+
        | Weighted     | :math:`\mu_W`            | SUPERLINEAR |
        | intermediate | or :math:`\mu / \mu_W`   +-------------+
        |              |                          | LINEAR      |
        +--------------+--------------------------+-------------+
        | Intermediate | :math:`\mu_I`            | EQUAL       |
        |              | or :math:`\mu / \mu_I`   |             |
        +--------------+--------------------------+-------------+
    """

    SUPERLINEAR = 0
    LINEAR = 1
    EQUAL = 2


@dataclasses.dataclass
class StrategyParameters:
    """Define the external strategy parameters for :class:`CMA`.

    Parameters
    ----------
    n : optional
        Dimensionality of the search space \
            (i.e. the number of variables of a solution).
    population_size : optional
        The population size.
        Overrided if ``weights`` is provided.
    mu : optional
        The parent population size.
        Overrided if ``weights`` is provided.
    weights : optional
        The recombination weights.
    c_m : optional
        The learning rate used for the mean update.
    c_sigma : optional
        The learning rate used for the step size update.
    d_sigma : optional
        The step size damping parameter to handle selection noise.
    c_c : optional
        The learning rate for the evolution path update.
    c_1 : optional
        The learning rate for the rank-one update used for updating \
            the covariance matrix.
        Can be a callable that takes ``mu_eff`` as argument.
    c_mu : optional
        The learning rate for the rank-mu update used for updating \
            the covariance matrix.
        Can be a callable that takes ``mu_eff`` as argument.
    recombination_type : optional
        How to initialize ``weights`` if not provided.
    alpha_cov : optional
        Used to initialize ``c_1`` and ``c_mu`` if not provided.
    active : optional
        Initialize parameters (that were not provided) to use
        active CMA.

    Attributes
    ----------
    mu_eff
        The variance effective number of the positive weights.

    Raises
    ------
    ValueError
        Value provided for a parameter is invalid.

    Notes
    -----
    All optional parameters from 1 to 9 are initialized according to the \
        default values from :cite:`2016:cma-es-tutorial` if not provided. \
        The equations referenced in the following table can be found there.

    .. list-table:: Parameter summary
        :widths: 2 6 2 15 16
        :header-rows: 1

        * -
          - Parameter
          - Symbol
          - Validation
          - Default
        * - 1
          - n
          - :math:`n`
          - :math:`n \\geq 1`
          -
        * - 2
          - population_size
          - :math:`\\lambda`
          - :math:`\\lambda \\geq 2`
          - :math:`\\begin{cases} |w| & \\text{weights provided} \\\\ \
                                  4 + \\lfloor 3 \\log n \\rfloor \
                                  & \\text{otherwise} \
                   \\end{cases}`
        * - 3
          - mu
          - :math:`\\mu`
          - :math:`\\mu \\leq \\lambda`
          - :math:`\\begin{cases} |\\{ w_i : w_i > 0 \\}| \
                                  & \\text{weights provided} \\\\ \
                                  \\lfloor \\lambda / 4 \\rfloor & \
                                  \\text{equal recombination type} \\\\ \
                                  \\lfloor \\lambda / 2 \\rfloor & \
                                  \\text{otherwise} \
                    \\end{cases}`
        * - 4
          - weights
          - :math:`w_{1,2 \\cdots, \\lambda}`
          - :math:`w_1 \\geq \\cdots \\geq w_\\lambda > 0`, :math:`w_1 > 0`
          - Eq. (53)
        * - 5
          - c_m
          - :math:`c_m`
          - :math:`c_m \\leq 1`
          - Eq. (54)
        * - 6
          - c_sigma
          - :math:`c_\\sigma`
          - :math:`c_\\sigma \\leq 1`
          - Eq. (55)
        * - 7
          - d_sigma
          - :math:`d_\\sigma`
          - :math:`d_\\sigma \\approx 1`
          - Eq. (55)
        * - 8
          - c_c
          - :math:`c_c`
          - :math:`c_c \\leq 1`
          - Eq. (56)
        * - 9
          - c_1
          - :math:`c_1`
          - :math:`c_1 \\leq 1 - c_\\mu`
          - Eq. (57)
        * - 10
          - c_mu
          - :math:`c_\\mu`
          - :math:`c_\\mu \\leq 1 - c_1`
          - Eq. (58)

    Following the recommendations from p. 32 of \
    :cite:`2016:cma-es-tutorial`, for most problems prefer the default \
    values for parameters 4 to 10; and don't set values for parameter \
    2 smaller than its default would be.
    """

    n: int
    population_size: typing.Optional[int] = None
    mu: typing.Optional[int] = None
    weights: typing.Optional[np.ndarray] = None
    c_sigma: typing.Optional[float] = None
    d_sigma: typing.Optional[float] = None
    c_c: typing.Optional[float] = None
    c_1: typing.Union[float, typing.Callable[[float], float], None] = None
    c_mu: typing.Union[float, typing.Callable[[float], float], None] = None
    c_m: float = 1.0
    mu_eff: float = dataclasses.field(init=False)
    recombination_type: dataclasses.InitVar[
        RecombinationType
    ] = RecombinationType.SUPERLINEAR
    alpha_cov: dataclasses.InitVar[float] = 2.0
    # TODO: Make this True by default
    active: dataclasses.InitVar[bool] = False

    def __post_init__(
        self,
        recombination_type: RecombinationType,
        alpha_cov: float,
        active: bool,
    ) -> None:
        """Initialize and validate parameters."""
        self._post_init_population_size()

        self._post_init_weights(recombination_type)

        self._post_init_mu_eff()

        self._post_init_c_1_mu(alpha_cov)

        self._post_init_scale_weights(active)

        if self.c_m > 1.0:
            raise ValueError("Invalid value for c_m")

    def _post_init_population_size(self) -> None:
        """Initialize and/or validate population size."""
        if self.n < 1:
            raise ValueError("Invalid value for n")

        if self.population_size is None:
            # Eq. (48) [2016:cma-es-tutorial]
            # Eq. (44) [2011:cma-es-tutorial] (same)
            lambda_ = 4 + math.floor(3 * math.log(self.n))
            # Heuristic for small search spaces from [2008:shark]
            lambda_ = max(5, min(lambda_, self.n))
            self.population_size = lambda_
        elif self.population_size < 2:
            raise ValueError("Invalid value for population_size")

    def _post_init_weights(
        self, recombination_type: RecombinationType
    ) -> None:
        """Initialize and/or validate weights."""
        # No custom weights were provided
        if self.weights is None:
            # Implementation allows a custom mu or uses a default
            # depending on the provided value for recombination_type
            if self.mu is not None and self.mu > self.population_size:
                raise ValueError("Invalid value for mu")
            # The following uses the same values as in [2008:shark]
            if recombination_type == RecombinationType.SUPERLINEAR:
                if self.mu is None:
                    self.mu = self.population_size // 2
                # Same as in Eq. (49) [2016:cma-es-tutorial]
                self.weights = math.log(
                    (self.population_size + 1.0) / 2.0
                ) - np.log1p(np.arange(self.population_size))
            elif recombination_type == RecombinationType.LINEAR:
                if self.mu is None:
                    self.mu = self.population_size // 2
                self.weights = np.repeat(self.mu, self.population_size)
                self.weights[: self.mu] -= np.arange(self.mu)
                self.weights[self.mu :] -= np.arange(
                    self.population_size - 1, self.mu, -1.0
                )
            elif recombination_type == RecombinationType.EQUAL:
                if self.mu is None:
                    self.mu = self.population_size // 4
                self.weights = np.ones(self.population_size)
                self.weights[self.mu :] *= -1.0
        # Custom weights were provided
        # If population_size and mu are provided they will be overrided
        else:
            if self.weights[0] < 0.0 or np.any(
                self.weights[1:] > self.weights[:-1]
            ):
                raise ValueError("Invalid value for weights")
            self.population_size = self.weights.shape[0]
            self.mu = np.sum(self.weights > 0.0)

    def _post_init_mu_eff(self) -> None:
        """Initialize mu_eff, mu_eff_neg, c_sigma, d_sigma and c_c."""
        # Page 31 [2016:cma-es-tutorial]
        # Page 27 [2011:cma-es-tutorial] (deprecated)
        self.mu_eff = np.sum(self.weights[: self.mu]) ** 2 / np.sum(
            self.weights[: self.mu] * self.weights[: self.mu]
        )

        # Page 31 [2016:cma-es-tutorial] (new)
        self.mu_eff_neg = np.sum(self.weights[self.mu :]) ** 2 / np.sum(
            self.weights[self.mu :] * self.weights[self.mu :]
        )

        if self.c_sigma is None:
            # Eq. (55) [2016:cma-es-tutorial]
            # Eq. (56) [2011:cma-es-tutorial] (same)
            self.c_sigma = (self.mu_eff + 2.0) / (self.n + self.mu_eff + 5.0)
        elif self.c_sigma > 1.0:
            raise ValueError("Invalid value for c_sigma")

        if self.d_sigma is None:
            # Eq. (55) [2016:cma-es-tutorial]
            # Eq. (46) [2011:cma-es-tutorial] (same)
            tmp = (self.mu_eff - 1.0) / (self.n + 1.0)
            self.d_sigma = 1.0 + 2.0 * max(0.0, tmp - 1.0) + self.c_sigma
        elif not np.close(1.0, self.d_sigma):
            raise ValueError("Invalid value for d_sigma")

        if self.c_c is None:
            # Eq. (56) [2016:cma-es-tutorial]
            # Eq. (47) [2011:cma-es-tutorial] (same)
            self.c_c = (4.0 + self.mu_eff / self.n) / (
                self.n + 4.0 + 2.0 * self.mu_eff / self.n
            )
        elif self.c_c > 1.0:
            raise ValueError("Invalid value for c_c")

    def _post_init_c_1_mu(self, alpha_cov) -> None:
        """Initialize c_1 and c_mu if not provided."""
        if self.c_1 is None:
            # Eq. (57) [2016:cma-es-tutorial]
            # Eq. (48) [2011:cma-es-tutorial] (same if alpha_cov = 2)
            self.c_1 = alpha_cov / ((self.n + 1.3) ** 2 + self.mu_eff)
        elif callable(self.c_1):
            # Allows the user to compute dynamically
            self.c_1 = self.c_1(self.mu_eff)
        if self.c_mu is None:
            # Eq. (58) [2016:cma-es-tutorial]
            # Eq. (49) [2011:cma-es-tutorial] (same, alpha_mu is now alpha_cov)
            tmp_num = self.mu_eff - 2.0 + (1.0 / self.mu_eff)
            tmp_den = (self.n + 2.0) ** 2 + alpha_cov * self.mu_eff / 2.0
            self.c_mu = min(1 - self.c_1, alpha_cov * tmp_num / tmp_den)
        elif callable(self.c_mu):
            self.c_mu = self.c_mu(self.mu_eff)

        if self.c_1 > 1.0 - self.c_mu:
            raise ValueError("Invalid value for c_1")
        if self.c_mu > 1.0 - self.c_1:
            raise ValueError("Invalid value for c_mu")

    def _post_init_scale_weights(self, active: bool) -> None:
        """Scale the weigths."""
        # Scale positive weights to that they sum to 1
        # Eq. (53) [2016:cma-es-tutorial]
        # Eq. (45) [2011:cma-es-tutorial] (same)
        self.weights[: self.mu] /= np.sum(self.weights[: self.mu])

        if active:
            # Scale negative weights so that they sum to -alpha_mu_eff_neg
            # Eq. (53) [2016:cma-es-tutorial] (new)
            # TODO: Need to clarify what could be a good unit test for this?
            alpha_mu = 1.0 + self.c_1 / self.c_mu
            alpha_mu_eff_neg = 1.0 + (2.0 * self.mu_eff) / (
                self.mu_eff_neg + 2.0
            )
            alpha_pos_def = (1.0 - self.c_1 - self.c_mu) / (self.n * self.c_mu)
            alpha_min = min(alpha_mu, alpha_mu_eff_neg, alpha_pos_def)

            self.weights[self.mu :] *= alpha_min / np.sum(
                np.abs(self.weights[self.mu :])
            )
        else:
            self.weights[self.mu :] *= 0.0


@dataclasses.dataclass
class StoppingConditions:
    """Define stopping conditions for :class:`CMA`.

    Parameters
    ----------
    n : optional
        The number of variables of a solution.
        Required if ``tol_stagnation`` or ``max_fevals`` are not provided.
    population_size : optional
        The population size.
        Required if ``stagnation`` is not provided.
    initial_sigma : optional
        The initial step size. Used to initialize ``tol_x`` if not provided.
    ftarget : optional
        The target value of the objective function.
    max_fevals : optional
        Maximum number of function evaluations.
    tol_conditioncov : optional
        Tolerance for condition number of covariance matrix.
    tol_facupx : optional
        Tolerance for step size increase (divergence).
    tol_upsigma : optional
        Tolerance for creeping behaviour.
    tol_flatfitness : optional
        Tolerance for iterations with flat fitness.
    tol_fun : optional
        Tolerance for function value.
    tol_funhist : optional
        Tolerance for function value history.
    tol_funrel : optional
        Relative tolerance for function value.
    tol_stagnation : optional
        Tolerance for iterations with no improvement.

    Notes
    -----
        The termination criteria from :cite:`2019:pycma`. Currently, only a
        subset is supported in :class:`CMA`.
    """

    n: dataclasses.InitVar[typing.Optional[int]] = None
    population_size: dataclasses.InitVar[typing.Optional[int]] = None
    initial_sigma: dataclasses.InitVar[typing.Optional[float]] = None

    ftarget: float = np.NINF
    max_fevals: typing.Optional[int] = None
    tol_conditioncov: float = 1e14
    tol_facupx: float = 1e3
    tol_upsigma: float = 1e20
    tol_flatfitness: int = 1
    tol_fun: float = 1e-12
    tol_funhist: float = 1e-12
    tol_funrel: float = 0.0
    tol_stagnation: typing.Optional[int] = None
    tol_x: typing.Optional[float] = None

    def __post_init__(self, n, population_size, initial_sigma):
        """Initialize values that were not provided."""
        if n is None and (
            self.tol_stagnation is None or self.max_fevals is None
        ):
            raise ValueError(
                "n is requierd if max_fevals or tol_stagnation \
                are not provided"
            )

        if population_size is None and self.tol_stagnation is None:
            raise ValueError(
                "population_size is required if tol_stagnation \
                is not provided"
            )

        if self.max_fevals is None:
            self.max_fevals = 1000 * n * n

        if self.tol_stagnation is None:
            self.tol_stagnation = 100 * (
                1 + math.floor(n ** 1.5 / population_size)
            )

        if self.tol_x is None:
            # The default from [2016:cma-es-tutorial]
            if initial_sigma is not None:
                self.tol_x = 1e-12 * initial_sigma
            # The default from [2019:pycma]
            else:
                self.tol_x = 1e-11


class CMA(AbstractOptimizer):
    """The CMA-ES algorithm for single-objective real-valued optimization.

    Parameters
    ----------
    initial_point
        The initial search point.
    initial_sigma
        The initial step size.
    strategy_parameters : optional
        The external (exogenous / exposed) evolution strategy parameters.
    initial_cov : optional
        The initial covariance matrix. Default is the identity.
    rng : optional
        The random number generator.

    Raises
    ------
    numpy.linalg.LinAlgError
        The provided covariance matrix is not positive definite.

    Notes
    -----
    Based on :cite:`2016:cma-es-tutorial` and \
        the reference implementations from :cite:`2019:pycma`  \
            and :cite:`2008:shark`.
    """

    _initial_sigma: float
    _rng: np.random.Generator
    _mean: np.ndarray
    _cov_B: np.ndarray
    _cov_D: np.ndarray
    _cov_C: np.ndarray
    _eigeneval: int
    _path_cov: np.ndarray
    _path_sigma: np.ndarray
    _exp_norm_chi: float
    _params: StrategyParameters
    _dtype: np.dtype
    _best_solution: typing.Optional[np.ndarray]
    _best_value: typing.Optional[float]
    _function_values: typing.Optional[np.ndarray]

    def name(self) -> str:
        """Return the (prefixed) name of the CMA-ES variant."""
        # TODO: Annotate with relevant prefixes
        return "CMA-ES"

    def __init__(
        self,
        initial_point: np.ndarray,
        initial_sigma: float,
        strategy_parameters: typing.Optional[StrategyParameters] = None,
        stopping_conditions: typing.Optional[StoppingConditions] = None,
        initial_cov: typing.Optional[np.ndarray] = None,
        rng: typing.Optional[np.random.Generator] = None,
    ) -> None:
        """Initialize the optimizer."""
        super().__init__(initial_point)

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

        self._generation = 0
        self._n = initial_point.shape[0]
        self._mean = initial_point.reshape(self._n, 1)
        self._dtype = initial_point.dtype
        self._exp_norm_chi = anguilla.util.exp_norm_chi(float(self._n))
        self._path_cov = np.zeros(self._n)
        self._path_sigma = np.zeros(self._n)
        self._sigma = initial_sigma
        self._best_solution = None
        self._best_value = None

        if strategy_parameters is not None:
            if self._n != strategy_parameters.n:
                raise ValueError(
                    "Dimensionality disagreement between \
                    the initial_point and the given strategy_params."
                )
            self._params = strategy_parameters
        else:
            self._params = StrategyParameters(n=self._n)

        if stopping_conditions is None:
            self._stopping_conditions = StoppingConditions(
                n=self._n, population_size=self._params.population_size
            )
        else:
            self._stopping_conditions = stopping_conditions

        # Page 29 [2016:cma-es-tutorial]
        # Page 26 [2016:cma-es-tutorial] (same)
        self._eigeneval = 0
        if initial_cov is None:
            self._cov_D = np.eye(self._n, dtype=self._dtype)
            self._cov_B = np.eye(self._n, dtype=self._dtype)
            self._cov_C = np.eye(self._n, dtype=self._dtype)
        else:
            if initial_cov.shape[0] != self._n:
                raise ValueError(
                    "Dimensionality disagreement between  the \
                                  initial point and the initial covariance \
                                  matrix."
                )
            self._cov_C = initial_cov.copy()
            # Obtain B and D by eigendecomposition, such that C = B D^2 B'
            # Sec B.2, p. 32 of [2016:cma-es-tutorial]
            D_sq, self._cov_B = np.linalg.eigh(self._cov_C)
            self._cov_D = np.diag(np.sqrt(D_sq))
            self._eigeneval += 1

        self._function_values = np.zeros(self._params.population_size)

    def reset(self) -> None:
        """Re-initialize the optimizer."""
        raise NotImplementedError()

    def ask(self) -> np.ndarray:
        """Create new individuals (candidate solutions / search points) \
            through mutation.

        Notes
        -----
        The mutation operator samples candidate solutions (individuals) \
        from the multivariate Gaussian distribution that models the parent \
        population. See p. 8 of :cite:`2016:cma-es-tutorial`.

        Returns
        -------
        np.ndarray
            The new candidate solutions. Currently, an array with shape \
            ``(n, population_size)``.
        """
        # TODO: Could refactor when Numpy adds support for passing
        #       a factorization of the covariance matrix to the
        #       Generator.multivariate_normal method.
        #       See: https://github.com/numpy/numpy/issues/17158
        #            https://git.io/JkiaQ (numpy source)
        # TODO: Decide if parameter passing must be extended for
        #       decoupled ranking. See p. 34 [2013:oo-optimizers].
        # TODO: Refactor to use shape (population_size, n).

        shape = (self._n, self._params.population_size)
        # Eq. (38) [2016:cma-es-tutorial]
        # Eq. (35) [2011:cma-es-tutorial] (same)
        z = self._rng.standard_normal(size=shape)
        # Eq. (39) [2016:cma-es-tutorial]
        # Eq. (36) [2011:cma-es-tutorial] (same)
        BD = self._cov_B @ self._cov_D
        # Eq. (40) [2016:cma-es-tutorial]
        # Eq. (37) [2011:cma-es-tutorial] (same)
        x = self._mean + self._sigma * (BD @ z)
        assert x.shape == (self._n, self._params.population_size)

        return x

    def rank(
        self, solutions: np.ndarray, function_values: np.ndarray
    ) -> np.ndarray:
        """Rank the individuals (candidate solutions) according \
            to their fitness (ordering of their function values).

        Parameters
        ----------
        solutions
            The candidate solutions.
        function_values
            The values of the solutions computed by the objective function.

        Returns
        -------
        np.ndarray
            The indices of the solutions sorted according to their ranking (i.e. \
            the first element is the index of the best solution).

        Notes
        -----
        Implements the idea of decoupled ranking presented in \
        :cite:`2013:oo-optimizers`. This allows to implement
        different alternatives of uncertainty handling (i.e. noise) by
        subclassing and overriding this method. The default \
        implementation is the one used by :cite:`2008:shark`.
        """
        self._function_values[:] = function_values

        # TODO: Add the noise handling implemented by [2008:shark].
        #       Currently uses the simple ranking without accounting for
        #       uncertainty.

        # Sort candidate solutions by their fitness
        ranked_indices = np.argsort(function_values)

        # TODO: Fix counting of function evaluations
        #       when uncertainty handling is implemented.
        #       See p. 34 [2013:oo-optimizers] and [2008:shark].
        self._fevals += self._params.population_size

        best_index = ranked_indices[0]
        if (
            self._best_value is None
            or self._best_value > function_values[best_index]
        ):
            self._best_value = function_values[best_index]
            self._best_solution = np.copy(solutions[:, best_index])

        return ranked_indices

    def tell(self, solutions: np.ndarray, ranked_indices: np.ndarray) -> None:
        """Perform the selection and recombination of the individuals \
            (candidate solutions) and the self-adaptation of the internal \
            parameters (step size and covariance matrix).

        Parameters
        ----------
        solutions
            The candidate solutions.
        ranked_indices
            The indices of the solutions sorted according to their ranking (i.e. \
            the first element is the index of the best solution).
        """
        population_size = self._params.population_size
        mu = self._params.mu
        mu_eff = self._params.mu_eff
        c_1 = self._params.c_1
        c_m = self._params.c_m
        c_mu = self._params.c_mu
        c_c = self._params.c_c
        c_sigma = self._params.c_sigma
        d_sigma = self._params.d_sigma
        weights = self._params.weights

        # TODO: Remove assertions
        assert ranked_indices.shape[0] == population_size

        self._generation += 1

        # Selection
        selected_indices = ranked_indices[:mu]

        # Recombination
        # Eq. (39) [2016:cma-es-tutorial]
        # Eq. (36) [2011:cma-es-tutorial] (same)
        y = (solutions - self._mean) / self._sigma
        # Eq. (41) [2016:cma-es-tutorial]
        # Eq. (38) [2011:cma-es-tutorial] (same)
        yw = y[:, selected_indices] @ weights[:mu]
        assert yw.shape == (self._n,)

        # Eq. (42) [2016:cma-es-tutorial]
        # Eq. (39) [2011:cma-es-tutorial] (deprecated)
        tmp = (
            (solutions[:, selected_indices] - self._mean) @ weights[:mu]
        ).reshape(self._n, 1)
        assert tmp.shape == self._mean.shape
        self._mean += c_m * tmp.reshape(self._n, 1)

        # Transformation to obtain back z
        #
        #       x                            = mean + sigma * BDz
        #   <=> (x-mean) / sigma             = BDz
        #   <=> D^-1 B^-1 ((x-mean) / sigma) = z
        #   <=> D^-1 B' ((x-mean) / sigma)   = z
        BD_inv = np.diag(1.0 / np.diag(self._cov_D)) @ self._cov_B.T
        z = BD_inv @ y
        assert z.shape == solutions.shape

        # Perform self-adaptation of the internal (endogenous) parameters:
        # 1) Step size
        # 2) Covariance matrix

        # Step size adaptation

        # Step size (conjugate) evolution path
        # Eq. (43) [2016:cma-es-tutorial]
        # Eq. (40) [2011:cma-es-tutorial] (same)
        self._path_sigma = (1.0 - c_sigma) * self._path_sigma + math.sqrt(
            c_sigma * (2.0 - c_sigma) * mu_eff
        ) * self._cov_B @ (z[:, selected_indices] @ weights[:mu])
        assert self._path_sigma.shape == (self._n,)

        # Step size
        # Eq. (44) [2016:cma-es-tutorial]
        # Eq. (41) [2011:cma-es-tutorial] (same)
        tmp = (np.linalg.norm(self._path_sigma) / self._exp_norm_chi) - 1.0
        self._sigma = self._sigma * math.exp((c_sigma / d_sigma) * tmp)

        # Covariance matrix adaptation

        # The following coefficients prevent the covariance matrix
        # evolution path from getting large quickly when
        # the step size evolution path increases rapidly
        # Page [2015:es-overview]
        # Page 28 [2016:cma-es-tutorial]
        # Page 25 [2011:cma-es-tutorial] (same)
        h_sigma_lhs_num = np.linalg.norm(self._path_sigma)
        h_sigma_lhs_den = math.sqrt(
            1.0 - math.pow(1.0 - c_sigma, 2.0 * (self._generation + 1))
        )
        h_sigma_lhs = h_sigma_lhs_num / h_sigma_lhs_den
        h_sigma_rhs = (1.4 + 2.0 / (float(self._n) + 1.0)) * self._exp_norm_chi
        h_sigma = 1.0 if h_sigma_lhs < h_sigma_rhs else 0.0
        delta_h_sigma = (1.0 - h_sigma) * c_c * (2.0 - c_c)

        # Covariance matrix evolution path
        # Perform cumulation
        # Eq. (45) [2016:cma-es-tutorial]
        # Eq. (42) [2011:cma-es-tutorial] (same)
        tmp = math.sqrt(c_c * (2.0 - c_c) * mu_eff)
        self._path_cov = (1.0 - c_c) * self._path_cov + h_sigma * tmp * yw
        assert self._path_cov.shape == (self._n,)

        # Decay
        # Eq. (47) [2016:cma-es-tutorial]
        # Eq. (43) [2011:cma-es-tutorial] (deprecated)
        self._cov_C *= 1.0 + c_1 * delta_h_sigma - c_1 - c_mu * np.sum(weights)

        # Rank-one update
        # Align cov_C to the distribution of the selected / successful steps
        # to increase likelihood of sampling in the direction of the
        # evolution path [2016:cma-es-tutorial] [2015:es-overview]
        # Eq. (47) [2016:cma-es-tutorial]
        # Eq. (43) [2011:cma-es-tutorial] (deprecated)
        self._cov_C += c_1 * np.outer(self._path_cov, self._path_cov)

        # Eq. (46) [2016:cma-es-tutorial] (new)
        weights_circ = weights.copy()
        tmp = np.linalg.norm(self._cov_B @ z[:, ranked_indices])
        weights_circ[mu:] /= float(self._n) / (tmp * tmp)

        # Rank-mu update
        # Eq. (47) [2016:cma-es-tutorial] uses range(population_size)
        # Eq. (43) [2011:cma-es-tutorial] uses range(mu)
        # Both versions are the same obviating Eq. (46)
        tmp = (
            y[:, ranked_indices]
            @ np.diag(weights_circ)
            @ y[:, ranked_indices].T
        )
        assert tmp.shape == self._cov_C.shape
        self._cov_C += c_mu * tmp

        # Update B and D using the so-called lazy-update scheme
        # to reduce computation complexity from O(n^3) to O(n^2)
        # See page 18 [2015:es-overview] and
        # Sec. B.2. [2016:cma-es-tutorial]
        # Sec. B.2. [2011:cma-es-tutorial] (same)
        threshold = max(1.0, math.floor(1.0 / 10.0 * self._n * (c_1 + c_mu)))
        if self._fevals - self._eigeneval > threshold:
            # Enforce symmetry.
            # Taken from p. 37 of [2016:cma-es-tutorial]
            self._covC = np.triu(self._cov_C) + np.triu(self._cov_C, 1).T
            # Compute the eigendecomposition
            cov_D_sq, self._cov_B = np.linalg.eigh(self._covC)
            self._cov_D = np.diag(np.sqrt(cov_D_sq))
            self._eigeneval += 1

    def stop(self) -> bool:
        """Check for stopping conditions.

        Returns
        -------
        bool
            True if any stopping conditions

        Notes
        -----
        Currently does not implement the interface from :cite:`2019:pycma`.
        """

        # TODO: Comply with the interface of [2019:pycma] (i.e. return
        #       details on the conditions that triggered the stop).
        # TODO: Add support for more stoping conditions
        conditions = self._stopping_conditions

        return self._fevals > conditions.max_fevals or (
            self._fevals > 0 and self._best_value < conditions.ftarget
        )
