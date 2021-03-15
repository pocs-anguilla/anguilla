#pragma once

#ifndef ANGUILLA_UPMO_PARAMETERS_HPP
#define ANGUILLA_UPMO_PARAMETERS_HPP

// STL
#include <optional>

namespace anguilla {
namespace upmo {

constexpr auto DEF_D = std::nullopt;
constexpr auto DEF_P_THRESHOLD = 0.44;
constexpr auto DEF_P_TARGET_SUCC = std::nullopt;
constexpr auto DEF_C_P = std::nullopt;

// See default values in table 1 of [2016:mo-cma-es].
constexpr auto DEF_P_EXTREME = 1.0 / 5.0;
constexpr auto DEF_SIGMA_MIN = 1e-20;
constexpr auto DEF_ALPHA = 3;
constexpr auto DEF_C_COV = std::nullopt; // 2 / (d^2.1 + 3)
constexpr auto DEF_C_R = std::nullopt;   // c_cov / 2

static constexpr const char* PARAMETERS_DOCSTRING = R"pbdoc(
Parameters for UP-MO-CMA-ES.

Parameters
----------
n_dimensions
    Dimensionality of the search space.
initial_step_size
    Initial step size.
n_offspring: optional
    Number of offspring per parent.
d: optional
    Step size damping parameter.
p_target_succ: optional
    Target success probability.
c_p: optional
    Success rate averaging parameter.
p_threshold: optional
    Smoothed success rate threshold.
c_cov: optional
    Covariance matrix learning rate.
p_extreme: optional
    Extreme point probability.
sigma_min: optional
    Selected point convergence threshold.
alpha: optional
    Interior point probability weight.
c_r: optional
    Covariance matrix recombination learning rate.

Notes
-----
Implements the default values defined in Table 1, p. 3 \
:cite:`2016:up-mo-cma-es`, Table 1, p. 5 :cite:`2007:mo-cma-es` \
and p. 489 :cite:`2010:mo-cma-es`.)pbdoc";

template <typename T> class Parameters {
  public:
    Parameters(std::size_t nDimensions, T initialStepSize, std::optional<T> d,
               std::optional<T> pTargetSucc, std::optional<T> cP, T pThreshold,
               std::optional<T> cCov, T pExtreme, T sigmaMin, T alpha,
               std::optional<T> cR) {
        auto n = (T)nDimensions;
        this->initialStepSize = initialStepSize;
        this->d = d.value_or(1.0 + n / 2.0);
        this->pTargetSucc = pTargetSucc.value_or(0.5);
        this->cP = cP.value_or(this->pTargetSucc / (2.0 + this->pTargetSucc));
        this->pThreshold = pThreshold;
        this->cCov = cCov.value_or(2.0 / (std::pow(n, 2.1) + 3.0));
        this->pExtreme = pExtreme;
        this->sigmaMin = sigmaMin;
        this->alpha = alpha;
        this->cR = cR.value_or(this->cCov / 2.0);
    }

    T d;
    T initialStepSize;
    T sigmaMin;
    T alpha;
    T pTargetSucc;
    T pThreshold;
    T pExtreme;
    T cP;
    T cCov;
    T cR;
};

} // namespace upmo
} // namespace anguilla

#endif // ANGUILLA_UPMO_PARAMETERS_HPP
