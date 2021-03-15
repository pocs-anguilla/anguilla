#pragma once

#ifndef ANGUILLA_MOCMA_HPP
#define ANGUILLA_MOCMA_HPP

// STL
#include <cmath>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <optional>
#include <stdexcept>

// PyBind11
#include <pybind11/stl.h>

// Xtensor
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xslice.hpp>

// Xtensor Blas
#include <xtensor-blas/xlinalg.hpp>

// {fmt}
#include <fmt/core.h>

// Anguilla
#include <anguilla/dominance/dominance.hpp>
#include <anguilla/hypervolume/hvc2d.hpp>
#include <anguilla/optimizers/selection.hpp>

namespace anguilla {
namespace optimizers {

static constexpr const char* MO_SUCCESS_NOTION_DOCSTRING = R"pbdoc(
Define the notion of success of an offspring.
)pbdoc";

enum struct SuccessNotion : bool { PopulationBased, IndividualBased };

static constexpr const char* MO_PARAMETERS_DOCSTRING = R"pbdoc(
Define parameters for MO-CMA-ES.

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
    c_c: optional
        Evolution path learning rate.
    c_cov: optional
        Covariance matrix learning rate.

    Notes
    -----
    Implements the default values defined in Table 1, p. 5 \
    :cite:`2007:mo-cma-es` and p. 489 :cite:`2010:mo-cma-es`.
)pbdoc";

template <typename T> struct MOParameters {
  public:
    MOParameters(std::size_t nDimensions, T initialStepSize = 1.0,
                 std::size_t nOffspringPerParent = 1U,
                 std::optional<T> d = std::nullopt,
                 std::optional<T> pTargetSucc = std::nullopt,
                 std::optional<T> cP = std::nullopt, T pThreshold = 0.44,
                 std::optional<T> cC = std::nullopt,
                 std::optional<T> cCov = std::nullopt) {
        auto nDimensionsT = (T)nDimensions;
        auto nOffspringPerParentT = (T)nOffspringPerParent;
        this->nDimensions = nDimensions;
        this->nOffspringPerParent = nOffspringPerParent;
        this->initialStepSize = initialStepSize;
        this->d = d.value_or(1.0 + nDimensionsT / (2.0 * nOffspringPerParentT));
        this->pTargetSucc = pTargetSucc.value_or(
            1.0 / (5.0 + std::sqrt(nOffspringPerParentT) / 2.0));
        auto tmp = this->pTargetSucc * nOffspringPerParentT;
        this->cP = cP.value_or(tmp / (2.0 + tmp));
        this->pThreshold = pThreshold;
        this->cC = cC.value_or(2.0 / (nDimensionsT + 2.0));
        this->cCov = cCov.value_or(2.0 / (nDimensionsT * nDimensionsT + 6.0));
    }

    MOParameters(const MOParameters<T>&) = default;

    std::size_t nDimensions;
    std::size_t nOffspringPerParent;
    T initialStepSize;
    T d;
    T pTargetSucc;
    T cP;
    T pThreshold;
    T cC;
    T cCov;
};

static constexpr const char* MO_SOLUTION_DOCSTRING = R"pbdoc(
Define solution information for MO-CMA-ES.

    Parameters
    ----------
    points
        The current Pareto set approximation.
    fitness
        The current Pareto front approximation.
    step_size
        The current Pareto set step sizes. Used for experiments.
)pbdoc";

template <typename T> struct MOSolution {
    MOSolution(xt::xtensor<T, 2U> point, xt::xtensor<T, 2U> fitness,
               xt::xtensor<T, 2U> stepSize)
        : point(point), fitness(fitness), stepSize(stepSize) {}
    xt::xtensor<T, 2U> point;
    xt::xtensor<T, 2U> fitness;
    xt::xtensor<T, 2U> stepSize;
};

static constexpr const char* MO_STOPPING_CONDITIONS_DOCSTRING = R"pbdoc(
Define stopping criteria for MO-CMA-ES.

    Parameters
    ----------
    max_generations: optional
        Maximum number of generations.
    max_evaluations: optional
        Maximum number of function evaluations.
    target_indicator_value: optional
        Target indicator value.
    triggered: optional
        Indicates if any condition was triggered, when returned as an output.
    is_output: optional
        Indicates the class is used as an output.

    Raises
    ------
    ValueError
        No stopping condition was provided.

    Notes
    -----
    The class can be used as an input to specify stopping conditions and as an
    output to define the conditions that triggered the stop.
)pbdoc";

template <typename T> struct MOStoppingConditions {
  public:
    MOStoppingConditions(std::optional<std::size_t> maxGenerations,
                         std::optional<std::size_t> maxEvaluations,
                         std::optional<T> targetIndicatorValue,
                         bool triggered = false, bool isOutput = false)
        : maxGenerations(maxGenerations), maxEvaluations(maxEvaluations),
          targetIndicatorValue(targetIndicatorValue), triggered(triggered),
          isOutput(isOutput) {
        if (!isOutput && !maxGenerations.has_value() &&
            !maxEvaluations.has_value() && !targetIndicatorValue.has_value()) {
            throw std::invalid_argument(
                "At least one stopping condition must be provided");
        }
    }

    std::optional<std::size_t> maxGenerations;
    std::optional<std::size_t> maxEvaluations;
    std::optional<T> targetIndicatorValue;
    bool triggered;
    bool isOutput;
};

template <typename T> struct MOPopulation {
    MOPopulation(std::size_t nParents, std::size_t nOffspring,
                 std::size_t nDimensions, std::size_t nObjectives, T stepSize,
                 T pSucc)
        : nParents(nParents), nOffspring(nOffspring),
          nIndividuals(nParents + nOffspring),
          point(xt::empty<T>({nIndividuals, nDimensions})),
          fitness(xt::empty<T>({nIndividuals, nObjectives})),
          penalizedFitness(xt::empty<T>({nIndividuals, nObjectives})),
          rank(xt::ones_like(
              xt::xtensor<T, 2U>::from_shape({nIndividuals, 1U}))),
          stepSize(xt::full_like(
              xt::xtensor<T, 2U>::from_shape({nIndividuals, 1U}), stepSize)),
          pSucc(xt::full_like(
              xt::xtensor<T, 2U>::from_shape({nIndividuals, 1U}), pSucc)),
          z(xt::empty<T>({nOffspring, nDimensions})),
          pC(xt::zeros_like(
              xt::xtensor<T, 2U>::from_shape({nIndividuals, nDimensions}))),
          cov(xt::eye<T>({nIndividuals, nDimensions, nDimensions}, 0)),
          parentIndex(
              xt::zeros_like(xt::xtensor<T, 1U>::from_shape({nOffspring}))) {}

    auto allRange() { return xt::range(0, nIndividuals); }

    auto parentRange() { return xt::range(0, nParents); }

    auto offspringRange() { return xt::range(nParents, nIndividuals); }

    // See: https://git.io/JqWUd
    template <class S> auto pointView(S&& slice) {
        return xt::view(point, slice, xt::all());
    }

    template <class S> auto fitnessView(S&& slice) {
        return xt::view(fitness, slice, xt::all());
    }

    template <class S> auto penalizedFitnessView(S&& slice) {
        return xt::view(penalizedFitness, slice, xt::all());
    }

    template <class S> auto rankView(S&& slice) {
        return xt::view(rank, slice);
    }

    template <class S> auto stepSizeView(S&& slice) {
        return xt::view(stepSize, slice, xt::all());
    }

    template <class S> auto pSuccView(S&& slice) {
        return xt::view(pSucc, slice, xt::all());
    }

    template <class S> auto pCView(S&& slice) {
        return xt::view(pC, slice, xt::all());
    }

    template <class S> auto covView(S&& slice) {
        return xt::view(cov, slice, xt::all(), xt::all());
    }

    std::size_t nParents;
    std::size_t nOffspring;
    std::size_t nIndividuals;
    // The search points.
    xt::xtensor<T, 2U> point;
    // The objective points.
    xt::xtensor<T, 2U> fitness;
    // The penalized objective points
    xt::xtensor<T, 2U> penalizedFitness;
    // The non-domination ranks.
    xt::xtensor<std::size_t, 1U> rank;
    // The step sizes.
    xt::xtensor<T, 2U> stepSize;
    // The smoothed probabilities of success.
    xt::xtensor<T, 2U> pSucc;
    // The samples used to mutate the parent points.
    xt::xtensor<T, 2U> z;
    // The covariance adaptation evolution paths.
    xt::xtensor<T, 2U> pC;
    // The covariance matrices.
    xt::xtensor<T, 3U> cov;
    // The parent indices.
    xt::xtensor<std::size_t, 1U> parentIndex;
};

template <typename T> constexpr auto square(T x) { return x * x; }

template <typename T>
constexpr auto matrix(xt::xtensor<T, 3U> A, std::size_t idx) {
    // Selects a matrix from a 3D array.
    // Similar to xt::row but for 3D arrays.
    return xt::view(A, idx, xt::all(), xt::all());
}

template <typename T>
auto choleskyUpdate(const xt::xtensor<T, 2U>& L, T alpha, T beta,
                    const xt::xtensor<T, 1U>& v) -> xt::xtensor<T, 2U> {
    // Algorithm 3.1, p.3, [2015:efficient-rank1-update]
    // Computes the updated Cholesky factor L' of alpha * A + beta * vv.T
    // Here using a port of Shark's implementation, which is based on Eigen's
    // implementation.
    // Shark: https://git.io/JqgKP
    // Eigen: https://bit.ly/30JTzXQ

    const auto n = L.shape(0U);
    const T alphaSqrt = std::sqrt(alpha);
    xt::xtensor<T, 2U> Lp = L;

    if (beta == (T)0.0) {
        Lp *= alphaSqrt;
        return Lp;
    }

    xt::xtensor<T, 1U> omega = v;
    T b = 1.0;
    for (auto j = 0U; j != n; ++j) {
        const T Ljj = alphaSqrt * L(j, j);
        const T dj = Ljj * Ljj;
        const T wj = omega(j);
        const T swj2 = beta * square<>(wj);
        const T gamma = dj * b + swj2;
        const T x = dj + swj2 / b;

        if (x <= (T)0.0) {
            throw std::runtime_error("Update makes matrix indefinite");
        }

        const T nLjj = std::sqrt(x);
        Lp(j, j) = nLjj;
        b += swj2 / dj;

        // Update the terms of L
        if (j + 1U < n) {
            xt::view(Lp, xt::range(j + 1U, n), j) *= alphaSqrt;
            xt::view(omega, xt::range(j + 1U, n)) -=
                (wj / Ljj) * xt::view(L, xt::range(j + 1U, n), j);
            if (gamma == (T)0.0) {
                continue;
            }
            xt::view(Lp, xt::range(j + 1U, n), j) *= nLjj / Ljj;
            xt::view(Lp, xt::range(j + 1U, n), j) +=
                (nLjj * beta * wj / gamma) *
                xt::view(omega, xt::range(j + 1U, n));
        }
    }
    return Lp;
}

static constexpr const char* MOCMA_DOCSTRING = R"pbdoc(
The MO-CMA-ES multi-objective optimizer.

    Parameters
    ----------
    parent_points
        The search points of the initial population.
    parent_fitness
        The objective points of the initial population.
    n_offspring: optional
        The number of offspring. Defaults to the same number of parents.
    initial_step_size: optional
        The initial step size. Ignored if `parameters` is provided.
    success_notion: optional
        The notion of success (either `individual` or `population`).
    max_generations: optional
        Maximum number of generations to trigger stop.
    max_evaluations: optional
        Maximum number of function evaluations to trigger stop.
    target_indicator_value: optional
        Target value of the indicator to trigger stop.
    parameters: optional
        The external parameters. Allows to provide custom values other than \
        the recommended in the literature.
    seed: optional
        A seed for the random number generator.

    Raises
    ------
    ValueError
        A parameter was provided with an invalid value.

    Notes
    -----
    The implementation supports the algorithms presented in Algorithm 4, \
    p. 12, :cite:`2007:mo-cma-es` and Algorithm 1, p. 488 \
    :cite:`2010:mo-cma-es`.
)pbdoc";

template <typename T, typename RandomEngine = xt::random::default_engine_type>
class MOCMA {
  public:
    using SeedType = typename RandomEngine::result_type;

    MOCMA(xt::xtensor<T, 2U> parentPoints, xt::xtensor<T, 2U> parentFitness,
          std::optional<std::size_t> nOffspring = std::nullopt,
          T initialStepSize = 1.0,
          std::string successNotion = std::string("population"),
          std::optional<std::size_t> maxGenerations = std::nullopt,
          std::optional<std::size_t> maxEvaluations = std::nullopt,
          std::optional<T> targetIndicatorValue = std::nullopt,
          std::optional<MOParameters<T>> parameters = std::nullopt,
          std::optional<SeedType> seed = std::nullopt)
        : m_successNotion(successNotion.compare("population") == 0
                              ? SuccessNotion::PopulationBased
                              : SuccessNotion::IndividualBased),
          m_nParents(parentPoints.shape(0U)),
          m_nOffspring(nOffspring.value_or(m_nParents)),
          m_parameters(parameters.value_or(
              MOParameters<T>(parentPoints.shape(1U), initialStepSize))),
          m_stoppingConditions(maxGenerations, maxEvaluations,
                               targetIndicatorValue),
          m_population(parentPoints.shape(0U), m_nOffspring,
                       parentPoints.shape(1U), parentFitness.shape(1U),
                       m_parameters.initialStepSize, m_parameters.pTargetSucc),
          m_generationCount(0), m_evaluationCount(0), m_askCalled(false) {
        static_assert(std::is_floating_point<T>::value,
                      "MOCMA is not meant to be instantiated "
                      "with a non floating point type.");
        if (parentPoints.shape(0U) != parentFitness.shape(0U)) {
            throw std::invalid_argument("Parent points and fitness have "
                                        "different number of elements.");
        }
        auto parentRange = m_population.parentRange();
        m_population.pointView(parentRange) = parentPoints;
        m_population.fitnessView(parentRange) = parentFitness;
        m_population.penalizedFitnessView(parentRange) = parentFitness;
        m_population.rankView(parentRange) =
            std::get<0U>(dominance::nonDominatedSort<T>(
                m_population.penalizedFitnessView(parentRange)));
        m_randomEngine.seed(seed.value_or(time(NULL)));
    }

    auto name() const { return std::string("MO-CMA-ES"); }

    auto qualifiedName() const -> std::string {
        return fmt::format(
            "({1}+{2})-{0}-{3}", name(), m_nParents, m_nOffspring,
            m_successNotion == SuccessNotion::PopulationBased ? "P" : "I");
    }

    auto generationCount() const { return m_generationCount; }

    auto evaluationCount() const { return m_evaluationCount; }

    auto parameters() const -> const MOParameters<T>& { return m_parameters; }

    auto stoppingConditions() const -> const MOStoppingConditions<T>& {
        return m_stoppingConditions;
    }

    auto population() const -> const MOPopulation<T>& { return m_population; }

    auto successNotion() const -> const SuccessNotion& {
        return m_successNotion;
    }

    bool isSteadyState() const { return m_nOffspring == 1U; }

    auto stop() -> MOStoppingConditions<T> {
        auto result = MOStoppingConditions<T>(std::nullopt, std::nullopt,
                                              std::nullopt, false, true);
        if (m_stoppingConditions.maxGenerations.has_value() &&
            m_generationCount >= m_stoppingConditions.maxGenerations.value()) {
            result.maxGenerations = m_generationCount;
            result.triggered = true;
        }
        if (m_stoppingConditions.maxEvaluations.has_value() &&
            m_evaluationCount >= m_stoppingConditions.maxEvaluations.value()) {
            result.maxEvaluations = m_evaluationCount;
            result.triggered = true;
        }
        if (m_stoppingConditions.targetIndicatorValue.has_value()) {
            throw std::runtime_error("Not implemented.");
        }
        return result;
    }

    auto best() -> MOSolution<T> {
        auto parentRange = m_population.parentRange();
        return MOSolution<T>(m_population.pointView(parentRange),
                             m_population.fitnessView(parentRange),
                             m_population.stepSizeView(parentRange));
    }

    auto ask() -> xt::xtensor<T, 2U> {
        // Update ask-and-tell state machine.
        m_askCalled = true;
        if (m_nParents == m_nOffspring) {
            // Each parent produces an offspring
            // Algorithm 1, line 4b [2010:mo-cma-es].
            m_population.parentIndex =
                xt::arange<std::size_t>(0U, m_nOffspring);
        } else {
            // Parents with a non-domination rank of 1 are selected to reproduce
            // uniformly at random with replacement. Algorithm 1, line 4a
            // [2010:mo-cma-es].
            const auto parentRanks =
                m_population.rankView(m_population.parentRange());
            const auto selectedParents = xt::squeeze(
                xt::from_indices(xt::argwhere(xt::equal(parentRanks, 1U))));
            m_population.parentIndex = xt::random::choice(
                selectedParents, m_nOffspring, true, m_randomEngine);
        }
        // We use Algorithm 4.1 from [2015:efficient-rank1-update]
        // adapted for Algorithm 1 from [2010:mo-cma-es].
        auto parentIndices = m_population.parentIndex;
        auto offspringIndices = xt::arange<std::size_t>(
            m_population.nParents, m_population.nIndividuals);
        // Perform mutation of the parents chosen to reproduce.
        auto parentPoints = m_population.pointView(xt::keep(parentIndices));
        auto offspringPoints =
            m_population.pointView(xt::keep(offspringIndices));
        auto parentStepSize =
            m_population.stepSizeView(xt::keep(parentIndices));
        auto parentCov = m_population.covView(xt::keep(parentIndices));
        m_population.z = xt::random::randn<T>(m_population.z.shape(), 0.0, 1.0,
                                              m_randomEngine);
        xt::xtensor<T, 2> tmp = xt::empty<T>(offspringPoints.shape());
        for (auto i = 0U; i != tmp.shape(0U); i++) {
            xt::row(tmp, i) =
                xt::linalg::dot(xt::view(m_population.cov, parentIndices(i),
                                         xt::all(), xt::all()),
                                xt::row(m_population.z, i));
        }
        offspringPoints = parentPoints + parentStepSize * tmp;
        // Copy data from the parent.
        m_population.stepSizeView(xt::keep(offspringIndices)) = parentStepSize;
        m_population.pSuccView(xt::keep(offspringIndices)) =
            m_population.pSuccView(xt::keep(parentIndices));
        m_population.covView(xt::keep(offspringIndices)) = parentCov;
        return offspringPoints;
    }

    void tell(
        const xt::xtensor<T, 2U> fitness,
        const std::optional<xt::xtensor<T, 2U>> penalizedFitness = std::nullopt,
        const std::optional<std::size_t> evaluationCount = std::nullopt) {
        if (!m_askCalled) {
            throw std::runtime_error("Tell called before ask.");
        }
        // Update ask-and-tell state machine
        m_askCalled = false;
        // Update counters
        if (evaluationCount.has_value()) {
            m_evaluationCount += evaluationCount.value();
        } else {
            m_evaluationCount += fitness.shape(0U);
        }
        m_generationCount += 1;
        // Copy provided fitness data into the population container
        auto offspringRange = m_population.offspringRange();
        m_population.fitnessView(offspringRange) = fitness;
        m_population.penalizedFitnessView(offspringRange) =
            penalizedFitness.value_or(fitness);
        // Compute non-domination ranks
        m_population.rank = std::get<0U>(
            dominance::nonDominatedSort<T>(m_population.penalizedFitness));
        // Compute indicator-based selection
        auto selected = hvi::selection(m_population.penalizedFitness,
                                       m_population.rank, m_nParents);
        // Perform adaptation
        for (auto i = 0U; i != m_nOffspring; i++) {
            auto oidx = m_nParents + i;
            auto pidx = m_population.parentIndex(i);
            // Updates should only occur if individuals are selected and
            // successful (see [2008:shark], URL: https://git.io/Jty6G).
            T offspringIsSuccessful = 0.0;
            // Offspring adaptation
            if (m_successNotion == SuccessNotion::IndividualBased &&
                selected(oidx) &&
                m_population.rank(oidx) <= m_population.rank(pidx)) {
                // [2010:mo-cma-es] Section 3.1, p. 489
                offspringIsSuccessful = 1.0;
                updateStepSize(oidx, offspringIsSuccessful);
                updateCov(oidx, xt::row(m_population.z, i));
            } else {
                if (m_successNotion == SuccessNotion::PopulationBased &&
                    selected(oidx)) {
                    // [2010:mo-cma-es] Section 3.2, p. 489
                    offspringIsSuccessful = 1.0;
                    updateStepSize(oidx, offspringIsSuccessful);
                    updateCov(oidx, xt::row(m_population.z, i));
                }
            }
            // Parent adaptation
            if (selected(pidx)) {
                updateStepSize(pidx, offspringIsSuccessful);
            }
        }
        // Perform selection
        // [2007:mo-cma-es] Algorithm 4, line 20
        const auto parentIndices = xt::arange<std::size_t>(0U, m_nParents);
        const auto selectedIndices = xt::squeeze(
            xt::from_indices(xt::argwhere(xt::equal(selected, true))));
        // The search points.
        m_population.pointView(xt::keep(parentIndices)) =
            m_population.pointView(xt::keep(selectedIndices));
        // The objective points.
        m_population.fitnessView(xt::keep(parentIndices)) =
            m_population.fitnessView(xt::keep(selectedIndices));
        // The penalized objective points
        m_population.penalizedFitnessView(xt::keep(parentIndices)) =
            m_population.penalizedFitnessView(xt::keep(selectedIndices));
        // The non-domination ranks
        m_population.rankView(xt::keep(parentIndices)) =
            m_population.rankView(xt::keep(selectedIndices));
        // The step sizes.
        m_population.stepSizeView(xt::keep(parentIndices)) =
            m_population.stepSizeView(xt::keep(selectedIndices));
        // The smoothed probabilities of success.
        m_population.pSuccView(xt::keep(parentIndices)) =
            m_population.pSuccView(xt::keep(selectedIndices));
        // The covariance adaptation evolution paths.
        m_population.pCView(xt::keep(parentIndices)) =
            m_population.pCView(xt::keep(selectedIndices));
        // The covariance matrices.
        m_population.covView(xt::keep(parentIndices)) =
            m_population.covView(xt::keep(selectedIndices));
    }

  private:
    void updateStepSize(std::size_t idx, T offspringIsSuccessful) {
        // Algorithm 1, lines 9-10, 17-18, p. 2 [2010:mo-cma-es]
        // Update the smoothed probability of success
        m_population.pSucc(idx) =
            (1.0 - m_parameters.cP) * m_population.pSucc(idx) +
            offspringIsSuccessful * m_parameters.cP;
        // Update the step size
        m_population.stepSize(idx) =
            m_population.stepSize(idx) *
            std::exp((1.0 / m_parameters.d) *
                     ((m_population.pSucc(idx) - m_parameters.pTargetSucc) /
                      (1.0 - m_parameters.pTargetSucc)));
    }

    void updateCov(std::size_t idx, const xt::xtensor<T, 1U>& z) {
        if (m_population.pSucc(idx) < m_parameters.pThreshold) {
            // Algorithm 4.1, line 19, p.5. [2015:efficient-rank1-update]
            // Update evolution path
            xt::row(m_population.pC, idx) =
                (1.0 - m_parameters.cC) * xt::row(m_population.pC, idx) +
                std::sqrt(m_parameters.cC * (2.0 - m_parameters.cC)) * z;
            // Update Cholesky factor of the covariance matrix
            matrix(m_population.cov, idx) = choleskyUpdate<T>(
                matrix(m_population.cov, idx), 1.0 - m_parameters.cCov,
                m_parameters.cCov, xt::row(m_population.pC, idx));
        } else {
            // Algorithm 4.1, line 15, p.5. [2015:efficient-rank1-update]
            const T cL =
                1.0 - m_parameters.cCov +
                m_parameters.cCov * m_parameters.cC * (2.0 - m_parameters.cC);
            // Update evolution path
            xt::row(m_population.pC, idx) =
                (1.0 - m_parameters.cC) * xt::row(m_population.pC, idx);
            // Update Cholesky factor of the covariance matrix
            matrix(m_population.cov, idx) = choleskyUpdate<T>(
                matrix(m_population.cov, idx), cL, m_parameters.cCov,
                xt::row(m_population.pC, idx));
        }
    }

    SuccessNotion m_successNotion;
    std::size_t m_nParents;
    std::size_t m_nOffspring;
    RandomEngine m_randomEngine;
    MOParameters<T> m_parameters;
    MOStoppingConditions<T> m_stoppingConditions;
    MOPopulation<T> m_population;
    std::size_t m_generationCount;
    std::size_t m_evaluationCount;
    bool m_askCalled;
};

} // namespace optimizers
} // namespace anguilla

#endif // ANGUILLA_MOCMA_HPP
