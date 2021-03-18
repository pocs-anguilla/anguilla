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
          lastStep(xt::empty<T>({nOffspring, nDimensions})),
          lastZ(xt::empty<T>({nOffspring, nDimensions})),
          pC(xt::zeros_like(
              xt::xtensor<T, 2U>::from_shape({nIndividuals, nDimensions}))),
          cov(xt::eye<T>({nIndividuals, nDimensions, nDimensions}, 0)),
          parentIdx(
              xt::zeros_like(xt::xtensor<T, 1U>::from_shape({nOffspring}))) {}

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
    // The last mutative step
    xt::xtensor<T, 2U> lastStep;
    // The samples used to mutate the parent points.
    xt::xtensor<T, 2U> lastZ;
    // The covariance adaptation evolution paths.
    xt::xtensor<T, 2U> pC;
    // The covariance matrices.
    xt::xtensor<T, 3U> cov;
    // The parent indices.
    xt::xtensor<std::size_t, 1U> parentIdx;
};

template <typename T> constexpr auto square(T x) { return x * x; }

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
          m_nIndividuals(m_nParents + m_nOffspring),
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
        xt::view(m_population.point, xt::range(0U, m_nParents), xt::all()) =
            parentPoints;
        xt::view(m_population.fitness, xt::range(0U, m_nParents), xt::all()) =
            parentFitness;
        xt::view(m_population.penalizedFitness, xt::range(0U, m_nParents),
                 xt::all()) = parentFitness;
        xt::view(m_population.rank, xt::range(0U, m_nParents)) =
            std::get<0U>(dominance::nonDominatedSort<T>(
                xt::view(m_population.rank, xt::range(0U, m_nParents))));
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
        return MOSolution<T>(
            xt::view(m_population.point, xt::range(0U, m_nParents), xt::all()),
            xt::view(m_population.fitness, xt::range(0U, m_nParents),
                     xt::all()),
            xt::view(m_population.stepSize, xt::range(0U, m_nParents),
                     xt::all()));
    }

    auto ask() -> xt::xtensor<T, 2U> {
        // Update ask-and-tell state machine.
        m_askCalled = true;
        if (m_nParents == m_nOffspring) {
            // Each parent produces an offspring.
            // Algorithm 1, line 4b [2010:mo-cma-es].
            m_population.parentIdx = xt::arange<std::size_t>(0U, m_nOffspring);
        } else {
            // Parents with a non-domination rank of 1 are selected to reproduce
            // uniformly at random with replacement. Algorithm 1, line 4a
            // [2010:mo-cma-es].
            const auto parentRanks =
                xt::view(m_population.rank, xt::range(0U, m_nParents));
            const auto eligibleParents = xt::squeeze(
                xt::from_indices(xt::argwhere(xt::equal(parentRanks, 1U))));
            m_population.parentIdx = xt::random::choice(
                eligibleParents, m_nOffspring, true, m_randomEngine);
        }
        // We use Algorithm 4.1 from [2015:efficient-rank1-update].
        // adapted for Algorithm 1 from [2010:mo-cma-es].
        const auto parentIdx = m_population.parentIdx;
        const auto offspringIdx =
            xt::arange<std::size_t>(m_nParents, m_nIndividuals);
        // Perform mutation of the parents chosen to reproduce.
        // See also: [2008:shark] https://git.io/JqhW7
        m_population.lastZ = xt::random::randn<T>(m_population.lastZ.shape(),
                                                  0.0, 1.0, m_randomEngine);
        /*for (auto i = 0U; i != m_nOffspring; i++) {
            const std::size_t pidx = parentIdx(i);
            const std::size_t oidx = m_nParents + i;
            // Compute the mutative step
            xt::row(m_population.lastStep, i) = xt::linalg::dot(
                xt::view(m_population.cov, pidx, xt::all(), xt::all()),
                xt::row(m_population.lastZ, i));
            // Mutate parent point and assign it to its offspring point
            xt::row(m_population.point, oidx) =
                xt::row(m_population.point, pidx) +
                m_population.stepSize(pidx, 0) *
                    xt::row(m_population.lastStep, i);
            // Copy parent data
            m_population.stepSize(oidx, 0) = m_population.stepSize(pidx, 0);
            m_population.pSucc(oidx, 0) = m_population.stepSize(pidx, 0);
            xt::row(m_population.pC, oidx) = xt::row(m_population.pC, pidx);
            xt::view(m_population.cov, oidx, xt::all(), xt::all()) =
                xt::view(m_population.cov, pidx, xt::all(), xt::all());
        }*/
        for (auto i = 0U; i != m_nOffspring; i++) {
            // Compute the mutative step
            xt::view(m_population.lastStep, i, xt::all()) = xt::linalg::dot(
                xt::view(m_population.cov, parentIdx(i), xt::all(), xt::all()),
                xt::view(m_population.lastZ, i, xt::all()));
        }
        xt::view(m_population.point, xt::keep(offspringIdx), xt::all()) =
            xt::view(m_population.point, xt::keep(parentIdx), xt::all()) +
            xt::view(m_population.stepSize, xt::keep(parentIdx), xt::all()) *
                m_population.lastStep;
        // Copy data from the parent.
        // Step size.
        xt::view(m_population.stepSize, xt::keep(offspringIdx), xt::all()) =
            xt::view(m_population.stepSize, xt::keep(parentIdx), xt::all());
        // Smoothed probability of success.
        xt::view(m_population.pSucc, xt::keep(offspringIdx), xt::all()) =
            xt::view(m_population.pSucc, xt::keep(parentIdx), xt::all());
        // Evolution path.
        xt::view(m_population.pC, xt::keep(offspringIdx), xt::all()) =
            xt::view(m_population.pC, xt::keep(parentIdx), xt::all());
        // Covariance matrix Cholesky factor.
        xt::view(m_population.cov, xt::keep(offspringIdx), xt::all(),
                 xt::all()) = xt::view(m_population.cov, xt::keep(parentIdx),
                                       xt::all(), xt::all());
        return xt::view(m_population.point, xt::keep(offspringIdx), xt::all());
    }

    void tell(
        const xt::xtensor<T, 2U> fitness,
        const std::optional<xt::xtensor<T, 2U>> penalizedFitness = std::nullopt,
        const std::optional<std::size_t> evaluationCount = std::nullopt) {
        if (!m_askCalled) {
            throw std::runtime_error("Tell called before ask.");
        }
        if (fitness.shape(0U) != m_nOffspring) {
            throw std::invalid_argument(
                "Fitness shape doesn't match number of offspring.");
        }
        if (penalizedFitness.has_value() &&
            penalizedFitness.value().shape(0U) != m_nOffspring) {
            throw std::invalid_argument(
                "Penalized fitness shape doesn't match number of offspring.");
        }
        // Update ask-and-tell state machine
        m_askCalled = false;
        // Update counters
        if (evaluationCount.has_value()) {
            m_evaluationCount += evaluationCount.value();
        } else {
            m_evaluationCount += m_nOffspring;
        }
        m_generationCount += 1;
        // Copy provided fitness data into the population container
        xt::view(m_population.fitness, xt::range(m_nParents, m_nIndividuals),
                 xt::all()) = fitness;
        xt::view(m_population.penalizedFitness,
                 xt::range(m_nParents, m_nIndividuals), xt::all()) =
            penalizedFitness.value_or(fitness);
        // Compute non-domination ranks
        m_population.rank = std::get<0U>(
            dominance::nonDominatedSort<T>(m_population.penalizedFitness));
        // Compute indicator-based selection
        const auto selected = hvi::selection(m_population.penalizedFitness,
                                             m_population.rank, m_nParents);
        // Perform adaptation
        for (auto i = 0U; i != m_nOffspring; i++) {
            auto oidx = m_nParents + i;
            auto pidx = m_population.parentIdx(i);
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
                updateCov(oidx, xt::row(m_population.lastStep, i));
            } else {
                if (m_successNotion == SuccessNotion::PopulationBased &&
                    selected(oidx)) {
                    // [2010:mo-cma-es] Section 3.2, p. 489
                    offspringIsSuccessful = 1.0;
                    updateStepSize(oidx, offspringIsSuccessful);
                    updateCov(oidx, xt::row(m_population.lastStep, i));
                }
            }
            // Parent adaptation
            if (selected(pidx)) {
                updateStepSize(pidx, offspringIsSuccessful);
            }
        }
        // Perform selection
        // [2007:mo-cma-es] Algorithm 4, line 20
        const auto parentIdx = xt::arange<std::size_t>(0U, m_nParents);
        const auto selectedIdx = xt::squeeze(
            xt::from_indices(xt::argwhere(xt::equal(selected, true))));
        // The search points.
        xt::view(m_population.point, xt::keep(parentIdx), xt::all()) =
            xt::view(m_population.point, xt::keep(selectedIdx), xt::all());
        // The objective points.
        xt::view(m_population.fitness, xt::keep(parentIdx), xt::all()) =
            xt::view(m_population.fitness, xt::keep(selectedIdx), xt::all());
        // The penalized objective points
        xt::view(m_population.penalizedFitness, xt::keep(parentIdx),
                 xt::all()) = xt::view(m_population.penalizedFitness,
                                       xt::keep(selectedIdx), xt::all());
        // The non-domination ranks
        xt::view(m_population.rank, xt::keep(parentIdx)) =
            xt::view(m_population.rank, xt::keep(selectedIdx));
        // The step sizes.
        xt::view(m_population.stepSize, xt::keep(parentIdx), xt::all()) =
            xt::view(m_population.stepSize, xt::keep(selectedIdx), xt::all());
        // The smoothed probabilities of success.
        xt::view(m_population.pSucc, xt::keep(parentIdx), xt::all()) =
            xt::view(m_population.pSucc, xt::keep(selectedIdx), xt::all());
        // The covariance adaptation evolution paths.
        xt::view(m_population.pC, xt::keep(parentIdx), xt::all()) =
            xt::view(m_population.pC, xt::keep(selectedIdx), xt::all());
        // The covariance matrices.
        xt::view(m_population.cov, xt::keep(parentIdx), xt::all(), xt::all()) =
            xt::view(m_population.cov, xt::keep(selectedIdx), xt::all(),
                     xt::all());
    }

  private:
    void updateStepSize(std::size_t idx, T offspringIsSuccessful) {
        // Algorithm 1, lines 9-10, 17-18, p. 2 [2010:mo-cma-es]
        // Update the smoothed probability of success
        m_population.pSucc(idx, 0U) =
            (1.0 - m_parameters.cP) * m_population.pSucc(idx, 0U) +
            offspringIsSuccessful * m_parameters.cP;
        // Update the step size
        m_population.stepSize(idx, 0U) *=
            std::exp((1.0 / m_parameters.d) *
                     ((m_population.pSucc(idx, 0U) - m_parameters.pTargetSucc) /
                      (1.0 - m_parameters.pTargetSucc)));
    }

    void updateCov(std::size_t idx, const xt::xtensor<T, 1U>& lastStep) {
        const T pCUpdateWeight = m_parameters.cC * (2.0 - m_parameters.cC);
        // Update the evolution path
        xt::row(m_population.pC, idx) *= 1.0 - m_parameters.cC;
        if (m_population.pSucc(idx, 0) < m_parameters.pThreshold) {
            // Algorithm 4.1, line 19, p.5. [2015:efficient-rank1-update]
            // [2008:Shark] See: https://git.io/Jqphk
            // Update the evolution path
            xt::row(m_population.pC, idx) +=
                std::sqrt(pCUpdateWeight) * lastStep;
            // Update the Cholesky factor of the covariance matrix
            xt::view(m_population.cov, idx, xt::all(), xt::all()) =
                choleskyUpdate<T>(
                    xt::view(m_population.cov, idx, xt::all(), xt::all()),
                    1.0 - m_parameters.cCov, m_parameters.cCov,
                    xt::row(m_population.pC, idx));
        } else {
            // Algorithm 4.1, line 15, p.5. [2015:efficient-rank1-update]
            // This is the roundUpdate from [2008:Shark] See:
            // https://git.io/JqpA7
            // Update the Cholesky factor of the covariance matrix
            xt::view(m_population.cov, idx, xt::all(), xt::all()) =
                choleskyUpdate<T>(
                    xt::view(m_population.cov, idx, xt::all(), xt::all()),
                    1.0 - m_parameters.cCov + pCUpdateWeight, m_parameters.cCov,
                    xt::row(m_population.pC, idx));
        }
    }

    SuccessNotion m_successNotion;
    std::size_t m_nParents;
    std::size_t m_nOffspring;
    std::size_t m_nIndividuals;
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
