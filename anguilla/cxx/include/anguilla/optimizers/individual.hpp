#pragma once

#ifndef ANGUILLA_OPTIMIZERS_INDIVIDUAL_HPP_
#define ANGUILLA_OPTIMIZERS_INDIVIDUAL_HPP_

// PyBind11
#include <pybind11/stl.h>

// Xtensor
#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor.hpp>

// STL
#include <cstddef>

namespace anguilla {
namespace optimizers {

template <typename T> struct Individual {
    Individual(
        xt::xtensor<T, 1U> point, xt::xtensor<T, 1U> fitness,
        std::optional<xt::xtensor<T, 1U>> penalizedFitness = std::nullopt,
        T stepSize = 1.0, T pSucc = 0.5,
        std::optional<xt::xtensor<T, 1U>> z = std::nullopt,
        std::size_t parentIndex = 0U)
        : point(point), fitness(fitness),
          penalizedFitness(penalizedFitness.value_or(fitness)),
          stepSize(stepSize), pSucc(pSucc),
          z(z.value_or(xt::zeros<T>(point.shape()))), parentIndex(parentIndex),
          rank(0U), contribution(0.0), selected(false) {
        // cov is initialized to be the identity matrix
        cov = xt::eye(point.shape(0U));
    }

    Individual(const Individual<T>&) = default;

    auto& getCov() { return cov; }

    // Search point
    const xt::xtensor<T, 1U> point;
    // Fitness
    const xt::xtensor<T, 1U> fitness;
    // Penalized fitness
    const xt::xtensor<T, 1U> penalizedFitness;
    // Covariance matrix
    xt::xtensor<T, 2U> cov;
    // Step size
    T stepSize;
    // Smooth probability of success
    T pSucc;
    // The random sample ~N(0,1) used to create the search point through
    // mutation.
    xt::xtensor<T, 1U> z;
    // Parent index
    std::size_t parentIndex;
    // Non-dominated rank
    std::size_t rank;
    // Contribution
    T contribution;
    // Selected for next generation
    bool selected;
};

} // namespace optimizers
} // namespace anguilla

#endif // ANGUILLA_OPTIMIZERS_INDIVIDUAL_HPP_
