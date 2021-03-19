#pragma once

#include <optional>
#ifndef ANGUILLA_HYPERVOLUME_HYPERVOLUME_HPP_
#define ANGUILLA_HYPERVOLUME_HYPERVOLUME_HPP_

// PyBind11
#include <pybind11/stl.h>

// Xtensor
#include <xtensor/xtensor.hpp>

// Anguilla
#include <anguilla/hypervolume/hv2d.hpp>
#include <anguilla/hypervolume/hv3d.hpp>
#include <anguilla/hypervolume/hvc2d.hpp>
#include <anguilla/hypervolume/hvc3d.hpp>
#include <anguilla/hypervolume/hvkd.hpp>
namespace ag = anguilla;

namespace anguilla {
namespace hypervolume {

/* Public interface */
template <typename T>
T calculate(const xt::xtensor<T, 2U>& inputPoints,
            const std::optional<xt::xtensor<T, 1U>>& inputReference,
            const bool ignoreDominated = false);

template <typename T>
[[nodiscard]] auto contributions(const xt::xtensor<T, 2U>& inputPoints)
    -> xt::xtensor<T, 1U>;

/* Implementations */
template <typename T>
T calculate(const xt::xtensor<T, 2U>& inputPoints,
            const std::optional<xt::xtensor<T, 1U>>& inputReference,
            const bool ignoreDominated) {

    const std::size_t d = inputPoints.shape(1U);
    if (d < 2U) {
        throw std::runtime_error("HV not defined for 1-D or less");
    }
    switch (d) {
    case 2U:
        return ag::hv2d::calculate<T>(inputPoints, inputReference,
                                      ignoreDominated);
    case 3U:
        return ag::hv3d::calculate<T, ag::hv3d::BTreeMap<T>>(
            inputPoints, inputReference, ignoreDominated);
    default:
        return ag::hvkd::calculate<T>(inputPoints, inputReference,
                                      ignoreDominated);
    }
}

template <typename T>
[[nodiscard]] auto contributions(const xt::xtensor<T, 2U>& inputPoints)
    -> xt::xtensor<T, 1> {
    const std::size_t d = inputPoints.shape(1U);
    if (d < 2U) {
        throw std::runtime_error(
            "HV contributions not defined for 1-D or less");
    }
    switch (d) {
    case 2U:
        return ag::hvc2d::contributions<T>(inputPoints);
    case 3U:
        return ag::hvc3d::contributions<T, ag::hvc3d::BTreeMap<T>>(
            inputPoints, std::nullopt, true);
    default:
        throw std::runtime_error("Not implemented");
    }
}

} // namespace hypervolume
} // namespace anguilla

#endif // ANGUILLA_HYPERVOLUME_HYPERVOLUME_HPP_
