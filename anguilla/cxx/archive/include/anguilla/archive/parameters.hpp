#pragma once

#ifndef ANGUILLA_ARCHIVE_PARAMETERS_HPP
#define ANGUILLA_ARCHIVE_PARAMETERS_HPP

// STL
#include <optional>

namespace archive {

template <typename T>
class Parameters {
   public:
    Parameters(
        std::size_t nDimensions, T initialStepSize, std::optional<T> d, std::optional<T> pTargetSucc, std::optional<T> cP, T pThreshold, std::optional<T> cCov, T pExtreme, T sigmaMin, T alpha, std::optional<T> cR) {
        auto n = (T)nDimensions;
        this->initialStepSize = initialStepSize;
        this->d = !d.has_value() ? 1.0 + n / 2.0 : d.value();
        this->pTargetSucc = !pTargetSucc.has_value() ? 0.5 : pTargetSucc.value();
        this->cP = !cP.has_value() ? this->pTargetSucc / (2.0 + this->pTargetSucc) : cP.value();
        this->pThreshold = pThreshold;
        this->cCov = !cCov.has_value() ? 2.0 / (std::pow(n, 2.1) + 3.0) : cCov.value();
        this->pExtreme = pExtreme;
        this->sigmaMin = sigmaMin;
        this->alpha = alpha;
        this->cR = !cR.has_value() ? this->cCov / 2.0 : cR.value();
    }

    T initialStepSize;
    T d;
    T pTargetSucc;
    T cP;
    T pThreshold;
    T cCov;
    T pExtreme;
    T sigmaMin;
    T alpha;
    T cR;
};
}  // namespace archive

#endif  // ANGUILLA_ARCHIVE_PARAMETERS_HPP
