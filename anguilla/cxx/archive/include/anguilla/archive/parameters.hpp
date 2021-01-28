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
