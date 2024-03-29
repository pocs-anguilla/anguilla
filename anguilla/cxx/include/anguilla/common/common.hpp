#pragma once

#ifndef ANGUILLA_HYPERVOLUME_COMMON_HPP
#define ANGUILLA_HYPERVOLUME_COMMON_HPP

// PyBind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

// STL
#include <memory>
#include <vector>

/* Begin external snippets
 - Source 0:
 https://github.com/pybind/pybind11/issues/1042#issuecomment-508582847
 - Source 1:
 https://github.com/pybind/pybind11/issues/1042#issuecomment-642215028
 - Source 2:
 https://github.com/pybind/pybind11/issues/1042#issuecomment-509529335
*/

template <typename S> using __PyArrayOf = py::array_t<typename S::value_type>;

template <typename S>
[[nodiscard]] inline __PyArrayOf<S> to_pyarray(const S& seq) { // copies
    return py::array(static_cast<py::ssize_t>(seq.size()), seq.data());
}
template <typename S,
          typename = std::enable_if_t<std::is_rvalue_reference_v<S&&>>>
[[nodiscard]] inline __PyArrayOf<S> as_pyarray(S&& seq) { // moves
    auto size = seq.size();
    auto data = seq.data();
    std::unique_ptr<S> seqPtr = std::make_unique<S>(std::move(seq));
    auto capsule = py::capsule(seqPtr.get(), [](void* ptr) {
        std::unique_ptr<S>(static_cast<S*>(ptr));
    });
    seqPtr.release();
    return py::array(static_cast<py::ssize_t>(size), data, capsule);
}
/* End external snippets */

#endif // ANGUILLA_HYPERVOLUME_COMMON_HPP
