#ifndef PYSHARK_LINALG_BASE_H
#define PYSHARK_LINALG_BASE_H

// Inspired on:
// https://github.com/tdegeus/pybind11_examples/blob/master/09_numpy_cpp-custom-matrix/pybind_matrix.h

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <shark/LinAlg/Base.h>

template <typename T>
using PySharkMatrix = shark::blas::matrix<T, shark::blas::row_major>;

template <typename T> using PySharkVector = shark::blas::vector<T>;

template <typename T> using PySharkVectorList = std::vector<PySharkVector<T>>;

namespace py = pybind11;

namespace pybind11 {
namespace detail {

template <typename T> struct type_caster<PySharkVector<T>> {
  public:
    PYBIND11_TYPE_CASTER(PySharkVector<T>, _("PySharkVector"));

    // Python to CPP
    bool load(py::handle src, bool convert) {
        if (!convert && !py::array_t<T>::check_(src)) {
            return false;
        }

        auto buffer =
            py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(
                src);
        if (!buffer) {
            return false;
        }

        // 1-D
        const auto dims = buffer.ndim();
        if (dims != 1) {
            return false;
        }
        const auto size = buffer.shape()[0];
        constexpr std::size_t stride = 1U;

        value =
            remora::dense_vector_adaptor<T>((T*)buffer.data(), size, stride);

        return true;
    }

    // CPP to Python
    static py::handle cast(const PySharkVector<T>& src,
                           py::return_value_policy policy, py::handle parent) {
        auto raw = src.raw_storage();

        std::vector<size_t> shape(1);
        shape[0] = src.size();

        std::vector<size_t> stride(1);
        stride[0] = raw[1];

        py::array a(std::move(shape), std::move(stride), std::move(raw[0]));

        return a.release();
    }
};

template <typename T> struct type_caster<PySharkMatrix<T>> {
  public:
    PYBIND11_TYPE_CASTER(PySharkMatrix<T>, _("PySharkMatrix"));

    // Python to CPP
    bool load(py::handle src, bool convert) {
        if (!convert && !py::array_t<T>::check_(src)) {
            return false;
        }

        auto buffer =
            py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(
                src);
        if (!buffer) {
            return false;
        }

        // 2-D matrices only
        const auto dims = buffer.ndim();
        if (dims != 2) {
            return false;
        }

        const auto size1 = buffer.shape()[0];
        const auto size2 = buffer.shape()[1];
        constexpr std::size_t leading_dimension = 0U;
        value = remora::dense_matrix_adaptor<T>((T*)buffer.data(), size1, size2,
                                                leading_dimension);

        return true;
    }

    // CPP to Python
    static py::handle cast(const PySharkMatrix<T>& src,
                           py::return_value_policy policy, py::handle parent) {
        auto raw = src.raw_storage();

        std::vector<size_t> shape(2);
        shape[0] = src.size1();
        shape[1] = src.size2();

        std::vector<size_t> stride(2);
        stride[0] = 1U;
        stride[1] = 1U;

        py::array a(std::move(shape), std::move(stride), std::move(raw[0]));

        return a.release();
    }
};
} // namespace detail
} // namespace pybind11
#endif
