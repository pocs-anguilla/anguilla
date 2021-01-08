#ifndef ANGUILLA_HYPERVOLUME_COMMON_HPP
#define ANGUILLA_HYPERVOLUME_COMMON_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <vector>
namespace py = pybind11;

template <typename T>
struct Point2D {
    Point2D(const Point2D<T> &other) : x(other.x), y(other.y) {}
    Point2D(T x = std::numeric_limits<T>::signaling_NaN(),
            T y = std::numeric_limits<T>::signaling_NaN()) : x(x), y(y) {}
    T x;
    T y;

    inline bool operator==(const Point2D<T> &other) const {
        return (x == other.x) && (y == other.y);
    }
};

template <typename T>
struct Point3D {
    Point3D(const Point3D<T> &other) : x(other.x), y(other.y), z(other.z) {}
    Point3D(T x = std::numeric_limits<T>::signaling_NaN(),
            T y = std::numeric_limits<T>::signaling_NaN(),
            T z = std::numeric_limits<T>::signaling_NaN()) : x(x), y(y), z(z) {}
    T x;
    T y;
    T z;

    inline bool operator==(const Point3D<T> &other) const {
        return (x == other.x) && (y == other.y) && (z == other.z);
    }
};

template <typename T>
using Point = py::array_t<T, py::array::c_style>;

template <typename T>
using PointList = std::vector<py::array_t<T, py::array::c_style>>;

template <typename T>
using Point2DList = std::vector<Point2D<T>>;

template <typename T>
using Point3DList = std::vector<Point3D<T>>;

namespace pybind11 {
namespace detail {
// Point2D
template <typename T>
struct type_caster<Point2D<T>> {
   public:
    PYBIND11_TYPE_CASTER(Point2D<T>, _("Point2D<T>"));

    // Python to CPP
    bool load(py::handle src, bool convert) {
        if (!convert && !py::array_t<T>::check_(src)) {
            return false;
        }

        auto buffer = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(src);
        if (!buffer) {
            return false;
        }

        // 1-D
        const auto dims = buffer.ndim();
        if (dims != 1) {
            return false;
        }
        const T *data = buffer.data();
        value = Point2D(data[0], data[1]);
        return true;
    }

    // CPP to Python
    static py::handle cast(const Point2D<T> &src,
                           py::return_value_policy policy, py::handle parent) {
        std::vector<size_t> shape(1);
        shape[0] = 2U;

        std::vector<size_t> stride(1);
        stride[0] = sizeof(T);

        py::array a(std::move(shape), std::move(stride), std::move(&(src->x)));

        return a.release();
    }
};

// Point3D
template <typename T>
struct type_caster<Point3D<T>> {
   public:
    PYBIND11_TYPE_CASTER(Point3D<T>, _("Point3D<T>"));

    // Python to CPP
    bool load(py::handle src, bool convert) {
        if (!convert && !py::array_t<T>::check_(src)) {
            return false;
        }

        auto buffer = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(src);
        if (!buffer) {
            return false;
        }

        // 1-D
        const auto dims = buffer.ndim();
        if (dims != 1) {
            return false;
        }
        const T *data = buffer.data();
        value = Point3D(data[0], data[1], data[2]);
        return true;
    }

    // CPP to Python
    static py::handle cast(const Point3D<T> &src,
                           py::return_value_policy policy, py::handle parent) {
        std::vector<size_t> shape(1);
        shape[0] = 3U;

        std::vector<size_t> stride(1);
        stride[0] = sizeof(T);

        py::array a(std::move(shape), std::move(stride), std::move(&(src->x)));

        return a.release();
    }
};
}  // namespace detail
}  // namespace pybind11

#endif  // ANGUILLA_HYPERVOLUME_COMMON_HPP
