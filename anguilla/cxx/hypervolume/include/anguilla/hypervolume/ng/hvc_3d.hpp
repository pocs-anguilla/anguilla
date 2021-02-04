#pragma once

#ifndef ANGUILLA_HYPERVOLUME_NG_HVC_3D_HPP
#define ANGUILLA_HYPERVOLUME_NG_HVC_3D_HPP

// Boost Intrusive
#include <boost/intrusive/any_hook.hpp>
#include <boost/intrusive/avl_set.hpp>
#include <boost/intrusive/avltree_algorithms.hpp>

namespace anguilla {
namespace hv {

template <typename T>
struct IndexedPoint3D : public boost::intrusive::avl_set_base_hook<boost::intrusive::optimize_size<true>> {
    T x;
    T y;
};

};  // namespace hv
};  // namespace anguilla

#endif  // ANGUILLA_HYPERVOLUME_NG_HVC_3D_HPP
