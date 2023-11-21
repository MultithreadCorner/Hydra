// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2020, Arm Limited and Contributors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HYDRA_EIGEN_MATH_FUNCTIONS_SVE_H
#define HYDRA_EIGEN_MATH_FUNCTIONS_SVE_H

namespace hydra_Eigen {
namespace internal {

template <>
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_UNUSED PacketXf pexp<PacketXf>(const PacketXf& x) {
  return pexp_float(x);
}

template <>
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_UNUSED PacketXf plog<PacketXf>(const PacketXf& x) {
  return plog_float(x);
}

template <>
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_UNUSED PacketXf psin<PacketXf>(const PacketXf& x) {
  return psin_float(x);
}

template <>
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_UNUSED PacketXf pcos<PacketXf>(const PacketXf& x) {
  return pcos_float(x);
}

// Hyperbolic Tangent function.
template <>
HYDRA_EIGEN_STRONG_INLINE HYDRA_EIGEN_UNUSED PacketXf ptanh<PacketXf>(const PacketXf& x) {
  return internal::generic_fast_tanh_float(x);
}
}  // end namespace internal
}  // end namespace hydra_Eigen

#endif  // HYDRA_EIGEN_MATH_FUNCTIONS_SVE_H
