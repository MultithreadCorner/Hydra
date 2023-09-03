// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
// Copyright (C) 2019 Rasmus Munk Larsen <rmlarsen@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HYDRA_EIGEN_GENERIC_TYPE_CASTING_H
#define HYDRA_EIGEN_GENERIC_TYPE_CASTING_H

namespace hydra_Eigen {

namespace internal {

template<>
struct scalar_cast_op<float, hydra_Eigen::half> {
  HYDRA_EIGEN_EMPTY_STRUCT_CTOR(scalar_cast_op)
  typedef hydra_Eigen::half result_type;
  HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_STRONG_INLINE hydra_Eigen::half operator() (const float& a) const {
    #if (defined(HYDRA_EIGEN_HAS_CUDA_FP16) && defined(HYDRA_EIGEN_CUDA_ARCH) && HYDRA_EIGEN_CUDA_ARCH >= 300) || \
      (defined(HYDRA_EIGEN_HAS_HIP_FP16) && defined(HYDRA_EIGEN_HIP_DEVICE_COMPILE))
      return __float2half(a);
    #else
      return hydra_Eigen::half(a);
    #endif
  }
};

template<>
struct functor_traits<scalar_cast_op<float, hydra_Eigen::half> >
{ enum { Cost = NumTraits<float>::AddCost, PacketAccess = false }; };


template<>
struct scalar_cast_op<int, hydra_Eigen::half> {
  HYDRA_EIGEN_EMPTY_STRUCT_CTOR(scalar_cast_op)
  typedef hydra_Eigen::half result_type;
  HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_STRONG_INLINE hydra_Eigen::half operator() (const int& a) const {
    #if (defined(HYDRA_EIGEN_HAS_CUDA_FP16) && defined(HYDRA_EIGEN_CUDA_ARCH) && HYDRA_EIGEN_CUDA_ARCH >= 300) || \
      (defined(HYDRA_EIGEN_HAS_HIP_FP16) && defined(HYDRA_EIGEN_HIP_DEVICE_COMPILE))
      return __float2half(static_cast<float>(a));
    #else
      return hydra_Eigen::half(static_cast<float>(a));
    #endif
  }
};

template<>
struct functor_traits<scalar_cast_op<int, hydra_Eigen::half> >
{ enum { Cost = NumTraits<float>::AddCost, PacketAccess = false }; };


template<>
struct scalar_cast_op<hydra_Eigen::half, float> {
  HYDRA_EIGEN_EMPTY_STRUCT_CTOR(scalar_cast_op)
  typedef float result_type;
  HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_STRONG_INLINE float operator() (const hydra_Eigen::half& a) const {
    #if (defined(HYDRA_EIGEN_HAS_CUDA_FP16) && defined(HYDRA_EIGEN_CUDA_ARCH) && HYDRA_EIGEN_CUDA_ARCH >= 300) || \
      (defined(HYDRA_EIGEN_HAS_HIP_FP16) && defined(HYDRA_EIGEN_HIP_DEVICE_COMPILE))
      return __half2float(a);
    #else
      return static_cast<float>(a);
    #endif
  }
};

template<>
struct functor_traits<scalar_cast_op<hydra_Eigen::half, float> >
{ enum { Cost = NumTraits<float>::AddCost, PacketAccess = false }; };


template<>
struct scalar_cast_op<float, hydra_Eigen::bfloat16> {
  HYDRA_EIGEN_EMPTY_STRUCT_CTOR(scalar_cast_op)
  typedef hydra_Eigen::bfloat16 result_type;
  HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_STRONG_INLINE hydra_Eigen::bfloat16 operator() (const float& a) const {
    return hydra_Eigen::bfloat16(a);
  }
};

template<>
struct functor_traits<scalar_cast_op<float, hydra_Eigen::bfloat16> >
{ enum { Cost = NumTraits<float>::AddCost, PacketAccess = false }; };


template<>
struct scalar_cast_op<int, hydra_Eigen::bfloat16> {
  HYDRA_EIGEN_EMPTY_STRUCT_CTOR(scalar_cast_op)
  typedef hydra_Eigen::bfloat16 result_type;
  HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_STRONG_INLINE hydra_Eigen::bfloat16 operator() (const int& a) const {
    return hydra_Eigen::bfloat16(static_cast<float>(a));
  }
};

template<>
struct functor_traits<scalar_cast_op<int, hydra_Eigen::bfloat16> >
{ enum { Cost = NumTraits<float>::AddCost, PacketAccess = false }; };


template<>
struct scalar_cast_op<hydra_Eigen::bfloat16, float> {
  HYDRA_EIGEN_EMPTY_STRUCT_CTOR(scalar_cast_op)
  typedef float result_type;
  HYDRA_EIGEN_DEVICE_FUNC HYDRA_EIGEN_STRONG_INLINE float operator() (const hydra_Eigen::bfloat16& a) const {
    return static_cast<float>(a);
  }
};

template<>
struct functor_traits<scalar_cast_op<hydra_Eigen::bfloat16, float> >
{ enum { Cost = NumTraits<float>::AddCost, PacketAccess = false }; };


}
}

#endif  // HYDRA_EIGEN_GENERIC_TYPE_CASTING_H
