// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#pragma once

#include <limits>

#include <hydra/detail/external/thrust/detail/type_traits.h>

HYDRA_THRUST_BEGIN_NS

template <typename T>
struct numeric_limits : std::numeric_limits<T> {};

HYDRA_THRUST_END_NS

