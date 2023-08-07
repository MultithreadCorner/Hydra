// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/cpp14_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2014

#include <hydra/detail/external/hydra_thrust/system/cuda/pointer.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/execution_policy.h>

HYDRA_THRUST_NAMESPACE_BEGIN

namespace system { namespace cuda
{

struct ready_event;

template <typename T>
struct ready_future;

struct unique_eager_event;

template <typename T>
struct unique_eager_future;

template <typename... Events>
__host__
unique_eager_event when_all(Events&&... evs);

}} // namespace system::cuda

namespace cuda
{

using hydra_thrust::system::cuda::ready_event;

using hydra_thrust::system::cuda::ready_future;

using hydra_thrust::system::cuda::unique_eager_event;
using event = unique_eager_event;

using hydra_thrust::system::cuda::unique_eager_future;
template <typename T> using future = unique_eager_future<T>;

using hydra_thrust::system::cuda::when_all;

} // namespace cuda

template <typename DerivedPolicy>
__host__ 
hydra_thrust::cuda::unique_eager_event
unique_eager_event_type(
  hydra_thrust::cuda::execution_policy<DerivedPolicy> const&
) noexcept;

template <typename T, typename DerivedPolicy>
__host__ 
hydra_thrust::cuda::unique_eager_future<T>
unique_eager_future_type(
  hydra_thrust::cuda::execution_policy<DerivedPolicy> const&
) noexcept;

HYDRA_THRUST_NAMESPACE_END

#include <hydra/detail/external/hydra_thrust/system/cuda/detail/future.inl>

#endif // C++14

