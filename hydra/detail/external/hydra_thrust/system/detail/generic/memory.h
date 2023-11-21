/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


/*! \file generic/memory.h
 *  \brief Generic implementation of memory functions.
 *         Calling some of these is an error. They have no implementation.
 */

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/tag.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/pointer.h>
#include <hydra/detail/external/hydra_thrust/pair.h>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{

template<typename DerivedPolicy, typename Size>
__host__ __device__
void malloc(hydra_thrust::execution_policy<DerivedPolicy> &, Size);

template<typename T, typename DerivedPolicy>
__host__ __device__
hydra_thrust::pointer<T,DerivedPolicy> malloc(hydra_thrust::execution_policy<DerivedPolicy> &s, std::size_t n);

template<typename DerivedPolicy, typename Pointer>
__host__ __device__
void free(hydra_thrust::execution_policy<DerivedPolicy> &, Pointer);

template<typename Pointer1, typename Pointer2>
__host__ __device__
void assign_value(tag, Pointer1, Pointer2);

template<typename DerivedPolicy, typename Pointer>
__host__ __device__
void get_value(hydra_thrust::execution_policy<DerivedPolicy> &, Pointer);

template<typename DerivedPolicy, typename Pointer1, typename Pointer2>
__host__ __device__
void iter_swap(hydra_thrust::execution_policy<DerivedPolicy>&, Pointer1, Pointer2);

} // end generic
} // end detail
} // end system
HYDRA_THRUST_NAMESPACE_END

#include <hydra/detail/external/hydra_thrust/system/detail/generic/memory.inl>

