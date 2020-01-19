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


#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/tag.h>

namespace hydra_thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename UnaryOperation>
__host__ __device__
  void tabulate(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                ForwardIterator first,
                ForwardIterator last,
                UnaryOperation unary_op);


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace hydra_thrust

#include <hydra/detail/external/hydra_thrust/system/detail/generic/tabulate.inl>
