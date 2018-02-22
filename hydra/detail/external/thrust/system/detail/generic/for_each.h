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

/*! \file for_each.h
 *  \brief Generic implementation of for_each & for_each_n.
 *         It is an error to call these functions; they have no implementation.
 */

#pragma once

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/detail/execution_policy.h>
#include <hydra/detail/external/thrust/system/detail/generic/tag.h>
#include <hydra/detail/external/thrust/detail/static_assert.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy,
         typename InputIterator,
         typename UnaryFunction>
__host__ __device__
InputIterator for_each(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy>&,
                       InputIterator first,
                       InputIterator ,
                       UnaryFunction )
{
  // unimplemented
  HYDRA_THRUST_STATIC_ASSERT( (HYDRA_EXTERNAL_NS::thrust::detail::depend_on_instantiation<InputIterator, false>::value) );
  return first;
} // end for_each()


template<typename DerivedPolicy,
         typename InputIterator,
         typename Size,
         typename UnaryFunction>
__host__ __device__
InputIterator for_each_n(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy>&,
                         InputIterator first,
                         Size ,
                         UnaryFunction )
{
  // unimplemented
  HYDRA_THRUST_STATIC_ASSERT( (HYDRA_EXTERNAL_NS::thrust::detail::depend_on_instantiation<InputIterator, false>::value) );
  return first;
} // end for_each_n()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust

HYDRA_EXTERNAL_NAMESPACE_END
