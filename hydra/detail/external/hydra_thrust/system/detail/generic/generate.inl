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
#include <hydra/detail/external/hydra_thrust/system/detail/generic/generate.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/internal_functional.h>
#include <hydra/detail/external/hydra_thrust/for_each.h>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{

template<typename ExecutionPolicy,
         typename ForwardIterator,
         typename Generator>
__host__ __device__
  void generate(hydra_thrust::execution_policy<ExecutionPolicy> &exec,
                ForwardIterator first,
                ForwardIterator last,
                Generator gen)
{
  // this static assert is necessary due to a workaround in generate_functor
  // it takes a const reference to accept temporaries from proxy iterators
  // and then const_casts the constness away
  //
  // this had the weird side effect of allowing generate (and fill, and whatever
  // else is implemented in terms of generate) to fill through const iterators.
  // this might become unnecessary once Thrust is C++11-and-above only, since the
  // other solution is to take an rvalue reference in a second overload of
  // operator() of the function object, but until we support pre-11, this is a
  // nice solution that validates the const_cast and doesn't take away any
  // functionality.
  HYDRA_THRUST_STATIC_ASSERT_MSG(
    !hydra_thrust::detail::is_const<
      typename hydra_thrust::detail::remove_reference<
        typename hydra_thrust::iterator_traits<ForwardIterator>::reference
      >::type
    >::value
  , "generating to `const` iterators is not allowed"
  );
  hydra_thrust::for_each(exec, first, last, typename hydra_thrust::detail::generate_functor<ExecutionPolicy,Generator>::type(gen));
} // end generate()

template<typename ExecutionPolicy,
         typename OutputIterator,
         typename Size,
         typename Generator>
__host__ __device__
  OutputIterator generate_n(hydra_thrust::execution_policy<ExecutionPolicy> &exec,
                            OutputIterator first,
                            Size n,
                            Generator gen)
{
  // this static assert is necessary due to a workaround in generate_functor
  // it takes a const reference to accept temporaries from proxy iterators
  // and then const_casts the constness away
  //
  // this had the weird side effect of allowing generate (and fill, and whatever
  // else is implemented in terms of generate) to fill through const iterators.
  // this might become unnecessary once Thrust is C++11-and-above only, since the
  // other solution is to take an rvalue reference in a second overload of
  // operator() of the function object, but until we support pre-11, this is a
  // nice solution that validates the const_cast and doesn't take away any
  // functionality.
  HYDRA_THRUST_STATIC_ASSERT_MSG(
    !hydra_thrust::detail::is_const<
      typename hydra_thrust::detail::remove_reference<
        typename hydra_thrust::iterator_traits<OutputIterator>::reference
      >::type
    >::value
  , "generating to `const` iterators is not allowed"
  );
  return hydra_thrust::for_each_n(exec, first, n, typename hydra_thrust::detail::generate_functor<ExecutionPolicy,Generator>::type(gen));
} // end generate()

} // end namespace generic
} // end namespace detail
} // end namespace system
HYDRA_THRUST_NAMESPACE_END

