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

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/sequence.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/tabulate.h>

namespace hydra_thrust
{
namespace system
{
namespace detail
{
namespace generic
{
namespace sequence_detail
{


template<typename T>
struct sequence_functor
{
  T init, step;

  __host__ __device__
  sequence_functor(T init, T step)
    : init(init), step(step)
  {}

  template<typename Index>
  __host__ __device__
  T operator()(Index i) const
  {
    return static_cast<T>(init + step * i);
  }
};


} // end sequence_detail


template<typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
  void sequence(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                ForwardIterator first,
                ForwardIterator last)
{
  typedef typename hydra_thrust::iterator_traits<ForwardIterator>::value_type T;

  hydra_thrust::sequence(exec, first, last, T(0));
} // end sequence()


template<typename DerivedPolicy, typename ForwardIterator, typename T>
__host__ __device__
  void sequence(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                ForwardIterator first,
                ForwardIterator last,
                T init)
{
  hydra_thrust::sequence(exec, first, last, init, T(1));
} // end sequence()


template<typename DerivedPolicy, typename ForwardIterator, typename T>
__host__ __device__
  void sequence(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                ForwardIterator first,
                ForwardIterator last,
                T init,
                T step)
{
  // XXX TODO use a placeholder expression here
  hydra_thrust::tabulate(exec, first, last, sequence_detail::sequence_functor<T>(init, step));
} // end sequence()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace hydra_thrust

