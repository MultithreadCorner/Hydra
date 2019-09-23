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

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/system/detail/generic/uninitialized_fill.h>
#include <hydra/detail/external/thrust/fill.h>
#include <hydra/detail/external/thrust/detail/internal_functional.h>
#include <hydra/detail/external/thrust/detail/type_traits.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{
namespace detail
{

template<typename DerivedPolicy,
         typename ForwardIterator,
         typename T>
__hydra_host__ __hydra_device__
  void uninitialized_fill(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                          ForwardIterator first,
                          ForwardIterator last,
                          const T &x,
                          HYDRA_EXTERNAL_NS::thrust::detail::true_type) // has_trivial_copy_constructor
{
  HYDRA_EXTERNAL_NS::thrust::fill(exec, first, last, x);
} // end uninitialized_fill()

template<typename DerivedPolicy,
         typename ForwardIterator,
         typename T>
__hydra_host__ __hydra_device__
  void uninitialized_fill(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                          ForwardIterator first,
                          ForwardIterator last,
                          const T &x,
                          HYDRA_EXTERNAL_NS::thrust::detail::false_type) // has_trivial_copy_constructor
{
  typedef typename iterator_traits<ForwardIterator>::value_type ValueType;

  HYDRA_EXTERNAL_NS::thrust::for_each(exec, first, last, HYDRA_EXTERNAL_NS::thrust::detail::uninitialized_fill_functor<ValueType>(x));
} // end uninitialized_fill()

template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Size,
         typename T>
__hydra_host__ __hydra_device__
  ForwardIterator uninitialized_fill_n(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                                       ForwardIterator first,
                                       Size n,
                                       const T &x,
                                       HYDRA_EXTERNAL_NS::thrust::detail::true_type) // has_trivial_copy_constructor
{
  return HYDRA_EXTERNAL_NS::thrust::fill_n(exec, first, n, x);
} // end uninitialized_fill()

template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Size,
         typename T>
__hydra_host__ __hydra_device__
  ForwardIterator uninitialized_fill_n(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                                       ForwardIterator first,
                                       Size n,
                                       const T &x,
                                       HYDRA_EXTERNAL_NS::thrust::detail::false_type) // has_trivial_copy_constructor
{
  typedef typename iterator_traits<ForwardIterator>::value_type ValueType;

  return HYDRA_EXTERNAL_NS::thrust::for_each_n(exec, first, n, HYDRA_EXTERNAL_NS::thrust::detail::uninitialized_fill_functor<ValueType>(x));
} // end uninitialized_fill()

} // end detail

template<typename DerivedPolicy,
         typename ForwardIterator,
         typename T>
__hydra_host__ __hydra_device__
  void uninitialized_fill(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                          ForwardIterator first,
                          ForwardIterator last,
                          const T &x)
{
  typedef typename iterator_traits<ForwardIterator>::value_type ValueType;

  typedef HYDRA_EXTERNAL_NS::thrust::detail::has_trivial_copy_constructor<ValueType> ValueTypeHasTrivialCopyConstructor;

  HYDRA_EXTERNAL_NS::thrust::system::detail::generic::detail::uninitialized_fill(exec, first, last, x,
    ValueTypeHasTrivialCopyConstructor());
} // end uninitialized_fill()

template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Size,
         typename T>
__hydra_host__ __hydra_device__
  ForwardIterator uninitialized_fill_n(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                                       ForwardIterator first,
                                       Size n,
                                       const T &x)
{
  typedef typename iterator_traits<ForwardIterator>::value_type ValueType;

  typedef HYDRA_EXTERNAL_NS::thrust::detail::has_trivial_copy_constructor<ValueType> ValueTypeHasTrivialCopyConstructor;

  return HYDRA_EXTERNAL_NS::thrust::system::detail::generic::detail::uninitialized_fill_n(exec, first, n, x,
    ValueTypeHasTrivialCopyConstructor());
} // end uninitialized_fill()

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust

HYDRA_EXTERNAL_NAMESPACE_END
