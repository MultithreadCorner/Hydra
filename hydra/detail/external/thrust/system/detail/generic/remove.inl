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


/*! \file remove.inl
 *  \brief Inline file for remove.h
 */

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/system/detail/generic/remove.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/detail/copy_if.h>
#include <hydra/detail/external/thrust/detail/internal_functional.h>
#include <hydra/detail/external/thrust/detail/temporary_array.h>
#include <hydra/detail/external/thrust/remove.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename T>
__hydra_host__ __hydra_device__
  ForwardIterator remove(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                         ForwardIterator first,
                         ForwardIterator last,
                         const T &value)
{
  HYDRA_EXTERNAL_NS::thrust::detail::equal_to_value<T> pred(value);

  // XXX consider using a placeholder here
  return HYDRA_EXTERNAL_NS::thrust::remove_if(exec, first, last, pred);
} // end remove()


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename T>
__hydra_host__ __hydra_device__
  OutputIterator remove_copy(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                             InputIterator first,
                             InputIterator last,
                             OutputIterator result,
                             const T &value)
{
  HYDRA_EXTERNAL_NS::thrust::detail::equal_to_value<T> pred(value);

  // XXX consider using a placeholder here
  return HYDRA_EXTERNAL_NS::thrust::remove_copy_if(exec, first, last, result, pred);
} // end remove_copy()


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Predicate>
__hydra_host__ __hydra_device__
  ForwardIterator remove_if(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<ForwardIterator>::value_type InputType;

  // create temporary storage for an intermediate result
  HYDRA_EXTERNAL_NS::thrust::detail::temporary_array<InputType,DerivedPolicy> temp(exec, first, last);

  // remove into temp
  return HYDRA_EXTERNAL_NS::thrust::remove_copy_if(exec, temp.begin(), temp.end(), temp.begin(), first, pred);
} // end remove_if()


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
__hydra_host__ __hydra_device__
  ForwardIterator remove_if(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<ForwardIterator>::value_type InputType;

  // create temporary storage for an intermediate result
  HYDRA_EXTERNAL_NS::thrust::detail::temporary_array<InputType,DerivedPolicy> temp(exec, first, last);

  // remove into temp
  return HYDRA_EXTERNAL_NS::thrust::remove_copy_if(exec, temp.begin(), temp.end(), stencil, first, pred);
} // end remove_if() 


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename Predicate>
__hydra_host__ __hydra_device__
  OutputIterator remove_copy_if(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                Predicate pred)
{
  return HYDRA_EXTERNAL_NS::thrust::remove_copy_if(exec, first, last, first, result, pred);
} // end remove_copy_if()


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
__hydra_host__ __hydra_device__
  OutputIterator remove_copy_if(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                                InputIterator1 first,
                                InputIterator1 last,
                                InputIterator2 stencil,
                                OutputIterator result,
                                Predicate pred)
{
  return HYDRA_EXTERNAL_NS::thrust::copy_if(exec, first, last, stencil, result, HYDRA_EXTERNAL_NS::thrust::detail::not1(pred));
} // end remove_copy_if()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust

HYDRA_EXTERNAL_NAMESPACE_END
