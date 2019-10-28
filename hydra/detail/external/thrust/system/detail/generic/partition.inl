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
#include <hydra/detail/external/thrust/system/detail/generic/partition.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/pair.h>

#include <hydra/detail/external/thrust/remove.h>
#include <hydra/detail/external/thrust/count.h>
#include <hydra/detail/external/thrust/advance.h>
#include <hydra/detail/external/thrust/partition.h>
#include <hydra/detail/external/thrust/sort.h>
#include <hydra/detail/external/thrust/iterator/transform_iterator.h>

#include <hydra/detail/external/thrust/detail/internal_functional.h>
#include <hydra/detail/external/thrust/detail/temporary_array.h>

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
         typename Predicate>
__hydra_host__ __hydra_device__
  ForwardIterator stable_partition(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                                   ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<ForwardIterator>::value_type InputType;

  // copy input to temp buffer
  HYDRA_EXTERNAL_NS::thrust::detail::temporary_array<InputType,DerivedPolicy> temp(exec, first, last);

  // count the size of the true partition
  typename HYDRA_EXTERNAL_NS::thrust::iterator_difference<ForwardIterator>::type num_true = HYDRA_EXTERNAL_NS::thrust::count_if(exec, first,last,pred);

  // point to the beginning of the false partition
  ForwardIterator out_false = first;
  HYDRA_EXTERNAL_NS::thrust::advance(out_false, num_true);

  return HYDRA_EXTERNAL_NS::thrust::stable_partition_copy(exec, temp.begin(), temp.end(), first, out_false, pred).first;
} // end stable_partition()


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
__hydra_host__ __hydra_device__
  ForwardIterator stable_partition(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                                   ForwardIterator first,
                                   ForwardIterator last,
                                   InputIterator stencil,
                                   Predicate pred)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<ForwardIterator>::value_type InputType;

  // copy input to temp buffer
  HYDRA_EXTERNAL_NS::thrust::detail::temporary_array<InputType,DerivedPolicy> temp(exec, first, last);

  // count the size of the true partition
  InputIterator stencil_last = stencil;
  HYDRA_EXTERNAL_NS::thrust::advance(stencil_last, temp.size());
  typename HYDRA_EXTERNAL_NS::thrust::iterator_difference<InputIterator>::type num_true = HYDRA_EXTERNAL_NS::thrust::count_if(exec, stencil, stencil_last, pred);

  // point to the beginning of the false partition
  ForwardIterator out_false = first;
  HYDRA_EXTERNAL_NS::thrust::advance(out_false, num_true);

  return HYDRA_EXTERNAL_NS::thrust::stable_partition_copy(exec, temp.begin(), temp.end(), stencil, first, out_false, pred).first;
} // end stable_partition()


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
__hydra_host__ __hydra_device__
  HYDRA_EXTERNAL_NS::thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                          InputIterator first,
                          InputIterator last,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred)
{
  HYDRA_EXTERNAL_NS::thrust::detail::unary_negate<Predicate> not_pred(pred);

  // remove_copy_if the true partition to out_true
  OutputIterator1 end_of_true_partition = HYDRA_EXTERNAL_NS::thrust::remove_copy_if(exec, first, last, out_true, not_pred);

  // remove_copy_if the false partition to out_false
  OutputIterator2 end_of_false_partition = HYDRA_EXTERNAL_NS::thrust::remove_copy_if(exec, first, last, out_false, pred);

  return HYDRA_EXTERNAL_NS::thrust::make_pair(end_of_true_partition, end_of_false_partition);
} // end stable_partition_copy()


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
__hydra_host__ __hydra_device__
  HYDRA_EXTERNAL_NS::thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                          InputIterator1 first,
                          InputIterator1 last,
                          InputIterator2 stencil,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred)
{
  HYDRA_EXTERNAL_NS::thrust::detail::unary_negate<Predicate> not_pred(pred);

  // remove_copy_if the true partition to out_true
  OutputIterator1 end_of_true_partition = HYDRA_EXTERNAL_NS::thrust::remove_copy_if(exec, first, last, stencil, out_true, not_pred);

  // remove_copy_if the false partition to out_false
  OutputIterator2 end_of_false_partition = HYDRA_EXTERNAL_NS::thrust::remove_copy_if(exec, first, last, stencil, out_false, pred);

  return HYDRA_EXTERNAL_NS::thrust::make_pair(end_of_true_partition, end_of_false_partition);
} // end stable_partition_copy()


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Predicate>
__hydra_host__ __hydra_device__
  ForwardIterator partition(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  return HYDRA_EXTERNAL_NS::thrust::stable_partition(exec, first, last, pred);
} // end partition()


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
__hydra_host__ __hydra_device__
  ForwardIterator partition(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred)
{
  return HYDRA_EXTERNAL_NS::thrust::stable_partition(exec, first, last, stencil, pred);
} // end partition()


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
__hydra_host__ __hydra_device__
  HYDRA_EXTERNAL_NS::thrust::pair<OutputIterator1,OutputIterator2>
    partition_copy(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                   InputIterator first,
                   InputIterator last,
                   OutputIterator1 out_true,
                   OutputIterator2 out_false,
                   Predicate pred)
{
  return HYDRA_EXTERNAL_NS::thrust::stable_partition_copy(exec,first,last,out_true,out_false,pred);
} // end partition_copy()


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
__hydra_host__ __hydra_device__
  HYDRA_EXTERNAL_NS::thrust::pair<OutputIterator1,OutputIterator2>
    partition_copy(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                   InputIterator1 first,
                   InputIterator1 last,
                   InputIterator2 stencil,
                   OutputIterator1 out_true,
                   OutputIterator2 out_false,
                   Predicate pred)
{
  return HYDRA_EXTERNAL_NS::thrust::stable_partition_copy(exec,first,last,stencil,out_true,out_false,pred);
} // end partition_copy()


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Predicate>
__hydra_host__ __hydra_device__
  ForwardIterator partition_point(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                                  ForwardIterator first,
                                  ForwardIterator last,
                                  Predicate pred)
{
  return HYDRA_EXTERNAL_NS::thrust::find_if_not(exec, first, last, pred);
} // end partition_point()


template<typename DerivedPolicy,
         typename InputIterator,
         typename Predicate>
__hydra_host__ __hydra_device__
  bool is_partitioned(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                      InputIterator first,
                      InputIterator last,
                      Predicate pred)
{
  return HYDRA_EXTERNAL_NS::thrust::is_sorted(exec,
                           HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(first, HYDRA_EXTERNAL_NS::thrust::detail::not1(pred)),
                           HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(last,  HYDRA_EXTERNAL_NS::thrust::detail::not1(pred)));
} // end is_partitioned()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust

HYDRA_EXTERNAL_NAMESPACE_END
