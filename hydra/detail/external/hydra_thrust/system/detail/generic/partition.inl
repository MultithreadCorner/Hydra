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
#include <hydra/detail/external/hydra_thrust/system/detail/generic/partition.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/pair.h>

#include <hydra/detail/external/hydra_thrust/remove.h>
#include <hydra/detail/external/hydra_thrust/count.h>
#include <hydra/detail/external/hydra_thrust/advance.h>
#include <hydra/detail/external/hydra_thrust/partition.h>
#include <hydra/detail/external/hydra_thrust/sort.h>
#include <hydra/detail/external/hydra_thrust/iterator/transform_iterator.h>

#include <hydra/detail/external/hydra_thrust/detail/internal_functional.h>
#include <hydra/detail/external/hydra_thrust/detail/temporary_array.h>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator stable_partition(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                                   ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred)
{
  typedef typename hydra_thrust::iterator_traits<ForwardIterator>::value_type InputType;

  // copy input to temp buffer
  hydra_thrust::detail::temporary_array<InputType,DerivedPolicy> temp(exec, first, last);

  // count the size of the true partition
  typename hydra_thrust::iterator_difference<ForwardIterator>::type num_true = hydra_thrust::count_if(exec, first,last,pred);

  // point to the beginning of the false partition
  ForwardIterator out_false = first;
  hydra_thrust::advance(out_false, num_true);

  return hydra_thrust::stable_partition_copy(exec, temp.begin(), temp.end(), first, out_false, pred).first;
} // end stable_partition()


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator stable_partition(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                                   ForwardIterator first,
                                   ForwardIterator last,
                                   InputIterator stencil,
                                   Predicate pred)
{
  typedef typename hydra_thrust::iterator_traits<ForwardIterator>::value_type InputType;

  // copy input to temp buffer
  hydra_thrust::detail::temporary_array<InputType,DerivedPolicy> temp(exec, first, last);

  // count the size of the true partition
  InputIterator stencil_last = stencil;
  hydra_thrust::advance(stencil_last, temp.size());
  typename hydra_thrust::iterator_difference<InputIterator>::type num_true = hydra_thrust::count_if(exec, stencil, stencil_last, pred);

  // point to the beginning of the false partition
  ForwardIterator out_false = first;
  hydra_thrust::advance(out_false, num_true);

  return hydra_thrust::stable_partition_copy(exec, temp.begin(), temp.end(), stencil, first, out_false, pred).first;
} // end stable_partition()


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
__host__ __device__
  hydra_thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                          InputIterator first,
                          InputIterator last,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred)
{
  hydra_thrust::detail::unary_negate<Predicate> not_pred(pred);

  // remove_copy_if the true partition to out_true
  OutputIterator1 end_of_true_partition = hydra_thrust::remove_copy_if(exec, first, last, out_true, not_pred);

  // remove_copy_if the false partition to out_false
  OutputIterator2 end_of_false_partition = hydra_thrust::remove_copy_if(exec, first, last, out_false, pred);

  return hydra_thrust::make_pair(end_of_true_partition, end_of_false_partition);
} // end stable_partition_copy()


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
__host__ __device__
  hydra_thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                          InputIterator1 first,
                          InputIterator1 last,
                          InputIterator2 stencil,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred)
{
  hydra_thrust::detail::unary_negate<Predicate> not_pred(pred);

  // remove_copy_if the true partition to out_true
  OutputIterator1 end_of_true_partition = hydra_thrust::remove_copy_if(exec, first, last, stencil, out_true, not_pred);

  // remove_copy_if the false partition to out_false
  OutputIterator2 end_of_false_partition = hydra_thrust::remove_copy_if(exec, first, last, stencil, out_false, pred);

  return hydra_thrust::make_pair(end_of_true_partition, end_of_false_partition);
} // end stable_partition_copy()


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator partition(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  return hydra_thrust::stable_partition(exec, first, last, pred);
} // end partition()


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator partition(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred)
{
  return hydra_thrust::stable_partition(exec, first, last, stencil, pred);
} // end partition()


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
__host__ __device__
  hydra_thrust::pair<OutputIterator1,OutputIterator2>
    partition_copy(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                   InputIterator first,
                   InputIterator last,
                   OutputIterator1 out_true,
                   OutputIterator2 out_false,
                   Predicate pred)
{
  return hydra_thrust::stable_partition_copy(exec,first,last,out_true,out_false,pred);
} // end partition_copy()


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
__host__ __device__
  hydra_thrust::pair<OutputIterator1,OutputIterator2>
    partition_copy(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                   InputIterator1 first,
                   InputIterator1 last,
                   InputIterator2 stencil,
                   OutputIterator1 out_true,
                   OutputIterator2 out_false,
                   Predicate pred)
{
  return hydra_thrust::stable_partition_copy(exec,first,last,stencil,out_true,out_false,pred);
} // end partition_copy()


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator partition_point(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                                  ForwardIterator first,
                                  ForwardIterator last,
                                  Predicate pred)
{
  return hydra_thrust::find_if_not(exec, first, last, pred);
} // end partition_point()


template<typename DerivedPolicy,
         typename InputIterator,
         typename Predicate>
__host__ __device__
  bool is_partitioned(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                      InputIterator first,
                      InputIterator last,
                      Predicate pred)
{
  return hydra_thrust::is_sorted(exec,
                           hydra_thrust::make_transform_iterator(first, hydra_thrust::detail::not1(pred)),
                           hydra_thrust::make_transform_iterator(last,  hydra_thrust::detail::not1(pred)));
} // end is_partitioned()


} // end namespace generic
} // end namespace detail
} // end namespace system
HYDRA_THRUST_NAMESPACE_END

