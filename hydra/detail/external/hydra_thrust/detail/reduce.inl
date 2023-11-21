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

#include <hydra/detail/external/hydra_thrust/reduce.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/reduce.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/reduce_by_key.h>
#include <hydra/detail/external/hydra_thrust/system/detail/adl/reduce.h>
#include <hydra/detail/external/hydra_thrust/system/detail/adl/reduce_by_key.h>

HYDRA_THRUST_NAMESPACE_BEGIN


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator>
__host__ __device__
  typename hydra_thrust::iterator_traits<InputIterator>::value_type
    reduce(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec, InputIterator first, InputIterator last)
{
  using hydra_thrust::system::detail::generic::reduce;
  return reduce(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first, last);
} // end reduce()


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename T>
__host__ __device__
  T reduce(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
           InputIterator first,
           InputIterator last,
           T init)
{
  using hydra_thrust::system::detail::generic::reduce;
  return reduce(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first, last, init);
} // end reduce()


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename T,
         typename BinaryFunction>
__host__ __device__
  T reduce(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
           InputIterator first,
           InputIterator last,
           T init,
           BinaryFunction binary_op)
{
  using hydra_thrust::system::detail::generic::reduce;
  return reduce(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first, last, init, binary_op);
} // end reduce()


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
__host__ __device__
  hydra_thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                InputIterator1 keys_first,
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output)
{
  using hydra_thrust::system::detail::generic::reduce_by_key;
  return reduce_by_key(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), keys_first, keys_last, values_first, keys_output, values_output);
} // end reduce_by_key()


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
__host__ __device__
  hydra_thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                InputIterator1 keys_first,
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred)
{
  using hydra_thrust::system::detail::generic::reduce_by_key;
  return reduce_by_key(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), keys_first, keys_last, values_first, keys_output, values_output, binary_pred);
} // end reduce_by_key()


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
__host__ __device__
  hydra_thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                InputIterator1 keys_first,
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred,
                BinaryFunction binary_op)
{
  using hydra_thrust::system::detail::generic::reduce_by_key;
  return reduce_by_key(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
} // end reduce_by_key()


template<typename InputIterator>
typename hydra_thrust::iterator_traits<InputIterator>::value_type
  reduce(InputIterator first,
         InputIterator last)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<InputIterator>::type System;

  System system;

  return hydra_thrust::reduce(select_system(system), first, last);
}


template<typename InputIterator,
         typename T>
   T reduce(InputIterator first,
            InputIterator last,
            T init)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<InputIterator>::type System;

  System system;

  return hydra_thrust::reduce(select_system(system), first, last, init);
}


template<typename InputIterator,
         typename T,
         typename BinaryFunction>
   T reduce(InputIterator first,
            InputIterator last,
            T init,
            BinaryFunction binary_op)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<InputIterator>::type System;

  System system;

  return hydra_thrust::reduce(select_system(system), first, last, init, binary_op);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
  hydra_thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(InputIterator1 keys_first,
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename hydra_thrust::iterator_system<InputIterator2>::type  System2;
  typedef typename hydra_thrust::iterator_system<OutputIterator1>::type System3;
  typedef typename hydra_thrust::iterator_system<OutputIterator2>::type System4;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return hydra_thrust::reduce_by_key(select_system(system1,system2,system3,system4), keys_first, keys_last, values_first, keys_output, values_output);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
  hydra_thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(InputIterator1 keys_first,
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename hydra_thrust::iterator_system<InputIterator2>::type  System2;
  typedef typename hydra_thrust::iterator_system<OutputIterator1>::type System3;
  typedef typename hydra_thrust::iterator_system<OutputIterator2>::type System4;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return hydra_thrust::reduce_by_key(select_system(system1,system2,system3,system4), keys_first, keys_last, values_first, keys_output, values_output, binary_pred);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
  hydra_thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(InputIterator1 keys_first,
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred,
                BinaryFunction binary_op)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename hydra_thrust::iterator_system<InputIterator2>::type  System2;
  typedef typename hydra_thrust::iterator_system<OutputIterator1>::type System3;
  typedef typename hydra_thrust::iterator_system<OutputIterator2>::type System4;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return hydra_thrust::reduce_by_key(select_system(system1,system2,system3,system4), keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
}


HYDRA_THRUST_NAMESPACE_END

