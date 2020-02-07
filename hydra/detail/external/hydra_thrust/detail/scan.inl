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


/*! \file scan.inl
 *  \brief Inline file for scan.h.
 */

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/scan.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/scan.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/scan_by_key.h>
#include <hydra/detail/external/hydra_thrust/system/detail/adl/scan.h>
#include <hydra/detail/external/hydra_thrust/system/detail/adl/scan_by_key.h>

namespace hydra_thrust
{


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator>
__host__ __device__
  OutputIterator inclusive_scan(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result)
{
  using hydra_thrust::system::detail::generic::inclusive_scan;
  return inclusive_scan(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first, last, result);
} // end inclusive_scan() 


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename AssociativeOperator>
__host__ __device__
  OutputIterator inclusive_scan(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                AssociativeOperator binary_op)
{
  using hydra_thrust::system::detail::generic::inclusive_scan;
  return inclusive_scan(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first, last, result, binary_op);
} // end inclusive_scan()


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator>
__host__ __device__
  OutputIterator exclusive_scan(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result)
{
  using hydra_thrust::system::detail::generic::exclusive_scan;
  return exclusive_scan(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first, last, result);
} // end exclusive_scan()


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename T>
__host__ __device__
  OutputIterator exclusive_scan(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init)
{
  using hydra_thrust::system::detail::generic::exclusive_scan;
  return exclusive_scan(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first, last, result, init);
} // end exclusive_scan()


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator>
__host__ __device__
  OutputIterator exclusive_scan(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init,
                                AssociativeOperator binary_op)
{
  using hydra_thrust::system::detail::generic::exclusive_scan;
  return exclusive_scan(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first, last, result, init, binary_op);
} // end exclusive_scan()


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
__host__ __device__
  OutputIterator inclusive_scan_by_key(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result)
{
  using hydra_thrust::system::detail::generic::inclusive_scan_by_key;
  return inclusive_scan_by_key(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first1, last1, first2, result);
} // end inclusive_scan_by_key()


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate>
__host__ __device__
  OutputIterator inclusive_scan_by_key(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryPredicate binary_pred)
{
  using hydra_thrust::system::detail::generic::inclusive_scan_by_key;
  return inclusive_scan_by_key(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first1, last1, first2, result, binary_pred);
} // end inclusive_scan_by_key()


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate,
         typename AssociativeOperator>
__host__ __device__
  OutputIterator inclusive_scan_by_key(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryPredicate binary_pred,
                                       AssociativeOperator binary_op)
{
  using hydra_thrust::system::detail::generic::inclusive_scan_by_key;
  return inclusive_scan_by_key(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first1, last1, first2, result, binary_pred, binary_op);
} // end inclusive_scan_by_key()


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
__host__ __device__
  OutputIterator exclusive_scan_by_key(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result)
{
  using hydra_thrust::system::detail::generic::exclusive_scan_by_key;
  return exclusive_scan_by_key(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first1, last1, first2, result);
} // end exclusive_scan_by_key()


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T>
__host__ __device__
  OutputIterator exclusive_scan_by_key(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init)
{
  using hydra_thrust::system::detail::generic::exclusive_scan_by_key;
  return exclusive_scan_by_key(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first1, last1, first2, result, init);
} // end exclusive_scan_by_key()


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate>
__host__ __device__
  OutputIterator exclusive_scan_by_key(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init,
                                       BinaryPredicate binary_pred)
{
  using hydra_thrust::system::detail::generic::exclusive_scan_by_key;
  return exclusive_scan_by_key(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first1, last1, first2, result, init, binary_pred);
} // end exclusive_scan_by_key()


__hydra_thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate,
         typename AssociativeOperator>
__host__ __device__
  OutputIterator exclusive_scan_by_key(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init,
                                       BinaryPredicate binary_pred,
                                       AssociativeOperator binary_op)
{
  using hydra_thrust::system::detail::generic::exclusive_scan_by_key;
  return exclusive_scan_by_key(hydra_thrust::detail::derived_cast(hydra_thrust::detail::strip_const(exec)), first1, last1, first2, result, init, binary_pred, binary_op);
} // end exclusive_scan_by_key()


template<typename InputIterator,
         typename OutputIterator>
  OutputIterator inclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<InputIterator>::type  System1;
  typedef typename hydra_thrust::iterator_system<OutputIterator>::type System2;

  System1 system1;
  System2 system2;

  return hydra_thrust::inclusive_scan(select_system(system1,system2), first, last, result);
} // end inclusive_scan()


template<typename InputIterator,
         typename OutputIterator,
         typename BinaryFunction>
  OutputIterator inclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                BinaryFunction binary_op)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<InputIterator>::type  System1;
  typedef typename hydra_thrust::iterator_system<OutputIterator>::type System2;

  System1 system1;
  System2 system2;

  return hydra_thrust::inclusive_scan(select_system(system1,system2), first, last, result, binary_op);
} // end inclusive_scan()


template<typename InputIterator,
         typename OutputIterator>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<InputIterator>::type  System1;
  typedef typename hydra_thrust::iterator_system<OutputIterator>::type System2;

  System1 system1;
  System2 system2;

  return hydra_thrust::exclusive_scan(select_system(system1,system2), first, last, result);
} // end exclusive_scan()


template<typename InputIterator,
         typename OutputIterator,
         typename T>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<InputIterator>::type  System1;
  typedef typename hydra_thrust::iterator_system<OutputIterator>::type System2;

  System1 system1;
  System2 system2;

  return hydra_thrust::exclusive_scan(select_system(system1,system2), first, last, result, init);
} // end exclusive_scan()


template<typename InputIterator,
         typename OutputIterator,
         typename T,
         typename BinaryFunction>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init,
                                BinaryFunction binary_op)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<InputIterator>::type  System1;
  typedef typename hydra_thrust::iterator_system<OutputIterator>::type System2;

  System1 system1;
  System2 system2;

  return hydra_thrust::exclusive_scan(select_system(system1,system2), first, last, result, init, binary_op);
} // end exclusive_scan()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator inclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<InputIterator1>::type System1;
  typedef typename hydra_thrust::iterator_system<InputIterator2>::type System2;
  typedef typename hydra_thrust::iterator_system<OutputIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return hydra_thrust::inclusive_scan_by_key(select_system(system1,system2,system3), first1, last1, first2, result);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate>
  OutputIterator inclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryPredicate binary_pred)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<InputIterator1>::type System1;
  typedef typename hydra_thrust::iterator_system<InputIterator2>::type System2;
  typedef typename hydra_thrust::iterator_system<OutputIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return hydra_thrust::inclusive_scan_by_key(select_system(system1,system2,system3), first1, last1, first2, result, binary_pred);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate,
         typename AssociativeOperator>
  OutputIterator inclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryPredicate binary_pred,
                                       AssociativeOperator binary_op)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<InputIterator1>::type System1;
  typedef typename hydra_thrust::iterator_system<InputIterator2>::type System2;
  typedef typename hydra_thrust::iterator_system<OutputIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return hydra_thrust::inclusive_scan_by_key(select_system(system1,system2,system3), first1, last1, first2, result, binary_pred, binary_op);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
  OutputIterator exclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<InputIterator1>::type System1;
  typedef typename hydra_thrust::iterator_system<InputIterator2>::type System2;
  typedef typename hydra_thrust::iterator_system<OutputIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return hydra_thrust::exclusive_scan_by_key(select_system(system1,system2,system3), first1, last1, first2, result);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T>
  OutputIterator exclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<InputIterator1>::type System1;
  typedef typename hydra_thrust::iterator_system<InputIterator2>::type System2;
  typedef typename hydra_thrust::iterator_system<OutputIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return hydra_thrust::exclusive_scan_by_key(select_system(system1,system2,system3), first1, last1, first2, result, init);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate>
  OutputIterator exclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init,
                                       BinaryPredicate binary_pred)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<InputIterator1>::type System1;
  typedef typename hydra_thrust::iterator_system<InputIterator2>::type System2;
  typedef typename hydra_thrust::iterator_system<OutputIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return hydra_thrust::exclusive_scan_by_key(select_system(system1,system2,system3), first1, last1, first2, result, init, binary_pred);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate,
         typename AssociativeOperator>
  OutputIterator exclusive_scan_by_key(InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init,
                                       BinaryPredicate binary_pred,
                                       AssociativeOperator binary_op)
{
  using hydra_thrust::system::detail::generic::select_system;

  typedef typename hydra_thrust::iterator_system<InputIterator1>::type System1;
  typedef typename hydra_thrust::iterator_system<InputIterator2>::type System2;
  typedef typename hydra_thrust::iterator_system<OutputIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return hydra_thrust::exclusive_scan_by_key(select_system(system1,system2,system3), first1, last1, first2, result, init, binary_pred, binary_op);
}


} // end namespace hydra_thrust

