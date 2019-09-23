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


/*! \file sort.inl
 *  \brief Inline file for sort.h.
 */

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/sort.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/thrust/system/detail/generic/sort.h>
#include <hydra/detail/external/thrust/system/detail/adl/sort.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename RandomAccessIterator>
__hydra_host__ __hydra_device__
  void sort(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
            RandomAccessIterator first,
            RandomAccessIterator last)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::sort;
  return sort(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), first, last);
} // end sort()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__hydra_host__ __hydra_device__
  void sort(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
            RandomAccessIterator first,
            RandomAccessIterator last,
            StrictWeakOrdering comp)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::sort;
  return sort(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), first, last, comp);
} // end sort()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename RandomAccessIterator>
__hydra_host__ __hydra_device__
  void stable_sort(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   RandomAccessIterator first,
                   RandomAccessIterator last)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::stable_sort;
  return stable_sort(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), first, last);
} // end stable_sort()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__hydra_host__ __hydra_device__
  void stable_sort(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   RandomAccessIterator first,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::stable_sort;
  return stable_sort(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), first, last, comp);
} // end stable_sort()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__hydra_host__ __hydra_device__
  void sort_by_key(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::sort_by_key;
  return sort_by_key(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), keys_first, keys_last, values_first);
} // end sort_by_key()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__hydra_host__ __hydra_device__
  void sort_by_key(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first,
                   StrictWeakOrdering comp)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::sort_by_key;
  return sort_by_key(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), keys_first, keys_last, values_first, comp);
} // end sort_by_key()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__hydra_host__ __hydra_device__
  void stable_sort_by_key(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                          RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::stable_sort_by_key;
  return stable_sort_by_key(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), keys_first, keys_last, values_first);
} // end stable_sort_by_key()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__hydra_host__ __hydra_device__
  void stable_sort_by_key(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                          RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first,
                          StrictWeakOrdering comp)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::stable_sort_by_key;
  return stable_sort_by_key(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), keys_first, keys_last, values_first, comp);
} // end stable_sort_by_key()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator>
__hydra_host__ __hydra_device__
  bool is_sorted(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                 ForwardIterator first,
                 ForwardIterator last)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::is_sorted;
  return is_sorted(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), first, last);
} // end is_sorted()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator, typename Compare>
__hydra_host__ __hydra_device__
  bool is_sorted(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                 ForwardIterator first,
                 ForwardIterator last,
                 Compare comp)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::is_sorted;
  return is_sorted(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), first, last, comp);
} // end is_sorted()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator>
__hydra_host__ __hydra_device__
  ForwardIterator is_sorted_until(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                  ForwardIterator first,
                                  ForwardIterator last)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::is_sorted_until;
  return is_sorted_until(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), first, last);
} // end is_sorted_until()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator, typename Compare>
__hydra_host__ __hydra_device__
  ForwardIterator is_sorted_until(const HYDRA_EXTERNAL_NS::thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                  ForwardIterator first,
                                  ForwardIterator last,
                                  Compare comp)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::is_sorted_until;
  return is_sorted_until(HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(HYDRA_EXTERNAL_NS::thrust::detail::strip_const(exec)), first, last, comp);
} // end is_sorted_until()


///////////////
// Key Sorts //
///////////////

template<typename RandomAccessIterator>
  void sort(RandomAccessIterator first,
            RandomAccessIterator last)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;

  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<RandomAccessIterator>::type System;

  System system;

  return HYDRA_EXTERNAL_NS::thrust::sort(select_system(system), first, last);
} // end sort()


template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  __hydra_host__ __hydra_device__
  void sort(RandomAccessIterator first,
            RandomAccessIterator last,
            StrictWeakOrdering comp)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;

  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<RandomAccessIterator>::type System;

  System system;

  return HYDRA_EXTERNAL_NS::thrust::sort(select_system(system), first, last, comp);
} // end sort()


template<typename RandomAccessIterator>
  void stable_sort(RandomAccessIterator first,
                   RandomAccessIterator last)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;

  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<RandomAccessIterator>::type System;

  System system;

  return HYDRA_EXTERNAL_NS::thrust::stable_sort(select_system(system), first, last);
} // end stable_sort() 


template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_sort(RandomAccessIterator first,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;

  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<RandomAccessIterator>::type System;

  System system;

  return HYDRA_EXTERNAL_NS::thrust::stable_sort(select_system(system), first, last, comp);
} // end stable_sort()



/////////////////////
// Key-Value Sorts //
/////////////////////

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void sort_by_key(RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;

  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<RandomAccessIterator1>::type System1;
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<RandomAccessIterator2>::type System2;

  System1 system1;
  System2 system2;

  return HYDRA_EXTERNAL_NS::thrust::sort_by_key(select_system(system1,system2), keys_first, keys_last, values_first);
} // end sort_by_key()


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void sort_by_key(RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first,
                   StrictWeakOrdering comp)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;

  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<RandomAccessIterator1>::type System1;
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<RandomAccessIterator2>::type System2;

  System1 system1;
  System2 system2;

  return HYDRA_EXTERNAL_NS::thrust::sort_by_key(select_system(system1,system2), keys_first, keys_last, values_first, comp);
} // end sort_by_key()


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void stable_sort_by_key(RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;

  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<RandomAccessIterator1>::type System1;
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<RandomAccessIterator2>::type System2;

  System1 system1;
  System2 system2;

  return HYDRA_EXTERNAL_NS::thrust::stable_sort_by_key(select_system(system1,system2), keys_first, keys_last, values_first);
} // end stable_sort_by_key()


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_sort_by_key(RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first,
                          StrictWeakOrdering comp)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;

  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<RandomAccessIterator1>::type System1;
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<RandomAccessIterator2>::type System2;

  System1 system1;
  System2 system2;

  return HYDRA_EXTERNAL_NS::thrust::stable_sort_by_key(select_system(system1,system2), keys_first, keys_last, values_first, comp);
} // end stable_sort_by_key()


template<typename ForwardIterator>
  bool is_sorted(ForwardIterator first,
                 ForwardIterator last)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
  
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return HYDRA_EXTERNAL_NS::thrust::is_sorted(select_system(system), first, last);
} // end is_sorted()


template<typename ForwardIterator,
         typename Compare>
  bool is_sorted(ForwardIterator first,
                 ForwardIterator last,
                 Compare comp)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
  
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return HYDRA_EXTERNAL_NS::thrust::is_sorted(select_system(system), first, last, comp);
} // end is_sorted()


template<typename ForwardIterator>
  ForwardIterator is_sorted_until(ForwardIterator first,
                                  ForwardIterator last)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
  
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return HYDRA_EXTERNAL_NS::thrust::is_sorted_until(select_system(system), first, last);
} // end is_sorted_until()


template<typename ForwardIterator,
         typename Compare>
  ForwardIterator is_sorted_until(ForwardIterator first,
                                  ForwardIterator last,
                                  Compare comp)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
  
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return HYDRA_EXTERNAL_NS::thrust::is_sorted_until(select_system(system), first, last, comp);
} // end is_sorted_until()


} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust


HYDRA_EXTERNAL_NAMESPACE_END
