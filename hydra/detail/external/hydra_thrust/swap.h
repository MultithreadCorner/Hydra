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

/*! \file swap.h
 *  \brief Functions for swapping the value of elements
 */

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/execution_policy.h>

HYDRA_THRUST_NAMESPACE_BEGIN

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup swap
 *  \{
 */

/*! \p swap assigns the contents of \c a to \c b and the
 *  contents of \c b to \c a. This is used as a primitive operation
 *  by many other algorithms.
 *  
 *  \param a The first value of interest. After completion,
 *           the value of b will be returned here.
 *  \param b The second value of interest. After completion,
 *           the value of a will be returned here.
 *
 *  \tparam Assignable is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>.
 *
 *  The following code snippet demonstrates how to use \p swap to
 *  swap the contents of two variables.
 *
 *  \code
 *  #include <hydra/detail/external/hydra_thrust/swap.h>
 *  ...
 *  int x = 1;
 *  int y = 2;
 *  hydra_thrust::swap(x,h);
 *
 *  // x == 2, y == 1
 *  \endcode
 */
template<typename Assignable1, typename Assignable2>
__host__ __device__ 
inline void swap(Assignable1 &a, Assignable2 &b);

/*! \} // swap
 */

/*! \} // utility
 */


/*! \addtogroup copying
 *  \{
 */


/*! \p swap_ranges swaps each of the elements in the range <tt>[first1, last1)</tt>
 *  with the corresponding element in the range <tt>[first2, first2 + (last1 - first1))</tt>.
 *  That is, for each integer \c n such that <tt>0 <= n < (last1 - first1)</tt>, it swaps
 *  <tt>*(first1 + n)</tt> and <tt>*(first2 + n)</tt>. The return value is
 *  <tt>first2 + (last1 - first1)</tt>.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the first sequence to swap.
 *  \param last1 One position past the last element of the first sequence to swap.
 *  \param first2 The beginning of the second sequence to swap.
 *  \return An iterator pointing to one position past the last element of the second
 *          sequence to swap.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator1's \c value_type must be convertible to \p ForwardIterator2's \c value_type.
 *  \tparam ForwardIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator2's \c value_type must be convertible to \p ForwardIterator1's \c value_type.
 *
 *  \pre \p first1 may equal \p first2, but the range <tt>[first1, last1)</tt> shall not overlap the range <tt>[first2, first2 + (last1 - first1))</tt> otherwise.
 *
 *  The following code snippet demonstrates how to use \p swap_ranges to
 *  swap the contents of two \c hydra_thrust::device_vectors using the \p hydra_thrust::device execution
 *  policy for parallelization:
 *
 *  \code
 *  #include <hydra/detail/external/hydra_thrust/swap.h>
 *  #include <hydra/detail/external/hydra_thrust/device_vector.h>
 *  #include <hydra/detail/external/hydra_thrust/execution_policy.h>
 *  ...
 *  hydra_thrust::device_vector<int> v1(2), v2(2);
 *  v1[0] = 1;
 *  v1[1] = 2;
 *  v2[0] = 3;
 *  v2[1] = 4;
 *
 *  hydra_thrust::swap_ranges(hydra_thrust::device, v1.begin(), v1.end(), v2.begin());
 *
 *  // v1[0] == 3, v1[1] == 4, v2[0] == 1, v2[1] == 2
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/swap_ranges
 *  \see \c swap
 */
template<typename DerivedPolicy,
         typename ForwardIterator1,
         typename ForwardIterator2>
__host__ __device__
  ForwardIterator2 swap_ranges(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                               ForwardIterator1 first1,
                               ForwardIterator1 last1,
                               ForwardIterator2 first2);


/*! \p swap_ranges swaps each of the elements in the range <tt>[first1, last1)</tt>
 *  with the corresponding element in the range <tt>[first2, first2 + (last1 - first1))</tt>.
 *  That is, for each integer \c n such that <tt>0 <= n < (last1 - first1)</tt>, it swaps
 *  <tt>*(first1 + n)</tt> and <tt>*(first2 + n)</tt>. The return value is
 *  <tt>first2 + (last1 - first1)</tt>.
 *
 *  \param first1 The beginning of the first sequence to swap.
 *  \param last1 One position past the last element of the first sequence to swap.
 *  \param first2 The beginning of the second sequence to swap.
 *  \return An iterator pointing to one position past the last element of the second
 *          sequence to swap.
 *
 *  \tparam ForwardIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator1's \c value_type must be convertible to \p ForwardIterator2's \c value_type.
 *  \tparam ForwardIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator2's \c value_type must be convertible to \p ForwardIterator1's \c value_type.
 *
 *  \pre \p first1 may equal \p first2, but the range <tt>[first1, last1)</tt> shall not overlap the range <tt>[first2, first2 + (last1 - first1))</tt> otherwise.
 *
 *  The following code snippet demonstrates how to use \p swap_ranges to
 *  swap the contents of two \c hydra_thrust::device_vectors.
 *
 *  \code
 *  #include <hydra/detail/external/hydra_thrust/swap.h>
 *  #include <hydra/detail/external/hydra_thrust/device_vector.h>
 *  ...
 *  hydra_thrust::device_vector<int> v1(2), v2(2);
 *  v1[0] = 1;
 *  v1[1] = 2;
 *  v2[0] = 3;
 *  v2[1] = 4;
 *
 *  hydra_thrust::swap_ranges(v1.begin(), v1.end(), v2.begin());
 *
 *  // v1[0] == 3, v1[1] == 4, v2[0] == 1, v2[1] == 2
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/swap_ranges
 *  \see \c swap
 */
template<typename ForwardIterator1,
         typename ForwardIterator2>
  ForwardIterator2 swap_ranges(ForwardIterator1 first1,
                               ForwardIterator1 last1,
                               ForwardIterator2 first2);


/*! \} // copying
 */

HYDRA_THRUST_NAMESPACE_END

#include <hydra/detail/external/hydra_thrust/detail/swap.inl>
