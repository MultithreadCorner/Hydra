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


/*! \file tabulate.h
 *  \brief Fills a range with the tabulation of a function
 */

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/execution_policy.h>

namespace hydra_thrust
{


/*! \addtogroup transformations
 *  \{
 */


/*! \p tabulate fills the range <tt>[first, last)</tt> with the value of a function applied to each
 *     element's index.
 *
 *  For each iterator \c i in the range <tt>[first, last)</tt>, \p tabulate performs the assignment
 *  <tt>*i = unary_op(i - first)</tt>.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the range.
 *  \param last The end of the range.
 *  \param unary_op The unary operation to apply.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="http://www.sgi.com/tech/stl/ForwardIterator.html">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable,
 *          and if \c x and \c y are objects of \c ForwardIterator's \c value_type, then <tt>x + y</tt> is defined,
 *          and if \c T is \p ForwardIterator's \c value_type, then <tt>T(0)</tt> is defined.
 *  \tparam UnaryOperation is a model of <a href="http://www.sgi.com/tech/stl/UnaryFunction.html">Unary Function</a>
 *                         and \c UnaryFunction's \c result_type is convertible to \c OutputIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p tabulate to generate the first \c n non-positive integers
 *  using the \p hydra_thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <hydra/detail/external/hydra_thrust/tabulate.h>
 *  #include <hydra/detail/external/hydra_thrust/functional.h>
 *  #include <hydra/detail/external/hydra_thrust/execution_policy.h>
 *  ...
 *  const int N = 10;
 *  int A[N];
 *  hydra_thrust::tabulate(hydra_thrust::host, A, A + 10, hydra_thrust::negate<int>());
 *  // A is now {0, -1, -2, -3, -4, -5, -6, -7, -8, -9}
 *  \endcode
 *
 *  \see hydra_thrust::fill
 *  \see hydra_thrust::generate
 *  \see hydra_thrust::sequence
 */
template<typename DerivedPolicy, typename ForwardIterator, typename UnaryOperation>
__host__ __device__
  void tabulate(const hydra_thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                ForwardIterator first,
                ForwardIterator last,
                UnaryOperation unary_op);


/*! \p tabulate fills the range <tt>[first, last)</tt> with the value of a function applied to each
 *     element's index.
 *
 *  For each iterator \c i in the range <tt>[first, last)</tt>, \p tabulate performs the assignment
 *  <tt>*i = unary_op(i - first)</tt>.
 *
 *  \param first The beginning of the range.
 *  \param last The end of the range.
 *  \param unary_op The unary operation to apply.
 *
 *  \tparam ForwardIterator is a model of <a href="http://www.sgi.com/tech/stl/ForwardIterator.html">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable,
 *          and if \c x and \c y are objects of \c ForwardIterator's \c value_type, then <tt>x + y</tt> is defined,
 *          and if \c T is \p ForwardIterator's \c value_type, then <tt>T(0)</tt> is defined.
 *  \tparam UnaryOperation is a model of <a href="http://www.sgi.com/tech/stl/UnaryFunction.html">Unary Function</a>
 *                         and \c UnaryFunction's \c result_type is convertible to \c OutputIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p tabulate to generate the first \c n non-positive integers:
 *
 *  \code
 *  #include <hydra/detail/external/hydra_thrust/tabulate.h>
 *  #include <hydra/detail/external/hydra_thrust/functional.h>
 *  ...
 *  const int N = 10;
 *  int A[N];
 *  hydra_thrust::tabulate(A, A + 10, hydra_thrust::negate<int>());
 *  // A is now {0, -1, -2, -3, -4, -5, -6, -7, -8, -9}
 *  \endcode
 *
 *  \see hydra_thrust::fill
 *  \see hydra_thrust::generate
 *  \see hydra_thrust::sequence
 */
template<typename ForwardIterator, typename UnaryOperation>
  void tabulate(ForwardIterator first,
                ForwardIterator last,
                UnaryOperation unary_op);


/*! \} // end transformations
 */


} // end namespace hydra_thrust

#include <hydra/detail/external/hydra_thrust/detail/tabulate.inl>

