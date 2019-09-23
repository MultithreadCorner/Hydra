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
#include <hydra/detail/external/thrust/system/detail/generic/inner_product.h>
#include <hydra/detail/external/thrust/functional.h>
#include <hydra/detail/external/thrust/detail/internal_functional.h>
#include <hydra/detail/external/thrust/transform_reduce.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputType>
__hydra_host__ __hydra_device__
OutputType inner_product(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                         InputIterator1 first1,
                         InputIterator1 last1,
                         InputIterator2 first2,
                         OutputType init)
{
  HYDRA_EXTERNAL_NS::thrust::plus<OutputType>       binary_op1;
  HYDRA_EXTERNAL_NS::thrust::multiplies<OutputType> binary_op2;
  return HYDRA_EXTERNAL_NS::thrust::inner_product(exec, first1, last1, first2, init, binary_op1, binary_op2);
} // end inner_product()


template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputType, typename BinaryFunction1, typename BinaryFunction2>
__hydra_host__ __hydra_device__
OutputType inner_product(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                         InputIterator1 first1,
                         InputIterator1 last1,
                         InputIterator2 first2,
                         OutputType init, 
                         BinaryFunction1 binary_op1,
                         BinaryFunction2 binary_op2)
{
  typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<HYDRA_EXTERNAL_NS::thrust::tuple<InputIterator1,InputIterator2> > ZipIter;

  ZipIter first = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(first1,first2));

  // only the first iterator in the tuple is relevant for the purposes of last
  ZipIter last  = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(last1, first2));

  return HYDRA_EXTERNAL_NS::thrust::transform_reduce(exec, first, last, HYDRA_EXTERNAL_NS::thrust::detail::zipped_binary_op<OutputType,BinaryFunction2>(binary_op2), init, binary_op1);
} // end inner_product()


} // end generic
} // end detail
} // end system
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust

HYDRA_EXTERNAL_NAMESPACE_END
