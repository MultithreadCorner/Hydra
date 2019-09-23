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
#include <hydra/detail/external/thrust/system/detail/generic/mismatch.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/detail/internal_functional.h>
#include <hydra/detail/external/thrust/find.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2>
__hydra_host__ __hydra_device__
  HYDRA_EXTERNAL_NS::thrust::pair<InputIterator1, InputIterator2>
    mismatch(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
             InputIterator1 first1,
             InputIterator1 last1,
             InputIterator2 first2)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_value<InputIterator1>::type InputType1;
  
  // XXX use a placeholder expression here
  return HYDRA_EXTERNAL_NS::thrust::mismatch(exec, first1, last1, first2, HYDRA_EXTERNAL_NS::thrust::detail::equal_to<InputType1>());
} // end mismatch()


template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
__hydra_host__ __hydra_device__
  HYDRA_EXTERNAL_NS::thrust::pair<InputIterator1, InputIterator2>
    mismatch(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
             InputIterator1 first1,
             InputIterator1 last1,
             InputIterator2 first2,
             BinaryPredicate pred)
{
  // Contributed by Erich Elsen
  typedef HYDRA_EXTERNAL_NS::thrust::tuple<InputIterator1,InputIterator2> IteratorTuple;
  typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<IteratorTuple>          ZipIterator;
  
  ZipIterator zipped_first = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(first1,first2));
  ZipIterator zipped_last  = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(last1, first2));
  
  ZipIterator result = HYDRA_EXTERNAL_NS::thrust::find_if_not(exec, zipped_first, zipped_last, HYDRA_EXTERNAL_NS::thrust::detail::tuple_binary_predicate<BinaryPredicate>(pred));
  
  return HYDRA_EXTERNAL_NS::thrust::make_pair(HYDRA_EXTERNAL_NS::thrust::get<0>(result.get_iterator_tuple()),
                           HYDRA_EXTERNAL_NS::thrust::get<1>(result.get_iterator_tuple()));
} // end mismatch()


} // end generic
} // end detail
} // end system
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust

HYDRA_EXTERNAL_NAMESPACE_END
