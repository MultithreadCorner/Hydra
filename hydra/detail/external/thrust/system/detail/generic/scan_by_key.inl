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
#include <hydra/detail/external/thrust/system/detail/generic/scan_by_key.h>
#include <hydra/detail/external/thrust/functional.h>
#include <hydra/detail/external/thrust/transform.h>
#include <hydra/detail/external/thrust/replace.h>
#include <hydra/detail/external/thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/detail/temporary_array.h>
#include <hydra/detail/external/thrust/detail/internal_functional.h>
#include <hydra/detail/external/thrust/scan.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{
namespace detail
{


template <typename OutputType, typename HeadFlagType, typename AssociativeOperator>
struct segmented_scan_functor
{
  AssociativeOperator binary_op;
  
  typedef typename HYDRA_EXTERNAL_NS::thrust::tuple<OutputType, HeadFlagType> result_type;
  
  __hydra_host__ __hydra_device__
  segmented_scan_functor(AssociativeOperator _binary_op) : binary_op(_binary_op) {}
  
  __hydra_host__ __hydra_device__
  result_type operator()(result_type a, result_type b)
  {
    return result_type(HYDRA_EXTERNAL_NS::thrust::get<1>(b) ? HYDRA_EXTERNAL_NS::thrust::get<0>(b) : binary_op(HYDRA_EXTERNAL_NS::thrust::get<0>(a), HYDRA_EXTERNAL_NS::thrust::get<0>(b)),
                       HYDRA_EXTERNAL_NS::thrust::get<1>(a) | HYDRA_EXTERNAL_NS::thrust::get<1>(b));
  }
};


} // end namespace detail


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
__hydra_host__ __hydra_device__
  OutputIterator inclusive_scan_by_key(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<InputIterator1>::value_type InputType1;
  return HYDRA_EXTERNAL_NS::thrust::inclusive_scan_by_key(exec, first1, last1, first2, result, HYDRA_EXTERNAL_NS::thrust::equal_to<InputType1>());
}


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate>
__hydra_host__ __hydra_device__
  OutputIterator inclusive_scan_by_key(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryPredicate binary_pred)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<OutputIterator>::value_type OutputType;
  return HYDRA_EXTERNAL_NS::thrust::inclusive_scan_by_key(exec, first1, last1, first2, result, binary_pred, HYDRA_EXTERNAL_NS::thrust::plus<OutputType>());
}


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate,
         typename AssociativeOperator>
__hydra_host__ __hydra_device__
  OutputIterator inclusive_scan_by_key(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryPredicate binary_pred,
                                       AssociativeOperator binary_op)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<OutputIterator>::value_type OutputType;
  typedef unsigned int HeadFlagType;

  const size_t n = last1 - first1;

  if(n != 0)
  {
    // compute head flags
    HYDRA_EXTERNAL_NS::thrust::detail::temporary_array<HeadFlagType,DerivedPolicy> flags(exec, n);
    flags[0] = 1; HYDRA_EXTERNAL_NS::thrust::transform(exec, first1, last1 - 1, first1 + 1, flags.begin() + 1, HYDRA_EXTERNAL_NS::thrust::detail::not2(binary_pred));

    // scan key-flag tuples, 
    // For additional details refer to Section 2 of the following paper
    //    S. Sengupta, M. Harris, and M. Garland. "Efficient parallel scan algorithms for GPUs"
    //    NVIDIA Technical Report NVR-2008-003, December 2008
    //    http://mgarland.org/files/papers/nvr-2008-003.pdf
    HYDRA_EXTERNAL_NS::thrust::inclusive_scan(exec,
                           HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(first2, flags.begin())),
                           HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(first2, flags.begin())) + n,
                           HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(result, flags.begin())),
                           detail::segmented_scan_functor<OutputType, HeadFlagType, AssociativeOperator>(binary_op));
  }

  return result + n;
}


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
__hydra_host__ __hydra_device__
  OutputIterator exclusive_scan_by_key(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<OutputIterator>::value_type OutputType;
  return HYDRA_EXTERNAL_NS::thrust::exclusive_scan_by_key(exec, first1, last1, first2, result, OutputType(0));
}


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T>
__hydra_host__ __hydra_device__
  OutputIterator exclusive_scan_by_key(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<InputIterator1>::value_type InputType1;
  return HYDRA_EXTERNAL_NS::thrust::exclusive_scan_by_key(exec, first1, last1, first2, result, init, HYDRA_EXTERNAL_NS::thrust::equal_to<InputType1>());
}


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate>
__hydra_host__ __hydra_device__
  OutputIterator exclusive_scan_by_key(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init,
                                       BinaryPredicate binary_pred)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<OutputIterator>::value_type OutputType;
  return HYDRA_EXTERNAL_NS::thrust::exclusive_scan_by_key(exec, first1, last1, first2, result, init, binary_pred, HYDRA_EXTERNAL_NS::thrust::plus<OutputType>());
}


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate,
         typename AssociativeOperator>
__hydra_host__ __hydra_device__
  OutputIterator exclusive_scan_by_key(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init,
                                       BinaryPredicate binary_pred,
                                       AssociativeOperator binary_op)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<OutputIterator>::value_type OutputType;
  typedef unsigned int HeadFlagType;

  const size_t n = last1 - first1;

  if(n != 0)
  {
    InputIterator2 last2 = first2 + n;

    // compute head flags
    HYDRA_EXTERNAL_NS::thrust::detail::temporary_array<HeadFlagType,DerivedPolicy> flags(exec, n);
    flags[0] = 1; HYDRA_EXTERNAL_NS::thrust::transform(exec, first1, last1 - 1, first1 + 1, flags.begin() + 1, HYDRA_EXTERNAL_NS::thrust::detail::not2(binary_pred));

    // shift input one to the right and initialize segments with init
    HYDRA_EXTERNAL_NS::thrust::detail::temporary_array<OutputType,DerivedPolicy> temp(exec, n);
    HYDRA_EXTERNAL_NS::thrust::replace_copy_if(exec, first2, last2 - 1, flags.begin() + 1, temp.begin() + 1, HYDRA_EXTERNAL_NS::thrust::negate<HeadFlagType>(), init);
    temp[0] = init;

    // scan key-flag tuples, 
    // For additional details refer to Section 2 of the following paper
    //    S. Sengupta, M. Harris, and M. Garland. "Efficient parallel scan algorithms for GPUs"
    //    NVIDIA Technical Report NVR-2008-003, December 2008
    //    http://mgarland.org/files/papers/nvr-2008-003.pdf
    HYDRA_EXTERNAL_NS::thrust::inclusive_scan(exec,
                           HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(temp.begin(), flags.begin())),
                           HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(temp.begin(), flags.begin())) + n,
                           HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::make_tuple(result,       flags.begin())),
                           detail::segmented_scan_functor<OutputType, HeadFlagType, AssociativeOperator>(binary_op));
  }

  return result + n;
}


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust

HYDRA_EXTERNAL_NAMESPACE_END
