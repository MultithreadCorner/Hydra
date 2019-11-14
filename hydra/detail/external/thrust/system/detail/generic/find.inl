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
#include <hydra/detail/external/thrust/find.h>
#include <hydra/detail/external/thrust/reduce.h>

#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/detail/external/thrust/detail/minmax.h>
#include <hydra/detail/external/thrust/iterator/counting_iterator.h>
#include <hydra/detail/external/thrust/iterator/transform_iterator.h>
#include <hydra/detail/external/thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/thrust/detail/internal_functional.h>


// Contributed by Erich Elsen

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy, typename InputIterator, typename T>
__hydra_host__ __hydra_device__
InputIterator find(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                   InputIterator first,
                   InputIterator last,
                   const T& value)
{
  // XXX consider a placeholder expression here
  return HYDRA_EXTERNAL_NS::thrust::find_if(exec, first, last, HYDRA_EXTERNAL_NS::thrust::detail::equal_to_value<T>(value));
} // end find()


template<typename TupleType>
struct find_if_functor
{
  __hydra_host__ __hydra_device__
  TupleType operator()(const TupleType& lhs, const TupleType& rhs) const
  {
    // select the smallest index among true results
    if(HYDRA_EXTERNAL_NS::thrust::get<0>(lhs) && HYDRA_EXTERNAL_NS::thrust::get<0>(rhs))
    {
      return TupleType(true, (HYDRA_EXTERNAL_NS::thrust::min)(HYDRA_EXTERNAL_NS::thrust::get<1>(lhs), HYDRA_EXTERNAL_NS::thrust::get<1>(rhs)));
    }
    else if(HYDRA_EXTERNAL_NS::thrust::get<0>(lhs))
    {
      return lhs;
    }
    else
    {
      return rhs;
    }
  }
};
    

template<typename DerivedPolicy, typename InputIterator, typename Predicate>
__hydra_host__ __hydra_device__
InputIterator find_if(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                      InputIterator first,
                      InputIterator last,
                      Predicate pred)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<InputIterator>::difference_type difference_type;
  typedef typename HYDRA_EXTERNAL_NS::thrust::tuple<bool,difference_type> result_type;
  
  // empty sequence
  if(first == last) return last;
  
  const difference_type n = HYDRA_EXTERNAL_NS::thrust::distance(first, last);
  
  // this implementation breaks up the sequence into separate intervals
  // in an attempt to early-out as soon as a value is found
  
  // TODO incorporate sizeof(InputType) into interval_threshold and round to multiple of 32
  const difference_type interval_threshold = 1 << 20;
  const difference_type interval_size = (HYDRA_EXTERNAL_NS::thrust::min)(interval_threshold, n);
  
  // force transform_iterator output to bool
  typedef HYDRA_EXTERNAL_NS::thrust::transform_iterator<Predicate, InputIterator, bool> XfrmIterator;
  typedef HYDRA_EXTERNAL_NS::thrust::tuple<XfrmIterator, HYDRA_EXTERNAL_NS::thrust::counting_iterator<difference_type> > IteratorTuple;
  typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<IteratorTuple> ZipIterator;
  
  IteratorTuple iter_tuple = HYDRA_EXTERNAL_NS::thrust::make_tuple(XfrmIterator(first, pred),
                                                HYDRA_EXTERNAL_NS::thrust::counting_iterator<difference_type>(0));
  
  ZipIterator begin = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(iter_tuple);
  ZipIterator end   = begin + n;
  
  for(ZipIterator interval_begin = begin; interval_begin < end; interval_begin += interval_size)
  {
    ZipIterator interval_end = interval_begin + interval_size;
    if(end < interval_end)
    {
      interval_end = end;
    } // end if
    
    result_type result = HYDRA_EXTERNAL_NS::thrust::reduce(exec,
                                        interval_begin, interval_end,
                                        result_type(false,interval_end - begin),
                                        find_if_functor<result_type>());
    
    // see if we found something
    if(HYDRA_EXTERNAL_NS::thrust::get<0>(result))
    {
      return first + HYDRA_EXTERNAL_NS::thrust::get<1>(result);
    }
  }
  
  //nothing was found if we reach here...
  return first + n;
}


template<typename DerivedPolicy, typename InputIterator, typename Predicate>
__hydra_host__ __hydra_device__
InputIterator find_if_not(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec,
                          InputIterator first,
                          InputIterator last,
                          Predicate pred)
{
  return HYDRA_EXTERNAL_NS::thrust::find_if(exec, first, last, HYDRA_EXTERNAL_NS::thrust::detail::not1(pred));
} // end find()


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust

HYDRA_EXTERNAL_NAMESPACE_END
