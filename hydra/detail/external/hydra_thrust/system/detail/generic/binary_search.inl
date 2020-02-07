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


/*! \file binary_search.inl
 *  \brief Inline file for binary_search.h
 */

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/distance.h>
#include <hydra/detail/external/hydra_thrust/binary_search.h>
#include <hydra/detail/external/hydra_thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/binary_search.h>

#include <hydra/detail/external/hydra_thrust/for_each.h>
#include <hydra/detail/external/hydra_thrust/detail/function.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/scalar/binary_search.h>

#include <hydra/detail/external/hydra_thrust/detail/temporary_array.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

namespace hydra_thrust
{
namespace detail
{


// XXX WAR circular #inclusion with this forward declaration
template<typename,typename> class temporary_array;


} // end detail
namespace system
{
namespace detail
{
namespace generic
{
namespace detail
{


// short names to avoid nvcc bug
struct lbf
{
  template<typename RandomAccessIterator, typename T, typename StrictWeakOrdering>
  __host__ __device__
  typename hydra_thrust::iterator_traits<RandomAccessIterator>::difference_type
    operator()(RandomAccessIterator begin, RandomAccessIterator end, const T& value, StrictWeakOrdering comp)
  {
    return hydra_thrust::system::detail::generic::scalar::lower_bound(begin, end, value, comp) - begin;
  }
};


struct ubf
{
  template<typename RandomAccessIterator, typename T, typename StrictWeakOrdering>
  __host__ __device__
  typename hydra_thrust::iterator_traits<RandomAccessIterator>::difference_type
    operator()(RandomAccessIterator begin, RandomAccessIterator end, const T& value, StrictWeakOrdering comp)
  {
    return hydra_thrust::system::detail::generic::scalar::upper_bound(begin, end, value, comp) - begin;
  }
};


struct bsf
{
  template<typename RandomAccessIterator, typename T, typename StrictWeakOrdering>
  __host__ __device__
  bool operator()(RandomAccessIterator begin, RandomAccessIterator end, const T& value, StrictWeakOrdering comp)
  {
    RandomAccessIterator iter = hydra_thrust::system::detail::generic::scalar::lower_bound(begin, end, value, comp);
    
    hydra_thrust::detail::wrapped_function<StrictWeakOrdering,bool> wrapped_comp(comp);
    
    return iter != end && !wrapped_comp(value, *iter);
  }
};


template<typename ForwardIterator, typename StrictWeakOrdering, typename BinarySearchFunction>
struct binary_search_functor
{
  ForwardIterator begin;
  ForwardIterator end;
  StrictWeakOrdering comp;
  BinarySearchFunction func;
  
  __host__ __device__
  binary_search_functor(ForwardIterator begin, ForwardIterator end, StrictWeakOrdering comp, BinarySearchFunction func)
    : begin(begin), end(end), comp(comp), func(func) {}
  
  template<typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    hydra_thrust::get<1>(t) = func(begin, end, hydra_thrust::get<0>(t), comp);
  }
}; // binary_search_functor


// Vector Implementation
template<typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering, typename BinarySearchFunction>
__host__ __device__
OutputIterator binary_search(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                             ForwardIterator begin, 
                             ForwardIterator end,
                             InputIterator values_begin, 
                             InputIterator values_end,
                             OutputIterator output,
                             StrictWeakOrdering comp,
                             BinarySearchFunction func)
{
  hydra_thrust::for_each(exec,
                   hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(values_begin, output)),
                   hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(values_end, output + hydra_thrust::distance(values_begin, values_end))),
                   detail::binary_search_functor<ForwardIterator, StrictWeakOrdering, BinarySearchFunction>(begin, end, comp, func));
  
  return output + hydra_thrust::distance(values_begin, values_end);
}

   

// Scalar Implementation
template<typename OutputType, typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering, typename BinarySearchFunction>
__host__ __device__
OutputType binary_search(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                         ForwardIterator begin,
                         ForwardIterator end,
                         const T& value, 
                         StrictWeakOrdering comp,
                         BinarySearchFunction func)
{
  // use the vectorized path to implement the scalar version
  
  // allocate device buffers for value and output
  hydra_thrust::detail::temporary_array<T,DerivedPolicy>          d_value(exec,1);
  hydra_thrust::detail::temporary_array<OutputType,DerivedPolicy> d_output(exec,1);
  
  // copy value to device
  d_value[0] = value;
  
  // perform the query
  hydra_thrust::system::detail::generic::detail::binary_search(exec, begin, end, d_value.begin(), d_value.end(), d_output.begin(), comp, func);
  
  // copy result to host and return
  return d_output[0];
}


// this functor differs from hydra_thrust::less<T>
// because it allows the types of lhs & rhs to differ
// which is required by the binary search functions
// XXX use C++14 hydra_thrust::less<> when it's ready
struct binary_search_less
{
  template<typename T1, typename T2>
  __host__ __device__
  bool operator()(const T1& lhs, const T2& rhs) const
  {
    return lhs < rhs;
  }
};

   
} // end namespace detail


//////////////////////
// Scalar Functions //
//////////////////////


template<typename DerivedPolicy, typename ForwardIterator, typename T>
__host__ __device__
ForwardIterator lower_bound(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                            ForwardIterator begin,
                            ForwardIterator end,
                            const T& value)
{
  namespace p = hydra_thrust::placeholders;
  return hydra_thrust::lower_bound(exec, begin, end, value, detail::binary_search_less());
}

template<typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
__host__ __device__
ForwardIterator lower_bound(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                            ForwardIterator begin,
                            ForwardIterator end,
                            const T& value, 
                            StrictWeakOrdering comp)
{
  typedef typename hydra_thrust::iterator_traits<ForwardIterator>::difference_type difference_type;
  
  return begin + detail::binary_search<difference_type>(exec, begin, end, value, comp, detail::lbf());
}


template<typename DerivedPolicy, typename ForwardIterator, typename T>
__host__ __device__
ForwardIterator upper_bound(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                            ForwardIterator begin,
                            ForwardIterator end,
                            const T& value)
{
  namespace p = hydra_thrust::placeholders;
  return hydra_thrust::upper_bound(exec, begin, end, value, detail::binary_search_less());
}


template<typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
__host__ __device__
ForwardIterator upper_bound(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                            ForwardIterator begin,
                            ForwardIterator end,
                            const T& value, 
                            StrictWeakOrdering comp)
{
  typedef typename hydra_thrust::iterator_traits<ForwardIterator>::difference_type difference_type;
  
  return begin + detail::binary_search<difference_type>(exec, begin, end, value, comp, detail::ubf());
}


template<typename DerivedPolicy, typename ForwardIterator, typename T>
__host__ __device__
bool binary_search(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                   ForwardIterator begin,
                   ForwardIterator end,
                   const T& value)
{
  return hydra_thrust::binary_search(exec, begin, end, value, detail::binary_search_less());
}


template<typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
__host__ __device__
bool binary_search(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                   ForwardIterator begin,
                   ForwardIterator end,
                   const T& value, 
                   StrictWeakOrdering comp)
{
  return detail::binary_search<bool>(exec, begin, end, value, comp, detail::bsf());
}


//////////////////////
// Vector Functions //
//////////////////////


template<typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator>
__host__ __device__
OutputIterator lower_bound(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                           ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output)
{
  namespace p = hydra_thrust::placeholders;
  return hydra_thrust::lower_bound(exec, begin, end, values_begin, values_end, output, detail::binary_search_less());
}


template<typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
__host__ __device__
OutputIterator lower_bound(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                           ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output,
                           StrictWeakOrdering comp)
{
  return detail::binary_search(exec, begin, end, values_begin, values_end, output, comp, detail::lbf());
}


template<typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator>
__host__ __device__
OutputIterator upper_bound(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                           ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output)
{
  namespace p = hydra_thrust::placeholders;
  return hydra_thrust::upper_bound(exec, begin, end, values_begin, values_end, output, detail::binary_search_less());
}


template<typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
__host__ __device__
OutputIterator upper_bound(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                           ForwardIterator begin, 
                           ForwardIterator end,
                           InputIterator values_begin, 
                           InputIterator values_end,
                           OutputIterator output,
                           StrictWeakOrdering comp)
{
  return detail::binary_search(exec, begin, end, values_begin, values_end, output, comp, detail::ubf());
}


template<typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator>
__host__ __device__
OutputIterator binary_search(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                             ForwardIterator begin, 
                             ForwardIterator end,
                             InputIterator values_begin, 
                             InputIterator values_end,
                             OutputIterator output)
{
  namespace p = hydra_thrust::placeholders;
  return hydra_thrust::binary_search(exec, begin, end, values_begin, values_end, output, detail::binary_search_less());
}


template<typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator, typename StrictWeakOrdering>
__host__ __device__
OutputIterator binary_search(hydra_thrust::execution_policy<DerivedPolicy> &exec,
                             ForwardIterator begin, 
                             ForwardIterator end,
                             InputIterator values_begin, 
                             InputIterator values_end,
                             OutputIterator output,
                             StrictWeakOrdering comp)
{
  return detail::binary_search(exec, begin, end, values_begin, values_end, output, comp, detail::bsf());
}


template<typename DerivedPolicy, typename ForwardIterator, typename LessThanComparable>
__host__ __device__
hydra_thrust::pair<ForwardIterator,ForwardIterator>
equal_range(hydra_thrust::execution_policy<DerivedPolicy> &exec,
            ForwardIterator first,
            ForwardIterator last,
            const LessThanComparable &value)
{
  return hydra_thrust::equal_range(exec, first, last, value, detail::binary_search_less());
}


template<typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
__host__ __device__
hydra_thrust::pair<ForwardIterator,ForwardIterator>
equal_range(hydra_thrust::execution_policy<DerivedPolicy> &exec,
            ForwardIterator first,
            ForwardIterator last,
            const T &value,
            StrictWeakOrdering comp)
{
  ForwardIterator lb = hydra_thrust::lower_bound(exec, first, last, value, comp);
  ForwardIterator ub = hydra_thrust::upper_bound(exec, first, last, value, comp);
  return hydra_thrust::make_pair(lb, ub);
}


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace hydra_thrust

