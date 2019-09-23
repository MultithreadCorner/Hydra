/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once


#if HYDRA_THRUST_DEVICE_COMPILER == HYDRA_THRUST_DEVICE_COMPILER_NVCC
#include <hydra/detail/external/thrust/system/cuda/config.h>

#include <hydra/detail/external/thrust/system/cuda/detail/execution_policy.h>
#include <hydra/detail/external/thrust/detail/minmax.h>
#include <hydra/detail/external/thrust/distance.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN

HYDRA_THRUST_BEGIN_NS
namespace cuda_cub {

// XXX forward declare to circumvent circular depedency
template <class Derived,
          class InputIt,
          class Predicate>
InputIt __hydra_host__ __hydra_device__
find_if(execution_policy<Derived>& policy,
        InputIt                    first,
        InputIt                    last,
        Predicate                  predicate);

template <class Derived,
          class InputIt,
          class Predicate>
InputIt __hydra_host__ __hydra_device__
find_if_not(execution_policy<Derived>& policy,
            InputIt                    first,
            InputIt                    last,
            Predicate                  predicate);

template <class Derived,
          class InputIt,
          class T>
InputIt __hydra_host__ __hydra_device__
find(execution_policy<Derived> &policy,
     InputIt                    first,
     InputIt                    last,
     T const& value);

}; // namespace cuda_cub
HYDRA_THRUST_END_NS

HYDRA_EXTERNAL_NAMESPACE_END

#include <hydra/detail/external/thrust/system/cuda/detail/reduce.h>
#include <hydra/detail/external/thrust/iterator/zip_iterator.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN

HYDRA_THRUST_BEGIN_NS
namespace cuda_cub {

namespace __find_if {

  template <typename TupleType>
  struct functor
  {
    HYDRA_THRUST_DEVICE_FUNCTION TupleType
    operator()(const TupleType& lhs, const TupleType& rhs) const
    {
      // select the smallest index among true results
      if (HYDRA_EXTERNAL_NS::thrust::get<0>(lhs) && HYDRA_EXTERNAL_NS::thrust::get<0>(rhs))
      {
        return TupleType(true, (HYDRA_EXTERNAL_NS::thrust::min)(HYDRA_EXTERNAL_NS::thrust::get<1>(lhs), HYDRA_EXTERNAL_NS::thrust::get<1>(rhs)));
      }
      else if (HYDRA_EXTERNAL_NS::thrust::get<0>(lhs))
      {
        return lhs;
      }
      else
      {
        return rhs;
      }
    }
  };
}    // namespace __find_if

template <class Derived,
          class InputIt,
          class Size,
          class Predicate>
InputIt __hydra_host__ __hydra_device__
find_if_n(execution_policy<Derived>& policy,
          InputIt                    first,
          Size                       num_items,
          Predicate                  predicate)
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::tuple<bool,Size> result_type;
  
  // empty sequence
  if(num_items == 0) return first;
  
  // this implementation breaks up the sequence into separate intervals
  // in an attempt to early-out as soon as a value is found
  //
  // XXX compose find_if from a look-back prefix scan algorithm
  //     and abort kernel when the first element is found


  // TODO incorporate sizeof(InputType) into interval_threshold and round to multiple of 32
  const Size interval_threshold = 1 << 20;
  const Size interval_size = (HYDRA_EXTERNAL_NS::thrust::min)(interval_threshold, num_items);
  
  // force transform_iterator output to bool
  typedef transform_input_iterator_t<bool,
                                     InputIt,
                                     Predicate>
      XfrmIterator;
  typedef HYDRA_EXTERNAL_NS::thrust::tuple<XfrmIterator,
                        counting_iterator_t<Size> >
      IteratorTuple;
  typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<IteratorTuple> ZipIterator;

  IteratorTuple iter_tuple =
      HYDRA_EXTERNAL_NS::thrust::make_tuple(XfrmIterator(first, predicate),
                         counting_iterator_t<Size>(0));

  ZipIterator begin = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(iter_tuple);
  ZipIterator end   = begin + num_items;

  for (ZipIterator interval_begin = begin;
       interval_begin < end;
       interval_begin += interval_size)
  {
    ZipIterator interval_end = interval_begin + interval_size;
    if(end < interval_end)
    {
      interval_end = end;
    } // end if

    result_type result = reduce(policy,
                                interval_begin,
                                interval_end,
                                result_type(false, interval_end - begin),
                                __find_if::functor<result_type>());

    // see if we found something
    if(HYDRA_EXTERNAL_NS::thrust::get<0>(result))
    {
      return first + HYDRA_EXTERNAL_NS::thrust::get<1>(result);
    }
  }
  
  //nothing was found if we reach here...
  return first + num_items;
}

template <class Derived,
          class InputIt,
          class Predicate>
InputIt __hydra_host__ __hydra_device__
find_if(execution_policy<Derived>& policy,
        InputIt                    first,
        InputIt                    last,
        Predicate                  predicate)
{
  return cuda_cub::find_if_n(policy, first, HYDRA_EXTERNAL_NS::thrust::distance(first,last), predicate);
}

template <class Derived,
          class InputIt,
          class Predicate>
InputIt __hydra_host__ __hydra_device__
find_if_not(execution_policy<Derived>& policy,
            InputIt                    first,
            InputIt                    last,
            Predicate                  predicate)
{
  return cuda_cub::find_if(policy, first, last, HYDRA_EXTERNAL_NS::thrust::detail::not1(predicate));
}


template <class Derived,
          class InputIt,
          class T>
InputIt __hydra_host__ __hydra_device__
find(execution_policy<Derived> &policy,
     InputIt                    first,
     InputIt                    last,
     T const& value)
{
  return cuda_cub::find_if(policy,
                        first,
                        last,
                        HYDRA_EXTERNAL_NS::thrust::detail::equal_to_value<T>(value));
}


} // namespace cuda_cub
HYDRA_THRUST_END_NS

HYDRA_EXTERNAL_NAMESPACE_END
#endif
