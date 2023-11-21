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

#include <hydra/detail/external/hydra_thrust/detail/config.h>

#if HYDRA_THRUST_DEVICE_COMPILER == HYDRA_THRUST_DEVICE_COMPILER_NVCC
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/execution_policy.h>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace cuda_cub {

template <class Derived, class ItemsIt, class ResultIt>
ResultIt __host__ __device__
reverse_copy(execution_policy<Derived> &policy,
             ItemsIt                    first,
             ItemsIt                    last,
             ResultIt                   result);

template <class Derived, class ItemsIt>
void __host__ __device__
reverse(execution_policy<Derived> &policy,
        ItemsIt                    first,
        ItemsIt                    last);

}    // namespace cuda_cub
HYDRA_THRUST_NAMESPACE_END

#include <hydra/detail/external/hydra_thrust/advance.h>
#include <hydra/detail/external/hydra_thrust/distance.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/swap_ranges.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/copy.h>
#include <hydra/detail/external/hydra_thrust/iterator/reverse_iterator.h>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace cuda_cub {

template <class Derived,
          class ItemsIt,
          class ResultIt>
ResultIt __host__ __device__
reverse_copy(execution_policy<Derived> &policy,
             ItemsIt                    first,
             ItemsIt                    last,
             ResultIt                   result)
{
  return cuda_cub::copy(policy,
                        hydra_thrust::make_reverse_iterator(last),
                        hydra_thrust::make_reverse_iterator(first),
                        result);
}

template <class Derived,
          class ItemsIt>
void __host__ __device__
reverse(execution_policy<Derived> &policy,
        ItemsIt                    first,
        ItemsIt                    last)
{
  typedef typename hydra_thrust::iterator_difference<ItemsIt>::type difference_type;

  // find the midpoint of [first,last)
  difference_type N = hydra_thrust::distance(first, last);
  ItemsIt mid(first);
  hydra_thrust::advance(mid, N / 2);

  cuda_cub::swap_ranges(policy, first, mid, hydra_thrust::make_reverse_iterator(last));
}


}    // namespace cuda_cub
HYDRA_THRUST_NAMESPACE_END
#endif
