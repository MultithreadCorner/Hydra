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


#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/reduce_intervals.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/function.h>
#include <hydra/detail/external/hydra_thrust/detail/cstdint.h>

namespace hydra_thrust
{
namespace system
{
namespace omp
{
namespace detail
{

template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename BinaryFunction,
          typename Decomposition>
void reduce_intervals(execution_policy<DerivedPolicy> &,
                      InputIterator input,
                      OutputIterator output,
                      BinaryFunction binary_op,
                      Decomposition decomp)
{
  // we're attempting to launch an omp kernel, assert we're compiling with omp support
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to enable OpenMP support in your compiler.                  X
  // ========================================================================
  HYDRA_THRUST_STATIC_ASSERT_MSG(
    (hydra_thrust::detail::depend_on_instantiation<
      InputIterator, (HYDRA_THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == HYDRA_THRUST_TRUE)
    >::value)
  , "OpenMP compiler support is not enabled"
  );

#if (HYDRA_THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == HYDRA_THRUST_TRUE)
  typedef typename hydra_thrust::iterator_value<OutputIterator>::type OutputType;

  // wrap binary_op
  hydra_thrust::detail::wrapped_function<BinaryFunction,OutputType> wrapped_binary_op(binary_op);

  typedef hydra_thrust::detail::intptr_t index_type;

  index_type n = static_cast<index_type>(decomp.size());

#if (HYDRA_THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == HYDRA_THRUST_TRUE)
# pragma omp parallel for
#endif // HYDRA_THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE
  for(index_type i = 0; i < n; i++)
  {
    InputIterator begin = input + decomp[i].begin();
    InputIterator end   = input + decomp[i].end();

    if (begin != end)
    {
      OutputType sum = hydra_thrust::raw_reference_cast(*begin);

      ++begin;

      while (begin != end)
      {
        sum = wrapped_binary_op(sum, *begin);
        ++begin;
      }

      OutputIterator tmp = output + i;
      *tmp = sum;
    }
  }
#endif // HYDRA_THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE
}

} // end namespace detail
} // end namespace omp
} // end namespace system
} // end namespace hydra_thrust

