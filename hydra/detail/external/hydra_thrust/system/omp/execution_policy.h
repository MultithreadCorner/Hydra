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

#pragma once

/*! \file hydra_thrust/system/omp/execution_policy.h
 *  \brief Execution policies for Thrust's OpenMP system.
 */

#include <hydra/detail/external/hydra_thrust/detail/config.h>

// get the execution policies definitions first
#include <hydra/detail/external/hydra_thrust/system/omp/detail/execution_policy.h>

// get the definition of par
#include <hydra/detail/external/hydra_thrust/system/omp/detail/par.h>

// now get all the algorithm definitions

#include <hydra/detail/external/hydra_thrust/system/omp/detail/adjacent_difference.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/assign_value.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/binary_search.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/copy.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/copy_if.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/count.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/equal.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/extrema.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/fill.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/find.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/for_each.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/gather.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/generate.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/get_value.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/inner_product.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/iter_swap.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/logical.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/malloc_and_free.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/merge.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/mismatch.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/partition.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/reduce.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/reduce_by_key.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/remove.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/replace.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/reverse.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/scan.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/scan_by_key.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/scatter.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/sequence.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/set_operations.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/sort.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/swap_ranges.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/tabulate.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/transform.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/transform_reduce.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/transform_scan.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/uninitialized_copy.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/uninitialized_fill.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/unique.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/unique_by_key.h>


// define these entities here for the purpose of Doxygenating them
// they are actually defined elsewhere
#if 0
HYDRA_THRUST_NAMESPACE_BEGIN
namespace system
{
namespace omp
{


/*! \addtogroup execution_policies
 *  \{
 */


/*! \p hydra_thrust::omp::execution_policy is the base class for all Thrust parallel execution
 *  policies which are derived from Thrust's OpenMP backend system.
 */
template<typename DerivedPolicy>
struct execution_policy : hydra_thrust::execution_policy<DerivedPolicy>
{};


/*! \p omp::tag is a type representing Thrust's standard C++ backend system in C++'s type system.
 *  Iterators "tagged" with a type which is convertible to \p omp::tag assert that they may be
 *  "dispatched" to algorithm implementations in the \p omp system.
 */
struct tag : hydra_thrust::system::omp::execution_policy<tag> { unspecified };


/*! \p hydra_thrust::omp::par is the parallel execution policy associated with Thrust's OpenMP
 *  backend system.
 *
 *  Instead of relying on implicit algorithm dispatch through iterator system tags, users may
 *  directly target Thrust's OpenMP backend system by providing \p hydra_thrust::omp::par as an algorithm
 *  parameter.
 *
 *  Explicit dispatch can be useful in avoiding the introduction of data copies into containers such
 *  as \p hydra_thrust::omp::vector.
 *
 *  The type of \p hydra_thrust::omp::par is implementation-defined.
 *
 *  The following code snippet demonstrates how to use \p hydra_thrust::omp::par to explicitly dispatch an
 *  invocation of \p hydra_thrust::for_each to the OpenMP backend system:
 *
 *  \code
 *  #include <hydra/detail/external/hydra_thrust/for_each.h>
 *  #include <hydra/detail/external/hydra_thrust/system/omp/execution_policy.h>
 *  #include <cstdio>
 *
 *  struct printf_functor
 *  {
 *    __host__ __device__
 *    void operator()(int x)
 *    {
 *      printf("%d\n", x);
 *    }
 *  };
 *  ...
 *  int vec[3];
 *  vec[0] = 0; vec[1] = 1; vec[2] = 2;
 *
 *  hydra_thrust::for_each(hydra_thrust::omp::par, vec.begin(), vec.end(), printf_functor());
 *
 *  // 0 1 2 is printed to standard output in some unspecified order
 *  \endcode
 */
static const unspecified par;


/*! \}
 */


} // end cpp
} // end system
HYDRA_THRUST_NAMESPACE_END
#endif


