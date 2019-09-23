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

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/system/cpp/detail/execution_policy.h>
#include <hydra/detail/external/thrust/system/tbb/detail/execution_policy.h>
#include <hydra/detail/external/thrust/iterator/detail/any_system_tag.h>
#include <hydra/detail/external/thrust/detail/type_traits.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{
namespace system
{
// put the canonical tag in the same ns as the backend's entry points
namespace omp
{
namespace detail
{

// this awkward sequence of definitions arise
// from the desire both for tag to derive
// from execution_policy and for execution_policy
// to convert to tag (when execution_policy is not
// an ancestor of tag)

// forward declaration of tag
struct tag;

// forward declaration of execution_policy
template<typename> struct execution_policy;

// specialize execution_policy for tag
template<>
  struct execution_policy<tag>
    : HYDRA_EXTERNAL_NS::thrust::system::cpp::detail::execution_policy<tag>
{};

// tag's definition comes before the
// generic definition of execution_policy
struct tag : execution_policy<tag> {};

// allow conversion to tag when it is not a successor
template<typename Derived>
  struct execution_policy
    : HYDRA_EXTERNAL_NS::thrust::system::cpp::detail::execution_policy<Derived>
{
  typedef tag tag_type; 
  operator tag() const { return tag(); }
};


// overloads of select_system

// XXX select_system(tbb, omp) & select_system(omp, tbb) are ambiguous
//     because both convert to cpp without these overloads, which we
//     arbitrarily define in the omp backend

template<typename System1, typename System2>
inline __hydra_host__ __hydra_device__
  System1 select_system(execution_policy<System1> s, HYDRA_EXTERNAL_NS::thrust::system::tbb::detail::execution_policy<System2>)
{
  return HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(s);
} // end select_system()


template<typename System1, typename System2>
inline __hydra_host__ __hydra_device__
  System2 select_system(HYDRA_EXTERNAL_NS::thrust::system::tbb::detail::execution_policy<System1>, execution_policy<System2> s)
{
  return HYDRA_EXTERNAL_NS::thrust::detail::derived_cast(s);
} // end select_system()


} // end detail

// alias execution_policy and tag here
using HYDRA_EXTERNAL_NS::thrust::system::omp::detail::execution_policy;
using HYDRA_EXTERNAL_NS::thrust::system::omp::detail::tag;

} // end omp
} // end system

// alias items at top-level
namespace omp
{

using HYDRA_EXTERNAL_NS::thrust::system::omp::execution_policy;
using HYDRA_EXTERNAL_NS::thrust::system::omp::tag;

} // end omp
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust

HYDRA_EXTERNAL_NAMESPACE_END
