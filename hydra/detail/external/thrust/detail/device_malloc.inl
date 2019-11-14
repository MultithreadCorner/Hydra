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


/*! \file device_malloc.inl
 *  \brief Inline file for device_malloc.h.
 */

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/device_malloc.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/thrust/detail/malloc_and_free.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{


HYDRA_EXTERNAL_NS::thrust::device_ptr<void> device_malloc(const std::size_t n)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;

  typedef HYDRA_EXTERNAL_NS::thrust::iterator_system< HYDRA_EXTERNAL_NS::thrust::device_ptr<void> >::type system;

  // XXX lower to select_system(system) here
  system s;

  return HYDRA_EXTERNAL_NS::thrust::device_ptr<void>(HYDRA_EXTERNAL_NS::thrust::malloc(s, n).get());
} // end device_malloc()


template<typename T>
  HYDRA_EXTERNAL_NS::thrust::device_ptr<T> device_malloc(const std::size_t n)
{
  using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;

  typedef HYDRA_EXTERNAL_NS::thrust::iterator_system< HYDRA_EXTERNAL_NS::thrust::device_ptr<void> >::type system;

  // XXX lower to select_system(system) here
  system s;

  return HYDRA_EXTERNAL_NS::thrust::device_ptr<T>(HYDRA_EXTERNAL_NS::thrust::malloc<T>(s,n).get());
} // end device_malloc()


} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust


HYDRA_EXTERNAL_NAMESPACE_END
