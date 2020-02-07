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


/*! \file device_malloc.h
 *  \brief Allocates storage in device memory
 */

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/device_ptr.h>
#include <cstddef> // for std::size_t

namespace hydra_thrust
{

/*! \addtogroup allocation_functions Allocation Functions
 *  \ingroup memory_management_functions
 *  \{
 */

/*! This version of \p device_malloc allocates sequential device storage
 *  for bytes.
 *
 *  \param n The number of bytes to allocate sequentially
 *           in device memory.
 *  \return A \p device_ptr to the newly allocated memory.
 *
 *  The following code snippet demonstrates how to use \p device_malloc to
 *  allocate a range of device memory.
 *
 *  \code
 *  #include <hydra/detail/external/hydra_thrust/device_malloc.h>
 *  #include <hydra/detail/external/hydra_thrust/device_free.h>
 *  ...
 *  // allocate some memory with device_malloc
 *  const int N = 100;
 *  hydra_thrust::device_ptr<void> void_ptr = hydra_thrust::device_malloc(N);
 *
 *  // manipulate memory
 *  ...
 *
 *  // deallocate with device_free
 *  hydra_thrust::device_free(void_ptr);
 *  \endcode
 *
 *  \see device_ptr
 *  \see device_free
 */
inline hydra_thrust::device_ptr<void> device_malloc(const std::size_t n);

/*! This version of \p device_malloc allocates sequential device storage for
 *  new objects of the given type.
 *
 *  \param n The number of objects of type T to allocate
 *           sequentially in device memory.
 *  \return A \p device_ptr to the newly allocated memory.
 *
 *  The following code snippet demonstrates how to use \p device_malloc to
 *  allocate a range of device memory.
 *
 *  \code
 *  #include <hydra/detail/external/hydra_thrust/device_malloc.h>
 *  #include <hydra/detail/external/hydra_thrust/device_free.h>
 *  ...
 *  // allocate some integers with device_malloc
 *  const int N = 100;
 *  hydra_thrust::device_ptr<int> int_array = hydra_thrust::device_malloc<int>(N);
 *
 *  // manipulate integers
 *  ...
 *
 *  // deallocate with device_free
 *  hydra_thrust::device_free(int_array);
 *  \endcode
 *
 *  \see device_ptr
 *  \see device_free
 */
template<typename T>
  inline hydra_thrust::device_ptr<T> device_malloc(const std::size_t n);

/*! \}
 */

} // end hydra_thrust

#include <hydra/detail/external/hydra_thrust/detail/device_malloc.inl>

