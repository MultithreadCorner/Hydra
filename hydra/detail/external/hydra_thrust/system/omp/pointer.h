/*
 *  Copyright 2008-2020 NVIDIA Corporation
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

/*! \file hydra_thrust/system/omp/memory.h
 *  \brief Managing memory associated with Thrust's OpenMP system.
 */

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <type_traits>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/detail/pointer.h>
#include <hydra/detail/external/hydra_thrust/detail/reference.h>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace system { namespace omp
{

/*! \p omp::pointer stores a pointer to an object allocated in memory accessible
 *  by the \p omp system. This type provides type safety when dispatching
 *  algorithms on ranges resident in \p omp memory.
 *
 *  \p omp::pointer has pointer semantics: it may be dereferenced and
 *  manipulated with pointer arithmetic.
 *
 *  \p omp::pointer can be created with the function \p omp::malloc, or by
 *  explicitly calling its constructor with a raw pointer.
 *
 *  The raw pointer encapsulated by a \p omp::pointer may be obtained by eiter its
 *  <tt>get</tt> member function or the \p raw_pointer_cast function.
 *
 *  \note \p omp::pointer is not a "smart" pointer; it is the programmer's
 *        responsibility to deallocate memory pointed to by \p omp::pointer.
 *
 *  \tparam T specifies the type of the pointee.
 *
 *  \see omp::malloc
 *  \see omp::free
 *  \see raw_pointer_cast
 */
template <typename T>
using pointer = hydra_thrust::pointer<
  T,
  hydra_thrust::system::omp::tag,
  hydra_thrust::tagged_reference<T, hydra_thrust::system::omp::tag>
>;

/*! \p omp::universal_pointer stores a pointer to an object allocated in memory
 * accessible by the \p omp system and host systems.
 *
 *  \p omp::universal_pointer has pointer semantics: it may be dereferenced and
 *  manipulated with pointer arithmetic.
 *
 *  \p omp::universal_pointer can be created with \p omp::universal_allocator
 *  or by explicitly calling its constructor with a raw pointer.
 *
 *  The raw pointer encapsulated by a \p omp::universal_pointer may be obtained
 *  by eiter its <tt>get</tt> member function or the \p raw_pointer_cast
 *  function.
 *
 *  \note \p omp::universal_pointer is not a "smart" pointer; it is the
 *        programmer's responsibility to deallocate memory pointed to by
 *        \p omp::universal_pointer.
 *
 *  \tparam T specifies the type of the pointee.
 *
 *  \see omp::universal_allocator
 *  \see raw_pointer_cast
 */
template <typename T>
using universal_pointer = hydra_thrust::pointer<
  T,
  hydra_thrust::system::omp::tag,
  typename std::add_lvalue_reference<T>::type
>;

/*! \p reference is a wrapped reference to an object stored in memory available
 *  to the \p omp system. \p reference is the type of the result of
 *  dereferencing a \p omp::pointer.
 *
 *  \tparam T Specifies the type of the referenced object.
 */
template <typename T>
using reference = hydra_thrust::tagged_reference<T, hydra_thrust::system::omp::tag>;

}} // namespace system::omp

/*! \addtogroup system_backends Systems
 *  \ingroup system
 *  \{
 */

/*! \namespace hydra_thrust::omp
 *  \brief \p hydra_thrust::omp is a top-level alias for \p hydra_thrust::system::omp. */
namespace omp
{
using hydra_thrust::system::omp::pointer;
using hydra_thrust::system::omp::universal_pointer;
using hydra_thrust::system::omp::reference;
} // namespace omp

HYDRA_THRUST_NAMESPACE_END

