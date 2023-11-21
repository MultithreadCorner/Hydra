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

/*! \file hydra_thrust/system/tbb/memory.h
 *  \brief Managing memory associated with Thrust's TBB system.
 */

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <type_traits>
#include <hydra/detail/external/hydra_thrust/system/tbb/detail/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/detail/pointer.h>
#include <hydra/detail/external/hydra_thrust/detail/reference.h>

HYDRA_THRUST_NAMESPACE_BEGIN
namespace system { namespace tbb
{

/*! \p tbb::pointer stores a pointer to an object allocated in memory accessible
 *  by the \p tbb system. This type provides type safety when dispatching
 *  algorithms on ranges resident in \p tbb memory.
 *
 *  \p tbb::pointer has pointer semantics: it may be dereferenced and
 *  manipulated with pointer arithmetic.
 *
 *  \p tbb::pointer can be created with the function \p tbb::malloc, or by
 *  explicitly calling its constructor with a raw pointer.
 *
 *  The raw pointer encapsulated by a \p tbb::pointer may be obtained by eiter its
 *  <tt>get</tt> member function or the \p raw_pointer_cast function.
 *
 *  \note \p tbb::pointer is not a "smart" pointer; it is the programmer's
 *        responsibility to deallocate memory pointed to by \p tbb::pointer.
 *
 *  \tparam T specifies the type of the pointee.
 *
 *  \see tbb::malloc
 *  \see tbb::free
 *  \see raw_pointer_cast
 */
template <typename T>
using pointer = hydra_thrust::pointer<
  T,
  hydra_thrust::system::tbb::tag,
  hydra_thrust::tagged_reference<T, hydra_thrust::system::tbb::tag>
>;

/*! \p tbb::universal_pointer stores a pointer to an object allocated in memory
 * accessible by the \p tbb system and host systems.
 *
 *  \p tbb::universal_pointer has pointer semantics: it may be dereferenced and
 *  manipulated with pointer arithmetic.
 *
 *  \p tbb::universal_pointer can be created with \p tbb::universal_allocator
 *  or by explicitly calling its constructor with a raw pointer.
 *
 *  The raw pointer encapsulated by a \p tbb::universal_pointer may be obtained
 *  by eiter its <tt>get</tt> member function or the \p raw_pointer_cast
 *  function.
 *
 *  \note \p tbb::universal_pointer is not a "smart" pointer; it is the
 *        programmer's responsibility to deallocate memory pointed to by
 *        \p tbb::universal_pointer.
 *
 *  \tparam T specifies the type of the pointee.
 *
 *  \see tbb::universal_allocator
 *  \see raw_pointer_cast
 */
template <typename T>
using universal_pointer = hydra_thrust::pointer<
  T,
  hydra_thrust::system::tbb::tag,
  typename std::add_lvalue_reference<T>::type
>;

/*! \p reference is a wrapped reference to an object stored in memory available
 *  to the \p tbb system. \p reference is the type of the result of
 *  dereferencing a \p tbb::pointer.
 *
 *  \tparam T Specifies the type of the referenced object.
 */
template <typename T>
using reference = hydra_thrust::tagged_reference<T, hydra_thrust::system::tbb::tag>;

}} // namespace system::tbb

/*! \addtogroup system_backends Systems
 *  \ingroup system
 *  \{
 */

/*! \namespace hydra_thrust::tbb
 *  \brief \p hydra_thrust::tbb is a top-level alias for \p hydra_thrust::system::tbb. */
namespace tbb
{
using hydra_thrust::system::tbb::pointer;
using hydra_thrust::system::tbb::universal_pointer;
using hydra_thrust::system::tbb::reference;
} // namespace tbb

HYDRA_THRUST_NAMESPACE_END

