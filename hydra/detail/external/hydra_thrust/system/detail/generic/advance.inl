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
#include <hydra/detail/external/hydra_thrust/system/detail/generic/advance.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>

namespace hydra_thrust
{
namespace system
{
namespace detail
{
namespace generic
{
namespace detail
{

__hydra_thrust_exec_check_disable__
template<typename InputIterator, typename Distance>
__host__ __device__
void advance(InputIterator& i, Distance n, hydra_thrust::incrementable_traversal_tag)
{
  while(n)
  {
    ++i;
    --n;
  } // end while
} // end advance()

__hydra_thrust_exec_check_disable__
template<typename InputIterator, typename Distance>
__host__ __device__
void advance(InputIterator& i, Distance n, hydra_thrust::random_access_traversal_tag)
{
  i += n;
} // end advance()

} // end detail

template<typename InputIterator, typename Distance>
__host__ __device__
void advance(InputIterator& i, Distance n)
{
  // dispatch on iterator traversal
  hydra_thrust::system::detail::generic::detail::advance(i, n,
    typename hydra_thrust::iterator_traversal<InputIterator>::type());
} // end advance()

} // end namespace detail
} // end namespace generic
} // end namespace system
} // end namespace hydra_thrust

