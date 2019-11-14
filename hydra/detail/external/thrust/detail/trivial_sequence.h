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

/*! \file trivial_sequence.h
 *  \brief Container-like class for wrapping sequences.  The wrapped
 *         sequence always has trivial iterators, even when the input
 *         sequence does not.
 */


#pragma once

#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/detail/type_traits.h>
#include <hydra/detail/external/thrust/detail/execution_policy.h>
#include <hydra/detail/external/thrust/detail/temporary_array.h>
#include <hydra/detail/external/thrust/type_traits/is_contiguous_iterator.h>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{

namespace detail
{

// never instantiated
template<typename Iterator, typename DerivedPolicy, typename is_trivial> struct _trivial_sequence { };

// trivial case
template<typename Iterator, typename DerivedPolicy>
struct _trivial_sequence<Iterator, DerivedPolicy, HYDRA_EXTERNAL_NS::thrust::detail::true_type>
{
    typedef Iterator iterator_type;
    Iterator first, last;

    __hydra_host__ __hydra_device__
    _trivial_sequence(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &, Iterator _first, Iterator _last) : first(_first), last(_last)
    {
    }

    __hydra_host__ __hydra_device__
    iterator_type begin() { return first; }

    __hydra_host__ __hydra_device__
    iterator_type end()   { return last; }
};

// non-trivial case
template<typename Iterator, typename DerivedPolicy>
struct _trivial_sequence<Iterator, DerivedPolicy, HYDRA_EXTERNAL_NS::thrust::detail::false_type>
{
    typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_value<Iterator>::type iterator_value;
    typedef typename HYDRA_EXTERNAL_NS::thrust::detail::temporary_array<iterator_value, DerivedPolicy>::iterator iterator_type;
    
    HYDRA_EXTERNAL_NS::thrust::detail::temporary_array<iterator_value, DerivedPolicy> buffer;

    __hydra_host__ __hydra_device__
    _trivial_sequence(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec, Iterator first, Iterator last)
      : buffer(exec, first, last)
    {
    }

    __hydra_host__ __hydra_device__
    iterator_type begin() { return buffer.begin(); }

    __hydra_host__ __hydra_device__
    iterator_type end()   { return buffer.end(); }
};

template <typename Iterator, typename DerivedPolicy>
struct trivial_sequence
  : detail::_trivial_sequence<Iterator, DerivedPolicy, typename HYDRA_EXTERNAL_NS::thrust::is_contiguous_iterator<Iterator>::type>
{
    typedef _trivial_sequence<Iterator, DerivedPolicy, typename HYDRA_EXTERNAL_NS::thrust::is_contiguous_iterator<Iterator>::type> super_t;

    __hydra_host__ __hydra_device__
    trivial_sequence(HYDRA_EXTERNAL_NS::thrust::execution_policy<DerivedPolicy> &exec, Iterator first, Iterator last) : super_t(exec, first, last) { }
};

} // end namespace detail

} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust


HYDRA_EXTERNAL_NAMESPACE_END
