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

#include <hydra/detail/external/thrust/iterator/counting_iterator.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/detail/numeric_traits.h>
#include <hydra/detail/external/thrust/detail/type_traits.h>
#include <cstddef>

HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{

// forward declaration of counting_iterator
template <typename Incrementable, typename System, typename Traversal, typename Difference>
  class counting_iterator;

namespace detail
{

template <typename Incrementable, typename System, typename Traversal, typename Difference>
  struct counting_iterator_base
{
  typedef typename HYDRA_EXTERNAL_NS::thrust::detail::eval_if<
    // use any_system_tag if we are given use_default
    HYDRA_EXTERNAL_NS::thrust::detail::is_same<System,use_default>::value,
    HYDRA_EXTERNAL_NS::thrust::detail::identity_<HYDRA_EXTERNAL_NS::thrust::any_system_tag>,
    HYDRA_EXTERNAL_NS::thrust::detail::identity_<System>
  >::type system;

  typedef typename HYDRA_EXTERNAL_NS::thrust::detail::ia_dflt_help<
      Traversal,
      HYDRA_EXTERNAL_NS::thrust::detail::eval_if<
          HYDRA_EXTERNAL_NS::thrust::detail::is_numeric<Incrementable>::value,
          HYDRA_EXTERNAL_NS::thrust::detail::identity_<random_access_traversal_tag>,
          HYDRA_EXTERNAL_NS::thrust::iterator_traversal<Incrementable>
      >
  >::type traversal;

  // unlike Boost, we explicitly use std::ptrdiff_t as the difference type
  // for floating point counting_iterators
  typedef typename HYDRA_EXTERNAL_NS::thrust::detail::ia_dflt_help<
    Difference,
    HYDRA_EXTERNAL_NS::thrust::detail::eval_if<
      HYDRA_EXTERNAL_NS::thrust::detail::is_numeric<Incrementable>::value,
        HYDRA_EXTERNAL_NS::thrust::detail::eval_if<
          HYDRA_EXTERNAL_NS::thrust::detail::is_integral<Incrementable>::value,
          HYDRA_EXTERNAL_NS::thrust::detail::numeric_difference<Incrementable>,
          HYDRA_EXTERNAL_NS::thrust::detail::identity_<std::ptrdiff_t>
        >,
      HYDRA_EXTERNAL_NS::thrust::iterator_difference<Incrementable>
    >
  >::type difference;

  // our implementation departs from Boost's in that counting_iterator::dereference
  // returns a copy of its counter, rather than a reference to it. returning a reference
  // to the internal state of an iterator causes subtle bugs (consider the temporary
  // iterator created in the expression *(iter + i)) and has no compelling use case
  typedef HYDRA_EXTERNAL_NS::thrust::iterator_adaptor<
    counting_iterator<Incrementable, System, Traversal, Difference>, // self
    Incrementable,                                                  // Base
    Incrementable,                                                  // XXX we may need to pass const here as Boost does
    system,
    traversal,
    Incrementable,
    difference
  > type;
}; // end counting_iterator_base


template<typename Difference, typename Incrementable1, typename Incrementable2>
  struct iterator_distance
{
  __hydra_host__ __hydra_device__
  static Difference distance(Incrementable1 x, Incrementable2 y)
  {
    return y - x;
  }
};


template<typename Difference, typename Incrementable1, typename Incrementable2>
  struct number_distance
{
  __hydra_host__ __hydra_device__
  static Difference distance(Incrementable1 x, Incrementable2 y)
  {
      return static_cast<Difference>(numeric_distance(x,y));
  }
};


template<typename Difference, typename Incrementable1, typename Incrementable2, typename Enable = void>
  struct counting_iterator_equal
{
  __hydra_host__ __hydra_device__
  static bool equal(Incrementable1 x, Incrementable2 y)
  {
    return x == y;
  }
};


// specialization for floating point equality
template<typename Difference, typename Incrementable1, typename Incrementable2>
  struct counting_iterator_equal<
    Difference,
    Incrementable1,
    Incrementable2,
    typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<
      HYDRA_EXTERNAL_NS::thrust::detail::is_floating_point<Incrementable1>::value ||
      HYDRA_EXTERNAL_NS::thrust::detail::is_floating_point<Incrementable2>::value
    >::type
  >
{
  __hydra_host__ __hydra_device__
  static bool equal(Incrementable1 x, Incrementable2 y)
  {
    typedef number_distance<Difference,Incrementable1,Incrementable2> d;
    return d::distance(x,y) == 0;
  }
};


} // end detail
} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust


HYDRA_EXTERNAL_NAMESPACE_END

