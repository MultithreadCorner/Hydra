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

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/host_system_tag.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/device_system_tag.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/any_system_tag.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_categories.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/iterator_traversal_tags.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/is_iterator_category.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/iterator_category_with_system_and_traversal.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/iterator_category_to_traversal.h>

namespace hydra_thrust
{

namespace detail
{


// adapted from http://www.boost.org/doc/libs/1_37_0/libs/iterator/doc/iterator_facade.html#iterator-category
//
// in our implementation, R need not be a reference type to result in a category
// derived from forward_XXX_iterator_tag
//
// iterator-category(T,V,R) :=
//   if(T is convertible to input_host_iterator_tag
//      || T is convertible to output_host_iterator_tag
//      || T is convertible to input_device_iterator_tag
//      || T is convertible to output_device_iterator_tag
//   )
//     return T
//
//   else if (T is not convertible to incrementable_traversal_tag)
//     the program is ill-formed
//
//   else return a type X satisfying the following two constraints:
//
//     1. X is convertible to X1, and not to any more-derived
//        type, where X1 is defined by:
//
//        if (T is convertible to forward_traversal_tag)
//        {
//          if (T is convertible to random_access_traversal_tag)
//            X1 = random_access_host_iterator_tag
//          else if (T is convertible to bidirectional_traversal_tag)
//            X1 = bidirectional_host_iterator_tag
//          else
//            X1 = forward_host_iterator_tag
//        }
//        else
//        {
//          if (T is convertible to single_pass_traversal_tag
//              && R is convertible to V)
//            X1 = input_host_iterator_tag
//          else
//            X1 = T
//        }
//
//     2. category-to-traversal(X) is convertible to the most
//        derived traversal tag type to which X is also convertible,
//        and not to any more-derived traversal tag type.


template<typename System, typename Traversal, typename ValueParam, typename Reference>
  struct iterator_facade_default_category;


// Thrust's implementation of iterator_facade_default_category is slightly
// different from Boost's equivalent.
// Thrust does not check is_convertible<Reference, ValueParam> because Reference
// may not be a complete type at this point, and implementations of is_convertible
// typically require that both types be complete.
// Instead, it simply assumes that if is_convertible<Traversal, single_pass_traversal_tag>,
// then the category is input_iterator_tag


// this is the function for standard system iterators
template<typename Traversal, typename ValueParam, typename Reference>
  struct iterator_facade_default_category_std :
    hydra_thrust::detail::eval_if<
      hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::forward_traversal_tag>::value,
      hydra_thrust::detail::eval_if<
        hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::random_access_traversal_tag>::value,
        hydra_thrust::detail::identity_<std::random_access_iterator_tag>,
        hydra_thrust::detail::eval_if<
          hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::bidirectional_traversal_tag>::value,
          hydra_thrust::detail::identity_<std::bidirectional_iterator_tag>,
          hydra_thrust::detail::identity_<std::forward_iterator_tag>
        >
      >,
      hydra_thrust::detail::eval_if< // XXX note we differ from Boost here
        hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::single_pass_traversal_tag>::value,
        hydra_thrust::detail::identity_<std::input_iterator_tag>,
        hydra_thrust::detail::identity_<Traversal>
      >
    >
{
}; // end iterator_facade_default_category_std


// this is the function for host system iterators
template<typename Traversal, typename ValueParam, typename Reference>
  struct iterator_facade_default_category_host :
    hydra_thrust::detail::eval_if<
      hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::forward_traversal_tag>::value,
      hydra_thrust::detail::eval_if<
        hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::random_access_traversal_tag>::value,
        hydra_thrust::detail::identity_<hydra_thrust::random_access_host_iterator_tag>,
        hydra_thrust::detail::eval_if<
          hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::bidirectional_traversal_tag>::value,
          hydra_thrust::detail::identity_<hydra_thrust::bidirectional_host_iterator_tag>,
          hydra_thrust::detail::identity_<hydra_thrust::forward_host_iterator_tag>
        >
      >,
      hydra_thrust::detail::eval_if< // XXX note we differ from Boost here
        hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::single_pass_traversal_tag>::value,
        hydra_thrust::detail::identity_<hydra_thrust::input_host_iterator_tag>,
        hydra_thrust::detail::identity_<Traversal>
      >
    >
{
}; // end iterator_facade_default_category_host


// this is the function for device system iterators
template<typename Traversal, typename ValueParam, typename Reference>
  struct iterator_facade_default_category_device :
    hydra_thrust::detail::eval_if<
      hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::forward_traversal_tag>::value,
      hydra_thrust::detail::eval_if<
        hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::random_access_traversal_tag>::value,
        hydra_thrust::detail::identity_<hydra_thrust::random_access_device_iterator_tag>,
        hydra_thrust::detail::eval_if<
          hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::bidirectional_traversal_tag>::value,
          hydra_thrust::detail::identity_<hydra_thrust::bidirectional_device_iterator_tag>,
          hydra_thrust::detail::identity_<hydra_thrust::forward_device_iterator_tag>
        >
      >,
      hydra_thrust::detail::eval_if<
        hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::single_pass_traversal_tag>::value, // XXX note we differ from Boost here
        hydra_thrust::detail::identity_<hydra_thrust::input_device_iterator_tag>,
        hydra_thrust::detail::identity_<Traversal>
      >
    >
{
}; // end iterator_facade_default_category_device


// this is the function for any system iterators
template<typename Traversal, typename ValueParam, typename Reference>
  struct iterator_facade_default_category_any
{
  typedef hydra_thrust::detail::iterator_category_with_system_and_traversal<
    typename iterator_facade_default_category_std<Traversal, ValueParam, Reference>::type,
    hydra_thrust::any_system_tag,
    Traversal
  > type;
}; // end iterator_facade_default_category_any


template<typename System, typename Traversal, typename ValueParam, typename Reference>
  struct iterator_facade_default_category
      // check for any system
    : hydra_thrust::detail::eval_if<
        hydra_thrust::detail::is_convertible<System, hydra_thrust::any_system_tag>::value,
        iterator_facade_default_category_any<Traversal, ValueParam, Reference>,

        // check for host system
        hydra_thrust::detail::eval_if<
          hydra_thrust::detail::is_convertible<System, hydra_thrust::host_system_tag>::value,
          iterator_facade_default_category_host<Traversal, ValueParam, Reference>,

          // check for device system
          hydra_thrust::detail::eval_if<
            hydra_thrust::detail::is_convertible<System, hydra_thrust::device_system_tag>::value,
            iterator_facade_default_category_device<Traversal, ValueParam, Reference>,

            // if we don't recognize the system, get a standard iterator category
            // and combine it with System & Traversal
            hydra_thrust::detail::identity_<
              hydra_thrust::detail::iterator_category_with_system_and_traversal<
                typename iterator_facade_default_category_std<Traversal, ValueParam, Reference>::type,
                System,
                Traversal
              >
            >
          >
        >
      >
{};


template<typename System, typename Traversal, typename ValueParam, typename Reference>
  struct iterator_facade_category_impl
{
  typedef typename iterator_facade_default_category<
    System,Traversal,ValueParam,Reference
  >::type category;

  // we must be able to deduce both Traversal & System from category
  // otherwise, munge them all together
  typedef typename hydra_thrust::detail::eval_if<
    hydra_thrust::detail::and_<
      hydra_thrust::detail::is_same<
        Traversal,
        typename hydra_thrust::detail::iterator_category_to_traversal<category>::type
      >,
      hydra_thrust::detail::is_same<
        System,
        typename hydra_thrust::detail::iterator_category_to_system<category>::type
      >
    >::value,
    hydra_thrust::detail::identity_<category>,
    hydra_thrust::detail::identity_<hydra_thrust::detail::iterator_category_with_system_and_traversal<category,System,Traversal> >
  >::type type;
}; // end iterator_facade_category_impl


template<typename CategoryOrSystem,
         typename CategoryOrTraversal,
         typename ValueParam,
         typename Reference>
  struct iterator_facade_category
{
  typedef typename
  hydra_thrust::detail::eval_if<
    hydra_thrust::detail::is_iterator_category<CategoryOrTraversal>::value,
    hydra_thrust::detail::identity_<CategoryOrTraversal>, // categories are fine as-is
    iterator_facade_category_impl<CategoryOrSystem, CategoryOrTraversal, ValueParam, Reference>
  >::type type;
}; // end iterator_facade_category


} // end detail
} // end hydra_thrust

