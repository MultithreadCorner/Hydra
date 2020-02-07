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

#include <hydra/detail/external/hydra_thrust/iterator/transform_iterator.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_adaptor.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/result_of_adaptable_function.h>

namespace hydra_thrust
{

template <class UnaryFunction, class Iterator, class Reference, class Value>
  class transform_iterator;
  
namespace detail 
{

// Compute the iterator_adaptor instantiation to be used for transform_iterator
template <class UnaryFunc, class Iterator, class Reference, class Value>
struct transform_iterator_base
{
 private:
    // By default, dereferencing the iterator yields the same as the function.
    typedef typename hydra_thrust::detail::ia_dflt_help<
      Reference,
      hydra_thrust::detail::result_of_adaptable_function<UnaryFunc(typename hydra_thrust::iterator_value<Iterator>::type)>
    >::type reference;

    // To get the default for Value: remove any reference on the
    // result type, but retain any constness to signal
    // non-writability.  Note that if we adopt Thomas' suggestion
    // to key non-writability *only* on the Reference argument,
    // we'd need to strip constness here as well.
    typedef typename hydra_thrust::detail::ia_dflt_help<
      Value,
      hydra_thrust::detail::remove_reference<reference>
    >::type cv_value_type;

 public:
    typedef hydra_thrust::iterator_adaptor
    <
        transform_iterator<UnaryFunc, Iterator, Reference, Value>
      , Iterator
      , cv_value_type
      , hydra_thrust::use_default   // Leave the system alone
        //, hydra_thrust::use_default   // Leave the traversal alone
        // use the Iterator's category to let any system iterators remain random access even though
        // transform_iterator's reference type may not be a reference
        // XXX figure out why only iterators whose reference types are true references are random access
        , typename hydra_thrust::iterator_traits<Iterator>::iterator_category
      , reference
    > type;
};


} // end detail
} // end hydra_thrust

