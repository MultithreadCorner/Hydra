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

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/iterator_facade.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/detail/minimum_category.h>
#include <thrust/iterator/detail/minimum_system.h>
#include <thrust/tuple.h>
#include <thrust/detail/tuple/tuple_meta_transform.h>
#include <thrust/detail/tuple/tuple_transform.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/tuple_of_iterator_references.h>

namespace thrust
{

// forward declare zip_iterator for zip_iterator_base
template<typename IteratorTuple> class zip_iterator;

namespace detail
{


// Functors to be used with tuple algorithms
//
template<typename DiffType>
class advance_iterator
{
public:
  inline __host__ __device__
  advance_iterator(DiffType step) : m_step(step) {}
  
  __thrust_hd_warning_disable__
  template<typename Iterator>
  inline __host__ __device__
  void operator()(Iterator& it) const
  { it += m_step; }

private:
  DiffType m_step;
}; // end advance_iterator


struct increment_iterator
{
  __thrust_hd_warning_disable__
  template<typename Iterator>
  inline __host__ __device__
  void operator()(Iterator& it)
  { ++it; }
}; // end increment_iterator


struct decrement_iterator
{
  __thrust_hd_warning_disable__
  template<typename Iterator>
  inline __host__ __device__
  void operator()(Iterator& it)
  { --it; }
}; // end decrement_iterator


struct dereference_iterator
{
  template<typename Iterator>
  struct apply
  { 
    typedef typename
      iterator_traits<Iterator>::reference
    type;
  }; // end apply

  // XXX silence warnings of the form "calling a __host__ function from a __host__ __device__ function is not allowed
  __thrust_hd_warning_disable__
  template<typename Iterator>
  __host__ __device__
    typename apply<Iterator>::type operator()(Iterator const& it)
  {
    return *it;
  }
}; // end dereference_iterator


// The namespace tuple_impl_specific provides two meta-
// algorithms and two algorithms for tuples.
namespace tuple_impl_specific
{

// define apply1 for tuple_meta_transform_impl
template<typename UnaryMetaFunctionClass, class Arg>
  struct apply1
    : UnaryMetaFunctionClass::template apply<Arg>
{
}; // end apply1


// define apply2 for tuple_meta_accumulate_impl
template<typename UnaryMetaFunctionClass, class Arg1, class Arg2>
  struct apply2
    : UnaryMetaFunctionClass::template apply<Arg1,Arg2>
{
}; // end apply2


// Meta-accumulate algorithm for tuples. Note: The template 
// parameter StartType corresponds to the initial value in 
// ordinary accumulation.
//
template<class Tuple, class BinaryMetaFun, class StartType>
  struct tuple_meta_accumulate;

template<
    class BinaryMetaFun
  , typename StartType
>
  struct tuple_meta_accumulate<thrust::tuple<>,BinaryMetaFun,StartType>
{
   typedef typename thrust::detail::identity_<StartType>::type type;
};


template<
    class BinaryMetaFun
  , typename StartType
  , typename    T
  , typename... Types
>
  struct tuple_meta_accumulate<thrust::tuple<T,Types...>,BinaryMetaFun,StartType>
{
   typedef typename apply2<
       BinaryMetaFun
     , T
     , typename tuple_meta_accumulate<
           thrust::tuple<Types...>
         , BinaryMetaFun
         , StartType 
       >::type
   >::type type;
};


template<typename Fun>
inline __host__ __device__
Fun tuple_for_each_helper(Fun f)
{
  return f;
}

template<typename Fun, typename T, typename... Types>
inline __host__ __device__
Fun tuple_for_each_helper(Fun f, T& t, Types&... ts)
{
  f(t);
  return tuple_for_each_helper(f, ts...);
};

template<typename Fun, typename... Types, size_t... I>
inline __host__ __device__
Fun tuple_for_each(thrust::tuple<Types...>& t, Fun f, thrust::__index_sequence<I...>)
{
  return tuple_for_each_helper(f, thrust::get<I>(t)...);
};

// for_each algorithm for tuples.
template<typename Fun, typename... Types>
inline __host__ __device__
Fun tuple_for_each(thrust::tuple<Types...>& t, Fun f)
{ 
  return tuple_for_each(t, f, thrust::__make_index_sequence<thrust::tuple_size<thrust::tuple<Types...>>::value>{});    // XXX __index_sequence_for<Types...>{} upon variadic tuple
}


} // end tuple_impl_specific


// Metafunction to obtain the type of the tuple whose element types
// are the value_types of an iterator tupel.
//
template<typename IteratorTuple>
  struct tuple_of_value_types
    : tuple_meta_transform<
          IteratorTuple,
          iterator_value
        >
{
}; // end tuple_of_value_types


struct minimum_category_lambda
{
  template<typename T1, typename T2>
    struct apply : minimum_category<T1,T2>
  {};
};



// Metafunction to obtain the minimal traversal tag in a tuple
// of iterators.
//
template<typename IteratorTuple>
struct minimum_traversal_category_in_iterator_tuple
{
  typedef typename tuple_meta_transform<
      IteratorTuple
    , thrust::iterator_traversal
  >::type tuple_of_traversal_tags;
      
  typedef typename tuple_impl_specific::tuple_meta_accumulate<
      tuple_of_traversal_tags
    , minimum_category_lambda
    , thrust::random_access_traversal_tag
  >::type type;
};


struct minimum_system_lambda
{
  template<typename T1, typename T2>
    struct apply : minimum_system<T1,T2>
  {};
};



// Metafunction to obtain the minimal system tag in a tuple
// of iterators.
template<typename IteratorTuple>
struct minimum_system_in_iterator_tuple
{
  typedef typename thrust::detail::tuple_meta_transform<
    IteratorTuple,
    thrust::iterator_system
  >::type tuple_of_system_tags;

  typedef typename tuple_impl_specific::tuple_meta_accumulate<
    tuple_of_system_tags,
    minimum_system_lambda,
    thrust::any_system_tag
  >::type type;
};

namespace zip_iterator_base_ns
{


#ifdef THRUST_VARIADIC_TUPLE
template<typename Tuple, typename IndexSequence>
  struct tuple_of_iterator_references_helper;

template<typename Tuple, size_t... I>
  struct tuple_of_iterator_references_helper<Tuple, thrust::__index_sequence<I...>>
{
  typedef thrust::detail::tuple_of_iterator_references<
    typename thrust::tuple_element<I,Tuple>::type...
  > type;
};
#else
template<int i, typename Tuple>
  struct tuple_elements_helper
    : eval_if<
        (i < tuple_size<Tuple>::value),
        tuple_element<i,Tuple>,
        identity_<thrust::null_type>
      >
{};


template<typename Tuple>
  struct tuple_elements
{
  typedef typename tuple_elements_helper<0,Tuple>::type T0;
  typedef typename tuple_elements_helper<1,Tuple>::type T1;
  typedef typename tuple_elements_helper<2,Tuple>::type T2;
  typedef typename tuple_elements_helper<3,Tuple>::type T3;
  typedef typename tuple_elements_helper<4,Tuple>::type T4;
  typedef typename tuple_elements_helper<5,Tuple>::type T5;
  typedef typename tuple_elements_helper<6,Tuple>::type T6;
  typedef typename tuple_elements_helper<7,Tuple>::type T7;
  typedef typename tuple_elements_helper<8,Tuple>::type T8;
  typedef typename tuple_elements_helper<9,Tuple>::type T9;
};
#endif


template<typename IteratorTuple>
  struct tuple_of_iterator_references
{
  // get a thrust::tuple of the iterators' references
  typedef typename tuple_meta_transform<
    IteratorTuple,
    iterator_reference
  >::type tuple_of_references;

#ifdef THRUST_VARIADIC_TUPLE
  // map thrust::tuple<T...> to tuple_of_iterator_references<T...>
  typedef typename tuple_of_iterator_references_helper<
    tuple_of_references,
    thrust::__make_index_sequence<thrust::tuple_size<tuple_of_references>::value>
  >::type type;
#else
  // get at the individual tuple element types by name
  typedef tuple_elements<tuple_of_references> elements;

  // map thrust::tuple<T...> to tuple_of_iterator_references<T...>
  typedef thrust::detail::tuple_of_iterator_references<
    typename elements::T0,
    typename elements::T1,
    typename elements::T2,
    typename elements::T3,
    typename elements::T4,
    typename elements::T5,
    typename elements::T6,
    typename elements::T7,
    typename elements::T8,
    typename elements::T9
  > type;
#endif
};


} // end zip_iterator_base_ns

///////////////////////////////////////////////////////////////////
//
// Class zip_iterator_base
//
// Builds and exposes the iterator facade type from which the zip 
// iterator will be derived.
//
template<typename IteratorTuple>
  struct zip_iterator_base
{
 //private:
    // reference type is the type of the tuple obtained from the
    // iterators' reference types.
    typedef typename zip_iterator_base_ns::tuple_of_iterator_references<IteratorTuple>::type reference;

    // Boost's Value type is the same as reference type.
    //typedef reference value_type;
    typedef typename tuple_of_value_types<IteratorTuple>::type value_type;

    // Difference type is the first iterator's difference type
    typedef typename thrust::iterator_traits<
      typename thrust::tuple_element<0, IteratorTuple>::type
    >::difference_type difference_type;

    // Iterator system is the minimum system tag in the
    // iterator tuple
    typedef typename
    minimum_system_in_iterator_tuple<IteratorTuple>::type system;

    // Traversal category is the minimum traversal category in the
    // iterator tuple
    typedef typename
    minimum_traversal_category_in_iterator_tuple<IteratorTuple>::type traversal_category;
  
 public:
  
    // The iterator facade type from which the zip iterator will
    // be derived.
    typedef thrust::iterator_facade<
        zip_iterator<IteratorTuple>,
        value_type,  
        system,
        traversal_category,
        reference,
        difference_type
    > type;
}; // end zip_iterator_base

} // end detail

} // end thrust


