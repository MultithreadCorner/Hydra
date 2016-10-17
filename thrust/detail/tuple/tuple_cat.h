/*
 *  Copyright 2008-2014 NVIDIA Corporation
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

#include <thrust/detail/config.h>
#include <thrust/tuple.h>
#include <thrust/detail/type_traits.h>

#ifdef THRUST_VARIADIC_TUPLE
namespace thrust
{
template<class T>
using __decay_t = typename std::decay<T>::type;

template<class T>
struct tuple_traits
{
  using tuple_type = T;

  static const size_t size = thrust::tuple_size<tuple_type>::value; 

  template<size_t i>
  using element_type = typename thrust::tuple_element<i,tuple_type>::type;

  template<size_t i>
  __TUPLE_ANNOTATION
  static element_type<i>& get(tuple_type& t)
  {
    return thrust::get<i>(t);
  }

  template<size_t i>
  __TUPLE_ANNOTATION
  static const element_type<i>& get(const tuple_type& t)
  {
    return thrust::get<i>(t);
  }

  template<size_t i>
  __TUPLE_ANNOTATION
  static element_type<i>&& get(tuple_type&& t)
  {
    return thrust::get<i>(std::move(t));
  }
};


template<size_t i, class Tuple>
__TUPLE_ANNOTATION
auto __get(Tuple&& t)
  -> decltype(
       tuple_traits<__decay_t<Tuple>>::template get<i>(std::forward<Tuple>(t))
     )
{
  return tuple_traits<__decay_t<Tuple>>::template get<i>(std::forward<Tuple>(t));
}

template<bool b, class True, class False>
struct __lazy_conditional
{
  using type = typename True::type;
};


template<class True, class False>
struct __lazy_conditional<false, True, False>
{
  using type = typename False::type;
};


template<size_t I, class Tuple1, class... Tuples>
struct __tuple_cat_get_result
{
  using tuple1_type = typename std::decay<Tuple1>::type;
  static const size_t size1 = thrust::tuple_size<typename std::decay<Tuple1>::type>::value;

  using type = typename __lazy_conditional<
    (I < size1),
    thrust::tuple_element<I,tuple1_type>,
    __tuple_cat_get_result<I - size1, Tuples...>
  >::type;
};


template<size_t I, class Tuple1>
struct __tuple_cat_get_result<I,Tuple1>
  : thrust::tuple_element<I, typename std::decay<Tuple1>::type>
{};


template<size_t I, class Tuple1, class... Tuples>
__TUPLE_ANNOTATION
typename __tuple_cat_get_result<I,Tuple1,Tuples...>::type
  __tuple_cat_get(Tuple1&& t, Tuples&&... ts);


template<size_t I, class Tuple1, class... Tuples>
__TUPLE_ANNOTATION
typename __tuple_cat_get_result<I,Tuple1,Tuples...>::type
  __tuple_cat_get_impl(std::false_type, Tuple1&& t, Tuples&&...)
{
  return __get<I>(std::forward<Tuple1>(t));
}


template<size_t I, class Tuple1, class... Tuples>
__TUPLE_ANNOTATION
typename __tuple_cat_get_result<I,Tuple1,Tuples...>::type
  __tuple_cat_get_impl(std::true_type, Tuple1&&, Tuples&&... ts)
{
  const size_t J = I - thrust::tuple_size<typename std::decay<Tuple1>::type>::value;
  return __tuple_cat_get<J>(std::forward<Tuples>(ts)...);
}


template<size_t I, class Tuple1, class... Tuples>
__TUPLE_ANNOTATION
typename __tuple_cat_get_result<I,Tuple1,Tuples...>::type
  __tuple_cat_get(Tuple1&& t, Tuples&&... ts)
{
  auto recurse = typename std::conditional<
    I < thrust::tuple_size<typename std::decay<Tuple1>::type>::value,
    std::false_type,
    std::true_type
  >::type();

  return __tuple_cat_get_impl<I>(recurse, std::forward<Tuple1>(t), std::forward<Tuples>(ts)...);
}


template<size_t... I, class Function, class... Tuples>
__TUPLE_ANNOTATION
auto __tuple_cat_apply_impl(__index_sequence<I...>, Function f, Tuples&&... ts)
  -> decltype(
       f(__tuple_cat_get<I>(std::forward<Tuples>(ts)...)...)
     )
{
  return f(__tuple_cat_get<I>(std::forward<Tuples>(ts)...)...);
}


template<size_t Size, size_t... Sizes>
struct __sum
  : std::integral_constant<
      size_t,
      Size + __sum<Sizes...>::value
    >
{};


template<size_t Size> struct __sum<Size> : std::integral_constant<size_t, Size> {};


template<class Function, class... Tuples>
__TUPLE_ANNOTATION
auto tuple_cat_apply(Function f, Tuples&&... ts)
  -> decltype(
       __tuple_cat_apply_impl(
         __make_index_sequence<
           __sum<
             0u,
             thrust::tuple_size<typename std::decay<Tuples>::type>::value...
           >::value
         >(),
         f,
         std::forward<Tuples>(ts)...
       )
     )
{
  const size_t N = __sum<0u, thrust::tuple_size<typename std::decay<Tuples>::type>::value...>::value;
  return __tuple_cat_apply_impl(__make_index_sequence<N>(), f, std::forward<Tuples>(ts)...);
}


template<class Function, class Tuple, size_t... I>
__TUPLE_ANNOTATION
auto __tuple_apply_impl(Function f, Tuple&& t, __index_sequence<I...>)
  -> decltype(
       f(__get<I>(std::forward<Tuple>(t))...)
     )
{
  return f(__get<I>(std::forward<Tuple>(t))...);
}



template<class Function, class Tuple>
__TUPLE_ANNOTATION
auto tuple_apply(Function f, Tuple&& t)
  -> decltype(
       tuple_cat_apply(f, std::forward<Tuple>(t))
     )
{
  return tuple_cat_apply(f, std::forward<Tuple>(t));
}
    
template<class IndexSequence, class... Tuples>
struct tuple_cat_result_impl_impl;


template<size_t... I, class... Tuples>
struct tuple_cat_result_impl_impl<__index_sequence<I...>, Tuples...>
{
  using type = tuple<typename __tuple_cat_get_result<I, Tuples...>::type...>;
};


template<class... Tuples>
struct tuple_cat_result_impl
{
  static const size_t result_size = __sum<0u, tuple_size<Tuples>::value...>::value;

  using type = typename tuple_cat_result_impl_impl<
    thrust::__make_index_sequence<result_size>,
    Tuples...
  >::type;
};


template<class... Tuples>
using tuple_cat_result = typename tuple_cat_result_impl<typename std::decay<Tuples>::type...>::type;


template<class Tuple>
struct tuple_maker
{
  template<class... Args>
  __TUPLE_ANNOTATION
  Tuple operator()(Args&&... args)
  {
    return Tuple{std::forward<Args>(args)...};
  }
};


// XXX this doesn't forward tuple elements which are reference types correctly
//     because make_tuple() doesn't do that
template<class... Tuples>
__TUPLE_ANNOTATION
tuple_cat_result<Tuples...> tuple_cat(Tuples&&... tuples)
{
  return tuple_cat_apply(tuple_maker<tuple_cat_result<Tuples...>>{}, std::forward<Tuples>(tuples)...);
}
}
#else // THRUST_VARIADIC_TUPLE
#include <thrust/detail/tuple/tuple_helpers.h>

namespace thrust
{
namespace detail
{
namespace tuple_detail
{


// the type of concatenating two tuples
template<typename Tuple1, typename Tuple2, typename Enable = void>
struct tuple_cat_result2
  : tuple_cat_result2<
      typename tuple_append_result<     
        typename tuple_element<0,Tuple2>::type,
        Tuple1
      >::type,
      typename tuple_tail_result<
        Tuple2
      >::type
    >
{
};


// the type of concatenating a tuple with an empty tuple
template<typename Tuple1, typename Tuple2>
struct tuple_cat_result2<Tuple1, Tuple2, typename enable_if<tuple_size<Tuple2>::value == 0>::type>
  : identity_<Tuple1>
{};


template<typename Tuple1,
         typename Tuple2  = tuple<>,
         typename Tuple3  = tuple<>,
         typename Tuple4  = tuple<>,
         typename Tuple5  = tuple<>,
         typename Tuple6  = tuple<>,
         typename Tuple7  = tuple<>,
         typename Tuple8  = tuple<>,
         typename Tuple9  = tuple<>,
         typename Tuple10 = tuple<> >
struct tuple_cat_result;


template<typename Tuple1, typename Tuple2, typename Tuple3, typename Tuple4,
         typename Tuple5, typename Tuple6, typename Tuple7, typename Tuple8,
         typename Tuple9, typename Tuple10>
struct tuple_cat_result
{
  private:
    typedef typename tuple_cat_result2<
      Tuple9, Tuple10
    >::type tuple9_10;
  
    typedef typename tuple_cat_result2<
      Tuple8, tuple9_10
    >::type tuple8_10;
  
    typedef typename tuple_cat_result2<
      Tuple7, tuple8_10
    >::type tuple7_10;
  
    typedef typename tuple_cat_result2<
      Tuple6, tuple7_10
    >::type tuple6_10;
  
    typedef typename tuple_cat_result2<
      Tuple5, tuple6_10
    >::type tuple5_10;
  
    typedef typename tuple_cat_result2<
      Tuple4, tuple5_10
    >::type tuple4_10;
  
    typedef typename tuple_cat_result2<
      Tuple3, tuple4_10
    >::type tuple3_10;
  
    typedef typename tuple_cat_result2<
      Tuple2, tuple3_10
    >::type tuple2_10;

  public:
    typedef typename tuple_cat_result2<
      Tuple1, tuple2_10
    >::type type;
};



} // end tuple_detail


template<typename Tuple1 = thrust::tuple<>,
         typename Tuple2 = thrust::tuple<>,
         typename Tuple3 = thrust::tuple<>,
         typename Tuple4 = thrust::tuple<>,
         typename Tuple5 = thrust::tuple<>,
         typename Tuple6 = thrust::tuple<>,
         typename Tuple7 = thrust::tuple<>,
         typename Tuple8 = thrust::tuple<>,
         typename Tuple9 = thrust::tuple<>,
         typename Tuple10 = thrust::tuple<> >
struct tuple_cat_enable_if
  : enable_if<
      (tuple_size<Tuple1>::value +
       tuple_size<Tuple2>::value +
       tuple_size<Tuple3>::value +
       tuple_size<Tuple4>::value +
       tuple_size<Tuple5>::value +
       tuple_size<Tuple6>::value +
       tuple_size<Tuple7>::value +
       tuple_size<Tuple8>::value +
       tuple_size<Tuple9>::value +
       tuple_size<Tuple10>::value)
      <= 10,
      typename tuple_detail::tuple_cat_result<Tuple1,Tuple2,Tuple3,Tuple4,Tuple5,Tuple6,Tuple7,Tuple8,Tuple9,Tuple10>::type
    >
{};


} // end detail


// terminal case of tuple_cat()
template<typename Tuple>
inline __host__ __device__
typename detail::tuple_cat_enable_if<Tuple>::type
  tuple_cat(const Tuple& t, const thrust::tuple<> &)
{
  return t;
}


template<typename Tuple1, typename Tuple2>
inline __host__ __device__
typename detail::tuple_cat_enable_if<Tuple1,Tuple2>::type
  tuple_cat(const Tuple1 &t1, const Tuple2 &t2)
{
  typedef typename thrust::tuple_element<0,Tuple2>::type head_type;

  // recurse by appending the head of t2 to t1
  // and concatenating the result with the tail of t2
  namespace ns = thrust::detail::tuple_detail;
  return thrust::tuple_cat(ns::tuple_append<head_type>(t1, thrust::get<0>(t2)), ns::tuple_tail(t2));
}

template<typename Tuple1, typename... Tuples>
inline __host__ __device__
typename detail::tuple_cat_enable_if<Tuple1, Tuples...>::type
  tuple_cat(const Tuple1& t1, const Tuples&... ts)
{
    return thrust::tuple_cat(t1, thrust::tuple_cat(ts...));
}

} // end thrust
#endif // THRUST_VARIADIC_TUPLE
