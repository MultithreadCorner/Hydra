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
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/pair.h>
#include <hydra/detail/external/hydra_thrust/detail/reference_forward_declaration.h>

namespace hydra_thrust
{
namespace detail
{

#ifdef HYDRA_THRUST_VARIADIC_TUPLE

template<
  typename... Types
>
  class tuple_of_iterator_references
    : public hydra_thrust::tuple<Types...>
{
  private:
    typedef hydra_thrust::tuple<Types...> super_t;

  public:
    // allow implicit construction from tuple<refs>
    inline __host__ __device__
    tuple_of_iterator_references(const super_t &other)
      : super_t(other)
    {}

    // allow assignment from tuples
    // XXX might be worthwhile to guard this with an enable_if is_assignable
    template<typename... OtherTypes>
    inline __host__ __device__
    tuple_of_iterator_references &operator=(const hydra_thrust::tuple<OtherTypes...> &other)
    {
      super_t::operator=(other);
      return *this;
    }

     // allow assignment from tuples
    // XXX might be worthwhile to guard this with an enable_if is_assignable
    /*
    __hydra_thrust_exec_check_disable__
    template<typename U1, typename U2>
    inline __host__ __device__
    tuple_of_iterator_references &operator=(const detail::cons<U1,U2> &other)
    {
      super_t::operator=(other);
      return *this;
    }
    */

    // allow assignment from pairs
    // XXX might be worthwhile to guard this with an enable_if is_assignable
    template<typename U1, typename U2>
    inline __host__ __device__
    tuple_of_iterator_references &operator=(const hydra_thrust::pair<U1,U2> &other)
    {
      super_t::operator=(other);
      return *this;
    }

    // allow assignment from reference<tuple>
    // XXX perhaps we should generalize to reference<T>
    //     we could captures reference<pair> this way
    template<typename Pointer, typename Derived,
             typename... OtherTypes>
    inline __host__ __device__
// XXX gcc-4.2 crashes on is_assignable
//    typename hydra_thrust::detail::enable_if<
//      hydra_thrust::detail::is_assignable<
//        super_t,
//        const hydra_thrust::tuple<OtherTypes...>
//      >::value,
//      tuple_of_iterator_references &
//    >::type
    tuple_of_iterator_references &
    operator=(const hydra_thrust::reference<hydra_thrust::tuple<OtherTypes...>, Pointer, Derived> &other)
    {
      typedef hydra_thrust::tuple<OtherTypes...> tuple_type;

      // XXX perhaps this could be accelerated
      tuple_type other_tuple = other;
      super_t::operator=(other_tuple);
      return *this;
    }


    // duplicate hydra_thrust::tuple's constructors
#if 0   // C++11 constructor inheritance    -- not supported on gcc 4.7, try disabling if it causes problems
    using super_t::super_t;
#else
    inline __host__ __device__
    tuple_of_iterator_references() {}

    template<typename... UTypes>
    inline __host__ __device__
    tuple_of_iterator_references(typename access_traits<UTypes>::parameter_type... ts)
      : super_t(ts...)
    {}
#endif
};


} // end detail

// define tuple_size, tuple_element, etc.
template<class... Types>
struct tuple_size<detail::tuple_of_iterator_references<Types...>>
  : std::integral_constant<size_t, sizeof...(Types)>
{};

template<size_t i>
struct tuple_element<i, detail::tuple_of_iterator_references<>> {};


template<class Type1, class... Types>
struct tuple_element<0, detail::tuple_of_iterator_references<Type1,Types...>>
{
  using type = Type1;
};


template<size_t i, class Type1, class... Types>
struct tuple_element<i, detail::tuple_of_iterator_references<Type1,Types...>>
{
  using type = typename tuple_element<i - 1, detail::tuple_of_iterator_references<Types...>>::type;
};

template< typename ...T, typename ...U>
inline __host__ __device__
void swap(detail::tuple_of_iterator_references<T...> x,
		detail::tuple_of_iterator_references<U...> y)
{
  x.swap(y);
}

#else


template<
  typename T0, typename T1, typename T2,
  typename T3, typename T4, typename T5,
  typename T6, typename T7, typename T8,
  typename T9
>
  class tuple_of_iterator_references
    : public hydra_thrust::tuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
{
  private:
    typedef hydra_thrust::tuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9> super_t;

  public:
    // allow implicit construction from tuple<refs>
    inline __host__ __device__
    tuple_of_iterator_references(const super_t &other)
      : super_t(other)
    {}

    // allow assignment from tuples
    // XXX might be worthwhile to guard this with an enable_if is_assignable
    __hydra_thrust_exec_check_disable__
    template<typename U1, typename U2>
    inline __host__ __device__
    tuple_of_iterator_references &operator=(const detail::cons<U1,U2> &other)
    {
      super_t::operator=(other);
      return *this;
    }

    // allow assignment from pairs
    // XXX might be worthwhile to guard this with an enable_if is_assignable
    __hydra_thrust_exec_check_disable__
    template<typename U1, typename U2>
    inline __host__ __device__
    tuple_of_iterator_references &operator=(const hydra_thrust::pair<U1,U2> &other)
    {
      super_t::operator=(other);
      return *this;
    }

    // allow assignment from reference<tuple>
    // XXX perhaps we should generalize to reference<T>
    //     we could captures reference<pair> this way
    __hydra_thrust_exec_check_disable__
    template<typename U0, typename U1, typename U2,
             typename U3, typename U4, typename U5,
             typename U6, typename U7, typename U8,
             typename U9,
             typename Pointer, typename Derived>
    inline __host__ __device__
// XXX gcc-4.2 crashes on is_assignable
//    typename hydra_thrust::detail::enable_if<
//      hydra_thrust::detail::is_assignable<
//        super_t,
//        const hydra_thrust::tuple<U0,U1,U2,U3,U4,U5,U6,U7,U8,U9>
//      >::value,
//      tuple_of_iterator_references &
//    >::type
    tuple_of_iterator_references&
    operator=(const hydra_thrust::reference<hydra_thrust::tuple<U0,U1,U2,U3,U4,U5,U6,U7,U8,U9>, Pointer, Derived> &other)
    {
      typedef hydra_thrust::tuple<U0,U1,U2,U3,U4,U5,U6,U7,U8,U9> tuple_type;

      // XXX perhaps this could be accelerated
      tuple_type other_tuple = other;
      super_t::operator=(other_tuple);
      return *this;
    }


    // duplicate hydra_thrust::tuple's constructors
    inline __host__ __device__
    tuple_of_iterator_references() {}

    inline __host__ __device__
    tuple_of_iterator_references(typename access_traits<T0>::parameter_type t0)
      : super_t(t0,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()))
    {}

    inline __host__ __device__
    tuple_of_iterator_references(typename access_traits<T0>::parameter_type t0,
                                 typename access_traits<T1>::parameter_type t1)
      : super_t(t0, t1,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()))
    {}

    inline __host__ __device__
    tuple_of_iterator_references(typename access_traits<T0>::parameter_type t0,
                                 typename access_traits<T1>::parameter_type t1,
                                 typename access_traits<T2>::parameter_type t2)
      : super_t(t0, t1, t2,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()))
    {}

    inline __host__ __device__
    tuple_of_iterator_references(typename access_traits<T0>::parameter_type t0,
                                 typename access_traits<T1>::parameter_type t1,
                                 typename access_traits<T2>::parameter_type t2,
                                 typename access_traits<T3>::parameter_type t3)
      : super_t(t0, t1, t2, t3,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()))
    {}

    inline __host__ __device__
    tuple_of_iterator_references(typename access_traits<T0>::parameter_type t0,
                                 typename access_traits<T1>::parameter_type t1,
                                 typename access_traits<T2>::parameter_type t2,
                                 typename access_traits<T3>::parameter_type t3,
                                 typename access_traits<T4>::parameter_type t4)
      : super_t(t0, t1, t2, t3, t4,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()))
    {}

    inline __host__ __device__
    tuple_of_iterator_references(typename access_traits<T0>::parameter_type t0,
                                 typename access_traits<T1>::parameter_type t1,
                                 typename access_traits<T2>::parameter_type t2,
                                 typename access_traits<T3>::parameter_type t3,
                                 typename access_traits<T4>::parameter_type t4,
                                 typename access_traits<T5>::parameter_type t5)
      : super_t(t0, t1, t2, t3, t4, t5,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()))
    {}

    inline __host__ __device__
    tuple_of_iterator_references(typename access_traits<T0>::parameter_type t0,
                                 typename access_traits<T1>::parameter_type t1,
                                 typename access_traits<T2>::parameter_type t2,
                                 typename access_traits<T3>::parameter_type t3,
                                 typename access_traits<T4>::parameter_type t4,
                                 typename access_traits<T5>::parameter_type t5,
                                 typename access_traits<T6>::parameter_type t6)
      : super_t(t0, t1, t2, t3, t4, t5, t6,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()))
    {}

    inline __host__ __device__
    tuple_of_iterator_references(typename access_traits<T0>::parameter_type t0,
                                 typename access_traits<T1>::parameter_type t1,
                                 typename access_traits<T2>::parameter_type t2,
                                 typename access_traits<T3>::parameter_type t3,
                                 typename access_traits<T4>::parameter_type t4,
                                 typename access_traits<T5>::parameter_type t5,
                                 typename access_traits<T6>::parameter_type t6,
                                 typename access_traits<T7>::parameter_type t7)
      : super_t(t0, t1, t2, t3, t4, t5, t6, t7,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()))
    {}

    inline __host__ __device__
    tuple_of_iterator_references(typename access_traits<T0>::parameter_type t0,
                                 typename access_traits<T1>::parameter_type t1,
                                 typename access_traits<T2>::parameter_type t2,
                                 typename access_traits<T3>::parameter_type t3,
                                 typename access_traits<T4>::parameter_type t4,
                                 typename access_traits<T5>::parameter_type t5,
                                 typename access_traits<T6>::parameter_type t6,
                                 typename access_traits<T7>::parameter_type t7,
                                 typename access_traits<T8>::parameter_type t8)
      : super_t(t0, t1, t2, t3, t4, t5, t6, t7, t8,
                static_cast<const null_type&>(null_type()))
    {}

    inline __host__ __device__
    tuple_of_iterator_references(typename access_traits<T0>::parameter_type t0,
                                 typename access_traits<T1>::parameter_type t1,
                                 typename access_traits<T2>::parameter_type t2,
                                 typename access_traits<T3>::parameter_type t3,
                                 typename access_traits<T4>::parameter_type t4,
                                 typename access_traits<T5>::parameter_type t5,
                                 typename access_traits<T6>::parameter_type t6,
                                 typename access_traits<T7>::parameter_type t7,
                                 typename access_traits<T8>::parameter_type t8,
                                 typename access_traits<T9>::parameter_type t9)
      : super_t(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9)
    {}
};


// this overload of swap() permits swapping tuple_of_iterator_references returned as temporaries from
// iterator dereferences
template<
  typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9,
  typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9
>
inline __host__ __device__
void swap(tuple_of_iterator_references<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9> x,
          tuple_of_iterator_references<U0,U1,U2,U3,U4,U5,U6,U7,U8,U9> y)
{
  x.swap(y);
}
} // end detail
#endif

} // end hydra_hydra_thrust

