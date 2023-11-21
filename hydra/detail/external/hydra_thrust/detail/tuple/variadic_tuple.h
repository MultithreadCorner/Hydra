// Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <stddef.h> // XXX instead of <cstddef> to WAR clang issue
#include <type_traits>
#include <utility>

#if (defined(__CUDA_ARCH__) && CUDA_VERSION <= 7000) && (defined(__GNUC__) && !defined(__clang__))
//#  warning "Variadic tuple workaround for CUDA <= 7.0"
template <class T, class... Args>
struct std__is_constructible : std::integral_constant<bool,__is_constructible(T, Args...)> { };
#else
template <class T, class... Args>
struct std__is_constructible : std::is_constructible<T, Args...> { };
#endif

// allow the user to define an annotation to apply to these functions
// by default, it attempts to be constexpr
#ifndef __TUPLE_ANNOTATION
#  if __cplusplus <= 201103L
#    define __TUPLE_ANNOTATION __device__ __host__
#  else
#    define __TUPLE_ANNOTATION __device__ __host__
#  endif
#  define __TUPLE_ANNOTATION_NEEDS_UNDEF
#endif

// allow the user to define a namespace for these functions
#ifndef __TUPLE_NAMESPACE
#define __TUPLE_NAMESPACE hydra_thrust
#define __TUPLE_NAMESPACE_NEEDS_UNDEF
#endif


namespace __TUPLE_NAMESPACE
{

template<class... Types> class tuple;

//} // end namespace


// specializations of stuff in std come before their use
//namespace std
//{


template<size_t, class> struct tuple_element;


template<size_t i>
struct tuple_element<i, __TUPLE_NAMESPACE::tuple<>> {};


template<class Type1, class... Types>
struct tuple_element<0, __TUPLE_NAMESPACE::tuple<Type1,Types...>>
{
  using type = Type1;
};


template<size_t i, class Type1, class... Types>
struct tuple_element<i, __TUPLE_NAMESPACE::tuple<Type1,Types...>>
{
  using type = typename tuple_element<i - 1, __TUPLE_NAMESPACE::tuple<Types...>>::type;
};


template<class> struct tuple_size;


template<class... Types>
struct tuple_size<__TUPLE_NAMESPACE::tuple<Types...>>
  : std::integral_constant<size_t, sizeof...(Types)>
{};

template<size_t i, class... UTypes>
__TUPLE_ANNOTATION
typename hydra_thrust::tuple_element<i, __TUPLE_NAMESPACE::tuple<UTypes...>>::type &
get(__TUPLE_NAMESPACE::tuple<UTypes...>& t);


template<size_t i, class... UTypes>
__TUPLE_ANNOTATION
const typename hydra_thrust::tuple_element<i, __TUPLE_NAMESPACE::tuple<UTypes...>>::type &
get(const __TUPLE_NAMESPACE::tuple<UTypes...>& t);


template<size_t i, class... UTypes>
__TUPLE_ANNOTATION
typename hydra_thrust::tuple_element<i, __TUPLE_NAMESPACE::tuple<UTypes...>>::type &&
get(__TUPLE_NAMESPACE::tuple<UTypes...>&& t);

//} // end std


//namespace __TUPLE_NAMESPACE
//{

// define variadic "and" operator 
// prefix with "__tuple" to avoid collisions with other implementations 
template <typename... Conditions>
  struct __tuple_and;

template<>
  struct __tuple_and<>
    : public std::true_type
{
};

template <typename Condition, typename... Conditions>
  struct __tuple_and<Condition, Conditions...>
    : public std::integral_constant<
        bool,
        Condition::value && __tuple_and<Conditions...>::value>
{
};

// XXX this implementation is based on Howard Hinnant's "tuple leaf" construction in libcxx


// define index sequence in case it is missing
// prefix this stuff with "tuple" to avoid collisions with other implementations
template<size_t... I> struct __tuple_index_sequence {};

template<size_t Start, typename Indices, size_t End>
struct __tuple_make_index_sequence_impl;

template<size_t Start, size_t... Indices, size_t End>
struct __tuple_make_index_sequence_impl<
  Start,
  __tuple_index_sequence<Indices...>, 
  End
>
{
  typedef typename __tuple_make_index_sequence_impl<
    Start + 1,
    __tuple_index_sequence<Indices..., Start>,
    End
  >::type type;
};

template<size_t End, size_t... Indices>
struct __tuple_make_index_sequence_impl<End, __tuple_index_sequence<Indices...>, End>
{
  typedef __tuple_index_sequence<Indices...> type;
};

template<size_t N>
using __tuple_make_index_sequence = typename __tuple_make_index_sequence_impl<0, __tuple_index_sequence<>, N>::type;


template<class T>
struct __tuple_use_empty_base_class_optimization
  : std::integral_constant<
      bool,
      std::is_empty<T>::value
#if __cplusplus >= 201402L
      && !std::is_final<T>::value
#endif
    >
{};


template<class T, bool = __tuple_use_empty_base_class_optimization<T>::value>
class __tuple_leaf_base
{
  public:
#if defined(__CUDACC__)
#pragma nv_exec_check_disable
#endif
    __TUPLE_ANNOTATION
    ~__tuple_leaf_base() = default;

#if defined(__CUDACC__)
#pragma nv_exec_check_disable
#endif
    __TUPLE_ANNOTATION
    __tuple_leaf_base() = default;

#if defined(__CUDACC__)
#pragma nv_exec_check_disable
#endif
    template<class U>
    __TUPLE_ANNOTATION
    __tuple_leaf_base(U&& arg) : val_(std::forward<U>(arg)) {}

    __TUPLE_ANNOTATION
    const T& const_get() const
    {
      return val_;
    }

    __TUPLE_ANNOTATION
    T& mutable_get()
    {
      return val_;
    }

  private:
    T val_;
};

template<class T>
class __tuple_leaf_base<T,true> : public T
{
  public:
    __TUPLE_ANNOTATION
    __tuple_leaf_base() = default;

    template<class U>
    __TUPLE_ANNOTATION
    __tuple_leaf_base(U&& arg) : T(std::forward<U>(arg)) {}

    __TUPLE_ANNOTATION
    const T& const_get() const
    {
      return *this;
    }
  
    __TUPLE_ANNOTATION
    T& mutable_get()
    {
      return *this;
    }
};

template<size_t I, class T>
class __tuple_leaf : public __tuple_leaf_base<T>
{
  private:
    using super_t = __tuple_leaf_base<T>;

  public:
    __TUPLE_ANNOTATION
    __tuple_leaf() = default;

    template<class U,
             class = typename std::enable_if<
               std__is_constructible<T,U>::value
             >::type>
    __TUPLE_ANNOTATION
    __tuple_leaf(U&& arg) : super_t(std::forward<U>(arg)) {}

    __TUPLE_ANNOTATION
    __tuple_leaf(const __tuple_leaf& other) : super_t(other.const_get()) {}

    __TUPLE_ANNOTATION
    __tuple_leaf(__tuple_leaf&& other) : super_t(std::forward<T>(other.mutable_get())) {}

    template<class U,
             class = typename std::enable_if<
               std__is_constructible<T,U>::value
             >::type>
    __TUPLE_ANNOTATION
    __tuple_leaf(const __tuple_leaf<I,U>& other) : super_t(other.const_get()) {}


    template<class U,
             class = typename std::enable_if<
               std::is_assignable<T,U>::value
             >::type>
    __TUPLE_ANNOTATION
    __tuple_leaf& operator=(const __tuple_leaf<I,U>& other)
    {
      this->mutable_get() = other.const_get();
      return *this;
    }
    
    __TUPLE_ANNOTATION
    __tuple_leaf& operator=(const __tuple_leaf& other)
    {
      this->mutable_get() = other.const_get();
      return *this;
    }

    __TUPLE_ANNOTATION
    __tuple_leaf& operator=(__tuple_leaf&& other)
    {
      this->mutable_get() = std::forward<T>(other.mutable_get());
      return *this;
    }

    template<class U,
             class = typename std::enable_if<
               std::is_assignable<T,U&&>::value
             >::type>
    __TUPLE_ANNOTATION
    __tuple_leaf& operator=(__tuple_leaf<I,U>&& other)
    {
      this->mutable_get() = std::forward<U>(other.mutable_get());
      return *this;
    }

    __TUPLE_ANNOTATION
    int swap(__tuple_leaf& other)
    {
      using hydra_thrust::swap;
      swap(this->mutable_get(), other.mutable_get());
      return 0;
    }
};

template<class... Args>
struct __type_list {};

template<size_t i, class... Args>
struct __type_at_impl;

template<size_t i, class Arg0, class... Args>
struct __type_at_impl<i, Arg0, Args...>
{
  using type = typename __type_at_impl<i-1, Args...>::type;
};

template<class Arg0, class... Args>
struct __type_at_impl<0, Arg0,Args...>
{
  using type = Arg0;
};

template<size_t i, class... Args>
using __type_at = typename __type_at_impl<i,Args...>::type;

template<class IndexSequence, class... Args>
class __tuple_base;

template<size_t... I, class... Types>
class __tuple_base<__tuple_index_sequence<I...>, Types...>
  : public __tuple_leaf<I,Types>...
{
  public:
    using leaf_types = __type_list<__tuple_leaf<I,Types>...>;

    __TUPLE_ANNOTATION
    __tuple_base() = default;

    __TUPLE_ANNOTATION
    __tuple_base(const Types&... args)
      : __tuple_leaf<I,Types>(args)...
    {}


    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               __tuple_and<
                 std__is_constructible<Types,UTypes&&>...
               >::value
             >::type>
    __TUPLE_ANNOTATION
    explicit __tuple_base(UTypes&&... args)
      : __tuple_leaf<I,Types>(std::forward<UTypes>(args))...
    {}


    __TUPLE_ANNOTATION
    __tuple_base(const __tuple_base& other)
      : __tuple_leaf<I,Types>(other.template const_leaf<I>())...
    {}


    __TUPLE_ANNOTATION
    __tuple_base(__tuple_base&& other)
      : __tuple_leaf<I,Types>(std::move(other.template mutable_leaf<I>()))...
    {}


    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               __tuple_and<
                 std__is_constructible<Types,const UTypes&>...
                >::value
             >::type>
    __TUPLE_ANNOTATION
    __tuple_base(const __tuple_base<__tuple_index_sequence<I...>,UTypes...>& other)
      : __tuple_leaf<I,Types>(other.template const_leaf<I>())...
    {}


    //template<class... UTypes,
    //         class = typename std::enable_if<
    //           (sizeof...(Types) == sizeof...(UTypes)) &&
    //           __tuple_and<
    //             std__is_constructible<Types,const UTypes&>...
    //            >::value
    //         >::type>
    //__TUPLE_ANNOTATION
    //__tuple_base(const hydra_thrust::tuple<UTypes...>& other)
    //  : __tuple_base{hydra_thrust::get<I>(other)...}
    //{}


    __TUPLE_ANNOTATION
    __tuple_base& operator=(const __tuple_base& other)
    {
      swallow((mutable_leaf<I>() = other.template const_leaf<I>())...);
      return *this;
    }

    __TUPLE_ANNOTATION
    __tuple_base& operator=(__tuple_base&& other)
    {
      swallow((mutable_leaf<I>() = std::move(other.template mutable_leaf<I>()))...);
      return *this;
    }

    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               __tuple_and<
                 std::is_assignable<Types,const UTypes&>...
                >::value
             >::type>
    __TUPLE_ANNOTATION
    __tuple_base& operator=(const __tuple_base<__tuple_index_sequence<I...>,UTypes...>& other)
    {
      swallow((mutable_leaf<I>() = other.template const_leaf<I>())...);
      return *this;
    }

    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               __tuple_and<
                 std::is_assignable<Types,UTypes&&>...
               >::value
             >::type>
    __TUPLE_ANNOTATION
    __tuple_base& operator=(__tuple_base<__tuple_index_sequence<I...>,UTypes...>&& other)
    {
      swallow((mutable_leaf<I>() = std::move(other.template mutable_leaf<I>()))...);
      return *this;
    }

    template<class UType1, class UType2,
             class = typename std::enable_if<
               (sizeof...(Types) == 2) &&
               __tuple_and<
                 std::is_assignable<__type_at<                            0,Types...>,const UType1&>,
                 std::is_assignable<__type_at<sizeof...(Types) == 2 ? 1 : 0,Types...>,const UType2&>
               >::value
             >::type>
    __TUPLE_ANNOTATION
    __tuple_base& operator=(const hydra_thrust::pair<UType1,UType2>& p)
    {
      mutable_get<0>() = p.first;
      mutable_get<1>() = p.second;
      return *this;
    }

    template<class UType1, class UType2,
             class = typename std::enable_if<
               (sizeof...(Types) == 2) &&
               __tuple_and<
                 std::is_assignable<__type_at<                            0,Types...>,UType1&&>,
                 std::is_assignable<__type_at<sizeof...(Types) == 2 ? 1 : 0,Types...>,UType2&&>
               >::value
             >::type>
    __TUPLE_ANNOTATION
    __tuple_base& operator=(hydra_thrust::pair<UType1,UType2>&& p)
    {
      mutable_get<0>() = std::move(p.first);
      mutable_get<1>() = std::move(p.second);
      return *this;
    }

    template<size_t i>
    __TUPLE_ANNOTATION
    const __tuple_leaf<i,__type_at<i,Types...>>& const_leaf() const
    {
      return *this;
    }

    template<size_t i>
    __TUPLE_ANNOTATION
    __tuple_leaf<i,__type_at<i,Types...>>& mutable_leaf()
    {
      return *this;
    }

    template<size_t i>
    __TUPLE_ANNOTATION
    __tuple_leaf<i,__type_at<i,Types...>>&& move_leaf() &&
    {
      return std::move(*this);
    }

    __TUPLE_ANNOTATION
    void swap(__tuple_base& other)
    {
      swallow(__tuple_leaf<I,Types>::swap(other)...);
    }

    template<size_t i>
    __TUPLE_ANNOTATION
    const __type_at<i,Types...>& const_get() const
    {
      return const_leaf<i>().const_get();
    }

    template<size_t i>
    __TUPLE_ANNOTATION
    __type_at<i,Types...>& mutable_get()
    {
      return mutable_leaf<i>().mutable_get();
    }

    //// enable conversion to Tuple-like things
    //template<class... UTypes,
    //         class = typename std::enable_if<
    //           (sizeof...(Types) == sizeof...(UTypes)) &&
    //           __tuple_and<
    //             std__is_constructible<Types,const UTypes&>...
    //            >::value
    //         >::type>
    //__TUPLE_ANNOTATION
    //operator hydra_thrust::tuple<UTypes...> () const
    //{
    //  return hydra_thrust::tuple<UTypes...>(const_get<I>()...);
    //}

  private:
    template<class... Args>
    __TUPLE_ANNOTATION
    static void swallow(Args&&...) {}
};


template<class... Types>
class tuple
{
  private:
    using base_type = __tuple_base<__tuple_make_index_sequence<sizeof...(Types)>, Types...>;
    base_type base_;

    __TUPLE_ANNOTATION
    base_type& base()
    {
      return base_;
    }

    __TUPLE_ANNOTATION
    const base_type& base() const
    {
      return base_;
    }

  public:
    __TUPLE_ANNOTATION
    tuple() : base_{} {};

    __TUPLE_ANNOTATION
    explicit tuple(const Types&... args)
      : base_{args...}
    {}

    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               __tuple_and<
                 std__is_constructible<Types,UTypes&&>...
               >::value
             >::type>
    __TUPLE_ANNOTATION
    explicit tuple(UTypes&&... args)
      : base_{std::forward<UTypes>(args)...}
    {}

    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
                 __tuple_and<
                   std__is_constructible<Types,const UTypes&>...
                 >::value
             >::type>
    __TUPLE_ANNOTATION
    tuple(const tuple<UTypes...>& other)
      : base_{other.base()}
    {}

    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
                 __tuple_and<
                   std__is_constructible<Types,UTypes&&>...
                 >::value
             >::type>
    __TUPLE_ANNOTATION
    tuple(tuple<UTypes...>&& other)
      : base_{std::move(other.base())}
    {}

    template<class UType1, class UType2,
             class = typename std::enable_if<
               (sizeof...(Types) == 2) &&
               __tuple_and<
                 std__is_constructible<__type_at<                            0,Types...>,const UType1&>,
                 std__is_constructible<__type_at<sizeof...(Types) == 2 ? 1 : 0,Types...>,const UType2&>
               >::value
             >::type>
    __TUPLE_ANNOTATION
    tuple(const hydra_thrust::pair<UType1,UType2>& p)
      : base_{p.first, p.second}
    {}

    template<class UType1, class UType2,
             class = typename std::enable_if<
               (sizeof...(Types) == 2) &&
               __tuple_and<
                 std__is_constructible<__type_at<                            0,Types...>,UType1&&>,
                 std__is_constructible<__type_at<sizeof...(Types) == 2 ? 1 : 0,Types...>,UType2&&>
               >::value
             >::type>
    __TUPLE_ANNOTATION
    tuple(hydra_thrust::pair<UType1,UType2>&& p)
      : base_{std::move(p.first), std::move(p.second)}
    {}

    __TUPLE_ANNOTATION
    tuple(const tuple& other)
      : base_{other.base()}
    {}

    __TUPLE_ANNOTATION
    tuple(tuple&& other)
      : base_{std::move(other.base())}
    {}

    //template<class... UTypes,
    //         class = typename std::enable_if<
    //           (sizeof...(Types) == sizeof...(UTypes)) &&
    //             __tuple_and<
    //               std__is_constructible<Types,const UTypes&>...
    //             >::value
    //         >::type>
    //__TUPLE_ANNOTATION
    //tuple(const hydra_thrust::tuple<UTypes...>& other)
    //  : base_{other}
    //{}

    __TUPLE_ANNOTATION
    tuple& operator=(const tuple& other)
    {
      base().operator=(other.base());
      return *this;
    }

    __TUPLE_ANNOTATION
    tuple& operator=(tuple&& other)
    {
      base().operator=(std::move(other.base()));
      return *this;
    }

    // XXX needs enable_if
    template<class... UTypes>
    __TUPLE_ANNOTATION
    tuple& operator=(const tuple<UTypes...>& other)
    {
      base().operator=(other.base());
      return *this;
    }

    // XXX needs enable_if
    template<class... UTypes>
    __TUPLE_ANNOTATION
    tuple& operator=(tuple<UTypes...>&& other)
    {
      base().operator=(other.base());
      return *this;
    }

    template<class UType1, class UType2,
             class = typename std::enable_if<
               (sizeof...(Types) == 2) &&
               __tuple_and<
                 std::is_assignable<__type_at<                            0,Types...>,const UType1&>,
                 std::is_assignable<__type_at<sizeof...(Types) == 2 ? 1 : 0,Types...>,const UType2&>
               >::value
             >::type>
    __TUPLE_ANNOTATION
    tuple& operator=(const hydra_thrust::pair<UType1,UType2>& p)
    {
      base().operator=(p);
      return *this;
    }

    template<class UType1, class UType2,
             class = typename std::enable_if<
               (sizeof...(Types) == 2) &&
               __tuple_and<
                 std::is_assignable<__type_at<                            0,Types...>,UType1&&>,
                 std::is_assignable<__type_at<sizeof...(Types) == 2 ? 1 : 0,Types...>,UType2&&>
               >::value
             >::type>
    __TUPLE_ANNOTATION
    tuple& operator=(hydra_thrust::pair<UType1,UType2>&& p)
    {
      base().operator=(std::move(p));
      return *this;
    }

    __TUPLE_ANNOTATION
    void swap(tuple& other)
    {
      base().swap(other.base());
    }

    //// enable conversion to Tuple-like things
    //template<class... UTypes,
    //         class = typename std::enable_if<
    //           (sizeof...(Types) == sizeof...(UTypes)) &&
    //           __tuple_and<
    //             std__is_constructible<Types,const UTypes&>...
    //            >::value
    //         >::type>
    //__TUPLE_ANNOTATION
    //operator hydra_thrust::tuple<UTypes...> () const
    //{
    //  return static_cast<hydra_thrust::tuple<UTypes...>>(base());
    //}

  private:
    template<class... UTypes>
    friend class tuple;

    template<size_t i>
    __TUPLE_ANNOTATION
    const typename hydra_thrust::tuple_element<i,tuple>::type& const_get() const
    {
      return base().template const_get<i>();
    }

    template<size_t i>
    __TUPLE_ANNOTATION
    typename hydra_thrust::tuple_element<i,tuple>::type& mutable_get()
    {
      return base().template mutable_get<i>();
    }

  public:
    template<size_t i, class... UTypes>
    friend __TUPLE_ANNOTATION
    typename hydra_thrust::tuple_element<i, __TUPLE_NAMESPACE::tuple<UTypes...>>::type &
    hydra_thrust::get(__TUPLE_NAMESPACE::tuple<UTypes...>& t);


    template<size_t i, class... UTypes>
    friend __TUPLE_ANNOTATION
    const typename hydra_thrust::tuple_element<i, __TUPLE_NAMESPACE::tuple<UTypes...>>::type &
    hydra_thrust::get(const __TUPLE_NAMESPACE::tuple<UTypes...>& t);


    template<size_t i, class... UTypes>
    friend __TUPLE_ANNOTATION
    typename hydra_thrust::tuple_element<i, __TUPLE_NAMESPACE::tuple<UTypes...>>::type &&
    hydra_thrust::get(__TUPLE_NAMESPACE::tuple<UTypes...>&& t);
};


template<>
class tuple<>
{
  public:
    __TUPLE_ANNOTATION
    void swap(tuple&){}
};


template<class... Types>
__TUPLE_ANNOTATION
void swap(tuple<Types...>& a, tuple<Types...>& b)
{
  a.swap(b);
}


template<class... Types>
__TUPLE_ANNOTATION
tuple<typename std::decay<Types>::type...> make_tuple(Types&&... args)
{
  return tuple<typename std::decay<Types>::type...>(std::forward<Types>(args)...);
}


template<class... Types>
__TUPLE_ANNOTATION
tuple<Types&...> tie(Types&... args)
{
  return tuple<Types&...>(args...);
}


template<class... Args>
__TUPLE_ANNOTATION
__TUPLE_NAMESPACE::tuple<Args&&...> forward_as_tuple(Args&&... args)
{
  return __TUPLE_NAMESPACE::tuple<Args&&...>(std::forward<Args>(args)...);
}


struct __ignore_t
{
  template<class T>
  __TUPLE_ANNOTATION
  const __ignore_t operator=(T&&) const
  {
    return *this;
  }
};


constexpr __ignore_t ignore{};


template<size_t I, class T, class... Types>
struct __find_exactly_one_impl;


template<size_t I, class T, class U, class... Types>
struct __find_exactly_one_impl<I,T,U,Types...> : __find_exactly_one_impl<I+1, T, Types...> {};


template<size_t I, class T, class... Types>
struct __find_exactly_one_impl<I,T,T,Types...> : std::integral_constant<size_t, I>
{
  static_assert(__find_exactly_one_impl<I,T,Types...>::value == -1, "type can only occur once in type list");
};


template<size_t I, class T>
struct __find_exactly_one_impl<I,T> : std::integral_constant<int, -1> {};


template<class T, class... Types>
struct __find_exactly_one : __find_exactly_one_impl<0,T,Types...>
{
  static_assert(__find_exactly_one::value != -1, "type not found in type list");
};


//} // end namespace


// implement hydra_thrust::get()
//namespace std
//{


template<size_t i, class... UTypes>
__TUPLE_ANNOTATION
typename hydra_thrust::tuple_element<i, __TUPLE_NAMESPACE::tuple<UTypes...>>::type &
  get(__TUPLE_NAMESPACE::tuple<UTypes...>& t)
{
  return t.template mutable_get<i>();
}


template<size_t i, class... UTypes>
__TUPLE_ANNOTATION
const typename hydra_thrust::tuple_element<i, __TUPLE_NAMESPACE::tuple<UTypes...>>::type &
  get(const __TUPLE_NAMESPACE::tuple<UTypes...>& t)
{
  return t.template const_get<i>();
}


template<size_t i, class... UTypes>
__TUPLE_ANNOTATION
typename hydra_thrust::tuple_element<i, __TUPLE_NAMESPACE::tuple<UTypes...>>::type &&
  get(__TUPLE_NAMESPACE::tuple<UTypes...>&& t)
{
  using type = typename hydra_thrust::tuple_element<i, __TUPLE_NAMESPACE::tuple<UTypes...>>::type;

  auto&& leaf = static_cast<__TUPLE_NAMESPACE::__tuple_leaf<i,type>&&>(t.base());

  return static_cast<type&&>(leaf.mutable_get());
}


template<class T, class... Types>
__TUPLE_ANNOTATION
T& get(__TUPLE_NAMESPACE::tuple<Types...>& t)
{
  return hydra_thrust::get<__TUPLE_NAMESPACE::__find_exactly_one<T,Types...>::value>(t);
}


template<class T, class... Types>
__TUPLE_ANNOTATION
const T& get(const __TUPLE_NAMESPACE::tuple<Types...>& t)
{
  return hydra_thrust::get<__TUPLE_NAMESPACE::__find_exactly_one<T,Types...>::value>(t);
}


template<class T, class... Types>
__TUPLE_ANNOTATION
T&& get(__TUPLE_NAMESPACE::tuple<Types...>&& t)
{
  return hydra_thrust::get<__TUPLE_NAMESPACE::__find_exactly_one<T,Types...>::value>(std::move(t));
}


//} // end std


// implement relational operators
//namespace __TUPLE_NAMESPACE
//{


__TUPLE_ANNOTATION
  inline bool __tuple_all()
{
  return true;
}


__TUPLE_ANNOTATION
  inline bool __tuple_all(bool t)
{
  return t;
}


template<typename... Bools>
__TUPLE_ANNOTATION
  bool __tuple_all(bool t, Bools... ts)
{
  return t && __tuple_all(ts...);
}


template<class... TTypes, class... UTypes>
__TUPLE_ANNOTATION
  bool operator==(const tuple<TTypes...>& t, const tuple<UTypes...>& u);


template<class... TTypes, class... UTypes, size_t... I>
__TUPLE_ANNOTATION
  bool __tuple_eq(const tuple<TTypes...>& t, const tuple<UTypes...>& u, __tuple_index_sequence<I...>)
{
  return __tuple_all((hydra_thrust::get<I>(t) == hydra_thrust::get<I>(u))...);
}


template<class... TTypes, class... UTypes>
__TUPLE_ANNOTATION
  bool operator==(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return __tuple_eq(t, u, __tuple_make_index_sequence<sizeof...(TTypes)>{});
}


template<class... TTypes, class... UTypes>
__TUPLE_ANNOTATION
  bool operator<(const tuple<TTypes...>& t, const tuple<UTypes...>& u);


template<class... TTypes, class... UTypes>
__TUPLE_ANNOTATION
  bool __tuple_lt(const tuple<TTypes...>& t, const tuple<UTypes...>& u, __tuple_index_sequence<>)
{
  return false;
}


template<size_t I, class... TTypes, class... UTypes, size_t... Is>
__TUPLE_ANNOTATION
  bool __tuple_lt(const tuple<TTypes...>& t, const tuple<UTypes...>& u, __tuple_index_sequence<I, Is...>)
{
  return (   hydra_thrust::get<I>(t) < hydra_thrust::get<I>(u)
          || (!(hydra_thrust::get<I>(u) < hydra_thrust::get<I>(t))
              && __tuple_lt(t, u, typename __tuple_make_index_sequence_impl<I+1, __tuple_index_sequence<>, sizeof...(TTypes)>::type{})));
}


template<class... TTypes, class... UTypes>
__TUPLE_ANNOTATION
  bool operator<(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return __tuple_lt(t, u, __tuple_make_index_sequence<sizeof...(TTypes)>{});
}


template<class... TTypes, class... UTypes>
__TUPLE_ANNOTATION
  bool operator!=(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return !(t == u);
}


template<class... TTypes, class... UTypes>
__TUPLE_ANNOTATION
  bool operator>(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return u < t;
}


template<class... TTypes, class... UTypes>
__TUPLE_ANNOTATION
  bool operator<=(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return !(u < t);
}


template<class... TTypes, class... UTypes>
__TUPLE_ANNOTATION
  bool operator>=(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return !(t < u);
}


} // end namespace


#include <hydra/detail/external/hydra_thrust/detail/tuple/tuple_cat.h>

#ifdef __TUPLE_ANNOTATION_NEEDS_UNDEF
#undef __TUPLE_ANNOTATION
#undef __TUPLE_ANNOTATION_NEEDS_UNDEF
#endif

#ifdef __TUPLE_NAMESPACE_NEEDS_UNDEF
#undef __TUPLE_NAMESPACE
#undef __TUPLE_NAMESPACE_NEEDS_UNDEF
#endif

