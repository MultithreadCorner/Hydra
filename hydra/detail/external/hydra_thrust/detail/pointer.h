/*
 *  Copyright 2008-2021 NVIDIA Corporation
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

/*! \file
 *  \brief A pointer to a variable which resides in memory associated with a
 *  system.
 */

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_adaptor.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/iterator_traversal_tags.h>
#include <hydra/detail/external/hydra_thrust/type_traits/remove_cvref.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/pointer_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/reference_forward_declaration.h>
#include <ostream>
#include <cstddef>

HYDRA_THRUST_NAMESPACE_BEGIN

template <typename Element, typename Tag, typename Reference = use_default, typename Derived = use_default>
class pointer;

// Specialize `hydra_thrust::iterator_traits` to avoid problems with the name of
// pointer's constructor shadowing its nested pointer type. We do this before
// pointer is defined so the specialization is correctly used inside the
// definition.
template <typename Element, typename Tag, typename Reference, typename Derived>
struct iterator_traits<hydra_thrust::pointer<Element, Tag, Reference, Derived>>
{
  using pointer           = hydra_thrust::pointer<Element, Tag, Reference, Derived>;
  using iterator_category = typename pointer::iterator_category;
  using value_type        = typename pointer::value_type;
  using difference_type   = typename pointer::difference_type;
  using reference         = typename pointer::reference;
};

HYDRA_THRUST_NAMESPACE_END

namespace std
{

template <typename Element, typename Tag, typename Reference, typename Derived>
struct iterator_traits<HYDRA_THRUST_NS_QUALIFIER::pointer<Element, Tag, Reference, Derived>>
{
  using pointer           = HYDRA_THRUST_NS_QUALIFIER::pointer<Element, Tag, Reference, Derived>;
  using iterator_category = typename pointer::iterator_category;
  using value_type        = typename pointer::value_type;
  using difference_type   = typename pointer::difference_type;
  using reference         = typename pointer::reference;
};

} // namespace std

HYDRA_THRUST_NAMESPACE_BEGIN

namespace detail
{

// this metafunction computes the type of iterator_adaptor hydra_thrust::pointer should inherit from
template<typename Element, typename Tag, typename Reference, typename Derived>
  struct pointer_base
{
  // void pointers should have no element type
  // note that we remove_cv from the Element type to get the value_type
  typedef typename hydra_thrust::detail::eval_if<
    hydra_thrust::detail::is_void<typename hydra_thrust::remove_cvref<Element>::type>::value,
    hydra_thrust::detail::identity_<void>,
    hydra_thrust::detail::remove_cv<Element>
  >::type value_type;

  // if no Derived type is given, just use pointer
  typedef typename hydra_thrust::detail::eval_if<
    hydra_thrust::detail::is_same<Derived,use_default>::value,
    hydra_thrust::detail::identity_<pointer<Element,Tag,Reference,Derived> >,
    hydra_thrust::detail::identity_<Derived>
  >::type derived_type;

  // void pointers should have no reference type
  // if no Reference type is given, just use reference
  typedef typename hydra_thrust::detail::eval_if<
    hydra_thrust::detail::is_void<typename hydra_thrust::remove_cvref<Element>::type>::value,
    hydra_thrust::detail::identity_<void>,
    hydra_thrust::detail::eval_if<
      hydra_thrust::detail::is_same<Reference,use_default>::value,
      hydra_thrust::detail::identity_<reference<Element,derived_type> >,
      hydra_thrust::detail::identity_<Reference>
    >
  >::type reference_type;

  typedef hydra_thrust::iterator_adaptor<
    derived_type,                        // pass along the type of our Derived class to iterator_adaptor
    Element *,                           // we adapt a raw pointer
    value_type,                          // the value type
    Tag,                                 // system tag
    hydra_thrust::random_access_traversal_tag, // pointers have random access traversal
    reference_type,                      // pass along our Reference type
    std::ptrdiff_t
  > type;
}; // end pointer_base


} // end detail


// the base type for all of hydra_thrust's tagged pointers.
// for reasonable pointer-like semantics, derived types should reimplement the following:
// 1. no-argument constructor
// 2. constructor from OtherElement *
// 3. constructor from OtherPointer related by convertibility
// 4. constructor from OtherPointer to void
// 5. assignment from OtherPointer related by convertibility
// These should just call the corresponding members of pointer.
template<typename Element, typename Tag, typename Reference, typename Derived>
  class pointer
    : public hydra_thrust::detail::pointer_base<Element,Tag,Reference,Derived>::type
{
  private:
    typedef typename hydra_thrust::detail::pointer_base<Element,Tag,Reference,Derived>::type         super_t;

    typedef typename hydra_thrust::detail::pointer_base<Element,Tag,Reference,Derived>::derived_type derived_type;

    // friend iterator_core_access to give it access to dereference
    friend class hydra_thrust::iterator_core_access;

    __host__ __device__
    typename super_t::reference dereference() const;

    // don't provide access to this part of super_t's interface
    using super_t::base;
    using typename super_t::base_type;

  public:
    typedef typename super_t::base_type raw_pointer;

    // constructors

    __host__ __device__
    pointer();

    // NOTE: This is needed so that Thrust smart pointers can be used in
    // `std::unique_ptr`.
    __host__ __device__
    pointer(std::nullptr_t);

    // OtherValue shall be convertible to Value
    // XXX consider making the pointer implementation a template parameter which defaults to Element *
    template<typename OtherElement>
    __host__ __device__
    explicit pointer(OtherElement *ptr);

    // OtherPointer's element_type shall be convertible to Element
    // OtherPointer's system shall be convertible to Tag
    template<typename OtherPointer>
    __host__ __device__
    pointer(const OtherPointer &other,
            typename hydra_thrust::detail::enable_if_pointer_is_convertible<
              OtherPointer,
              pointer<Element,Tag,Reference,Derived>
            >::type * = 0);

    // OtherPointer's element_type shall be void
    // OtherPointer's system shall be convertible to Tag
    template<typename OtherPointer>
    __host__ __device__
    explicit
    pointer(const OtherPointer &other,
            typename hydra_thrust::detail::enable_if_void_pointer_is_system_convertible<
              OtherPointer,
              pointer<Element,Tag,Reference,Derived>
            >::type * = 0);

    // assignment

    // NOTE: This is needed so that Thrust smart pointers can be used in
    // `std::unique_ptr`.
    __host__ __device__
    derived_type& operator=(std::nullptr_t);

    // OtherPointer's element_type shall be convertible to Element
    // OtherPointer's system shall be convertible to Tag
    template<typename OtherPointer>
    __host__ __device__
    typename hydra_thrust::detail::enable_if_pointer_is_convertible<
      OtherPointer,
      pointer,
      derived_type &
    >::type
    operator=(const OtherPointer &other);

    // observers

    __host__ __device__
    Element *get() const;

    __host__ __device__
    Element *operator->() const;

    // NOTE: This is needed so that Thrust smart pointers can be used in
    // `std::unique_ptr`.
    __host__ __device__
    explicit operator bool() const;

    __host__ __device__
    static derived_type pointer_to(typename hydra_thrust::detail::pointer_traits_detail::pointer_to_param<Element>::type r)
    {
      return hydra_thrust::detail::pointer_traits<derived_type>::pointer_to(r);
    }
}; // end pointer

// Output stream operator
template<typename Element, typename Tag, typename Reference, typename Derived,
         typename charT, typename traits>
__host__
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &os,
           const pointer<Element, Tag, Reference, Derived> &p);

// NOTE: This is needed so that Thrust smart pointers can be used in
// `std::unique_ptr`.
template <typename Element, typename Tag, typename Reference, typename Derived>
__host__ __device__
bool operator==(std::nullptr_t, pointer<Element, Tag, Reference, Derived> p);

template <typename Element, typename Tag, typename Reference, typename Derived>
__host__ __device__
bool operator==(pointer<Element, Tag, Reference, Derived> p, std::nullptr_t);

template <typename Element, typename Tag, typename Reference, typename Derived>
__host__ __device__
bool operator!=(std::nullptr_t, pointer<Element, Tag, Reference, Derived> p);

template <typename Element, typename Tag, typename Reference, typename Derived>
__host__ __device__
bool operator!=(pointer<Element, Tag, Reference, Derived> p, std::nullptr_t);

HYDRA_THRUST_NAMESPACE_END

#include <hydra/detail/external/hydra_thrust/detail/pointer.inl>

