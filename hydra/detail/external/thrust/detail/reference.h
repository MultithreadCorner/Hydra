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

#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/detail/type_traits.h>
#include <hydra/detail/external/thrust/detail/use_default.h>
#include <hydra/detail/external/thrust/detail/reference_forward_declaration.h>
#include <ostream>


HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust
{
namespace detail
{

template<typename> struct is_wrapped_reference;

}

// the base type for all of thrust's system-annotated references.
// for reasonable reference-like semantics, derived types must reimplement the following:
// 1. constructor from pointer
// 2. copy constructor
// 3. templated copy constructor from other reference
// 4. templated assignment from other reference
// 5. assignment from value_type
template<typename Element, typename Pointer, typename Derived>
  class reference
{
  private:
    typedef typename HYDRA_EXTERNAL_NS::thrust::detail::eval_if<
      HYDRA_EXTERNAL_NS::thrust::detail::is_same<Derived,use_default>::value,
      HYDRA_EXTERNAL_NS::thrust::detail::identity_<reference>,
      HYDRA_EXTERNAL_NS::thrust::detail::identity_<Derived>
    >::type derived_type;

    // hint for is_wrapped_reference lets it know that this type (or a derived type)
    // is a wrapped reference
    struct wrapped_reference_hint {};
    template<typename> friend struct HYDRA_EXTERNAL_NS::thrust::detail::is_wrapped_reference;

  public:
    typedef Pointer                                              pointer;
    typedef typename HYDRA_EXTERNAL_NS::thrust::detail::remove_const<Element>::type value_type;

    __hydra_host__ __hydra_device__
    explicit reference(const pointer &ptr);

    template<typename OtherElement, typename OtherPointer, typename OtherDerived>
    __hydra_host__ __hydra_device__
    reference(const reference<OtherElement,OtherPointer,OtherDerived> &other,
              typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if_convertible<
                typename reference<OtherElement,OtherPointer,OtherDerived>::pointer,
                pointer
              >::type * = 0);

    __hydra_host__ __hydra_device__
    derived_type &operator=(const reference &other);

    // XXX this may need an enable_if
    template<typename OtherElement, typename OtherPointer, typename OtherDerived>
    __hydra_host__ __hydra_device__
    derived_type &operator=(const reference<OtherElement,OtherPointer,OtherDerived> &other);

    __hydra_host__ __hydra_device__
    derived_type &operator=(const value_type &x);

    __hydra_host__ __hydra_device__
    pointer operator&() const;

    __hydra_host__ __hydra_device__
    operator value_type () const;

    __hydra_host__ __hydra_device__
    void swap(derived_type &other);

    derived_type &operator++();

    value_type operator++(int);

    // XXX parameterize the type of rhs
    derived_type &operator+=(const value_type &rhs);

    derived_type &operator--();

    value_type operator--(int);

    // XXX parameterize the type of rhs
    derived_type &operator-=(const value_type &rhs);

    // XXX parameterize the type of rhs
    derived_type &operator*=(const value_type &rhs);

    // XXX parameterize the type of rhs
    derived_type &operator/=(const value_type &rhs);

    // XXX parameterize the type of rhs
    derived_type &operator%=(const value_type &rhs);

    // XXX parameterize the type of rhs
    derived_type &operator<<=(const value_type &rhs);

    // XXX parameterize the type of rhs
    derived_type &operator>>=(const value_type &rhs);

    // XXX parameterize the type of rhs
    derived_type &operator&=(const value_type &rhs);

    // XXX parameterize the type of rhs
    derived_type &operator|=(const value_type &rhs);

    // XXX parameterize the type of rhs
    derived_type &operator^=(const value_type &rhs);

  private:
    const pointer m_ptr;

    // allow access to m_ptr for other references
    template <typename OtherElement, typename OtherPointer, typename OtherDerived> friend class reference;

    template<typename System>
    __hydra_host__ __hydra_device__
    inline value_type strip_const_get_value(const System &system) const;

    template<typename OtherPointer>
    __hydra_host__ __hydra_device__
    inline void assign_from(OtherPointer src);

    // XXX this helper exists only to avoid warnings about null references from the other assign_from
    template<typename System1, typename System2, typename OtherPointer>
    inline __hydra_host__ __hydra_device__
    void assign_from(System1 *system1, System2 *system2, OtherPointer src);

    template<typename System, typename OtherPointer>
    __hydra_host__ __hydra_device__
    inline void strip_const_assign_value(const System &system, OtherPointer src);

    // XXX this helper exists only to avoid warnings about null references from the other swap
    template<typename System>
    inline __hydra_host__ __hydra_device__
    void swap(System *system, derived_type &other);

    // XXX this helper exists only to avoid warnings about null references from operator value_type ()
    template<typename System>
    inline __hydra_host__ __hydra_device__
    value_type convert_to_value_type(System *system) const;
}; // end reference

// Output stream operator
template<typename Element, typename Pointer, typename Derived,
         typename charT, typename traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &os,
           const reference<Element, Pointer, Derived> &y);

} // end HYDRA_EXTERNAL_NAMESPACE_BEGIN  namespace thrust

HYDRA_EXTERNAL_NAMESPACE_END

#include <hydra/detail/external/thrust/detail/reference.inl>

