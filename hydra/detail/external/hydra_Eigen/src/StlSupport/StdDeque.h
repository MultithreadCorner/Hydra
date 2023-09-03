// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Hauke Heibel <hauke.heibel@googlemail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HYDRA_EIGEN_STDDEQUE_H
#define HYDRA_EIGEN_STDDEQUE_H

#include "details.h"

/**
 * This section contains a convenience MACRO which allows an easy specialization of
 * std::deque such that for data types with alignment issues the correct allocator
 * is used automatically.
 */
#define HYDRA_EIGEN_DEFINE_STL_DEQUE_SPECIALIZATION(...) \
namespace std \
{ \
  template<> \
  class deque<__VA_ARGS__, std::allocator<__VA_ARGS__> >           \
    : public deque<__VA_ARGS__, HYDRA_EIGEN_ALIGNED_ALLOCATOR<__VA_ARGS__> > \
  { \
    typedef deque<__VA_ARGS__, HYDRA_EIGEN_ALIGNED_ALLOCATOR<__VA_ARGS__> > deque_base; \
  public: \
    typedef __VA_ARGS__ value_type; \
    typedef deque_base::allocator_type allocator_type; \
    typedef deque_base::size_type size_type;  \
    typedef deque_base::iterator iterator;  \
    explicit deque(const allocator_type& a = allocator_type()) : deque_base(a) {}  \
    template<typename InputIterator> \
    deque(InputIterator first, InputIterator last, const allocator_type& a = allocator_type()) : deque_base(first, last, a) {} \
    deque(const deque& c) : deque_base(c) {}  \
    explicit deque(size_type num, const value_type& val = value_type()) : deque_base(num, val) {} \
    deque(iterator start_, iterator end_) : deque_base(start_, end_) {}  \
    deque& operator=(const deque& x) {  \
      deque_base::operator=(x);  \
      return *this;  \
    } \
  }; \
}

// check whether we really need the std::deque specialization
#if !HYDRA_EIGEN_HAS_CXX11_CONTAINERS && !(defined(_GLIBCXX_DEQUE) && (!HYDRA_EIGEN_GNUC_AT_LEAST(4,1))) /* Note that before gcc-4.1 we already have: std::deque::resize(size_type,const T&). */

namespace std {

#define HYDRA_EIGEN_STD_DEQUE_SPECIALIZATION_BODY \
  public:  \
    typedef T value_type; \
    typedef typename deque_base::allocator_type allocator_type; \
    typedef typename deque_base::size_type size_type;  \
    typedef typename deque_base::iterator iterator;  \
    typedef typename deque_base::const_iterator const_iterator;  \
    explicit deque(const allocator_type& a = allocator_type()) : deque_base(a) {}  \
    template<typename InputIterator> \
    deque(InputIterator first, InputIterator last, const allocator_type& a = allocator_type()) \
    : deque_base(first, last, a) {} \
    deque(const deque& c) : deque_base(c) {}  \
    explicit deque(size_type num, const value_type& val = value_type()) : deque_base(num, val) {} \
    deque(iterator start_, iterator end_) : deque_base(start_, end_) {}  \
    deque& operator=(const deque& x) {  \
      deque_base::operator=(x);  \
      return *this;  \
    }

  template<typename T>
  class deque<T,HYDRA_EIGEN_ALIGNED_ALLOCATOR<T> >
    : public deque<HYDRA_EIGEN_WORKAROUND_MSVC_STL_SUPPORT(T),
                   hydra_Eigen::aligned_allocator_indirection<HYDRA_EIGEN_WORKAROUND_MSVC_STL_SUPPORT(T)> >
{
  typedef deque<HYDRA_EIGEN_WORKAROUND_MSVC_STL_SUPPORT(T),
                hydra_Eigen::aligned_allocator_indirection<HYDRA_EIGEN_WORKAROUND_MSVC_STL_SUPPORT(T)> > deque_base;
  HYDRA_EIGEN_STD_DEQUE_SPECIALIZATION_BODY

  void resize(size_type new_size)
  { resize(new_size, T()); }

#if defined(_DEQUE_)
  // workaround MSVC std::deque implementation
  void resize(size_type new_size, const value_type& x)
  {
    if (deque_base::size() < new_size)
      deque_base::_Insert_n(deque_base::end(), new_size - deque_base::size(), x);
    else if (new_size < deque_base::size())
      deque_base::erase(deque_base::begin() + new_size, deque_base::end());
  }
  void push_back(const value_type& x)
  { deque_base::push_back(x); } 
  void push_front(const value_type& x)
  { deque_base::push_front(x); }
  using deque_base::insert;  
  iterator insert(const_iterator position, const value_type& x)
  { return deque_base::insert(position,x); }
  void insert(const_iterator position, size_type new_size, const value_type& x)
  { deque_base::insert(position, new_size, x); }
#else
  // default implementation which should always work.
  void resize(size_type new_size, const value_type& x)
  {
    if (new_size < deque_base::size())
      deque_base::erase(deque_base::begin() + new_size, deque_base::end());
    else if (new_size > deque_base::size())
      deque_base::insert(deque_base::end(), new_size - deque_base::size(), x);
  }
#endif
  };
}

#endif // check whether specialization is actually required

#endif // HYDRA_EIGEN_STDDEQUE_H
