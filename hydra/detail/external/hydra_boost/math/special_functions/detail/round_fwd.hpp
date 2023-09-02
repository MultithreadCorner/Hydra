// Copyright John Maddock 2008.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_SPECIAL_ROUND_FWD_HPP
#define HYDRA_BOOST_MATH_SPECIAL_ROUND_FWD_HPP

#include <hydra/detail/external/hydra_boost/math/tools/config.hpp>
#include <hydra/detail/external/hydra_boost/math/tools/promotion.hpp>

#ifdef _MSC_VER
#pragma once
#endif

namespace hydra_boost
{
   namespace math
   { 

   template <class T, class Policy>
   typename tools::promote_args<T>::type trunc(const T& v, const Policy& pol);
   template <class T>
   typename tools::promote_args<T>::type trunc(const T& v);
   template <class T, class Policy>
   int itrunc(const T& v, const Policy& pol);
   template <class T>
   int itrunc(const T& v);
   template <class T, class Policy>
   long ltrunc(const T& v, const Policy& pol);
   template <class T>
   long ltrunc(const T& v);
   template <class T, class Policy>
   long long lltrunc(const T& v, const Policy& pol);
   template <class T>
   long long lltrunc(const T& v);
   template <class T, class Policy>
   typename tools::promote_args<T>::type round(const T& v, const Policy& pol);
   template <class T>
   typename tools::promote_args<T>::type round(const T& v);
   template <class T, class Policy>
   int iround(const T& v, const Policy& pol);
   template <class T>
   int iround(const T& v);
   template <class T, class Policy>
   long lround(const T& v, const Policy& pol);
   template <class T>
   long lround(const T& v);
   template <class T, class Policy>
   long long llround(const T& v, const Policy& pol);
   template <class T>
   long long llround(const T& v);
   template <class T, class Policy>
   T modf(const T& v, T* ipart, const Policy& pol);
   template <class T>
   T modf(const T& v, T* ipart);
   template <class T, class Policy>
   T modf(const T& v, int* ipart, const Policy& pol);
   template <class T>
   T modf(const T& v, int* ipart);
   template <class T, class Policy>
   T modf(const T& v, long* ipart, const Policy& pol);
   template <class T>
   T modf(const T& v, long* ipart);
   template <class T, class Policy>
   T modf(const T& v, long long* ipart, const Policy& pol);
   template <class T>
   T modf(const T& v, long long* ipart);
   }
}

#undef HYDRA_BOOST_MATH_STD_USING
#define HYDRA_BOOST_MATH_STD_USING HYDRA_BOOST_MATH_STD_USING_CORE\
   using hydra_boost::math::round;\
   using hydra_boost::math::iround;\
   using hydra_boost::math::lround;\
   using hydra_boost::math::trunc;\
   using hydra_boost::math::itrunc;\
   using hydra_boost::math::ltrunc;\
   using hydra_boost::math::modf;


#endif // HYDRA_BOOST_MATH_SPECIAL_ROUND_FWD_HPP

