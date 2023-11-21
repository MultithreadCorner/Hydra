//  Copyright (c) 2009 John Maddock
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_ICONV_HPP
#define HYDRA_BOOST_MATH_ICONV_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <type_traits>
#include <hydra/detail/external/hydra_boost/math/special_functions/round.hpp>

namespace hydra_boost { namespace math { namespace detail{

template <class T, class Policy>
inline int iconv_imp(T v, Policy const&, std::true_type const&)
{
   return static_cast<int>(v);
}

template <class T, class Policy>
inline int iconv_imp(T v, Policy const& pol, std::false_type const&)
{
   HYDRA_BOOST_MATH_STD_USING
   return iround(v, pol);
}

template <class T, class Policy>
inline int iconv(T v, Policy const& pol)
{
   typedef typename std::is_convertible<T, int>::type tag_type;
   return iconv_imp(v, pol, tag_type());
}


}}} // namespaces

#endif // HYDRA_BOOST_MATH_ICONV_HPP

