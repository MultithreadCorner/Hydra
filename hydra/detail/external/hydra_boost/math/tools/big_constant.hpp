
//  Copyright (c) 2011 John Maddock
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_TOOLS_BIG_CONSTANT_HPP
#define HYDRA_BOOST_MATH_TOOLS_BIG_CONSTANT_HPP

#include <hydra/detail/external/hydra_boost/math/tools/config.hpp>
#ifndef HYDRA_BOOST_MATH_STANDALONE
#include <hydra/detail/external/hydra_boost/lexical_cast.hpp>
#endif

#include <cstdlib>
#include <type_traits>
#include <limits>

namespace hydra_boost{ namespace math{ 

namespace tools{

template <class T>
struct numeric_traits : public std::numeric_limits< T > {};

#ifdef HYDRA_BOOST_MATH_USE_FLOAT128
typedef __float128 largest_float;
#define HYDRA_BOOST_MATH_LARGEST_FLOAT_C(x) x##Q
template <>
struct numeric_traits<__float128>
{
   static const int digits = 113;
   static const int digits10 = 33;
   static const int max_exponent = 16384;
   static const bool is_specialized = true;
};
#elif LDBL_DIG > DBL_DIG
typedef long double largest_float;
#define HYDRA_BOOST_MATH_LARGEST_FLOAT_C(x) x##L
#else
typedef double largest_float;
#define HYDRA_BOOST_MATH_LARGEST_FLOAT_C(x) x
#endif

template <class T>
inline constexpr T make_big_value(largest_float v, const char*, std::true_type const&, std::false_type const&) HYDRA_BOOST_MATH_NOEXCEPT(T)
{
   return static_cast<T>(v);
}
template <class T>
inline constexpr T make_big_value(largest_float v, const char*, std::true_type const&, std::true_type const&) HYDRA_BOOST_MATH_NOEXCEPT(T)
{
   return static_cast<T>(v);
}
#ifndef HYDRA_BOOST_MATH_NO_LEXICAL_CAST
template <class T>
inline T make_big_value(largest_float, const char* s, std::false_type const&, std::false_type const&)
{
   return hydra_boost::lexical_cast<T>(s);
}
#else
template <typename T>
inline T make_big_value(largest_float, const char* s, std::false_type const&, std::false_type const&)
{
   static_assert(sizeof(T) == 0, "Type is unsupported in standalone mode. Please disable and try again.");
}
#endif
template <class T>
inline constexpr T make_big_value(largest_float, const char* s, std::false_type const&, std::true_type const&) HYDRA_BOOST_MATH_NOEXCEPT(T)
{
   return T(s);
}

//
// For constants which might fit in a long double (if it's big enough):
//
#define HYDRA_BOOST_MATH_BIG_CONSTANT(T, D, x)\
   hydra_boost::math::tools::make_big_value<T>(\
      HYDRA_BOOST_MATH_LARGEST_FLOAT_C(x), \
      HYDRA_BOOST_STRINGIZE(x), \
      std::integral_constant<bool, (std::is_convertible<hydra_boost::math::tools::largest_float, T>::value) && \
      ((D <= hydra_boost::math::tools::numeric_traits<hydra_boost::math::tools::largest_float>::digits) \
          || std::is_floating_point<T>::value \
          || (hydra_boost::math::tools::numeric_traits<T>::is_specialized && \
          (hydra_boost::math::tools::numeric_traits<T>::digits10 <= hydra_boost::math::tools::numeric_traits<hydra_boost::math::tools::largest_float>::digits10))) >(), \
      std::is_constructible<T, const char*>())
//
// For constants too huge for any conceivable long double (and which generate compiler errors if we try and declare them as such):
//
#define HYDRA_BOOST_MATH_HUGE_CONSTANT(T, D, x)\
   hydra_boost::math::tools::make_big_value<T>(0.0L, HYDRA_BOOST_STRINGIZE(x), \
   std::integral_constant<bool, std::is_floating_point<T>::value || (hydra_boost::math::tools::numeric_traits<T>::is_specialized && hydra_boost::math::tools::numeric_traits<T>::max_exponent <= hydra_boost::math::tools::numeric_traits<hydra_boost::math::tools::largest_float>::max_exponent && hydra_boost::math::tools::numeric_traits<T>::digits <= hydra_boost::math::tools::numeric_traits<hydra_boost::math::tools::largest_float>::digits)>(), \
   std::is_constructible<T, const char*>())

}}} // namespaces

#endif

