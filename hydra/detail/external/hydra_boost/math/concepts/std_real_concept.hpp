//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// std_real_concept is an archetype for built-in Real types.

// The main purpose in providing this type is to verify
// that std lib functions are found via a using declaration
// bringing those functions into the current scope, and not
// just because they happen to be in global scope.
//
// If ::pow is found rather than std::pow say, then the code
// will silently compile, but truncation of long doubles to
// double will cause a significant loss of precision.
// A template instantiated with std_real_concept will *only*
// compile if it std::whatever is in scope.

#include <hydra/detail/external/hydra_boost/math/policies/policy.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/math_fwd.hpp>
#include <limits>
#include <ostream>
#include <istream>
#include <cmath>

#ifndef HYDRA_BOOST_MATH_STD_REAL_CONCEPT_HPP
#define HYDRA_BOOST_MATH_STD_REAL_CONCEPT_HPP

namespace hydra_boost{ namespace math{

namespace concepts
{

#ifdef HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   typedef double std_real_concept_base_type;
#else
   typedef long double std_real_concept_base_type;
#endif

class std_real_concept
{
public:
   // Constructors:
   std_real_concept() : m_value(0){}
   std_real_concept(char c) : m_value(c){}
#ifndef HYDRA_BOOST_NO_INTRINSIC_WCHAR_T
   std_real_concept(wchar_t c) : m_value(c){}
#endif
   std_real_concept(unsigned char c) : m_value(c){}
   std_real_concept(signed char c) : m_value(c){}
   std_real_concept(unsigned short c) : m_value(c){}
   std_real_concept(short c) : m_value(c){}
   std_real_concept(unsigned int c) : m_value(c){}
   std_real_concept(int c) : m_value(c){}
   std_real_concept(unsigned long c) : m_value(c){}
   std_real_concept(long c) : m_value(c){}
#if defined(__DECCXX) || defined(__SUNPRO_CC)
   std_real_concept(unsigned long long c) : m_value(static_cast<std_real_concept_base_type>(c)){}
   std_real_concept(long long c) : m_value(static_cast<std_real_concept_base_type>(c)){}
#endif
   std_real_concept(unsigned long long c) : m_value(static_cast<std_real_concept_base_type>(c)){}
   std_real_concept(long long c) : m_value(static_cast<std_real_concept_base_type>(c)){}
   std_real_concept(float c) : m_value(c){}
   std_real_concept(double c) : m_value(c){}
   std_real_concept(long double c) : m_value(c){}
#ifdef HYDRA_BOOST_MATH_USE_FLOAT128
   std_real_concept(HYDRA_BOOST_MATH_FLOAT128_TYPE c) : m_value(c){}
#endif

   // Assignment:
   std_real_concept& operator=(char c) { m_value = c; return *this; }
   std_real_concept& operator=(unsigned char c) { m_value = c; return *this; }
   std_real_concept& operator=(signed char c) { m_value = c; return *this; }
#ifndef HYDRA_BOOST_NO_INTRINSIC_WCHAR_T
   std_real_concept& operator=(wchar_t c) { m_value = c; return *this; }
#endif
   std_real_concept& operator=(short c) { m_value = c; return *this; }
   std_real_concept& operator=(unsigned short c) { m_value = c; return *this; }
   std_real_concept& operator=(int c) { m_value = c; return *this; }
   std_real_concept& operator=(unsigned int c) { m_value = c; return *this; }
   std_real_concept& operator=(long c) { m_value = c; return *this; }
   std_real_concept& operator=(unsigned long c) { m_value = c; return *this; }
#if defined(__DECCXX) || defined(__SUNPRO_CC)
   std_real_concept& operator=(unsigned long long c) { m_value = static_cast<std_real_concept_base_type>(c); return *this; }
   std_real_concept& operator=(long long c) { m_value = static_cast<std_real_concept_base_type>(c); return *this; }
#endif
   std_real_concept& operator=(long long c) { m_value = static_cast<std_real_concept_base_type>(c); return *this; }
   std_real_concept& operator=(unsigned long long c) { m_value = static_cast<std_real_concept_base_type>(c); return *this; }

   std_real_concept& operator=(float c) { m_value = c; return *this; }
   std_real_concept& operator=(double c) { m_value = c; return *this; }
   std_real_concept& operator=(long double c) { m_value = c; return *this; }

   // Access:
   std_real_concept_base_type value()const{ return m_value; }

   // Member arithmetic:
   std_real_concept& operator+=(const std_real_concept& other)
   { m_value += other.value(); return *this; }
   std_real_concept& operator-=(const std_real_concept& other)
   { m_value -= other.value(); return *this; }
   std_real_concept& operator*=(const std_real_concept& other)
   { m_value *= other.value(); return *this; }
   std_real_concept& operator/=(const std_real_concept& other)
   { m_value /= other.value(); return *this; }
   std_real_concept operator-()const
   { return -m_value; }
   std_real_concept const& operator+()const
   { return *this; }

private:
   std_real_concept_base_type m_value;
};

// Non-member arithmetic:
inline std_real_concept operator+(const std_real_concept& a, const std_real_concept& b)
{
   std_real_concept result(a);
   result += b;
   return result;
}
inline std_real_concept operator-(const std_real_concept& a, const std_real_concept& b)
{
   std_real_concept result(a);
   result -= b;
   return result;
}
inline std_real_concept operator*(const std_real_concept& a, const std_real_concept& b)
{
   std_real_concept result(a);
   result *= b;
   return result;
}
inline std_real_concept operator/(const std_real_concept& a, const std_real_concept& b)
{
   std_real_concept result(a);
   result /= b;
   return result;
}

// Comparison:
inline bool operator == (const std_real_concept& a, const std_real_concept& b)
{ return a.value() == b.value(); }
inline bool operator != (const std_real_concept& a, const std_real_concept& b)
{ return a.value() != b.value();}
inline bool operator < (const std_real_concept& a, const std_real_concept& b)
{ return a.value() < b.value(); }
inline bool operator <= (const std_real_concept& a, const std_real_concept& b)
{ return a.value() <= b.value(); }
inline bool operator > (const std_real_concept& a, const std_real_concept& b)
{ return a.value() > b.value(); }
inline bool operator >= (const std_real_concept& a, const std_real_concept& b)
{ return a.value() >= b.value(); }

} // namespace concepts
} // namespace math
} // namespace hydra_boost

namespace std{

// Non-member functions:
inline hydra_boost::math::concepts::std_real_concept acos(hydra_boost::math::concepts::std_real_concept a)
{ return std::acos(a.value()); }
inline hydra_boost::math::concepts::std_real_concept cos(hydra_boost::math::concepts::std_real_concept a)
{ return std::cos(a.value()); }
inline hydra_boost::math::concepts::std_real_concept asin(hydra_boost::math::concepts::std_real_concept a)
{ return std::asin(a.value()); }
inline hydra_boost::math::concepts::std_real_concept atan(hydra_boost::math::concepts::std_real_concept a)
{ return std::atan(a.value()); }
inline hydra_boost::math::concepts::std_real_concept atan2(hydra_boost::math::concepts::std_real_concept a, hydra_boost::math::concepts::std_real_concept b)
{ return std::atan2(a.value(), b.value()); }
inline hydra_boost::math::concepts::std_real_concept ceil(hydra_boost::math::concepts::std_real_concept a)
{ return std::ceil(a.value()); }
#ifndef HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
inline hydra_boost::math::concepts::std_real_concept fmod(hydra_boost::math::concepts::std_real_concept a, hydra_boost::math::concepts::std_real_concept b)
{ return fmodl(a.value(), b.value()); }
#else
inline hydra_boost::math::concepts::std_real_concept fmod(hydra_boost::math::concepts::std_real_concept a, hydra_boost::math::concepts::std_real_concept b)
{ return std::fmod(a.value(), b.value()); }
#endif
inline hydra_boost::math::concepts::std_real_concept cosh(hydra_boost::math::concepts::std_real_concept a)
{ return std::cosh(a.value()); }
inline hydra_boost::math::concepts::std_real_concept exp(hydra_boost::math::concepts::std_real_concept a)
{ return std::exp(a.value()); }
inline hydra_boost::math::concepts::std_real_concept fabs(hydra_boost::math::concepts::std_real_concept a)
{ return std::fabs(a.value()); }
inline hydra_boost::math::concepts::std_real_concept abs(hydra_boost::math::concepts::std_real_concept a)
{ return std::abs(a.value()); }
inline hydra_boost::math::concepts::std_real_concept floor(hydra_boost::math::concepts::std_real_concept a)
{ return std::floor(a.value()); }
inline hydra_boost::math::concepts::std_real_concept modf(hydra_boost::math::concepts::std_real_concept a, hydra_boost::math::concepts::std_real_concept* ipart)
{
   hydra_boost::math::concepts::std_real_concept_base_type ip;
   hydra_boost::math::concepts::std_real_concept_base_type result = std::modf(a.value(), &ip);
   *ipart = ip;
   return result;
}
inline hydra_boost::math::concepts::std_real_concept frexp(hydra_boost::math::concepts::std_real_concept a, int* expon)
{ return std::frexp(a.value(), expon); }
inline hydra_boost::math::concepts::std_real_concept ldexp(hydra_boost::math::concepts::std_real_concept a, int expon)
{ return std::ldexp(a.value(), expon); }
inline hydra_boost::math::concepts::std_real_concept log(hydra_boost::math::concepts::std_real_concept a)
{ return std::log(a.value()); }
inline hydra_boost::math::concepts::std_real_concept log10(hydra_boost::math::concepts::std_real_concept a)
{ return std::log10(a.value()); }
inline hydra_boost::math::concepts::std_real_concept tan(hydra_boost::math::concepts::std_real_concept a)
{ return std::tan(a.value()); }
inline hydra_boost::math::concepts::std_real_concept pow(hydra_boost::math::concepts::std_real_concept a, hydra_boost::math::concepts::std_real_concept b)
{ return std::pow(a.value(), b.value()); }
#if !defined(__SUNPRO_CC)
inline hydra_boost::math::concepts::std_real_concept pow(hydra_boost::math::concepts::std_real_concept a, int b)
{ return std::pow(a.value(), b); }
#else
inline hydra_boost::math::concepts::std_real_concept pow(hydra_boost::math::concepts::std_real_concept a, int b)
{ return std::pow(a.value(), static_cast<long double>(b)); }
#endif
inline hydra_boost::math::concepts::std_real_concept sin(hydra_boost::math::concepts::std_real_concept a)
{ return std::sin(a.value()); }
inline hydra_boost::math::concepts::std_real_concept sinh(hydra_boost::math::concepts::std_real_concept a)
{ return std::sinh(a.value()); }
inline hydra_boost::math::concepts::std_real_concept sqrt(hydra_boost::math::concepts::std_real_concept a)
{ return std::sqrt(a.value()); }
inline hydra_boost::math::concepts::std_real_concept tanh(hydra_boost::math::concepts::std_real_concept a)
{ return std::tanh(a.value()); }
inline hydra_boost::math::concepts::std_real_concept (nextafter)(hydra_boost::math::concepts::std_real_concept a, hydra_boost::math::concepts::std_real_concept b)
{ return (hydra_boost::math::nextafter)(a, b); }
//
// C++11 ism's
// Note that these must not actually call the std:: versions as that precludes using this
// header to test in C++03 mode, call the Boost versions instead:
//
inline hydra_boost::math::concepts::std_real_concept asinh(hydra_boost::math::concepts::std_real_concept a)
{ return hydra_boost::math::asinh(a.value(), hydra_boost::math::policies::make_policy(hydra_boost::math::policies::overflow_error<hydra_boost::math::policies::ignore_error>())); }
inline hydra_boost::math::concepts::std_real_concept acosh(hydra_boost::math::concepts::std_real_concept a)
{ return hydra_boost::math::acosh(a.value(), hydra_boost::math::policies::make_policy(hydra_boost::math::policies::overflow_error<hydra_boost::math::policies::ignore_error>())); }
inline hydra_boost::math::concepts::std_real_concept atanh(hydra_boost::math::concepts::std_real_concept a)
{ return hydra_boost::math::atanh(a.value(), hydra_boost::math::policies::make_policy(hydra_boost::math::policies::overflow_error<hydra_boost::math::policies::ignore_error>())); }
inline bool (isfinite)(hydra_boost::math::concepts::std_real_concept a)
{
   return (hydra_boost::math::isfinite)(a.value());
}


} // namespace std

#include <hydra/detail/external/hydra_boost/math/special_functions/round.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/trunc.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/modf.hpp>
#include <hydra/detail/external/hydra_boost/math/tools/precision.hpp>

namespace hydra_boost{ namespace math{ namespace concepts{

//
// Conversion and truncation routines:
//
template <class Policy>
inline int iround(const concepts::std_real_concept& v, const Policy& pol)
{
   return hydra_boost::math::iround(v.value(), pol);
}
inline int iround(const concepts::std_real_concept& v)
{
   return hydra_boost::math::iround(v.value(), policies::policy<>());
}

template <class Policy>
inline long lround(const concepts::std_real_concept& v, const Policy& pol)
{
   return hydra_boost::math::lround(v.value(), pol);
}
inline long lround(const concepts::std_real_concept& v)
{
   return hydra_boost::math::lround(v.value(), policies::policy<>());
}

template <class Policy>
inline long long llround(const concepts::std_real_concept& v, const Policy& pol)
{
   return hydra_boost::math::llround(v.value(), pol);
}
inline long long llround(const concepts::std_real_concept& v)
{
   return hydra_boost::math::llround(v.value(), policies::policy<>());
}

template <class Policy>
inline int itrunc(const concepts::std_real_concept& v, const Policy& pol)
{
   return hydra_boost::math::itrunc(v.value(), pol);
}
inline int itrunc(const concepts::std_real_concept& v)
{
   return hydra_boost::math::itrunc(v.value(), policies::policy<>());
}

template <class Policy>
inline long ltrunc(const concepts::std_real_concept& v, const Policy& pol)
{
   return hydra_boost::math::ltrunc(v.value(), pol);
}
inline long ltrunc(const concepts::std_real_concept& v)
{
   return hydra_boost::math::ltrunc(v.value(), policies::policy<>());
}

template <class Policy>
inline long long lltrunc(const concepts::std_real_concept& v, const Policy& pol)
{
   return hydra_boost::math::lltrunc(v.value(), pol);
}
inline long long lltrunc(const concepts::std_real_concept& v)
{
   return hydra_boost::math::lltrunc(v.value(), policies::policy<>());
}

// Streaming:
template <class charT, class traits>
inline std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& os, const std_real_concept& a)
{
   return os << a.value();
}
template <class charT, class traits>
inline std::basic_istream<charT, traits>& operator>>(std::basic_istream<charT, traits>& is, std_real_concept& a)
{
#if defined(__SGI_STL_PORT) || defined(_RWSTD_VER) || defined(__LIBCOMO__) || defined(_LIBCPP_VERSION)
   std::string s;
   std_real_concept_base_type d;
   is >> s;
   std::sscanf(s.c_str(), "%Lf", &d);
   a = d;
   return is;
#else
   std_real_concept_base_type v;
   is >> v;
   a = v;
   return is;
#endif
}

} // namespace concepts
}}

#include <hydra/detail/external/hydra_boost/math/tools/big_constant.hpp>

namespace hydra_boost{ namespace math{
namespace tools
{

template <>
inline concepts::std_real_concept make_big_value<concepts::std_real_concept>(hydra_boost::math::tools::largest_float val, const char*, std::false_type const&, std::false_type const&)
{
   return val;  // Can't use lexical_cast here, sometimes it fails....
}

template <>
inline concepts::std_real_concept max_value<concepts::std_real_concept>(HYDRA_BOOST_MATH_EXPLICIT_TEMPLATE_TYPE_SPEC(concepts::std_real_concept))
{
   return max_value<concepts::std_real_concept_base_type>();
}

template <>
inline concepts::std_real_concept min_value<concepts::std_real_concept>(HYDRA_BOOST_MATH_EXPLICIT_TEMPLATE_TYPE_SPEC(concepts::std_real_concept))
{
   return min_value<concepts::std_real_concept_base_type>();
}

template <>
inline concepts::std_real_concept log_max_value<concepts::std_real_concept>(HYDRA_BOOST_MATH_EXPLICIT_TEMPLATE_TYPE_SPEC(concepts::std_real_concept))
{
   return log_max_value<concepts::std_real_concept_base_type>();
}

template <>
inline concepts::std_real_concept log_min_value<concepts::std_real_concept>(HYDRA_BOOST_MATH_EXPLICIT_TEMPLATE_TYPE_SPEC(concepts::std_real_concept))
{
   return log_min_value<concepts::std_real_concept_base_type>();
}

template <>
inline concepts::std_real_concept epsilon(HYDRA_BOOST_MATH_EXPLICIT_TEMPLATE_TYPE_SPEC(concepts::std_real_concept))
{
   return tools::epsilon<concepts::std_real_concept_base_type>();
}

template <>
inline constexpr int digits<concepts::std_real_concept>(HYDRA_BOOST_MATH_EXPLICIT_TEMPLATE_TYPE_SPEC(concepts::std_real_concept)) noexcept
{ // Assume number of significand bits is same as std_real_concept_base_type,
  // unless std::numeric_limits<T>::is_specialized to provide digits.
   return digits<concepts::std_real_concept_base_type>();
}

template <>
inline double real_cast<double, concepts::std_real_concept>(concepts::std_real_concept r)
{
   return static_cast<double>(r.value());
}


} // namespace tools

#if defined(_MSC_VER) && (_MSC_VER <= 1310)
using concepts::itrunc;
using concepts::ltrunc;
using concepts::lltrunc;
using concepts::iround;
using concepts::lround;
using concepts::llround;
#endif

} // namespace math
} // namespace hydra_boost

//
// These must go at the end, as they include stuff that won't compile until
// after std_real_concept has been defined:
//
#include <hydra/detail/external/hydra_boost/math/special_functions/acosh.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/asinh.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/atanh.hpp>

#endif // HYDRA_BOOST_MATH_STD_REAL_CONCEPT_HPP
