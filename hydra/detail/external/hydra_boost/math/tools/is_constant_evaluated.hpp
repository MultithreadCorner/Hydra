//  Copyright John Maddock 2011-2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_TOOLS_IS_CONSTANT_EVALUATED_HPP
#define HYDRA_BOOST_MATH_TOOLS_IS_CONSTANT_EVALUATED_HPP

#include <hydra/detail/external/hydra_boost/math/tools/config.hpp>

#ifdef __has_include
# if __has_include(<version>)
#  include <version>
#  ifdef __cpp_lib_is_constant_evaluated
#   include <type_traits>
#   define HYDRA_BOOST_MATH_HAS_IS_CONSTANT_EVALUATED
#  endif
# endif
#endif

#ifdef __has_builtin
#  if __has_builtin(__builtin_is_constant_evaluated) && !defined(HYDRA_BOOST_NO_CXX14_CONSTEXPR) && !defined(HYDRA_BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX)
#    define HYDRA_BOOST_MATH_HAS_BUILTIN_IS_CONSTANT_EVALUATED
#  endif
#endif
//
// MSVC also supports __builtin_is_constant_evaluated if it's recent enough:
//
#if defined(_MSC_FULL_VER) && (_MSC_FULL_VER >= 192528326)
#  define HYDRA_BOOST_MATH_HAS_BUILTIN_IS_CONSTANT_EVALUATED
#endif
//
// As does GCC-9:
//
#if !defined(HYDRA_BOOST_NO_CXX14_CONSTEXPR) && (__GNUC__ >= 9) && !defined(HYDRA_BOOST_MATH_HAS_BUILTIN_IS_CONSTANT_EVALUATED)
#  define HYDRA_BOOST_MATH_HAS_BUILTIN_IS_CONSTANT_EVALUATED
#endif

#if defined(HYDRA_BOOST_MATH_HAS_IS_CONSTANT_EVALUATED) && !defined(HYDRA_BOOST_NO_CXX14_CONSTEXPR)
#  define HYDRA_BOOST_MATH_IS_CONSTANT_EVALUATED(x) std::is_constant_evaluated()
#elif defined(HYDRA_BOOST_MATH_HAS_BUILTIN_IS_CONSTANT_EVALUATED)
#  define HYDRA_BOOST_MATH_IS_CONSTANT_EVALUATED(x) __builtin_is_constant_evaluated()
#elif !defined(HYDRA_BOOST_NO_CXX14_CONSTEXPR) && (__GNUC__ >= 6)
#  define HYDRA_BOOST_MATH_IS_CONSTANT_EVALUATED(x) __builtin_constant_p(x)
#  define HYDRA_BOOST_MATH_USING_BUILTIN_CONSTANT_P
#else
#  define HYDRA_BOOST_MATH_IS_CONSTANT_EVALUATED(x) false
#  define HYDRA_BOOST_MATH_NO_CONSTEXPR_DETECTION
#endif

#endif // HYDRA_BOOST_MATH_TOOLS_IS_CONSTANT_EVALUATED_HPP
