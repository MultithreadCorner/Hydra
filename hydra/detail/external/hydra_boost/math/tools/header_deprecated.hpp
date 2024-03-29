//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_TOOLS_HEADER_DEPRECATED
#define HYDRA_BOOST_MATH_TOOLS_HEADER_DEPRECATED

#ifndef HYDRA_BOOST_MATH_STANDALONE

#   include <hydra/detail/external/hydra_boost/config/header_deprecated.hpp>
#   define HYDRA_BOOST_MATH_HEADER_DEPRECATED(expr) HYDRA_BOOST_HEADER_DEPRECATED(expr)

#else

#   ifdef _MSC_VER
// Expands to "This header is deprecated; use expr instead."
#       define HYDRA_BOOST_MATH_HEADER_DEPRECATED(expr) __pragma("This header is deprecated; use " expr " instead.")
#   else // GNU, Clang, Intel, IBM, etc.
// Expands to "This header is deprecated use expr instead"
#       define HYDRA_BOOST_MATH_HEADER_DEPRECATED_MESSAGE(expr) _Pragma(#expr)
#       define HYDRA_BOOST_MATH_HEADER_DEPRECATED(expr) HYDRA_BOOST_MATH_HEADER_DEPRECATED_MESSAGE(message "This header is deprecated use " expr " instead")
#   endif

#endif // HYDRA_BOOST_MATH_STANDALONE

#endif // HYDRA_BOOST_MATH_TOOLS_HEADER_DEPRECATED
