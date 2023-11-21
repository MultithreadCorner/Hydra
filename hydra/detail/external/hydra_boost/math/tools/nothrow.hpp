//  (C) Copyright Antony Polukhin 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_TOOLS_NOTHROW_HPP
#define HYDRA_BOOST_MATH_TOOLS_NOTHROW_HPP

#include <hydra/detail/external/hydra_boost/math/tools/is_standalone.hpp>

#ifndef HYDRA_BOOST_MATH_STANDALONE

#include <hydra/detail/external/hydra_boost/config.hpp>

#define HYDRA_BOOST_MATH_NOTHROW HYDRA_BOOST_NOEXCEPT_OR_NOTHROW

#else // Standalone mode - use noexcept or throw()

#if __cplusplus >= 201103L
#define HYDRA_BOOST_MATH_NOTHROW noexcept
#else
#define HYDRA_BOOST_MATH_NOTHROW throw()
#endif

#endif

#endif // HYDRA_BOOST_MATH_TOOLS_NOTHROW_HPP
