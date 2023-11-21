//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// We deliberately use assert in here:
//
// boost-no-inspect

#ifndef HYDRA_BOOST_MATH_TOOLS_ASSERT_HPP
#define HYDRA_BOOST_MATH_TOOLS_ASSERT_HPP

#include <hydra/detail/external/hydra_boost/math/tools/is_standalone.hpp>

#ifndef HYDRA_BOOST_MATH_STANDALONE

#include <hydra/detail/external/hydra_boost/assert.hpp>
#include <hydra/detail/external/hydra_boost/static_assert.hpp>
#define HYDRA_BOOST_MATH_ASSERT(expr) HYDRA_BOOST_ASSERT(expr)
#define HYDRA_BOOST_MATH_ASSERT_MSG(expr, msg) HYDRA_BOOST_ASSERT_MSG(expr, msg)
#define HYDRA_BOOST_MATH_STATIC_ASSERT(expr) HYDRA_BOOST_STATIC_ASSERT(expr)
#define HYDRA_BOOST_MATH_STATIC_ASSERT_MSG(expr, msg) HYDRA_BOOST_STATIC_ASSERT_MSG(expr, msg)

#else // Standalone mode - use cassert

#include <cassert>
#define HYDRA_BOOST_MATH_ASSERT(expr) assert(expr)
#define HYDRA_BOOST_MATH_ASSERT_MSG(expr, msg) assert((expr)&&(msg))
#define HYDRA_BOOST_MATH_STATIC_ASSERT(expr) static_assert(expr, #expr " failed")
#define HYDRA_BOOST_MATH_STATIC_ASSERT_MSG(expr, msg) static_assert(expr, msg)

#endif

#endif // HYDRA_BOOST_MATH_TOOLS_ASSERT_HPP
