//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_TOOLS_THROW_EXCEPTION_HPP
#define HYDRA_BOOST_MATH_TOOLS_THROW_EXCEPTION_HPP

#include <hydra/detail/external/hydra_boost/math/tools/is_standalone.hpp>

#ifndef HYDRA_BOOST_MATH_STANDALONE

#include <hydra/detail/external/hydra_boost/throw_exception.hpp>
#define HYDRA_BOOST_MATH_THROW_EXCEPTION(expr) hydra_boost::throw_exception(expr);

#else // Standalone mode - use standard library facilities

#define HYDRA_BOOST_MATH_THROW_EXCEPTION(expr) throw expr;

#endif // HYDRA_BOOST_MATH_STANDALONE

#endif // HYDRA_BOOST_MATH_TOOLS_THROW_EXCEPTION_HPP
