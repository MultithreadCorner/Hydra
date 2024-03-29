//  Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_TOOLS_IS_STANDALONE_HPP
#define HYDRA_BOOST_MATH_TOOLS_IS_STANDALONE_HPP

#ifdef __has_include
#if !__has_include(<hydra/detail/external/hydra_boost/config.hpp>) || !__has_include(<hydra/detail/external/hydra_boost/assert.hpp>) || !__has_include(<hydra/detail/external/hydra_boost/lexical_cast.hpp>) || \
    !__has_include(<hydra/detail/external/hydra_boost/throw_exception.hpp>) || !__has_include(<hydra/detail/external/hydra_boost/predef/other/endian.h>)
#   ifndef HYDRA_BOOST_MATH_STANDALONE
#       define HYDRA_BOOST_MATH_STANDALONE
#   endif
#endif
#endif

#endif // HYDRA_BOOST_MATH_TOOLS_IS_STANDALONE_HPP
