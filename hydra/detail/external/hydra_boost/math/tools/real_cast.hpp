//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_TOOLS_REAL_CAST_HPP
#define HYDRA_BOOST_MATH_TOOLS_REAL_CAST_HPP

#include <hydra/detail/external/hydra_boost/math/tools/config.hpp>

#ifdef _MSC_VER
#pragma once
#endif

namespace hydra_boost{ namespace math
{
  namespace tools
  {
    template <class To, class T>
    inline constexpr To real_cast(T t) noexcept(HYDRA_BOOST_MATH_IS_FLOAT(T) && HYDRA_BOOST_MATH_IS_FLOAT(To))
    {
       return static_cast<To>(t);
    }
  } // namespace tools
} // namespace math
} // namespace hydra_boost

#endif // HYDRA_BOOST_MATH_TOOLS_REAL_CAST_HPP



