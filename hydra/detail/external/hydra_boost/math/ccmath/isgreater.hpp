//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_CCMATH_ISGREATER_HPP
#define HYDRA_BOOST_MATH_CCMATH_ISGREATER_HPP

#include <cmath>
#include <limits>
#include <hydra/detail/external/hydra_boost/math/tools/is_constant_evaluated.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isnan.hpp>

namespace hydra_boost::math::ccmath {

template <typename T1, typename T2 = T1>
inline constexpr bool isgreater(T1 x, T2 y) noexcept
{
    if (HYDRA_BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        if (hydra_boost::math::ccmath::isnan(x) || hydra_boost::math::ccmath::isnan(y))
        {
            return false;
        }
        else
        {
            return x > y;
        }
    }
    else
    {
        using std::isgreater;
        return isgreater(x, y);
    }
}

} // Namespaces

#endif // HYDRA_BOOST_MATH_CCMATH_ISGREATER_HPP
