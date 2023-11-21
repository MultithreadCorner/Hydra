//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Constepxr implementation of fabs (see c.math.abs secion 26.8.2 of the ISO standard)

#ifndef HYDRA_BOOST_MATH_CCMATH_FABS
#define HYDRA_BOOST_MATH_CCMATH_FABS

#include <hydra/detail/external/hydra_boost/math/ccmath/abs.hpp>

namespace hydra_boost::math::ccmath {

template <typename T>
inline constexpr auto fabs(T x) noexcept
{
    return hydra_boost::math::ccmath::abs(x);
}

inline constexpr float fabsf(float x) noexcept
{
    return hydra_boost::math::ccmath::abs(x);
}

#ifndef HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
inline constexpr long double fabsl(long double x) noexcept
{
    return hydra_boost::math::ccmath::abs(x);
}
#endif

}

#endif // HYDRA_BOOST_MATH_CCMATH_FABS
