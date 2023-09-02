//  (C) Copyright John Maddock 2005-2021.
//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_CCMATH_HYPOT_HPP
#define HYDRA_BOOST_MATH_CCMATH_HYPOT_HPP

#include <cmath>
#include <array>
#include <limits>
#include <type_traits>
#include <hydra/detail/external/hydra_boost/math/tools/config.hpp>
#include <hydra/detail/external/hydra_boost/math/tools/is_constant_evaluated.hpp>
#include <hydra/detail/external/hydra_boost/math/tools/promotion.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/sqrt.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/abs.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isinf.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isnan.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/detail/swap.hpp>

namespace hydra_boost::math::ccmath {

namespace detail {

template <typename T>
constexpr T hypot_impl(T x, T y) noexcept
{
    x = hydra_boost::math::ccmath::abs(x);
    y = hydra_boost::math::ccmath::abs(y);

    if (y > x)
    {
        hydra_boost::math::ccmath::detail::swap(x, y);
    }

    if(x * std::numeric_limits<T>::epsilon() >= y)
    {
        return x;
    }

    T rat = y / x;
    return x * hydra_boost::math::ccmath::sqrt(1 + rat * rat);
}

} // Namespace detail

template <typename Real, std::enable_if_t<!std::is_integral_v<Real>, bool> = true>
constexpr Real hypot(Real x, Real y) noexcept
{
    if(HYDRA_BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        if (hydra_boost::math::ccmath::abs(x) == static_cast<Real>(0))
        {
            return hydra_boost::math::ccmath::abs(y);
        }
        else if (hydra_boost::math::ccmath::abs(y) == static_cast<Real>(0))
        {
            return hydra_boost::math::ccmath::abs(x);
        }
        // Return +inf even if the other argument is NaN
        else if (hydra_boost::math::ccmath::isinf(x) || hydra_boost::math::ccmath::isinf(y))
        {
            return std::numeric_limits<Real>::infinity();
        }
        else if (hydra_boost::math::ccmath::isnan(x))
        {
            return x;
        }
        else if (hydra_boost::math::ccmath::isnan(y))
        {
            return y;
        }
        
        return hydra_boost::math::ccmath::detail::hypot_impl(x, y);
    }
    else
    {
        using std::hypot;
        return hypot(x, y);
    }
}

template <typename T1, typename T2>
constexpr auto hypot(T1 x, T2 y) noexcept
{
    if(HYDRA_BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        using promoted_type = hydra_boost::math::tools::promote_args_2_t<T1, T2>;
        return hydra_boost::math::ccmath::hypot(static_cast<promoted_type>(x), static_cast<promoted_type>(y));
    }
    else
    {
        using std::hypot;
        return hypot(x, y);
    }
}

constexpr float hypotf(float x, float y) noexcept
{
    return hydra_boost::math::ccmath::hypot(x, y);
}

#ifndef HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
constexpr long double hypotl(long double x, long double y) noexcept
{
    return hydra_boost::math::ccmath::hypot(x, y);
}
#endif

} // Namespaces

#endif // HYDRA_BOOST_MATH_CCMATH_HYPOT_HPP
