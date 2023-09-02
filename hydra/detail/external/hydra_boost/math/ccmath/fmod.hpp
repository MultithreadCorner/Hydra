//  (C) Copyright Matt Borland 2021 - 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_CCMATH_FMOD_HPP
#define HYDRA_BOOST_MATH_CCMATH_FMOD_HPP

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <hydra/detail/external/hydra_boost/math/tools/promotion.hpp>
#include <hydra/detail/external/hydra_boost/math/tools/is_constant_evaluated.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/abs.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isinf.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isnan.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isfinite.hpp>

namespace hydra_boost::math::ccmath {

namespace detail {

template <typename T>
constexpr T fmod_impl(T x, T y)
{
    if (x == y)
    {
        return static_cast<T>(0);
    }
    else
    {
        while (x >= y)
        {
            x -= y;
        }

        return static_cast<T>(x);
    }
}

} // Namespace detail

template <typename Real, std::enable_if_t<!std::is_integral_v<Real>, bool> = true>
constexpr Real fmod(Real x, Real y)
{
    if(HYDRA_BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        if (hydra_boost::math::ccmath::abs(x) == static_cast<Real>(0) && y != static_cast<Real>(0))
        {
            return x;
        }
        else if (hydra_boost::math::ccmath::isinf(x) && !hydra_boost::math::ccmath::isnan(y))
        {
            return std::numeric_limits<Real>::quiet_NaN();
        }
        else if (hydra_boost::math::ccmath::abs(y) == static_cast<Real>(0) && !hydra_boost::math::ccmath::isnan(x))
        {
            return std::numeric_limits<Real>::quiet_NaN();
        }
        else if (hydra_boost::math::ccmath::isinf(y) && hydra_boost::math::ccmath::isfinite(x))
        {
            return x;
        }
        else if (hydra_boost::math::ccmath::isnan(x))
        {
            return x;
        }
        else if (hydra_boost::math::ccmath::isnan(y))
        {
            return y;
        }

        return hydra_boost::math::ccmath::detail::fmod_impl<Real>(x, y);
    }
    else
    {
        using std::fmod;
        return fmod(x, y);
    }
}

template <typename T1, typename T2>
constexpr auto fmod(T1 x, T2 y)
{
    if(HYDRA_BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        using promoted_type = hydra_boost::math::tools::promote_args_t<T1, T2>;
        return hydra_boost::math::ccmath::fmod(promoted_type(x), promoted_type(y));
    }
    else
    {
        using std::fmod;
        return fmod(x, y);
    }
}

constexpr float fmodf(float x, float y)
{
    return hydra_boost::math::ccmath::fmod(x, y);
}

#ifndef HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
constexpr long double fmodl(long double x, long double y)
{
    return hydra_boost::math::ccmath::fmod(x, y);
}
#endif

} // Namespaces

#endif // HYDRA_BOOST_MATH_CCMATH_FMOD_HPP
