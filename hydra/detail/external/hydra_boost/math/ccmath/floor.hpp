//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_CCMATH_FLOOR_HPP
#define HYDRA_BOOST_MATH_CCMATH_FLOOR_HPP

#include <cmath>
#include <limits>
#include <type_traits>
#include <hydra/detail/external/hydra_boost/math/tools/is_constant_evaluated.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/abs.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isinf.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isnan.hpp>

namespace hydra_boost::math::ccmath {

namespace detail {

template <typename T>
inline constexpr T floor_pos_impl(T arg) noexcept
{
    T result = 1;

    if(result < arg)
    {
        while(result < arg)
        {
            result *= 2;
        }
        while(result > arg)
        {
            --result;
        }

        return result;
    }
    else
    {
        return T(0);
    }
}

template <typename T>
inline constexpr T floor_neg_impl(T arg) noexcept
{
    T result = -1;

    if(result > arg)
    {
        while(result > arg)
        {
            result *= 2;
        }
        while(result < arg)
        {
            ++result;
        }
        if(result != arg)
        {
            --result;
        }
    }

    return result;
}

template <typename T>
inline constexpr T floor_impl(T arg) noexcept
{
    if(arg > 0)
    {
        return floor_pos_impl(arg);
    }
    else
    {
        return floor_neg_impl(arg);
    }
}

} // Namespace detail

template <typename Real, std::enable_if_t<!std::is_integral_v<Real>, bool> = true>
inline constexpr Real floor(Real arg) noexcept
{
    if(HYDRA_BOOST_MATH_IS_CONSTANT_EVALUATED(arg))
    {
        return hydra_boost::math::ccmath::abs(arg) == Real(0) ? arg :
               hydra_boost::math::ccmath::isinf(arg) ? arg :
               hydra_boost::math::ccmath::isnan(arg) ? arg :
               hydra_boost::math::ccmath::detail::floor_impl(arg);
    }
    else
    {
        using std::floor;
        return floor(arg);
    }
}

template <typename Z, std::enable_if_t<std::is_integral_v<Z>, bool> = true>
inline constexpr double floor(Z arg) noexcept
{
    return hydra_boost::math::ccmath::floor(static_cast<double>(arg));
}

inline constexpr float floorf(float arg) noexcept
{
    return hydra_boost::math::ccmath::floor(arg);
}

#ifndef HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
inline constexpr long double floorl(long double arg) noexcept
{
    return hydra_boost::math::ccmath::floor(arg);
}
#endif

} // Namespaces

#endif // HYDRA_BOOST_MATH_CCMATH_FLOOR_HPP
