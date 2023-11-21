//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_CCMATH_CEIL_HPP
#define HYDRA_BOOST_MATH_CCMATH_CEIL_HPP

#include <cmath>
#include <type_traits>
#include <hydra/detail/external/hydra_boost/math/tools/is_constant_evaluated.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/floor.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/abs.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isinf.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isnan.hpp>

namespace hydra_boost::math::ccmath {

namespace detail {

template <typename T>
inline constexpr T ceil_impl(T arg) noexcept
{
    T result = hydra_boost::math::ccmath::floor(arg);

    if(result == arg)
    {
        return result;
    }
    else
    {
        return result + 1;
    }
}

} // Namespace detail

template <typename Real, std::enable_if_t<!std::is_integral_v<Real>, bool> = true>
inline constexpr Real ceil(Real arg) noexcept
{
    if(HYDRA_BOOST_MATH_IS_CONSTANT_EVALUATED(arg))
    {
        return hydra_boost::math::ccmath::abs(arg) == Real(0) ? arg :
               hydra_boost::math::ccmath::isinf(arg) ? arg :
               hydra_boost::math::ccmath::isnan(arg) ? arg :
               hydra_boost::math::ccmath::detail::ceil_impl(arg);
    }
    else
    {
        using std::ceil;
        return ceil(arg);
    }
}

template <typename Z, std::enable_if_t<std::is_integral_v<Z>, bool> = true>
inline constexpr double ceil(Z arg) noexcept
{
    return hydra_boost::math::ccmath::ceil(static_cast<double>(arg));
}

inline constexpr float ceilf(float arg) noexcept
{
    return hydra_boost::math::ccmath::ceil(arg);
}

#ifndef HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
inline constexpr long double ceill(long double arg) noexcept
{
    return hydra_boost::math::ccmath::ceil(arg);
}
#endif

} // Namespaces

#endif // HYDRA_BOOST_MATH_CCMATH_CEIL_HPP
