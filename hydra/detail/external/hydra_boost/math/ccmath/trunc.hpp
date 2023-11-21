//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_CCMATH_TRUNC_HPP
#define HYDRA_BOOST_MATH_CCMATH_TRUNC_HPP

#include <cmath>
#include <type_traits>
#include <hydra/detail/external/hydra_boost/math/tools/is_constant_evaluated.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/abs.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isinf.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isnan.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/floor.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/ceil.hpp>

namespace hydra_boost::math::ccmath {

namespace detail {

template <typename T>
inline constexpr T trunc_impl(T arg) noexcept
{
    return (arg > 0) ? hydra_boost::math::ccmath::floor(arg) : hydra_boost::math::ccmath::ceil(arg);
}

} // Namespace detail

template <typename Real, std::enable_if_t<!std::is_integral_v<Real>, bool> = true>
inline constexpr Real trunc(Real arg) noexcept
{
    if(HYDRA_BOOST_MATH_IS_CONSTANT_EVALUATED(arg))
    {
        return hydra_boost::math::ccmath::abs(arg) == Real(0) ? arg :
               hydra_boost::math::ccmath::isinf(arg) ? arg :
               hydra_boost::math::ccmath::isnan(arg) ? arg :
               hydra_boost::math::ccmath::detail::trunc_impl(arg);
    }
    else
    {
        using std::trunc;
        return trunc(arg);
    }
}

template <typename Z, std::enable_if_t<std::is_integral_v<Z>, bool> = true>
inline constexpr double trunc(Z arg) noexcept
{
    return hydra_boost::math::ccmath::trunc(static_cast<double>(arg));
}

inline constexpr float truncf(float arg) noexcept
{
    return hydra_boost::math::ccmath::trunc(arg);
}

#ifndef HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
inline constexpr long double truncl(long double arg) noexcept
{
    return hydra_boost::math::ccmath::trunc(arg);
}
#endif

} // Namespaces

#endif // HYDRA_BOOST_MATH_CCMATH_TRUNC_HPP
