//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Constepxr implementation of abs (see c.math.abs secion 26.8.2 of the ISO standard)

#ifndef HYDRA_BOOST_MATH_CCMATH_ABS
#define HYDRA_BOOST_MATH_CCMATH_ABS

#include <cmath>
#include <type_traits>
#include <limits>
#include <hydra/detail/external/hydra_boost/math/tools/is_constant_evaluated.hpp>
#include <hydra/detail/external/hydra_boost/math/tools/assert.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isnan.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isinf.hpp>

#include <hydra/detail/external/hydra_boost/math/tools/is_standalone.hpp>
#ifndef HYDRA_BOOST_MATH_STANDALONE
#include <hydra/detail/external/hydra_boost/config.hpp>
#ifdef HYDRA_BOOST_NO_CXX17_IF_CONSTEXPR
#error "The header <hydra/detail/external/hydra_boost/math/norms.hpp> can only be used in C++17 and later."
#endif
#endif

namespace hydra_boost::math::ccmath {

namespace detail {

template <typename T> 
constexpr T abs_impl(T x) noexcept
{
    if ((hydra_boost::math::ccmath::isnan)(x))
    {
        return std::numeric_limits<T>::quiet_NaN();
    }
    else if (x == static_cast<T>(-0))
    {
        return static_cast<T>(0);
    }

    if constexpr (std::is_integral_v<T>)
    {
        HYDRA_BOOST_MATH_ASSERT(x != (std::numeric_limits<T>::min)());
    }
    
    return x >= 0 ? x : -x;
}

} // Namespace detail

template <typename T, std::enable_if_t<!std::is_unsigned_v<T>, bool> = true>
constexpr T abs(T x) noexcept
{
    if(HYDRA_BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        return detail::abs_impl<T>(x);
    }
    else
    {
        using std::abs;
        return abs(x);
    }
}

// If abs() is called with an argument of type X for which is_unsigned_v<X> is true and if X
// cannot be converted to int by integral promotion (7.3.7), the program is ill-formed.
template <typename T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
constexpr T abs(T x) noexcept
{
    if constexpr (std::is_convertible_v<T, int>)
    {
        return detail::abs_impl<int>(static_cast<int>(x));
    }
    else
    {
        static_assert(sizeof(T) == 0, "Taking the absolute value of an unsigned value not convertible to int is UB.");
        return T(0); // Unreachable, but suppresses warnings
    }
}

constexpr long int labs(long int j) noexcept
{
    return hydra_boost::math::ccmath::abs(j);
}

constexpr long long int llabs(long long int j) noexcept
{
    return hydra_boost::math::ccmath::abs(j);
}

} // Namespaces

#endif // HYDRA_BOOST_MATH_CCMATH_ABS
