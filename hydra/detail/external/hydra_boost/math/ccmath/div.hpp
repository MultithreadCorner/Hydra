//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_CCMATH_DIV_HPP
#define HYDRA_BOOST_MATH_CCMATH_DIV_HPP

#include <cmath>
#include <cstdlib>
#include <cinttypes>
#include <cstdint>
#include <type_traits>

#include <hydra/detail/external/hydra_boost/math/tools/is_standalone.hpp>
#ifndef HYDRA_BOOST_MATH_STANDALONE
#include <hydra/detail/external/hydra_boost/config.hpp>
#ifdef HYDRA_BOOST_NO_CXX17_IF_CONSTEXPR
#error "The header <hydra/detail/external/hydra_boost/math/norms.hpp> can only be used in C++17 and later."
#endif
#endif

namespace hydra_boost::math::ccmath {

namespace detail {

template <typename ReturnType, typename Z>
inline constexpr ReturnType div_impl(const Z x, const Z y) noexcept
{
    // std::div_t/ldiv_t/lldiv_t/imaxdiv_t can be defined as either { Z quot; Z rem; }; or { Z rem; Z quot; };
    // so don't use braced initialization to guarantee compatibility
    ReturnType ans {0, 0};

    ans.quot = x / y;
    ans.rem = x % y;

    return ans;
}

} // Namespace detail

// Used for types other than built-ins (e.g. boost multiprecision)
template <typename Z>
struct div_t
{
    Z quot;
    Z rem;
};

template <typename Z>
inline constexpr auto div(Z x, Z y) noexcept
{
    if constexpr (std::is_same_v<Z, int>)
    {
        return detail::div_impl<std::div_t>(x, y);
    }
    else if constexpr (std::is_same_v<Z, long>)
    {
        return detail::div_impl<std::ldiv_t>(x, y);
    }
    else if constexpr (std::is_same_v<Z, long long>)
    {
        return detail::div_impl<std::lldiv_t>(x, y);
    }
    else if constexpr (std::is_same_v<Z, std::intmax_t>)
    {
        return detail::div_impl<std::imaxdiv_t>(x, y);
    }
    else
    {
        return detail::div_impl<hydra_boost::math::ccmath::div_t<Z>>(x, y);
    }
}

inline constexpr std::ldiv_t ldiv(long x, long y) noexcept
{
    return detail::div_impl<std::ldiv_t>(x, y);
}

inline constexpr std::lldiv_t lldiv(long long x, long long y) noexcept
{
    return detail::div_impl<std::lldiv_t>(x, y);
}

inline constexpr std::imaxdiv_t imaxdiv(std::intmax_t x, std::intmax_t y) noexcept
{
    return detail::div_impl<std::imaxdiv_t>(x, y);
}

} // Namespaces

#endif // HYDRA_BOOST_MATH_CCMATH_DIV_HPP
