//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_CCMATH_MODF_HPP
#define HYDRA_BOOST_MATH_CCMATH_MODF_HPP

#include <cmath>
#include <limits>
#include <type_traits>
#include <hydra/detail/external/hydra_boost/math/tools/is_constant_evaluated.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/abs.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isinf.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isnan.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/trunc.hpp>

namespace hydra_boost::math::ccmath {

namespace detail {

template <typename Real>
inline constexpr Real modf_error_impl(Real x, Real* iptr)
{
    *iptr = x;
    return hydra_boost::math::ccmath::abs(x) == Real(0) ? x :
           x > Real(0) ? Real(0) : -Real(0);
}

template <typename Real>
inline constexpr Real modf_nan_impl(Real x, Real* iptr)
{
    *iptr = x;
    return x;
}

template <typename Real>
inline constexpr Real modf_impl(Real x, Real* iptr)
{
    *iptr = hydra_boost::math::ccmath::trunc(x);
    return (x - *iptr);
}

} // Namespace detail

template <typename Real>
inline constexpr Real modf(Real x, Real* iptr)
{
    if(HYDRA_BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        return hydra_boost::math::ccmath::abs(x) == Real(0) ? detail::modf_error_impl(x, iptr) :
               hydra_boost::math::ccmath::isinf(x) ? detail::modf_error_impl(x, iptr) :
               hydra_boost::math::ccmath::isnan(x) ? detail::modf_nan_impl(x, iptr) :
               hydra_boost::math::ccmath::detail::modf_impl(x, iptr);
    }
    else
    {
        using std::modf;
        return modf(x, iptr);
    }
}

inline constexpr float modff(float x, float* iptr)
{
    return hydra_boost::math::ccmath::modf(x, iptr);
}

#ifndef HYDRA_BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
inline constexpr long double modfl(long double x, long double* iptr)
{
    return hydra_boost::math::ccmath::modf(x, iptr);
}
#endif

} // Namespaces

#endif // HYDRA_BOOST_MATH_CCMATH_MODF_HPP
