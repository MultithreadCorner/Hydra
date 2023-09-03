//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_CCMATH_FPCLASSIFY
#define HYDRA_BOOST_MATH_CCMATH_FPCLASSIFY

#include <cmath>
#include <limits>
#include <type_traits>
#include <hydra/detail/external/hydra_boost/math/tools/is_constant_evaluated.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/fpclassify.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/abs.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isinf.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isnan.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isfinite.hpp>

namespace hydra_boost::math::ccmath {

template <typename T, std::enable_if_t<!std::is_integral_v<T>, bool> = true>
inline constexpr int fpclassify HYDRA_BOOST_PREVENT_MACRO_SUBSTITUTION(T x)
{
    if(HYDRA_BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        return (hydra_boost::math::ccmath::isnan)(x) ? FP_NAN :
               (hydra_boost::math::ccmath::isinf)(x) ? FP_INFINITE :
               hydra_boost::math::ccmath::abs(x) == T(0) ? FP_ZERO :
               hydra_boost::math::ccmath::abs(x) > 0 && hydra_boost::math::ccmath::abs(x) < (std::numeric_limits<T>::min)() ? FP_SUBNORMAL : FP_NORMAL;
    }
    else
    {
        using hydra_boost::math::fpclassify;
        return (fpclassify)(x);
    }
}

template <typename Z, std::enable_if_t<std::is_integral_v<Z>, bool> = true>
inline constexpr int fpclassify(Z x)
{
    return hydra_boost::math::ccmath::fpclassify(static_cast<double>(x));
}

}

#endif // HYDRA_BOOST_MATH_CCMATH_FPCLASSIFY
