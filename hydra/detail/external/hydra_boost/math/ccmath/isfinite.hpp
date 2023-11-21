//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_CCMATH_ISFINITE
#define HYDRA_BOOST_MATH_CCMATH_ISFINITE

#include <cmath>
#include <type_traits>
#include <hydra/detail/external/hydra_boost/math/tools/is_constant_evaluated.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isinf.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isnan.hpp>

#include <hydra/detail/external/hydra_boost/math/tools/is_standalone.hpp>
#ifndef HYDRA_BOOST_MATH_STANDALONE
#include <hydra/detail/external/hydra_boost/config.hpp>
#ifdef HYDRA_BOOST_NO_CXX17_IF_CONSTEXPR
#error "The header <hydra/detail/external/hydra_boost/math/norms.hpp> can only be used in C++17 and later."
#endif
#endif

namespace hydra_boost::math::ccmath {

template <typename T>
inline constexpr bool isfinite(T x)
{
    if(HYDRA_BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        // bool isfinite (IntegralType arg) is a set of overloads accepting the arg argument of any integral type
        // equivalent to casting the integral argument arg to double (e.g. static_cast<double>(arg))
        if constexpr (std::is_integral_v<T>)
        {
            return !hydra_boost::math::ccmath::isinf(static_cast<double>(x)) && !hydra_boost::math::ccmath::isnan(static_cast<double>(x));
        }
        else
        {
            return !hydra_boost::math::ccmath::isinf(x) && !hydra_boost::math::ccmath::isnan(x);
        }
    }
    else
    {
        using std::isfinite;

        if constexpr (!std::is_integral_v<T>)
        {
            return isfinite(x);
        }
        else
        {
            return isfinite(static_cast<double>(x));
        }
    }
}

}

#endif // HYDRA_BOOST_MATH_CCMATH_ISFINITE
