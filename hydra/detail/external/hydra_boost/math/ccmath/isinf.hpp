//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_CCMATH_ISINF
#define HYDRA_BOOST_MATH_CCMATH_ISINF

#include <cmath>
#include <limits>
#include <type_traits>
#include <hydra/detail/external/hydra_boost/math/tools/is_constant_evaluated.hpp>

#include <hydra/detail/external/hydra_boost/math/tools/is_standalone.hpp>
#ifndef HYDRA_BOOST_MATH_STANDALONE
#include <hydra/detail/external/hydra_boost/config.hpp>
#ifdef HYDRA_BOOST_NO_CXX17_IF_CONSTEXPR
#error "The header <hydra/detail/external/hydra_boost/math/norms.hpp> can only be used in C++17 and later."
#endif
#endif

namespace hydra_boost::math::ccmath {

template <typename T>
constexpr bool isinf(T x) noexcept
{
    if(HYDRA_BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        if constexpr (std::numeric_limits<T>::is_signed)
        {
            return x == std::numeric_limits<T>::infinity() || -x == std::numeric_limits<T>::infinity();
        }
        else
        {
            return x == std::numeric_limits<T>::infinity();
        }
    }
    else
    {
        using std::isinf;
        
        if constexpr (!std::is_integral_v<T>)
        {
            return isinf(x);
        }
        else
        {
            return isinf(static_cast<double>(x));
        }
    }
}

}

#endif // HYDRA_BOOST_MATH_CCMATH_ISINF
