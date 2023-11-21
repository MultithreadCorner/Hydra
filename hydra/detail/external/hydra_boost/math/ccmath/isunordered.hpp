//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_CCMATH_ISUNORDERED_HPP
#define HYDRA_BOOST_MATH_CCMATH_ISUNORDERED_HPP

#include <cmath>
#include <hydra/detail/external/hydra_boost/math/tools/is_constant_evaluated.hpp>
#include <hydra/detail/external/hydra_boost/math/ccmath/isnan.hpp>

namespace hydra_boost::math::ccmath {

template <typename T>
inline constexpr bool isunordered(const T x, const T y) noexcept
{
    if(HYDRA_BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        return hydra_boost::math::ccmath::isnan(x) || hydra_boost::math::ccmath::isnan(y);
    }
    else
    {
        using std::isunordered;
        return isunordered(x, y);
    }
}

} // Namespaces

#endif // HYDRA_BOOST_MATH_CCMATH_ISUNORDERED_HPP
