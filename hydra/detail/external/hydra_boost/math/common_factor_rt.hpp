//  (C) Copyright John Maddock 2017.

//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_COMMON_FACTOR_RT_HPP
#define HYDRA_BOOST_MATH_COMMON_FACTOR_RT_HPP

#ifndef HYDRA_BOOST_MATH_STANDALONE
#include <hydra/detail/external/hydra_boost/integer/common_factor_rt.hpp>
#include <hydra/detail/external/hydra_boost/math/tools/header_deprecated.hpp>

HYDRA_BOOST_MATH_HEADER_DEPRECATED("<hydra/detail/external/hydra_boost/integer/common_factor_rt.hpp>");

namespace hydra_boost {
   namespace math {
      using hydra_boost::integer::gcd;
      using hydra_boost::integer::lcm;
      using hydra_boost::integer::gcd_range;
      using hydra_boost::integer::lcm_range;
      using hydra_boost::integer::gcd_evaluator;
      using hydra_boost::integer::lcm_evaluator;
   }
}
#else
#error Common factor is not available in standalone mode because it requires boost.integer.
#endif // HYDRA_BOOST_MATH_STANDALONE

#endif  // HYDRA_BOOST_MATH_COMMON_FACTOR_RT_HPP
