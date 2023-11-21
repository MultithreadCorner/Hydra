//  Boost common_factor_ct.hpp header file  ----------------------------------//

//  (C) Copyright John Maddock 2017.
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//  See http://www.boost.org for updates, documentation, and revision history.

#ifndef HYDRA_BOOST_MATH_COMMON_FACTOR_CT_HPP
#define HYDRA_BOOST_MATH_COMMON_FACTOR_CT_HPP

#ifndef HYDRA_BOOST_MATH_STANDALONE
#include <hydra/detail/external/hydra_boost/integer/common_factor_ct.hpp>
#include <hydra/detail/external/hydra_boost/math/tools/header_deprecated.hpp>

HYDRA_BOOST_MATH_HEADER_DEPRECATED("<hydra/detail/external/hydra_boost/integer/common_factor_ct.hpp>");

namespace hydra_boost
{
namespace math
{

   using hydra_boost::integer::static_gcd;
   using hydra_boost::integer::static_lcm;
   using hydra_boost::integer::static_gcd_type;

}  // namespace math
}  // namespace hydra_boost
#else
#error Common factor is not available in standalone mode because it requires boost.integer.
#endif // HYDRA_BOOST_MATH_STANDALONE

#endif  // HYDRA_BOOST_MATH_COMMON_FACTOR_CT_HPP
