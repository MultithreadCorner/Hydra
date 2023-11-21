//  (C) Copyright John Maddock 2005.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_COMPLEX_ACOSH_INCLUDED
#define HYDRA_BOOST_MATH_COMPLEX_ACOSH_INCLUDED

#ifndef HYDRA_BOOST_MATH_COMPLEX_DETAILS_INCLUDED
#  include <hydra/detail/external/hydra_boost/math/complex/details.hpp>
#endif
#ifndef HYDRA_BOOST_MATH_COMPLEX_ATANH_INCLUDED
#  include <hydra/detail/external/hydra_boost/math/complex/acos.hpp>
#endif

namespace hydra_boost{ namespace math{

template<class T> 
[[deprecated("Replaced by C++11")]] inline std::complex<T> acosh(const std::complex<T>& z)
{
   //
   // We use the relation acosh(z) = +-i acos(z)
   // Choosing the sign of multiplier to give real(acosh(z)) >= 0
   // as well as compatibility with C99.
   //
   std::complex<T> result = hydra_boost::math::acos(z);
   if(!(hydra_boost::math::isnan)(result.imag()) && signbit(result.imag()))
      return detail::mult_i(result);
   return detail::mult_minus_i(result);
}

} } // namespaces

#endif // HYDRA_BOOST_MATH_COMPLEX_ACOSH_INCLUDED
