//  (C) Copyright John Maddock 2005.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_COMPLEX_ATAN_INCLUDED
#define HYDRA_BOOST_MATH_COMPLEX_ATAN_INCLUDED

#ifndef HYDRA_BOOST_MATH_COMPLEX_DETAILS_INCLUDED
#  include <hydra/detail/external/hydra_boost/math/complex/details.hpp>
#endif
#ifndef HYDRA_BOOST_MATH_COMPLEX_ATANH_INCLUDED
#  include <hydra/detail/external/hydra_boost/math/complex/atanh.hpp>
#endif

namespace hydra_boost{ namespace math{

template<class T> 
[[deprecated("Replaced by C++11")]] std::complex<T> atan(const std::complex<T>& x)
{
   //
   // We're using the C99 definition here; atan(z) = -i atanh(iz):
   //
   if(x.real() == 0)
   {
      if(x.imag() == 1)
         return std::complex<T>(0, std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : static_cast<T>(HUGE_VAL));
      if(x.imag() == -1)
         return std::complex<T>(0, std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity() : -static_cast<T>(HUGE_VAL));
   }
   return ::hydra_boost::math::detail::mult_minus_i(::hydra_boost::math::atanh(::hydra_boost::math::detail::mult_i(x)));
}

} } // namespaces

#endif // HYDRA_BOOST_MATH_COMPLEX_ATAN_INCLUDED
