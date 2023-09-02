//  (C) Copyright John Maddock 2005.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_COMPLEX_ASINH_INCLUDED
#define HYDRA_BOOST_MATH_COMPLEX_ASINH_INCLUDED

#ifndef HYDRA_BOOST_MATH_COMPLEX_DETAILS_INCLUDED
#  include <hydra/detail/external/hydra_boost/math/complex/details.hpp>
#endif
#ifndef HYDRA_BOOST_MATH_COMPLEX_ASIN_INCLUDED
#  include <hydra/detail/external/hydra_boost/math/complex/asin.hpp>
#endif

namespace hydra_boost{ namespace math{

template<class T> 
[[deprecated("Replaced by C++11")]] inline std::complex<T> asinh(const std::complex<T>& x)
{
   //
   // We use asinh(z) = i asin(-i z);
   // Note that C99 defines this the other way around (which is
   // to say asin is specified in terms of asinh), this is consistent
   // with C99 though:
   //
   return ::hydra_boost::math::detail::mult_i(::hydra_boost::math::asin(::hydra_boost::math::detail::mult_minus_i(x)));
}

} } // namespaces

#endif // HYDRA_BOOST_MATH_COMPLEX_ASINH_INCLUDED
