//  (C) Copyright John Maddock 2005.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_COMPLEX_FABS_INCLUDED
#define HYDRA_BOOST_MATH_COMPLEX_FABS_INCLUDED

#ifndef HYDRA_BOOST_MATH_HYPOT_INCLUDED
#  include <hydra/detail/external/hydra_boost/math/special_functions/hypot.hpp>
#endif

namespace hydra_boost{ namespace math{

template<class T> 
inline T fabs(const std::complex<T>& z)
{
   return ::hydra_boost::math::hypot(z.real(), z.imag());
}

} } // namespaces

#endif // HYDRA_BOOST_MATH_COMPLEX_FABS_INCLUDED
