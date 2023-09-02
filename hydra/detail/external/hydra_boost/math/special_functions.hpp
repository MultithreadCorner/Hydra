//  Copyright John Maddock 2006, 2007, 2012, 2014.
//  Copyright Paul A. Bristow 2006, 2007, 2012

//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file includes *all* the special functions.
// this may be useful if many are used
// - to avoid including each function individually.

#ifndef HYDRA_BOOST_MATH_SPECIAL_FUNCTIONS_HPP
#define HYDRA_BOOST_MATH_SPECIAL_FUNCTIONS_HPP

#include <hydra/detail/external/hydra_boost/math/special_functions/airy.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/acosh.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/asinh.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/atanh.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/bernoulli.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/bessel.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/bessel_prime.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/beta.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/binomial.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/cbrt.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/cos_pi.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/chebyshev.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/digamma.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/ellint_1.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/ellint_2.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/ellint_3.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/ellint_d.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/jacobi_theta.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/jacobi_zeta.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/heuman_lambda.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/ellint_rc.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/ellint_rd.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/ellint_rf.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/ellint_rj.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/ellint_rg.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/erf.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/expint.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/expm1.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/factorials.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/fpclassify.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/gamma.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/hermite.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/hypot.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/hypergeometric_1F0.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/hypergeometric_0F1.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/hypergeometric_2F0.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/hypergeometric_1F1.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/hypergeometric_pFq.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/jacobi_elliptic.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/laguerre.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/lanczos.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/legendre.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/log1p.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/math_fwd.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/next.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/owens_t.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/polygamma.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/powm1.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/sign.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/sin_pi.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/sinc.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/sinhc.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/spherical_harmonic.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/sqrt1pm1.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/zeta.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/modf.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/round.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/trunc.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/pow.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/next.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/owens_t.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/hankel.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/ulp.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/relative_difference.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/lambert_w.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/gegenbauer.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/jacobi.hpp>
#ifndef HYDRA_BOOST_NO_EXCEPTIONS
#include <hydra/detail/external/hydra_boost/math/special_functions/legendre_stieltjes.hpp>
#endif
#endif // HYDRA_BOOST_MATH_SPECIAL_FUNCTIONS_HPP
