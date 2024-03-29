//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_SPECIAL_ERF_HPP
#define HYDRA_BOOST_MATH_SPECIAL_ERF_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <hydra/detail/external/hydra_boost/math/special_functions/math_fwd.hpp>
#include <hydra/detail/external/hydra_boost/math/tools/config.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/gamma.hpp>
#include <hydra/detail/external/hydra_boost/math/tools/roots.hpp>
#include <hydra/detail/external/hydra_boost/math/policies/error_handling.hpp>
#include <hydra/detail/external/hydra_boost/math/tools/big_constant.hpp>

#if defined(__GNUC__) && defined(HYDRA_BOOST_MATH_USE_FLOAT128)
//
// This is the only way we can avoid
// warning: non-standard suffix on floating constant [-Wpedantic]
// when building with -Wall -pedantic.  Neither __extension__
// nor #pragma diagnostic ignored work :(
//
#pragma GCC system_header
#endif

namespace hydra_boost{ namespace math{

namespace detail
{

//
// Asymptotic series for large z:
//
template <class T>
struct erf_asympt_series_t
{
   erf_asympt_series_t(T z) : xx(2 * -z * z), tk(1)
   {
      HYDRA_BOOST_MATH_STD_USING
      result = -exp(-z * z) / sqrt(hydra_boost::math::constants::pi<T>());
      result /= z;
   }

   typedef T result_type;

   T operator()()
   {
      HYDRA_BOOST_MATH_STD_USING
      T r = result;
      result *= tk / xx;
      tk += 2;
      if( fabs(r) < fabs(result))
         result = 0;
      return r;
   }
private:
   T result;
   T xx;
   int tk;
};
//
// How large z has to be in order to ensure that the series converges:
//
template <class T>
inline float erf_asymptotic_limit_N(const T&)
{
   return (std::numeric_limits<float>::max)();
}
inline float erf_asymptotic_limit_N(const std::integral_constant<int, 24>&)
{
   return 2.8F;
}
inline float erf_asymptotic_limit_N(const std::integral_constant<int, 53>&)
{
   return 4.3F;
}
inline float erf_asymptotic_limit_N(const std::integral_constant<int, 64>&)
{
   return 4.8F;
}
inline float erf_asymptotic_limit_N(const std::integral_constant<int, 106>&)
{
   return 6.5F;
}
inline float erf_asymptotic_limit_N(const std::integral_constant<int, 113>&)
{
   return 6.8F;
}

template <class T, class Policy>
inline T erf_asymptotic_limit()
{
   typedef typename policies::precision<T, Policy>::type precision_type;
   typedef std::integral_constant<int,
      precision_type::value <= 0 ? 0 :
      precision_type::value <= 24 ? 24 :
      precision_type::value <= 53 ? 53 :
      precision_type::value <= 64 ? 64 :
      precision_type::value <= 113 ? 113 : 0
   > tag_type;
   return erf_asymptotic_limit_N(tag_type());
}

template <class T>
struct erf_series_near_zero
{
   typedef T result_type;
   T         term;
   T         zz;
   int       k;
   erf_series_near_zero(const T& z) : term(z), zz(-z * z), k(0) {}

   T operator()()
   {
      T result = term / (2 * k + 1);
      term *= zz / ++k;
      return result;
   }
};

template <class T, class Policy>
T erf_series_near_zero_sum(const T& x, const Policy& pol)
{
   //
   // We need Kahan summation here, otherwise the errors grow fairly quickly.
   // This method is *much* faster than the alternatives even so.
   //
   erf_series_near_zero<T> sum(x);
   std::uintmax_t max_iter = policies::get_max_series_iterations<Policy>();
   T result = constants::two_div_root_pi<T>() * tools::kahan_sum_series(sum, tools::digits<T>(), max_iter);
   policies::check_series_iterations<T>("hydra_boost::math::erf<%1%>(%1%, %1%)", max_iter, pol);
   return result;
}

template <class T, class Policy, class Tag>
T erf_imp(T z, bool invert, const Policy& pol, const Tag& t)
{
   HYDRA_BOOST_MATH_STD_USING

   HYDRA_BOOST_MATH_INSTRUMENT_CODE("Generic erf_imp called");

   if(z < 0)
   {
      if(!invert)
         return -erf_imp(T(-z), invert, pol, t);
      else
         return 1 + erf_imp(T(-z), false, pol, t);
   }

   T result;

   if(!invert && (z > detail::erf_asymptotic_limit<T, Policy>()))
   {
      detail::erf_asympt_series_t<T> s(z);
      std::uintmax_t max_iter = policies::get_max_series_iterations<Policy>();
      result = hydra_boost::math::tools::sum_series(s, policies::get_epsilon<T, Policy>(), max_iter, 1);
      policies::check_series_iterations<T>("hydra_boost::math::erf<%1%>(%1%, %1%)", max_iter, pol);
   }
   else
   {
      T x = z * z;
      if(z < 1.3f)
      {
         // Compute P:
         // This is actually good for z p to 2 or so, but the cutoff given seems
         // to be the best compromise.  Performance wise, this is way quicker than anything else...
         result = erf_series_near_zero_sum(z, pol);
      }
      else if(x > 1 / tools::epsilon<T>())
      {
         // http://functions.wolfram.com/06.27.06.0006.02
         invert = !invert;
         result = exp(-x) / (constants::root_pi<T>() * z);
      }
      else
      {
         // Compute Q:
         invert = !invert;
         result = z * exp(-x);
         result /= hydra_boost::math::constants::root_pi<T>();
         result *= upper_gamma_fraction(T(0.5f), x, policies::get_epsilon<T, Policy>());
      }
   }
   if(invert)
      result = 1 - result;
   return result;
}

template <class T, class Policy>
T erf_imp(T z, bool invert, const Policy& pol, const std::integral_constant<int, 53>& t)
{
   HYDRA_BOOST_MATH_STD_USING

   HYDRA_BOOST_MATH_INSTRUMENT_CODE("53-bit precision erf_imp called");

   if ((hydra_boost::math::isnan)(z))
      return policies::raise_denorm_error("hydra_boost::math::erf<%1%>(%1%)", "Expected a finite argument but got %1%", z, pol);

   if(z < 0)
   {
      if(!invert)
         return -erf_imp(T(-z), invert, pol, t);
      else if(z < T(-0.5))
         return 2 - erf_imp(T(-z), invert, pol, t);
      else
         return 1 + erf_imp(T(-z), false, pol, t);
   }

   T result;

   //
   // Big bunch of selection statements now to pick
   // which implementation to use,
   // try to put most likely options first:
   //
   if(z < T(0.5))
   {
      //
      // We're going to calculate erf:
      //
      if(z < T(1e-10))
      {
         if(z == 0)
         {
            result = T(0);
         }
         else
         {
            static const T c = HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.003379167095512573896158903121545171688);
            result = static_cast<T>(z * 1.125f + z * c);
         }
      }
      else
      {
         // Maximum Deviation Found:                     1.561e-17
         // Expected Error Term:                         1.561e-17
         // Maximum Relative Change in Control Points:   1.155e-04
         // Max Error found at double precision =        2.961182e-17

         static const T Y = 1.044948577880859375f;
         static const T P[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.0834305892146531832907),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, -0.338165134459360935041),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, -0.0509990735146777432841),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, -0.00772758345802133288487),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, -0.000322780120964605683831),
         };
         static const T Q[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 1.0),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.455004033050794024546),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.0875222600142252549554),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.00858571925074406212772),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.000370900071787748000569),
         };
         T zz = z * z;
         result = z * (Y + tools::evaluate_polynomial(P, zz) / tools::evaluate_polynomial(Q, zz));
      }
   }
   else if(invert ? (z < 28) : (z < 5.93f))
   {
      //
      // We'll be calculating erfc:
      //
      invert = !invert;
      if(z < 1.5f)
      {
         // Maximum Deviation Found:                     3.702e-17
         // Expected Error Term:                         3.702e-17
         // Maximum Relative Change in Control Points:   2.845e-04
         // Max Error found at double precision =        4.841816e-17
         static const T Y = 0.405935764312744140625f;
         static const T P[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, -0.098090592216281240205),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.178114665841120341155),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.191003695796775433986),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.0888900368967884466578),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.0195049001251218801359),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.00180424538297014223957),
         };
         static const T Q[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 1.0),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 1.84759070983002217845),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 1.42628004845511324508),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.578052804889902404909),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.12385097467900864233),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.0113385233577001411017),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.337511472483094676155e-5),
         };
         HYDRA_BOOST_MATH_INSTRUMENT_VARIABLE(Y);
         HYDRA_BOOST_MATH_INSTRUMENT_VARIABLE(P[0]);
         HYDRA_BOOST_MATH_INSTRUMENT_VARIABLE(Q[0]);
         HYDRA_BOOST_MATH_INSTRUMENT_VARIABLE(z);
         result = Y + tools::evaluate_polynomial(P, T(z - T(0.5))) / tools::evaluate_polynomial(Q, T(z - T(0.5)));
         HYDRA_BOOST_MATH_INSTRUMENT_VARIABLE(result);
         result *= exp(-z * z) / z;
         HYDRA_BOOST_MATH_INSTRUMENT_VARIABLE(result);
      }
      else if(z < 2.5f)
      {
         // Max Error found at double precision =        6.599585e-18
         // Maximum Deviation Found:                     3.909e-18
         // Expected Error Term:                         3.909e-18
         // Maximum Relative Change in Control Points:   9.886e-05
         static const T Y = 0.50672817230224609375f;
         static const T P[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, -0.0243500476207698441272),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.0386540375035707201728),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.04394818964209516296),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.0175679436311802092299),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.00323962406290842133584),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.000235839115596880717416),
         };
         static const T Q[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 1.0),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 1.53991494948552447182),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.982403709157920235114),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.325732924782444448493),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.0563921837420478160373),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.00410369723978904575884),
         };
         result = Y + tools::evaluate_polynomial(P, T(z - T(1.5))) / tools::evaluate_polynomial(Q, z - T(1.5));
         T hi, lo;
         int expon;
         hi = floor(ldexp(frexp(z, &expon), 26));
         hi = ldexp(hi, expon - 26);
         lo = z - hi;
         T sq = z * z;
         T err_sqr = ((hi * hi - sq) + 2 * hi * lo) + lo * lo;
         result *= exp(-sq) * exp(-err_sqr) / z;
      }
      else if(z < 4.5f)
      {
         // Maximum Deviation Found:                     1.512e-17
         // Expected Error Term:                         1.512e-17
         // Maximum Relative Change in Control Points:   2.222e-04
         // Max Error found at double precision =        2.062515e-17
         static const T Y = 0.5405750274658203125f;
         static const T P[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.00295276716530971662634),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.0137384425896355332126),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.00840807615555585383007),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.00212825620914618649141),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.000250269961544794627958),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.113212406648847561139e-4),
         };
         static const T Q[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 1.0),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 1.04217814166938418171),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.442597659481563127003),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.0958492726301061423444),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.0105982906484876531489),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.000479411269521714493907),
         };
         result = Y + tools::evaluate_polynomial(P, T(z - T(3.5))) / tools::evaluate_polynomial(Q, z - T(3.5));
         T hi, lo;
         int expon;
         hi = floor(ldexp(frexp(z, &expon), 26));
         hi = ldexp(hi, expon - 26);
         lo = z - hi;
         T sq = z * z;
         T err_sqr = ((hi * hi - sq) + 2 * hi * lo) + lo * lo;
         result *= exp(-sq) * exp(-err_sqr) / z;
      }
      else
      {
         // Max Error found at double precision =        2.997958e-17
         // Maximum Deviation Found:                     2.860e-17
         // Expected Error Term:                         2.859e-17
         // Maximum Relative Change in Control Points:   1.357e-05
         static const T Y = 0.5579090118408203125f;
         static const T P[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.00628057170626964891937),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 0.0175389834052493308818),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, -0.212652252872804219852),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, -0.687717681153649930619),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, -2.5518551727311523996),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, -3.22729451764143718517),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, -2.8175401114513378771),
         };
         static const T Q[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 1.0),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 2.79257750980575282228),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 11.0567237927800161565),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 15.930646027911794143),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 22.9367376522880577224),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 13.5064170191802889145),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 53, 5.48409182238641741584),
         };
         result = Y + tools::evaluate_polynomial(P, T(1 / z)) / tools::evaluate_polynomial(Q, T(1 / z));
         T hi, lo;
         int expon;
         hi = floor(ldexp(frexp(z, &expon), 26));
         hi = ldexp(hi, expon - 26);
         lo = z - hi;
         T sq = z * z;
         T err_sqr = ((hi * hi - sq) + 2 * hi * lo) + lo * lo;
         result *= exp(-sq) * exp(-err_sqr) / z;
      }
   }
   else
   {
      //
      // Any value of z larger than 28 will underflow to zero:
      //
      result = 0;
      invert = !invert;
   }

   if(invert)
   {
      result = 1 - result;
   }

   return result;
} // template <class T, class Lanczos>T erf_imp(T z, bool invert, const Lanczos& l, const std::integral_constant<int, 53>& t)


template <class T, class Policy>
T erf_imp(T z, bool invert, const Policy& pol, const std::integral_constant<int, 64>& t)
{
   HYDRA_BOOST_MATH_STD_USING

   HYDRA_BOOST_MATH_INSTRUMENT_CODE("64-bit precision erf_imp called");

   if(z < 0)
   {
      if(!invert)
         return -erf_imp(T(-z), invert, pol, t);
      else if(z < -0.5)
         return 2 - erf_imp(T(-z), invert, pol, t);
      else
         return 1 + erf_imp(T(-z), false, pol, t);
   }

   T result;

   //
   // Big bunch of selection statements now to pick which
   // implementation to use, try to put most likely options
   // first:
   //
   if(z < 0.5)
   {
      //
      // We're going to calculate erf:
      //
      if(z == 0)
      {
         result = 0;
      }
      else if(z < 1e-10)
      {
         static const T c = HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.003379167095512573896158903121545171688);
         result = z * 1.125 + z * c;
      }
      else
      {
         // Max Error found at long double precision =   1.623299e-20
         // Maximum Deviation Found:                     4.326e-22
         // Expected Error Term:                         -4.326e-22
         // Maximum Relative Change in Control Points:   1.474e-04
         static const T Y = 1.044948577880859375f;
         static const T P[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.0834305892146531988966),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, -0.338097283075565413695),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, -0.0509602734406067204596),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, -0.00904906346158537794396),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, -0.000489468651464798669181),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, -0.200305626366151877759e-4),
         };
         static const T Q[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 1.0),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.455817300515875172439),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.0916537354356241792007),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.0102722652675910031202),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.000650511752687851548735),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.189532519105655496778e-4),
         };
         result = z * (Y + tools::evaluate_polynomial(P, T(z * z)) / tools::evaluate_polynomial(Q, T(z * z)));
      }
   }
   else if(invert ? (z < 110) : (z < 6.6f))
   {
      //
      // We'll be calculating erfc:
      //
      invert = !invert;
      if(z < 1.5)
      {
         // Max Error found at long double precision =   3.239590e-20
         // Maximum Deviation Found:                     2.241e-20
         // Expected Error Term:                         -2.241e-20
         // Maximum Relative Change in Control Points:   5.110e-03
         static const T Y = 0.405935764312744140625f;
         static const T P[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, -0.0980905922162812031672),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.159989089922969141329),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.222359821619935712378),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.127303921703577362312),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.0384057530342762400273),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.00628431160851156719325),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.000441266654514391746428),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.266689068336295642561e-7),
         };
         static const T Q[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 1.0),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 2.03237474985469469291),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 1.78355454954969405222),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.867940326293760578231),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.248025606990021698392),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.0396649631833002269861),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.00279220237309449026796),
         };
         result = Y + tools::evaluate_polynomial(P, T(z - 0.5f)) / tools::evaluate_polynomial(Q, T(z - 0.5f));
         T hi, lo;
         int expon;
         hi = floor(ldexp(frexp(z, &expon), 32));
         hi = ldexp(hi, expon - 32);
         lo = z - hi;
         T sq = z * z;
         T err_sqr = ((hi * hi - sq) + 2 * hi * lo) + lo * lo;
         result *= exp(-sq) * exp(-err_sqr) / z;
      }
      else if(z < 2.5)
      {
         // Max Error found at long double precision =   3.686211e-21
         // Maximum Deviation Found:                     1.495e-21
         // Expected Error Term:                         -1.494e-21
         // Maximum Relative Change in Control Points:   1.793e-04
         static const T Y = 0.50672817230224609375f;
         static const T P[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, -0.024350047620769840217),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.0343522687935671451309),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.0505420824305544949541),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.0257479325917757388209),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.00669349844190354356118),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.00090807914416099524444),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.515917266698050027934e-4),
         };
         static const T Q[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 1.0),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 1.71657861671930336344),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 1.26409634824280366218),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.512371437838969015941),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.120902623051120950935),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.0158027197831887485261),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.000897871370778031611439),
         };
         result = Y + tools::evaluate_polynomial(P, T(z - 1.5f)) / tools::evaluate_polynomial(Q, T(z - 1.5f));
         T hi, lo;
         int expon;
         hi = floor(ldexp(frexp(z, &expon), 32));
         hi = ldexp(hi, expon - 32);
         lo = z - hi;
         T sq = z * z;
         T err_sqr = ((hi * hi - sq) + 2 * hi * lo) + lo * lo;
         result *= exp(-sq) * exp(-err_sqr) / z;
      }
      else if(z < 4.5)
      {
         // Maximum Deviation Found:                     1.107e-20
         // Expected Error Term:                         -1.106e-20
         // Maximum Relative Change in Control Points:   1.709e-04
         // Max Error found at long double precision =   1.446908e-20
         static const T Y  = 0.5405750274658203125f;
         static const T P[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.0029527671653097284033),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.0141853245895495604051),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.0104959584626432293901),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.00343963795976100077626),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.00059065441194877637899),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.523435380636174008685e-4),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.189896043050331257262e-5),
         };
         static const T Q[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 1.0),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 1.19352160185285642574),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.603256964363454392857),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.165411142458540585835),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.0259729870946203166468),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.00221657568292893699158),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.804149464190309799804e-4),
         };
         result = Y + tools::evaluate_polynomial(P, T(z - 3.5f)) / tools::evaluate_polynomial(Q, T(z - 3.5f));
         T hi, lo;
         int expon;
         hi = floor(ldexp(frexp(z, &expon), 32));
         hi = ldexp(hi, expon - 32);
         lo = z - hi;
         T sq = z * z;
         T err_sqr = ((hi * hi - sq) + 2 * hi * lo) + lo * lo;
         result *= exp(-sq) * exp(-err_sqr) / z;
      }
      else
      {
         // Max Error found at long double precision =   7.961166e-21
         // Maximum Deviation Found:                     6.677e-21
         // Expected Error Term:                         6.676e-21
         // Maximum Relative Change in Control Points:   2.319e-05
         static const T Y = 0.55825519561767578125f;
         static const T P[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.00593438793008050214106),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 0.0280666231009089713937),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, -0.141597835204583050043),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, -0.978088201154300548842),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, -5.47351527796012049443),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, -13.8677304660245326627),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, -27.1274948720539821722),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, -29.2545152747009461519),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, -16.8865774499799676937),
         };
         static const T Q[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 1.0),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 4.72948911186645394541),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 23.6750543147695749212),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 60.0021517335693186785),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 131.766251645149522868),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 178.167924971283482513),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 182.499390505915222699),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 104.365251479578577989),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 64, 30.8365511891224291717),
         };
         result = Y + tools::evaluate_polynomial(P, T(1 / z)) / tools::evaluate_polynomial(Q, T(1 / z));
         T hi, lo;
         int expon;
         hi = floor(ldexp(frexp(z, &expon), 32));
         hi = ldexp(hi, expon - 32);
         lo = z - hi;
         T sq = z * z;
         T err_sqr = ((hi * hi - sq) + 2 * hi * lo) + lo * lo;
         result *= exp(-sq) * exp(-err_sqr) / z;
      }
   }
   else
   {
      //
      // Any value of z larger than 110 will underflow to zero:
      //
      result = 0;
      invert = !invert;
   }

   if(invert)
   {
      result = 1 - result;
   }

   return result;
} // template <class T, class Lanczos>T erf_imp(T z, bool invert, const Lanczos& l, const std::integral_constant<int, 64>& t)


template <class T, class Policy>
T erf_imp(T z, bool invert, const Policy& pol, const std::integral_constant<int, 113>& t)
{
   HYDRA_BOOST_MATH_STD_USING

   HYDRA_BOOST_MATH_INSTRUMENT_CODE("113-bit precision erf_imp called");

   if(z < 0)
   {
      if(!invert)
         return -erf_imp(T(-z), invert, pol, t);
      else if(z < -0.5)
         return 2 - erf_imp(T(-z), invert, pol, t);
      else
         return 1 + erf_imp(T(-z), false, pol, t);
   }

   T result;

   //
   // Big bunch of selection statements now to pick which
   // implementation to use, try to put most likely options
   // first:
   //
   if(z < 0.5)
   {
      //
      // We're going to calculate erf:
      //
      if(z == 0)
      {
         result = 0;
      }
      else if(z < 1e-20)
      {
         static const T c = HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.003379167095512573896158903121545171688);
         result = z * 1.125 + z * c;
      }
      else
      {
         // Max Error found at long double precision =   2.342380e-35
         // Maximum Deviation Found:                     6.124e-36
         // Expected Error Term:                         -6.124e-36
         // Maximum Relative Change in Control Points:   3.492e-10
         static const T Y = 1.0841522216796875f;
         static const T P[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0442269454158250738961589031215451778),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, -0.35549265736002144875335323556961233),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, -0.0582179564566667896225454670863270393),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, -0.0112694696904802304229950538453123925),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, -0.000805730648981801146251825329609079099),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, -0.566304966591936566229702842075966273e-4),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, -0.169655010425186987820201021510002265e-5),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, -0.344448249920445916714548295433198544e-7),
         };
         static const T Q[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1.0),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.466542092785657604666906909196052522),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.100005087012526447295176964142107611),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0128341535890117646540050072234142603),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00107150448466867929159660677016658186),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.586168368028999183607733369248338474e-4),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.196230608502104324965623171516808796e-5),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.313388521582925207734229967907890146e-7),
         };
         result = z * (Y + tools::evaluate_polynomial(P, T(z * z)) / tools::evaluate_polynomial(Q, T(z * z)));
      }
   }
   else if(invert ? (z < 110) : (z < 8.65f))
   {
      //
      // We'll be calculating erfc:
      //
      invert = !invert;
      if(z < 1)
      {
         // Max Error found at long double precision =   3.246278e-35
         // Maximum Deviation Found:                     1.388e-35
         // Expected Error Term:                         1.387e-35
         // Maximum Relative Change in Control Points:   6.127e-05
         static const T Y = 0.371877193450927734375f;
         static const T P[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, -0.0640320213544647969396032886581290455),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.200769874440155895637857443946706731),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.378447199873537170666487408805779826),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.30521399466465939450398642044975127),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.146890026406815277906781824723458196),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0464837937749539978247589252732769567),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00987895759019540115099100165904822903),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00137507575429025512038051025154301132),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0001144764551085935580772512359680516),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.436544865032836914773944382339900079e-5),
         };
         static const T Q[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1.0),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 2.47651182872457465043733800302427977),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 2.78706486002517996428836400245547955),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1.87295924621659627926365005293130693),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.829375825174365625428280908787261065),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.251334771307848291593780143950311514),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0522110268876176186719436765734722473),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00718332151250963182233267040106902368),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.000595279058621482041084986219276392459),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.226988669466501655990637599399326874e-4),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.270666232259029102353426738909226413e-10),
         };
         result = Y + tools::evaluate_polynomial(P, T(z - 0.5f)) / tools::evaluate_polynomial(Q, T(z - 0.5f));
         T hi, lo;
         int expon;
         hi = floor(ldexp(frexp(z, &expon), 56));
         hi = ldexp(hi, expon - 56);
         lo = z - hi;
         T sq = z * z;
         T err_sqr = ((hi * hi - sq) + 2 * hi * lo) + lo * lo;
         result *= exp(-sq) * exp(-err_sqr) / z;
      }
      else if(z < 1.5)
      {
         // Max Error found at long double precision =   2.215785e-35
         // Maximum Deviation Found:                     1.539e-35
         // Expected Error Term:                         1.538e-35
         // Maximum Relative Change in Control Points:   6.104e-05
         static const T Y = 0.45658016204833984375f;
         static const T P[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, -0.0289965858925328393392496555094848345),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0868181194868601184627743162571779226),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.169373435121178901746317404936356745),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.13350446515949251201104889028133486),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0617447837290183627136837688446313313),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0185618495228251406703152962489700468),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00371949406491883508764162050169531013),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.000485121708792921297742105775823900772),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.376494706741453489892108068231400061e-4),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.133166058052466262415271732172490045e-5),
         };
         static const T Q[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1.0),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 2.32970330146503867261275580968135126),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 2.46325715420422771961250513514928746),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1.55307882560757679068505047390857842),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.644274289865972449441174485441409076),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.182609091063258208068606847453955649),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0354171651271241474946129665801606795),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00454060370165285246451879969534083997),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.000349871943711566546821198612518656486),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.123749319840299552925421880481085392e-4),
         };
         result = Y + tools::evaluate_polynomial(P, T(z - 1.0f)) / tools::evaluate_polynomial(Q, T(z - 1.0f));
         T hi, lo;
         int expon;
         hi = floor(ldexp(frexp(z, &expon), 56));
         hi = ldexp(hi, expon - 56);
         lo = z - hi;
         T sq = z * z;
         T err_sqr = ((hi * hi - sq) + 2 * hi * lo) + lo * lo;
         result *= exp(-sq) * exp(-err_sqr) / z;
      }
      else if(z < 2.25)
      {
         // Maximum Deviation Found:                     1.418e-35
         // Expected Error Term:                         1.418e-35
         // Maximum Relative Change in Control Points:   1.316e-04
         // Max Error found at long double precision =   1.998462e-35
         static const T Y = 0.50250148773193359375f;
         static const T P[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, -0.0201233630504573402185161184151016606),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0331864357574860196516686996302305002),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0716562720864787193337475444413405461),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0545835322082103985114927569724880658),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0236692635189696678976549720784989593),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00656970902163248872837262539337601845),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00120282643299089441390490459256235021),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.000142123229065182650020762792081622986),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.991531438367015135346716277792989347e-5),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.312857043762117596999398067153076051e-6),
         };
         static const T Q[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1.0),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 2.13506082409097783827103424943508554),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 2.06399257267556230937723190496806215),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1.18678481279932541314830499880691109),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.447733186643051752513538142316799562),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.11505680005657879437196953047542148),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.020163993632192726170219663831914034),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00232708971840141388847728782209730585),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.000160733201627963528519726484608224112),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.507158721790721802724402992033269266e-5),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.18647774409821470950544212696270639e-12),
         };
         result = Y + tools::evaluate_polynomial(P, T(z - 1.5f)) / tools::evaluate_polynomial(Q, T(z - 1.5f));
         T hi, lo;
         int expon;
         hi = floor(ldexp(frexp(z, &expon), 56));
         hi = ldexp(hi, expon - 56);
         lo = z - hi;
         T sq = z * z;
         T err_sqr = ((hi * hi - sq) + 2 * hi * lo) + lo * lo;
         result *= exp(-sq) * exp(-err_sqr) / z;
      }
      else if (z < 3)
      {
         // Maximum Deviation Found:                     3.575e-36
         // Expected Error Term:                         3.575e-36
         // Maximum Relative Change in Control Points:   7.103e-05
         // Max Error found at long double precision =   5.794737e-36
         static const T Y = 0.52896785736083984375f;
         static const T P[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, -0.00902152521745813634562524098263360074),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0145207142776691539346923710537580927),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0301681239582193983824211995978678571),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0215548540823305814379020678660434461),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00864683476267958365678294164340749949),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00219693096885585491739823283511049902),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.000364961639163319762492184502159894371),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.388174251026723752769264051548703059e-4),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.241918026931789436000532513553594321e-5),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.676586625472423508158937481943649258e-7),
         };
         static const T Q[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1.0),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1.93669171363907292305550231764920001),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1.69468476144051356810672506101377494),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.880023580986436640372794392579985511),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.299099106711315090710836273697708402),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0690593962363545715997445583603382337),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0108427016361318921960863149875360222),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00111747247208044534520499324234317695),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.686843205749767250666787987163701209e-4),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.192093541425429248675532015101904262e-5),
         };
         result = Y + tools::evaluate_polynomial(P, T(z - 2.25f)) / tools::evaluate_polynomial(Q, T(z - 2.25f));
         T hi, lo;
         int expon;
         hi = floor(ldexp(frexp(z, &expon), 56));
         hi = ldexp(hi, expon - 56);
         lo = z - hi;
         T sq = z * z;
         T err_sqr = ((hi * hi - sq) + 2 * hi * lo) + lo * lo;
         result *= exp(-sq) * exp(-err_sqr) / z;
      }
      else if(z < 3.5)
      {
         // Maximum Deviation Found:                     8.126e-37
         // Expected Error Term:                         -8.126e-37
         // Maximum Relative Change in Control Points:   1.363e-04
         // Max Error found at long double precision =   1.747062e-36
         static const T Y = 0.54037380218505859375f;
         static const T P[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, -0.0033703486408887424921155540591370375),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0104948043110005245215286678898115811),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0148530118504000311502310457390417795),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00816693029245443090102738825536188916),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00249716579989140882491939681805594585),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0004655591010047353023978045800916647),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.531129557920045295895085236636025323e-4),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.343526765122727069515775194111741049e-5),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.971120407556888763695313774578711839e-7),
         };
         static const T Q[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1.0),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1.59911256167540354915906501335919317),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1.136006830764025173864831382946934),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.468565867990030871678574840738423023),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.122821824954470343413956476900662236),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0209670914950115943338996513330141633),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00227845718243186165620199012883547257),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.000144243326443913171313947613547085553),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.407763415954267700941230249989140046e-5),
         };
         result = Y + tools::evaluate_polynomial(P, T(z - 3.0f)) / tools::evaluate_polynomial(Q, T(z - 3.0f));
         T hi, lo;
         int expon;
         hi = floor(ldexp(frexp(z, &expon), 56));
         hi = ldexp(hi, expon - 56);
         lo = z - hi;
         T sq = z * z;
         T err_sqr = ((hi * hi - sq) + 2 * hi * lo) + lo * lo;
         result *= exp(-sq) * exp(-err_sqr) / z;
      }
      else if(z < 5.5)
      {
         // Maximum Deviation Found:                     5.804e-36
         // Expected Error Term:                         -5.803e-36
         // Maximum Relative Change in Control Points:   2.475e-05
         // Max Error found at long double precision =   1.349545e-35
         static const T Y = 0.55000019073486328125f;
         static const T P[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00118142849742309772151454518093813615),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0072201822885703318172366893469382745),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0078782276276860110721875733778481505),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00418229166204362376187593976656261146),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00134198400587769200074194304298642705),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.000283210387078004063264777611497435572),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.405687064094911866569295610914844928e-4),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.39348283801568113807887364414008292e-5),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.248798540917787001526976889284624449e-6),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.929502490223452372919607105387474751e-8),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.156161469668275442569286723236274457e-9),
         };
         static const T Q[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1.0),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1.52955245103668419479878456656709381),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1.06263944820093830054635017117417064),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.441684612681607364321013134378316463),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.121665258426166960049773715928906382),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0232134512374747691424978642874321434),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00310778180686296328582860464875562636),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.000288361770756174705123674838640161693),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.177529187194133944622193191942300132e-4),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.655068544833064069223029299070876623e-6),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.11005507545746069573608988651927452e-7),
         };
         result = Y + tools::evaluate_polynomial(P, T(z - 4.5f)) / tools::evaluate_polynomial(Q, T(z - 4.5f));
         T hi, lo;
         int expon;
         hi = floor(ldexp(frexp(z, &expon), 56));
         hi = ldexp(hi, expon - 56);
         lo = z - hi;
         T sq = z * z;
         T err_sqr = ((hi * hi - sq) + 2 * hi * lo) + lo * lo;
         result *= exp(-sq) * exp(-err_sqr) / z;
      }
      else if(z < 7.5)
      {
         // Maximum Deviation Found:                     1.007e-36
         // Expected Error Term:                         1.007e-36
         // Maximum Relative Change in Control Points:   1.027e-03
         // Max Error found at long double precision =   2.646420e-36
         static const T Y = 0.5574436187744140625f;
         static const T P[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.000293236907400849056269309713064107674),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00225110719535060642692275221961480162),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00190984458121502831421717207849429799),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.000747757733460111743833929141001680706),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.000170663175280949889583158597373928096),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.246441188958013822253071608197514058e-4),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.229818000860544644974205957895688106e-5),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.134886977703388748488480980637704864e-6),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.454764611880548962757125070106650958e-8),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.673002744115866600294723141176820155e-10),
         };
         static const T Q[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1.0),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1.12843690320861239631195353379313367),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.569900657061622955362493442186537259),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.169094404206844928112348730277514273),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0324887449084220415058158657252147063),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00419252877436825753042680842608219552),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00036344133176118603523976748563178578),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.204123895931375107397698245752850347e-4),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.674128352521481412232785122943508729e-6),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.997637501418963696542159244436245077e-8),
         };
         result = Y + tools::evaluate_polynomial(P, T(z - 6.5f)) / tools::evaluate_polynomial(Q, T(z - 6.5f));
         T hi, lo;
         int expon;
         hi = floor(ldexp(frexp(z, &expon), 56));
         hi = ldexp(hi, expon - 56);
         lo = z - hi;
         T sq = z * z;
         T err_sqr = ((hi * hi - sq) + 2 * hi * lo) + lo * lo;
         result *= exp(-sq) * exp(-err_sqr) / z;
      }
      else if(z < 11.5)
      {
         // Maximum Deviation Found:                     8.380e-36
         // Expected Error Term:                         8.380e-36
         // Maximum Relative Change in Control Points:   2.632e-06
         // Max Error found at long double precision =   9.849522e-36
         static const T Y = 0.56083202362060546875f;
         static const T P[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.000282420728751494363613829834891390121),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00175387065018002823433704079355125161),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0021344978564889819420775336322920375),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00124151356560137532655039683963075661),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.000423600733566948018555157026862139644),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.914030340865175237133613697319509698e-4),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.126999927156823363353809747017945494e-4),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.110610959842869849776179749369376402e-5),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.55075079477173482096725348704634529e-7),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.119735694018906705225870691331543806e-8),
         };
         static const T Q[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1.0),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1.69889613396167354566098060039549882),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1.28824647372749624464956031163282674),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.572297795434934493541628008224078717),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.164157697425571712377043857240773164),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.0315311145224594430281219516531649562),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00405588922155632380812945849777127458),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.000336929033691445666232029762868642417),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.164033049810404773469413526427932109e-4),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.356615210500531410114914617294694857e-6),
         };
         result = Y + tools::evaluate_polynomial(P, T(z / 2 - 4.75f)) / tools::evaluate_polynomial(Q, T(z / 2 - 4.75f));
         T hi, lo;
         int expon;
         hi = floor(ldexp(frexp(z, &expon), 56));
         hi = ldexp(hi, expon - 56);
         lo = z - hi;
         T sq = z * z;
         T err_sqr = ((hi * hi - sq) + 2 * hi * lo) + lo * lo;
         result *= exp(-sq) * exp(-err_sqr) / z;
      }
      else
      {
         // Maximum Deviation Found:                     1.132e-35
         // Expected Error Term:                         -1.132e-35
         // Maximum Relative Change in Control Points:   4.674e-04
         // Max Error found at long double precision =   1.162590e-35
         static const T Y = 0.5632686614990234375f;
         static const T P[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.000920922048732849448079451574171836943),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 0.00321439044532288750501700028748922439),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, -0.250455263029390118657884864261823431),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, -0.906807635364090342031792404764598142),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, -8.92233572835991735876688745989985565),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, -21.7797433494422564811782116907878495),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, -91.1451915251976354349734589601171659),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, -144.1279109655993927069052125017673),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, -313.845076581796338665519022313775589),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, -273.11378811923343424081101235736475),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, -271.651566205951067025696102600443452),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, -60.0530577077238079968843307523245547),
         };
         static const T Q[] = {    
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1.0),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 3.49040448075464744191022350947892036),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 34.3563592467165971295915749548313227),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 84.4993232033879023178285731843850461),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 376.005865281206894120659401340373818),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 629.95369438888946233003926191755125),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1568.35771983533158591604513304269098),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1646.02452040831961063640827116581021),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 2299.96860633240298708910425594484895),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 1222.73204392037452750381340219906374),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 799.359797306084372350264298361110448),
            HYDRA_BOOST_MATH_BIG_CONSTANT(T, 113, 72.7415265778588087243442792401576737),
         };
         result = Y + tools::evaluate_polynomial(P, T(1 / z)) / tools::evaluate_polynomial(Q, T(1 / z));
         T hi, lo;
         int expon;
         hi = floor(ldexp(frexp(z, &expon), 56));
         hi = ldexp(hi, expon - 56);
         lo = z - hi;
         T sq = z * z;
         T err_sqr = ((hi * hi - sq) + 2 * hi * lo) + lo * lo;
         result *= exp(-sq) * exp(-err_sqr) / z;
      }
   }
   else
   {
      //
      // Any value of z larger than 110 will underflow to zero:
      //
      result = 0;
      invert = !invert;
   }

   if(invert)
   {
      result = 1 - result;
   }

   return result;
} // template <class T, class Lanczos>T erf_imp(T z, bool invert, const Lanczos& l, const std::integral_constant<int, 113>& t)

template <class T, class Policy, class tag>
struct erf_initializer
{
   struct init
   {
      init()
      {
         do_init(tag());
      }
      static void do_init(const std::integral_constant<int, 0>&){}
      static void do_init(const std::integral_constant<int, 53>&)
      {
         hydra_boost::math::erf(static_cast<T>(1e-12), Policy());
         hydra_boost::math::erf(static_cast<T>(0.25), Policy());
         hydra_boost::math::erf(static_cast<T>(1.25), Policy());
         hydra_boost::math::erf(static_cast<T>(2.25), Policy());
         hydra_boost::math::erf(static_cast<T>(4.25), Policy());
         hydra_boost::math::erf(static_cast<T>(5.25), Policy());
      }
      static void do_init(const std::integral_constant<int, 64>&)
      {
         hydra_boost::math::erf(static_cast<T>(1e-12), Policy());
         hydra_boost::math::erf(static_cast<T>(0.25), Policy());
         hydra_boost::math::erf(static_cast<T>(1.25), Policy());
         hydra_boost::math::erf(static_cast<T>(2.25), Policy());
         hydra_boost::math::erf(static_cast<T>(4.25), Policy());
         hydra_boost::math::erf(static_cast<T>(5.25), Policy());
      }
      static void do_init(const std::integral_constant<int, 113>&)
      {
         hydra_boost::math::erf(static_cast<T>(1e-22), Policy());
         hydra_boost::math::erf(static_cast<T>(0.25), Policy());
         hydra_boost::math::erf(static_cast<T>(1.25), Policy());
         hydra_boost::math::erf(static_cast<T>(2.125), Policy());
         hydra_boost::math::erf(static_cast<T>(2.75), Policy());
         hydra_boost::math::erf(static_cast<T>(3.25), Policy());
         hydra_boost::math::erf(static_cast<T>(5.25), Policy());
         hydra_boost::math::erf(static_cast<T>(7.25), Policy());
         hydra_boost::math::erf(static_cast<T>(11.25), Policy());
         hydra_boost::math::erf(static_cast<T>(12.5), Policy());
      }
      void force_instantiate()const{}
   };
   static const init initializer;
   static void force_instantiate()
   {
      initializer.force_instantiate();
   }
};

template <class T, class Policy, class tag>
const typename erf_initializer<T, Policy, tag>::init erf_initializer<T, Policy, tag>::initializer;

} // namespace detail

template <class T, class Policy>
inline typename tools::promote_args<T>::type erf(T z, const Policy& /* pol */)
{
   typedef typename tools::promote_args<T>::type result_type;
   typedef typename policies::evaluation<result_type, Policy>::type value_type;
   typedef typename policies::precision<result_type, Policy>::type precision_type;
   typedef typename policies::normalise<
      Policy, 
      policies::promote_float<false>, 
      policies::promote_double<false>, 
      policies::discrete_quantile<>,
      policies::assert_undefined<> >::type forwarding_policy;

   HYDRA_BOOST_MATH_INSTRUMENT_CODE("result_type = " << typeid(result_type).name());
   HYDRA_BOOST_MATH_INSTRUMENT_CODE("value_type = " << typeid(value_type).name());
   HYDRA_BOOST_MATH_INSTRUMENT_CODE("precision_type = " << typeid(precision_type).name());

   typedef std::integral_constant<int,
      precision_type::value <= 0 ? 0 :
      precision_type::value <= 53 ? 53 :
      precision_type::value <= 64 ? 64 :
      precision_type::value <= 113 ? 113 : 0
   > tag_type;

   HYDRA_BOOST_MATH_INSTRUMENT_CODE("tag_type = " << typeid(tag_type).name());

   detail::erf_initializer<value_type, forwarding_policy, tag_type>::force_instantiate(); // Force constants to be initialized before main

   return policies::checked_narrowing_cast<result_type, forwarding_policy>(detail::erf_imp(
      static_cast<value_type>(z),
      false,
      forwarding_policy(),
      tag_type()), "hydra_boost::math::erf<%1%>(%1%, %1%)");
}

template <class T, class Policy>
inline typename tools::promote_args<T>::type erfc(T z, const Policy& /* pol */)
{
   typedef typename tools::promote_args<T>::type result_type;
   typedef typename policies::evaluation<result_type, Policy>::type value_type;
   typedef typename policies::precision<result_type, Policy>::type precision_type;
   typedef typename policies::normalise<
      Policy, 
      policies::promote_float<false>, 
      policies::promote_double<false>, 
      policies::discrete_quantile<>,
      policies::assert_undefined<> >::type forwarding_policy;

   HYDRA_BOOST_MATH_INSTRUMENT_CODE("result_type = " << typeid(result_type).name());
   HYDRA_BOOST_MATH_INSTRUMENT_CODE("value_type = " << typeid(value_type).name());
   HYDRA_BOOST_MATH_INSTRUMENT_CODE("precision_type = " << typeid(precision_type).name());

   typedef std::integral_constant<int,
      precision_type::value <= 0 ? 0 :
      precision_type::value <= 53 ? 53 :
      precision_type::value <= 64 ? 64 :
      precision_type::value <= 113 ? 113 : 0
   > tag_type;

   HYDRA_BOOST_MATH_INSTRUMENT_CODE("tag_type = " << typeid(tag_type).name());

   detail::erf_initializer<value_type, forwarding_policy, tag_type>::force_instantiate(); // Force constants to be initialized before main

   return policies::checked_narrowing_cast<result_type, forwarding_policy>(detail::erf_imp(
      static_cast<value_type>(z),
      true,
      forwarding_policy(),
      tag_type()), "hydra_boost::math::erfc<%1%>(%1%, %1%)");
}

template <class T>
inline typename tools::promote_args<T>::type erf(T z)
{
   return hydra_boost::math::erf(z, policies::policy<>());
}

template <class T>
inline typename tools::promote_args<T>::type erfc(T z)
{
   return hydra_boost::math::erfc(z, policies::policy<>());
}

} // namespace math
} // namespace hydra_boost

#include <hydra/detail/external/hydra_boost/math/special_functions/detail/erf_inv.hpp>

#endif // HYDRA_BOOST_MATH_SPECIAL_ERF_HPP
