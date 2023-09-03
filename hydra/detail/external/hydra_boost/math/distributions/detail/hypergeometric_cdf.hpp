// Copyright 2008 John Maddock
//
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_DISTRIBUTIONS_DETAIL_HG_CDF_HPP
#define HYDRA_BOOST_MATH_DISTRIBUTIONS_DETAIL_HG_CDF_HPP

#include <hydra/detail/external/hydra_boost/math/policies/error_handling.hpp>
#include <hydra/detail/external/hydra_boost/math/distributions/detail/hypergeometric_pdf.hpp>
#include <cstdint>

namespace hydra_boost{ namespace math{ namespace detail{

   template <class T, class Policy>
   T hypergeometric_cdf_imp(std::uint64_t x, std::uint64_t r, std::uint64_t n, std::uint64_t N, bool invert, const Policy& pol)
   {
#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable:4267)
#endif
      HYDRA_BOOST_MATH_STD_USING
      T result = 0;
      T mode = floor(T(r + 1) * T(n + 1) / (N + 2));
      if(x < mode)
      {
         result = hypergeometric_pdf<T>(x, r, n, N, pol);
         T diff = result;
         const auto lower_limit = static_cast<std::uint64_t>((std::max)(INT64_C(0), static_cast<std::int64_t>(n + r) - static_cast<std::int64_t>(N)));
         while(diff > (invert ? T(1) : result) * tools::epsilon<T>())
         {
            diff = T(x) * T((N + x) - n - r) * diff / (T(1 + n - x) * T(1 + r - x));
            result += diff;
            HYDRA_BOOST_MATH_INSTRUMENT_VARIABLE(x);
            HYDRA_BOOST_MATH_INSTRUMENT_VARIABLE(diff);
            HYDRA_BOOST_MATH_INSTRUMENT_VARIABLE(result);
            if(x == lower_limit)
               break;
            --x;
         }
      }
      else
      {
         invert = !invert;
         const auto upper_limit = (std::min)(r, n);
         if(x != upper_limit)
         {
            ++x;
            result = hypergeometric_pdf<T>(x, r, n, N, pol);
            T diff = result;
            while((x <= upper_limit) && (diff > (invert ? T(1) : result) * tools::epsilon<T>()))
            {
               diff = T(n - x) * T(r - x) * diff / (T(x + 1) * T((N + x + 1) - n - r));
               result += diff;
               ++x;
               HYDRA_BOOST_MATH_INSTRUMENT_VARIABLE(x);
               HYDRA_BOOST_MATH_INSTRUMENT_VARIABLE(diff);
               HYDRA_BOOST_MATH_INSTRUMENT_VARIABLE(result);
            }
         }
      }
      if(invert)
         result = 1 - result;
      return result;
#ifdef _MSC_VER
#  pragma warning(pop)
#endif
   }

   template <class T, class Policy>
   inline T hypergeometric_cdf(std::uint64_t x, std::uint64_t r, std::uint64_t n, std::uint64_t N, bool invert, const Policy&)
   {
      HYDRA_BOOST_FPU_EXCEPTION_GUARD
      typedef typename tools::promote_args<T>::type result_type;
      typedef typename policies::evaluation<result_type, Policy>::type value_type;
      typedef typename policies::normalise<
         Policy,
         policies::promote_float<false>,
         policies::promote_double<false>,
         policies::discrete_quantile<>,
         policies::assert_undefined<> >::type forwarding_policy;

      value_type result;
      result = detail::hypergeometric_cdf_imp<value_type>(x, r, n, N, invert, forwarding_policy());
      if(result > 1)
      {
         result  = 1;
      }
      if(result < 0)
      {
         result = 0;
      }
      return policies::checked_narrowing_cast<result_type, forwarding_policy>(result, "hydra_boost::math::hypergeometric_cdf<%1%>(%1%,%1%,%1%,%1%)");
   }

}}} // namespaces

#endif
