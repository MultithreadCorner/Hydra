//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_STATS_EXPONENTIAL_HPP
#define HYDRA_BOOST_STATS_EXPONENTIAL_HPP

#include <hydra/detail/external/hydra_boost/math/distributions/fwd.hpp>
#include <hydra/detail/external/hydra_boost/math/constants/constants.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/log1p.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/expm1.hpp>
#include <hydra/detail/external/hydra_boost/math/distributions/complement.hpp>
#include <hydra/detail/external/hydra_boost/math/distributions/detail/common_error_handling.hpp>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4127) // conditional expression is constant
# pragma warning(disable: 4702) // unreachable code (return after domain_error throw).
#endif

#include <utility>
#include <cmath>

namespace hydra_boost{ namespace math{

namespace detail{
//
// Error check:
//
template <class RealType, class Policy>
inline bool verify_lambda(const char* function, RealType l, RealType* presult, const Policy& pol)
{
   if((l <= 0) || !(hydra_boost::math::isfinite)(l))
   {
      *presult = policies::raise_domain_error<RealType>(
         function,
         "The scale parameter \"lambda\" must be > 0, but was: %1%.", l, pol);
      return false;
   }
   return true;
}

template <class RealType, class Policy>
inline bool verify_exp_x(const char* function, RealType x, RealType* presult, const Policy& pol)
{
   if((x < 0) || (hydra_boost::math::isnan)(x))
   {
      *presult = policies::raise_domain_error<RealType>(
         function,
         "The random variable must be >= 0, but was: %1%.", x, pol);
      return false;
   }
   return true;
}

} // namespace detail

template <class RealType = double, class Policy = policies::policy<> >
class exponential_distribution
{
public:
   using value_type = RealType;
   using policy_type = Policy;

   explicit exponential_distribution(RealType l_lambda = 1)
      : m_lambda(l_lambda)
   {
      RealType err;
      detail::verify_lambda("hydra_boost::math::exponential_distribution<%1%>::exponential_distribution", l_lambda, &err, Policy());
   } // exponential_distribution

   RealType lambda()const { return m_lambda; }

private:
   RealType m_lambda;
};

using exponential = exponential_distribution<double>;

#ifdef __cpp_deduction_guides
template <class RealType>
exponential_distribution(RealType)->exponential_distribution<typename hydra_boost::math::tools::promote_args<RealType>::type>;
#endif

template <class RealType, class Policy>
inline std::pair<RealType, RealType> range(const exponential_distribution<RealType, Policy>& /*dist*/)
{ // Range of permissible values for random variable x.
  if (std::numeric_limits<RealType>::has_infinity)
  { 
    return std::pair<RealType, RealType>(static_cast<RealType>(0), std::numeric_limits<RealType>::infinity()); // 0 to + infinity.
  }
  else
  {
   using hydra_boost::math::tools::max_value;
   return std::pair<RealType, RealType>(static_cast<RealType>(0), max_value<RealType>()); // 0 to + max
  }
}

template <class RealType, class Policy>
inline std::pair<RealType, RealType> support(const exponential_distribution<RealType, Policy>& /*dist*/)
{ // Range of supported values for random variable x.
   // This is range where cdf rises from 0 to 1, and outside it, the pdf is zero.
   using hydra_boost::math::tools::max_value;
   using hydra_boost::math::tools::min_value;
   return std::pair<RealType, RealType>(min_value<RealType>(),  max_value<RealType>());
   // min_value<RealType>() to avoid a discontinuity at x = 0.
}

template <class RealType, class Policy>
inline RealType pdf(const exponential_distribution<RealType, Policy>& dist, const RealType& x)
{
   HYDRA_BOOST_MATH_STD_USING // for ADL of std functions

   static const char* function = "hydra_boost::math::pdf(const exponential_distribution<%1%>&, %1%)";

   RealType lambda = dist.lambda();
   RealType result = 0;
   if(0 == detail::verify_lambda(function, lambda, &result, Policy()))
      return result;
   if(0 == detail::verify_exp_x(function, x, &result, Policy()))
      return result;
   // Workaround for VC11/12 bug:
   if ((hydra_boost::math::isinf)(x))
      return 0;
   result = lambda * exp(-lambda * x);
   return result;
} // pdf

template <class RealType, class Policy>
inline RealType logpdf(const exponential_distribution<RealType, Policy>& dist, const RealType& x)
{
   HYDRA_BOOST_MATH_STD_USING // for ADL of std functions

   static const char* function = "hydra_boost::math::logpdf(const exponential_distribution<%1%>&, %1%)";

   RealType lambda = dist.lambda();
   RealType result = -std::numeric_limits<RealType>::infinity();
   if(0 == detail::verify_lambda(function, lambda, &result, Policy()))
      return result;
   if(0 == detail::verify_exp_x(function, x, &result, Policy()))
      return result;
   
   result = log(lambda) - lambda * x;
   return result;
} // logpdf

template <class RealType, class Policy>
inline RealType cdf(const exponential_distribution<RealType, Policy>& dist, const RealType& x)
{
   HYDRA_BOOST_MATH_STD_USING // for ADL of std functions

   static const char* function = "hydra_boost::math::cdf(const exponential_distribution<%1%>&, %1%)";

   RealType result = 0;
   RealType lambda = dist.lambda();
   if(0 == detail::verify_lambda(function, lambda, &result, Policy()))
      return result;
   if(0 == detail::verify_exp_x(function, x, &result, Policy()))
      return result;
   result = -hydra_boost::math::expm1(-x * lambda, Policy());

   return result;
} // cdf

template <class RealType, class Policy>
inline RealType logcdf(const exponential_distribution<RealType, Policy>& dist, const RealType& x)
{
   HYDRA_BOOST_MATH_STD_USING // for ADL of std functions

   static const char* function = "hydra_boost::math::logcdf(const exponential_distribution<%1%>&, %1%)";

   RealType result = 0;
   RealType lambda = dist.lambda();
   if(0 == detail::verify_lambda(function, lambda, &result, Policy()))
      return result;
   if(0 == detail::verify_exp_x(function, x, &result, Policy()))
      return result;
   result = hydra_boost::math::log1p(-exp(-x * lambda), Policy());

   return result;
} // cdf

template <class RealType, class Policy>
inline RealType quantile(const exponential_distribution<RealType, Policy>& dist, const RealType& p)
{
   HYDRA_BOOST_MATH_STD_USING // for ADL of std functions

   static const char* function = "hydra_boost::math::quantile(const exponential_distribution<%1%>&, %1%)";

   RealType result = 0;
   RealType lambda = dist.lambda();
   if(0 == detail::verify_lambda(function, lambda, &result, Policy()))
      return result;
   if(0 == detail::check_probability(function, p, &result, Policy()))
      return result;

   if(p == 0)
      return 0;
   if(p == 1)
      return policies::raise_overflow_error<RealType>(function, 0, Policy());

   result = -hydra_boost::math::log1p(-p, Policy()) / lambda;
   return result;
} // quantile

template <class RealType, class Policy>
inline RealType cdf(const complemented2_type<exponential_distribution<RealType, Policy>, RealType>& c)
{
   HYDRA_BOOST_MATH_STD_USING // for ADL of std functions

   static const char* function = "hydra_boost::math::cdf(const exponential_distribution<%1%>&, %1%)";

   RealType result = 0;
   RealType lambda = c.dist.lambda();
   if(0 == detail::verify_lambda(function, lambda, &result, Policy()))
      return result;
   if(0 == detail::verify_exp_x(function, c.param, &result, Policy()))
      return result;
   // Workaround for VC11/12 bug:
   if (c.param >= tools::max_value<RealType>())
      return 0;
   result = exp(-c.param * lambda);

   return result;
}

template <class RealType, class Policy>
inline RealType logcdf(const complemented2_type<exponential_distribution<RealType, Policy>, RealType>& c)
{
   HYDRA_BOOST_MATH_STD_USING // for ADL of std functions

   static const char* function = "hydra_boost::math::logcdf(const exponential_distribution<%1%>&, %1%)";

   RealType result = 0;
   RealType lambda = c.dist.lambda();
   if(0 == detail::verify_lambda(function, lambda, &result, Policy()))
      return result;
   if(0 == detail::verify_exp_x(function, c.param, &result, Policy()))
      return result;
   // Workaround for VC11/12 bug:
   if (c.param >= tools::max_value<RealType>())
      return 0;
   result = -c.param * lambda;

   return result;
}

template <class RealType, class Policy>
inline RealType quantile(const complemented2_type<exponential_distribution<RealType, Policy>, RealType>& c)
{
   HYDRA_BOOST_MATH_STD_USING // for ADL of std functions

   static const char* function = "hydra_boost::math::quantile(const exponential_distribution<%1%>&, %1%)";

   RealType result = 0;
   RealType lambda = c.dist.lambda();
   if(0 == detail::verify_lambda(function, lambda, &result, Policy()))
      return result;

   RealType q = c.param;
   if(0 == detail::check_probability(function, q, &result, Policy()))
      return result;

   if(q == 1)
      return 0;
   if(q == 0)
      return policies::raise_overflow_error<RealType>(function, 0, Policy());

   result = -log(q) / lambda;
   return result;
}

template <class RealType, class Policy>
inline RealType mean(const exponential_distribution<RealType, Policy>& dist)
{
   RealType result = 0;
   RealType lambda = dist.lambda();
   if(0 == detail::verify_lambda("hydra_boost::math::mean(const exponential_distribution<%1%>&)", lambda, &result, Policy()))
      return result;
   return 1 / lambda;
}

template <class RealType, class Policy>
inline RealType standard_deviation(const exponential_distribution<RealType, Policy>& dist)
{
   RealType result = 0;
   RealType lambda = dist.lambda();
   if(0 == detail::verify_lambda("hydra_boost::math::standard_deviation(const exponential_distribution<%1%>&)", lambda, &result, Policy()))
      return result;
   return 1 / lambda;
}

template <class RealType, class Policy>
inline RealType mode(const exponential_distribution<RealType, Policy>& /*dist*/)
{
   return 0;
}

template <class RealType, class Policy>
inline RealType median(const exponential_distribution<RealType, Policy>& dist)
{
   using hydra_boost::math::constants::ln_two;
   return ln_two<RealType>() / dist.lambda(); // ln(2) / lambda
}

template <class RealType, class Policy>
inline RealType skewness(const exponential_distribution<RealType, Policy>& /*dist*/)
{
   return 2;
}

template <class RealType, class Policy>
inline RealType kurtosis(const exponential_distribution<RealType, Policy>& /*dist*/)
{
   return 9;
}

template <class RealType, class Policy>
inline RealType kurtosis_excess(const exponential_distribution<RealType, Policy>& /*dist*/)
{
   return 6;
}

template <class RealType, class Policy>
inline RealType entropy(const exponential_distribution<RealType, Policy>& dist)
{
   using std::log;
   return 1 - log(dist.lambda());
}

} // namespace math
} // namespace hydra_boost

#ifdef _MSC_VER
# pragma warning(pop)
#endif

// This include must be at the end, *after* the accessors
// for this distribution have been defined, in order to
// keep compilers that support two-phase lookup happy.
#include <hydra/detail/external/hydra_boost/math/distributions/detail/derived_accessors.hpp>

#endif // HYDRA_BOOST_STATS_EXPONENTIAL_HPP
