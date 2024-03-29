//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_STATS_LOGNORMAL_HPP
#define HYDRA_BOOST_STATS_LOGNORMAL_HPP

// http://www.itl.nist.gov/div898/handbook/eda/section3/eda3669.htm
// http://mathworld.wolfram.com/LogNormalDistribution.html
// http://en.wikipedia.org/wiki/Lognormal_distribution

#include <hydra/detail/external/hydra_boost/math/distributions/fwd.hpp>
#include <hydra/detail/external/hydra_boost/math/distributions/normal.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/expm1.hpp>
#include <hydra/detail/external/hydra_boost/math/distributions/detail/common_error_handling.hpp>

#include <utility>

namespace hydra_boost{ namespace math
{
namespace detail
{

  template <class RealType, class Policy>
  inline bool check_lognormal_x(
        const char* function,
        RealType const& x,
        RealType* result, const Policy& pol)
  {
     if((x < 0) || !(hydra_boost::math::isfinite)(x))
     {
        *result = policies::raise_domain_error<RealType>(
           function,
           "Random variate is %1% but must be >= 0 !", x, pol);
        return false;
     }
     return true;
  }

} // namespace detail


template <class RealType = double, class Policy = policies::policy<> >
class lognormal_distribution
{
public:
   typedef RealType value_type;
   typedef Policy policy_type;

   lognormal_distribution(RealType l_location = 0, RealType l_scale = 1)
      : m_location(l_location), m_scale(l_scale)
   {
      RealType result;
      detail::check_scale("hydra_boost::math::lognormal_distribution<%1%>::lognormal_distribution", l_scale, &result, Policy());
      detail::check_location("hydra_boost::math::lognormal_distribution<%1%>::lognormal_distribution", l_location, &result, Policy());
   }

   RealType location()const
   {
      return m_location;
   }

   RealType scale()const
   {
      return m_scale;
   }
private:
   //
   // Data members:
   //
   RealType m_location;  // distribution location.
   RealType m_scale;     // distribution scale.
};

typedef lognormal_distribution<double> lognormal;

#ifdef __cpp_deduction_guides
template <class RealType>
lognormal_distribution(RealType)->lognormal_distribution<typename hydra_boost::math::tools::promote_args<RealType>::type>;
template <class RealType>
lognormal_distribution(RealType,RealType)->lognormal_distribution<typename hydra_boost::math::tools::promote_args<RealType>::type>;
#endif

template <class RealType, class Policy>
inline const std::pair<RealType, RealType> range(const lognormal_distribution<RealType, Policy>& /*dist*/)
{ // Range of permissible values for random variable x is >0 to +infinity.
   using hydra_boost::math::tools::max_value;
   return std::pair<RealType, RealType>(static_cast<RealType>(0), max_value<RealType>());
}

template <class RealType, class Policy>
inline const std::pair<RealType, RealType> support(const lognormal_distribution<RealType, Policy>& /*dist*/)
{ // Range of supported values for random variable x.
   // This is range where cdf rises from 0 to 1, and outside it, the pdf is zero.
   using hydra_boost::math::tools::max_value;
   return std::pair<RealType, RealType>(static_cast<RealType>(0),  max_value<RealType>());
}

template <class RealType, class Policy>
RealType pdf(const lognormal_distribution<RealType, Policy>& dist, const RealType& x)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std functions

   RealType mu = dist.location();
   RealType sigma = dist.scale();

   static const char* function = "hydra_boost::math::pdf(const lognormal_distribution<%1%>&, %1%)";

   RealType result = 0;
   if(0 == detail::check_scale(function, sigma, &result, Policy()))
      return result;
   if(0 == detail::check_location(function, mu, &result, Policy()))
      return result;
   if(0 == detail::check_lognormal_x(function, x, &result, Policy()))
      return result;

   if(x == 0)
      return 0;

   RealType exponent = log(x) - mu;
   exponent *= -exponent;
   exponent /= 2 * sigma * sigma;

   result = exp(exponent);
   result /= sigma * sqrt(2 * constants::pi<RealType>()) * x;

   return result;
}

template <class RealType, class Policy>
inline RealType cdf(const lognormal_distribution<RealType, Policy>& dist, const RealType& x)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std functions

   static const char* function = "hydra_boost::math::cdf(const lognormal_distribution<%1%>&, %1%)";

   RealType result = 0;
   if(0 == detail::check_scale(function, dist.scale(), &result, Policy()))
      return result;
   if(0 == detail::check_location(function, dist.location(), &result, Policy()))
      return result;
   if(0 == detail::check_lognormal_x(function, x, &result, Policy()))
      return result;

   if(x == 0)
      return 0;

   normal_distribution<RealType, Policy> norm(dist.location(), dist.scale());
   return cdf(norm, log(x));
}

template <class RealType, class Policy>
inline RealType quantile(const lognormal_distribution<RealType, Policy>& dist, const RealType& p)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std functions

   static const char* function = "hydra_boost::math::quantile(const lognormal_distribution<%1%>&, %1%)";

   RealType result = 0;
   if(0 == detail::check_scale(function, dist.scale(), &result, Policy()))
      return result;
   if(0 == detail::check_location(function, dist.location(), &result, Policy()))
      return result;
   if(0 == detail::check_probability(function, p, &result, Policy()))
      return result;

   if(p == 0)
      return 0;
   if(p == 1)
      return policies::raise_overflow_error<RealType>(function, 0, Policy());

   normal_distribution<RealType, Policy> norm(dist.location(), dist.scale());
   return exp(quantile(norm, p));
}

template <class RealType, class Policy>
inline RealType cdf(const complemented2_type<lognormal_distribution<RealType, Policy>, RealType>& c)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std functions

   static const char* function = "hydra_boost::math::cdf(const lognormal_distribution<%1%>&, %1%)";

   RealType result = 0;
   if(0 == detail::check_scale(function, c.dist.scale(), &result, Policy()))
      return result;
   if(0 == detail::check_location(function, c.dist.location(), &result, Policy()))
      return result;
   if(0 == detail::check_lognormal_x(function, c.param, &result, Policy()))
      return result;

   if(c.param == 0)
      return 1;

   normal_distribution<RealType, Policy> norm(c.dist.location(), c.dist.scale());
   return cdf(complement(norm, log(c.param)));
}

template <class RealType, class Policy>
inline RealType quantile(const complemented2_type<lognormal_distribution<RealType, Policy>, RealType>& c)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std functions

   static const char* function = "hydra_boost::math::quantile(const lognormal_distribution<%1%>&, %1%)";

   RealType result = 0;
   if(0 == detail::check_scale(function, c.dist.scale(), &result, Policy()))
      return result;
   if(0 == detail::check_location(function, c.dist.location(), &result, Policy()))
      return result;
   if(0 == detail::check_probability(function, c.param, &result, Policy()))
      return result;

   if(c.param == 1)
      return 0;
   if(c.param == 0)
      return policies::raise_overflow_error<RealType>(function, 0, Policy());

   normal_distribution<RealType, Policy> norm(c.dist.location(), c.dist.scale());
   return exp(quantile(complement(norm, c.param)));
}

template <class RealType, class Policy>
inline RealType mean(const lognormal_distribution<RealType, Policy>& dist)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std functions

   RealType mu = dist.location();
   RealType sigma = dist.scale();

   RealType result = 0;
   if(0 == detail::check_scale("hydra_boost::math::mean(const lognormal_distribution<%1%>&)", sigma, &result, Policy()))
      return result;
   if(0 == detail::check_location("hydra_boost::math::mean(const lognormal_distribution<%1%>&)", mu, &result, Policy()))
      return result;

   return exp(mu + sigma * sigma / 2);
}

template <class RealType, class Policy>
inline RealType variance(const lognormal_distribution<RealType, Policy>& dist)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std functions

   RealType mu = dist.location();
   RealType sigma = dist.scale();

   RealType result = 0;
   if(0 == detail::check_scale("hydra_boost::math::variance(const lognormal_distribution<%1%>&)", sigma, &result, Policy()))
      return result;
   if(0 == detail::check_location("hydra_boost::math::variance(const lognormal_distribution<%1%>&)", mu, &result, Policy()))
      return result;

   return hydra_boost::math::expm1(sigma * sigma, Policy()) * exp(2 * mu + sigma * sigma);
}

template <class RealType, class Policy>
inline RealType mode(const lognormal_distribution<RealType, Policy>& dist)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std functions

   RealType mu = dist.location();
   RealType sigma = dist.scale();

   RealType result = 0;
   if(0 == detail::check_scale("hydra_boost::math::mode(const lognormal_distribution<%1%>&)", sigma, &result, Policy()))
      return result;
   if(0 == detail::check_location("hydra_boost::math::mode(const lognormal_distribution<%1%>&)", mu, &result, Policy()))
      return result;

   return exp(mu - sigma * sigma);
}

template <class RealType, class Policy>
inline RealType median(const lognormal_distribution<RealType, Policy>& dist)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std functions
   RealType mu = dist.location();
   return exp(mu); // e^mu
}

template <class RealType, class Policy>
inline RealType skewness(const lognormal_distribution<RealType, Policy>& dist)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std functions

   //RealType mu = dist.location();
   RealType sigma = dist.scale();

   RealType ss = sigma * sigma;
   RealType ess = exp(ss);

   RealType result = 0;
   if(0 == detail::check_scale("hydra_boost::math::skewness(const lognormal_distribution<%1%>&)", sigma, &result, Policy()))
      return result;
   if(0 == detail::check_location("hydra_boost::math::skewness(const lognormal_distribution<%1%>&)", dist.location(), &result, Policy()))
      return result;

   return (ess + 2) * sqrt(hydra_boost::math::expm1(ss, Policy()));
}

template <class RealType, class Policy>
inline RealType kurtosis(const lognormal_distribution<RealType, Policy>& dist)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std functions

   //RealType mu = dist.location();
   RealType sigma = dist.scale();
   RealType ss = sigma * sigma;

   RealType result = 0;
   if(0 == detail::check_scale("hydra_boost::math::kurtosis(const lognormal_distribution<%1%>&)", sigma, &result, Policy()))
      return result;
   if(0 == detail::check_location("hydra_boost::math::kurtosis(const lognormal_distribution<%1%>&)", dist.location(), &result, Policy()))
      return result;

   return exp(4 * ss) + 2 * exp(3 * ss) + 3 * exp(2 * ss) - 3;
}

template <class RealType, class Policy>
inline RealType kurtosis_excess(const lognormal_distribution<RealType, Policy>& dist)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std functions

   // RealType mu = dist.location();
   RealType sigma = dist.scale();
   RealType ss = sigma * sigma;

   RealType result = 0;
   if(0 == detail::check_scale("hydra_boost::math::kurtosis_excess(const lognormal_distribution<%1%>&)", sigma, &result, Policy()))
      return result;
   if(0 == detail::check_location("hydra_boost::math::kurtosis_excess(const lognormal_distribution<%1%>&)", dist.location(), &result, Policy()))
      return result;

   return exp(4 * ss) + 2 * exp(3 * ss) + 3 * exp(2 * ss) - 6;
}

template <class RealType, class Policy>
inline RealType entropy(const lognormal_distribution<RealType, Policy>& dist)
{
   using std::log;
   RealType mu = dist.location();
   RealType sigma = dist.scale();
   return mu + log(constants::two_pi<RealType>()*constants::e<RealType>()*sigma*sigma)/2;
}

} // namespace math
} // namespace hydra_boost

// This include must be at the end, *after* the accessors
// for this distribution have been defined, in order to
// keep compilers that support two-phase lookup happy.
#include <hydra/detail/external/hydra_boost/math/distributions/detail/derived_accessors.hpp>

#endif // HYDRA_BOOST_STATS_STUDENTS_T_HPP


