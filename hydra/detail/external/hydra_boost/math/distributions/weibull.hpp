//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_STATS_WEIBULL_HPP
#define HYDRA_BOOST_STATS_WEIBULL_HPP

// http://www.itl.nist.gov/div898/handbook/eda/section3/eda3668.htm
// http://mathworld.wolfram.com/WeibullDistribution.html

#include <hydra/detail/external/hydra_boost/math/distributions/fwd.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/gamma.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/log1p.hpp>
#include <hydra/detail/external/hydra_boost/math/special_functions/expm1.hpp>
#include <hydra/detail/external/hydra_boost/math/distributions/detail/common_error_handling.hpp>
#include <hydra/detail/external/hydra_boost/math/distributions/complement.hpp>

#include <utility>

namespace hydra_boost{ namespace math
{
namespace detail{

template <class RealType, class Policy>
inline bool check_weibull_shape(
      const char* function,
      RealType shape,
      RealType* result, const Policy& pol)
{
   if((shape <= 0) || !(hydra_boost::math::isfinite)(shape))
   {
      *result = policies::raise_domain_error<RealType>(
         function,
         "Shape parameter is %1%, but must be > 0 !", shape, pol);
      return false;
   }
   return true;
}

template <class RealType, class Policy>
inline bool check_weibull_x(
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

template <class RealType, class Policy>
inline bool check_weibull(
      const char* function,
      RealType scale,
      RealType shape,
      RealType* result, const Policy& pol)
{
   return check_scale(function, scale, result, pol) && check_weibull_shape(function, shape, result, pol);
}

} // namespace detail

template <class RealType = double, class Policy = policies::policy<> >
class weibull_distribution
{
public:
   using value_type = RealType;
   using policy_type = Policy;

   explicit weibull_distribution(RealType l_shape, RealType l_scale = 1)
      : m_shape(l_shape), m_scale(l_scale)
   {
      RealType result;
      detail::check_weibull("hydra_boost::math::weibull_distribution<%1%>::weibull_distribution", l_scale, l_shape, &result, Policy());
   }

   RealType shape()const
   {
      return m_shape;
   }

   RealType scale()const
   {
      return m_scale;
   }
private:
   //
   // Data members:
   //
   RealType m_shape;     // distribution shape
   RealType m_scale;     // distribution scale
};

using weibull = weibull_distribution<double>;

#ifdef __cpp_deduction_guides
template <class RealType>
weibull_distribution(RealType)->weibull_distribution<typename hydra_boost::math::tools::promote_args<RealType>::type>;
template <class RealType>
weibull_distribution(RealType,RealType)->weibull_distribution<typename hydra_boost::math::tools::promote_args<RealType>::type>;
#endif

template <class RealType, class Policy>
inline std::pair<RealType, RealType> range(const weibull_distribution<RealType, Policy>& /*dist*/)
{ // Range of permissible values for random variable x.
   using hydra_boost::math::tools::max_value;
   return std::pair<RealType, RealType>(static_cast<RealType>(0), max_value<RealType>());
}

template <class RealType, class Policy>
inline std::pair<RealType, RealType> support(const weibull_distribution<RealType, Policy>& /*dist*/)
{ // Range of supported values for random variable x.
   // This is range where cdf rises from 0 to 1, and outside it, the pdf is zero.
   using hydra_boost::math::tools::max_value;
   using hydra_boost::math::tools::min_value;
   return std::pair<RealType, RealType>(min_value<RealType>(),  max_value<RealType>());
   // A discontinuity at x == 0, so only support down to min_value.
}

template <class RealType, class Policy>
inline RealType pdf(const weibull_distribution<RealType, Policy>& dist, const RealType& x)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std functions

   static const char* function = "hydra_boost::math::pdf(const weibull_distribution<%1%>, %1%)";

   RealType shape = dist.shape();
   RealType scale = dist.scale();

   RealType result = 0;
   if(false == detail::check_weibull(function, scale, shape, &result, Policy()))
      return result;
   if(false == detail::check_weibull_x(function, x, &result, Policy()))
      return result;

   if(x == 0)
   {
      if(shape == 1)
      {
         return 1 / scale;
      }
      if(shape > 1)
      {
         return 0;
      }
      return policies::raise_overflow_error<RealType>(function, 0, Policy());
   }
   result = exp(-pow(x / scale, shape));
   result *= pow(x / scale, shape - 1) * shape / scale;

   return result;
}

template <class RealType, class Policy>
inline RealType logpdf(const weibull_distribution<RealType, Policy>& dist, const RealType& x)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std functions

   static const char* function = "hydra_boost::math::logpdf(const weibull_distribution<%1%>, %1%)";

   RealType shape = dist.shape();
   RealType scale = dist.scale();

   RealType result = 0;
   if(false == detail::check_weibull(function, scale, shape, &result, Policy()))
      return result;
   if(false == detail::check_weibull_x(function, x, &result, Policy()))
      return result;

   if(x == 0)
   {
      if(shape == 1)
      {
         return log(1 / scale);
      }
      if(shape > 1)
      {
         return 0;
      }
      return policies::raise_overflow_error<RealType>(function, 0, Policy());
   }
   
   result = log(shape) - shape * log(scale) + log(x) * (shape - 1) - pow(x / scale, shape);

   return result;
}

template <class RealType, class Policy>
inline RealType cdf(const weibull_distribution<RealType, Policy>& dist, const RealType& x)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std functions

   static const char* function = "hydra_boost::math::cdf(const weibull_distribution<%1%>, %1%)";

   RealType shape = dist.shape();
   RealType scale = dist.scale();

   RealType result = 0;
   if(false == detail::check_weibull(function, scale, shape, &result, Policy()))
      return result;
   if(false == detail::check_weibull_x(function, x, &result, Policy()))
      return result;

   result = -hydra_boost::math::expm1(-pow(x / scale, shape), Policy());

   return result;
}

template <class RealType, class Policy>
inline RealType quantile(const weibull_distribution<RealType, Policy>& dist, const RealType& p)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std functions

   static const char* function = "hydra_boost::math::quantile(const weibull_distribution<%1%>, %1%)";

   RealType shape = dist.shape();
   RealType scale = dist.scale();

   RealType result = 0;
   if(false == detail::check_weibull(function, scale, shape, &result, Policy()))
      return result;
   if(false == detail::check_probability(function, p, &result, Policy()))
      return result;

   if(p == 1)
      return policies::raise_overflow_error<RealType>(function, 0, Policy());

   result = scale * pow(-hydra_boost::math::log1p(-p, Policy()), 1 / shape);

   return result;
}

template <class RealType, class Policy>
inline RealType cdf(const complemented2_type<weibull_distribution<RealType, Policy>, RealType>& c)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std functions

   static const char* function = "hydra_boost::math::cdf(const weibull_distribution<%1%>, %1%)";

   RealType shape = c.dist.shape();
   RealType scale = c.dist.scale();

   RealType result = 0;
   if(false == detail::check_weibull(function, scale, shape, &result, Policy()))
      return result;
   if(false == detail::check_weibull_x(function, c.param, &result, Policy()))
      return result;

   result = exp(-pow(c.param / scale, shape));

   return result;
}

template <class RealType, class Policy>
inline RealType quantile(const complemented2_type<weibull_distribution<RealType, Policy>, RealType>& c)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std functions

   static const char* function = "hydra_boost::math::quantile(const weibull_distribution<%1%>, %1%)";

   RealType shape = c.dist.shape();
   RealType scale = c.dist.scale();
   RealType q = c.param;

   RealType result = 0;
   if(false == detail::check_weibull(function, scale, shape, &result, Policy()))
      return result;
   if(false == detail::check_probability(function, q, &result, Policy()))
      return result;

   if(q == 0)
      return policies::raise_overflow_error<RealType>(function, 0, Policy());

   result = scale * pow(-log(q), 1 / shape);

   return result;
}

template <class RealType, class Policy>
inline RealType mean(const weibull_distribution<RealType, Policy>& dist)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std functions

   static const char* function = "hydra_boost::math::mean(const weibull_distribution<%1%>)";

   RealType shape = dist.shape();
   RealType scale = dist.scale();

   RealType result = 0;
   if(false == detail::check_weibull(function, scale, shape, &result, Policy()))
      return result;

   result = scale * hydra_boost::math::tgamma(1 + 1 / shape, Policy());
   return result;
}

template <class RealType, class Policy>
inline RealType variance(const weibull_distribution<RealType, Policy>& dist)
{
   RealType shape = dist.shape();
   RealType scale = dist.scale();

   static const char* function = "hydra_boost::math::variance(const weibull_distribution<%1%>)";

   RealType result = 0;
   if(false == detail::check_weibull(function, scale, shape, &result, Policy()))
   {
      return result;
   }
   result = hydra_boost::math::tgamma(1 + 1 / shape, Policy());
   result *= -result;
   result += hydra_boost::math::tgamma(1 + 2 / shape, Policy());
   result *= scale * scale;
   return result;
}

template <class RealType, class Policy>
inline RealType mode(const weibull_distribution<RealType, Policy>& dist)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std function pow.

   static const char* function = "hydra_boost::math::mode(const weibull_distribution<%1%>)";

   RealType shape = dist.shape();
   RealType scale = dist.scale();

   RealType result = 0;
   if(false == detail::check_weibull(function, scale, shape, &result, Policy()))
   {
      return result;
   }
   if(shape <= 1)
      return 0;
   result = scale * pow((shape - 1) / shape, 1 / shape);
   return result;
}

template <class RealType, class Policy>
inline RealType median(const weibull_distribution<RealType, Policy>& dist)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std function pow.

   static const char* function = "hydra_boost::math::median(const weibull_distribution<%1%>)";

   RealType shape = dist.shape(); // Wikipedia k
   RealType scale = dist.scale(); // Wikipedia lambda

   RealType result = 0;
   if(false == detail::check_weibull(function, scale, shape, &result, Policy()))
   {
      return result;
   }
   using hydra_boost::math::constants::ln_two;
   result = scale * pow(ln_two<RealType>(), 1 / shape);
   return result;
}

template <class RealType, class Policy>
inline RealType skewness(const weibull_distribution<RealType, Policy>& dist)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std functions

   static const char* function = "hydra_boost::math::skewness(const weibull_distribution<%1%>)";

   RealType shape = dist.shape();
   RealType scale = dist.scale();

   RealType result = 0;
   if(false == detail::check_weibull(function, scale, shape, &result, Policy()))
   {
      return result;
   }

   RealType g1 = hydra_boost::math::tgamma(1 + 1 / shape, Policy());
   RealType g2 = hydra_boost::math::tgamma(1 + 2 / shape, Policy());
   RealType g3 = hydra_boost::math::tgamma(1 + 3 / shape, Policy());
   RealType d = pow(g2 - g1 * g1, RealType(1.5));

   result = (2 * g1 * g1 * g1 - 3 * g1 * g2 + g3) / d;
   return result;
}

template <class RealType, class Policy>
inline RealType kurtosis_excess(const weibull_distribution<RealType, Policy>& dist)
{
   HYDRA_BOOST_MATH_STD_USING  // for ADL of std functions

   static const char* function = "hydra_boost::math::kurtosis_excess(const weibull_distribution<%1%>)";

   RealType shape = dist.shape();
   RealType scale = dist.scale();

   RealType result = 0;
   if(false == detail::check_weibull(function, scale, shape, &result, Policy()))
      return result;

   RealType g1 = hydra_boost::math::tgamma(1 + 1 / shape, Policy());
   RealType g2 = hydra_boost::math::tgamma(1 + 2 / shape, Policy());
   RealType g3 = hydra_boost::math::tgamma(1 + 3 / shape, Policy());
   RealType g4 = hydra_boost::math::tgamma(1 + 4 / shape, Policy());
   RealType g1_2 = g1 * g1;
   RealType g1_4 = g1_2 * g1_2;
   RealType d = g2 - g1_2;
   d *= d;

   result = -6 * g1_4 + 12 * g1_2 * g2 - 3 * g2 * g2 - 4 * g1 * g3 + g4;
   result /= d;
   return result;
}

template <class RealType, class Policy>
inline RealType kurtosis(const weibull_distribution<RealType, Policy>& dist)
{
   return kurtosis_excess(dist) + 3;
}

template <class RealType, class Policy>
inline RealType entropy(const weibull_distribution<RealType, Policy>& dist)
{
   using std::log;
   RealType k = dist.shape();
   RealType lambda = dist.scale();
   return constants::euler<RealType>()*(1-1/k) + log(lambda/k) + 1;
}

} // namespace math
} // namespace hydra_boost

// This include must be at the end, *after* the accessors
// for this distribution have been defined, in order to
// keep compilers that support two-phase lookup happy.
#include <hydra/detail/external/hydra_boost/math/distributions/detail/derived_accessors.hpp>

#endif // HYDRA_BOOST_STATS_WEIBULL_HPP


