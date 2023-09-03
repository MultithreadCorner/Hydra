// Copyright Nick Thompson, 2017
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

/*
 * This class performs exp-sinh quadrature on half infinite intervals.
 *
 * References:
 *
 * 1) Tanaka, Ken'ichiro, et al. "Function classes for double exponential integration formulas." Numerische Mathematik 111.4 (2009): 631-655.
 */

#ifndef HYDRA_BOOST_MATH_QUADRATURE_EXP_SINH_HPP
#define HYDRA_BOOST_MATH_QUADRATURE_EXP_SINH_HPP

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <hydra/detail/external/hydra_boost/math/quadrature/detail/exp_sinh_detail.hpp>

namespace hydra_boost{ namespace math{ namespace quadrature {

template<class Real, class Policy = policies::policy<> >
class exp_sinh
{
public:
   exp_sinh(size_t max_refinements = 9)
      : m_imp(std::make_shared<detail::exp_sinh_detail<Real, Policy>>(max_refinements)) {}

    template<class F>
    auto integrate(const F& f, Real a, Real b, Real tol = hydra_boost::math::tools::root_epsilon<Real>(), Real* error = nullptr, Real* L1 = nullptr, std::size_t* levels = nullptr) const ->decltype(std::declval<F>()(std::declval<Real>()));
    template<class F>
    auto integrate(const F& f, Real tol = hydra_boost::math::tools::root_epsilon<Real>(), Real* error = nullptr, Real* L1 = nullptr, std::size_t* levels = nullptr) const ->decltype(std::declval<F>()(std::declval<Real>()));

private:
    std::shared_ptr<detail::exp_sinh_detail<Real, Policy>> m_imp;
};

template<class Real, class Policy>
template<class F>
auto exp_sinh<Real, Policy>::integrate(const F& f, Real a, Real b, Real tolerance, Real* error, Real* L1, std::size_t* levels) const ->decltype(std::declval<F>()(std::declval<Real>()))
{
    typedef decltype(f(a)) K;
    static_assert(!std::is_integral<K>::value,
                  "The return type cannot be integral, it must be either a real or complex floating point type.");
    using std::abs;
    using hydra_boost::math::constants::half;
    using hydra_boost::math::quadrature::detail::exp_sinh_detail;

    static const char* function = "hydra_boost::math::quadrature::exp_sinh<%1%>::integrate";

    // Neither limit may be a NaN:
    if((hydra_boost::math::isnan)(a) || (hydra_boost::math::isnan)(b))
    {
       return static_cast<K>(policies::raise_domain_error(function, "NaN supplied as one limit of integration - sorry I don't know what to do", a, Policy()));
     }
    // Right limit is infinite:
    if ((hydra_boost::math::isfinite)(a) && (b >= hydra_boost::math::tools::max_value<Real>()))
    {
        // If a = 0, don't use an additional level of indirection:
        if (a == static_cast<Real>(0))
        {
            return m_imp->integrate(f, error, L1, function, tolerance, levels);
        }
        const auto u = [&](Real t)->K { return f(t + a); };
        return m_imp->integrate(u, error, L1, function, tolerance, levels);
    }

    if ((hydra_boost::math::isfinite)(b) && a <= -hydra_boost::math::tools::max_value<Real>())
    {
        const auto u = [&](Real t)->K { return f(b-t);};
        return m_imp->integrate(u, error, L1, function, tolerance, levels);
    }

    // Infinite limits:
    if ((a <= -hydra_boost::math::tools::max_value<Real>()) && (b >= hydra_boost::math::tools::max_value<Real>()))
    {
        return static_cast<K>(policies::raise_domain_error(function, "Use sinh_sinh quadrature for integration over the whole real line; exp_sinh is for half infinite integrals.", a, Policy()));
    }
    // If we get to here then both ends must necessarily be finite:
    return static_cast<K>(policies::raise_domain_error(function, "Use tanh_sinh quadrature for integration over finite domains; exp_sinh is for half infinite integrals.", a, Policy()));
}

template<class Real, class Policy>
template<class F>
auto exp_sinh<Real, Policy>::integrate(const F& f, Real tolerance, Real* error, Real* L1, std::size_t* levels) const ->decltype(std::declval<F>()(std::declval<Real>()))
{
    static const char* function = "hydra_boost::math::quadrature::exp_sinh<%1%>::integrate";
    using std::abs;
    if (abs(tolerance) > 1) {
        std::string msg = std::string(__FILE__) + ":" + std::to_string(__LINE__) + ":" + std::string(function) + ": The tolerance provided is unusually large; did you confuse it with a domain bound?";
        throw std::domain_error(msg);
    }
    return m_imp->integrate(f, error, L1, function, tolerance, levels);
}


}}}
#endif