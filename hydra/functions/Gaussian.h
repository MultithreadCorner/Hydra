
/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2021 Antonio Augusto Alves Junior
 *
 *   This file is part of Hydra Data Analysis Framework.
 *
 *   Hydra is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   Hydra is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with Hydra.  If not, see <http://www.gnu.org/licenses/>.
 *
 *---------------------------------------------------------------------------*/

/*
 * Gaussian.h
 *
 *  Created on: Dec 11, 2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GAUSSIAN_H_
#define GAUSSIAN_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/Pdf.h>
#include <hydra/Integrator.h>
#include <hydra/Distribution.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <assert.h>
#include <utility>

namespace hydra {

/**
 * \ingroup common_functions
 * \class Gaussian
 *
 * Gaussian functions are often used to represent the probability density function of a normally distributed random variable with
 * expected value \f$ \mu \f$ and variance \f$ \sigma \f$. In this case, the Gaussian is of the form:

\f[ g(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{ -\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2 }. \f]
 *
 */
template<typename ArgType, typename Signature=double(ArgType) >
class Gaussian: public BaseFunctor<Gaussian<ArgType>, Signature, 2>
{
	using BaseFunctor<Gaussian<ArgType>, Signature, 2>::_par;

public:

	Gaussian()=delete;

	Gaussian(Parameter const& mean, Parameter const& sigma ):
		BaseFunctor<Gaussian<ArgType>, Signature, 2>({mean, sigma})
		{}

	__hydra_host__ __hydra_device__
	Gaussian(Gaussian<ArgType> const& other ):
		BaseFunctor<Gaussian<ArgType>, Signature, 2>(other)
		{}

	__hydra_host__ __hydra_device__
	Gaussian<ArgType>& operator=(Gaussian<ArgType> const& other )
	{
		if(this==&other) return  *this;
		BaseFunctor<Gaussian<ArgType>, Signature, 2>::operator=(other);
		return  *this;
	}

	__hydra_host__ __hydra_device__
	inline double Evaluate(ArgType x)  const
	{
		double m2 = ( x - _par[0])*(x - _par[0] );
		double s2 = _par[1]*_par[1];
		return CHECK_VALUE( ::exp(-m2/(2.0 * s2 )), "par[0]=%f, par[1]=%f", _par[0], _par[1]);

	}

};

template<typename ArgType>
class IntegrationFormula< Gaussian<ArgType>, 1>
{

protected:

	inline std::pair<double, double>
	EvalFormula(Gaussian<ArgType>const& functor, double LowerLimit, double UpperLimit )const
	{
		double fraction = cumulative(functor[0], functor[1], UpperLimit)
							- cumulative(functor[0], functor[1], LowerLimit);

			return std::make_pair( CHECK_VALUE(fraction,
					" par[0] = %f par[1] = %f fLowerLimit = %f fUpperLimit = %f",
					functor[0], functor[1], LowerLimit, UpperLimit ) ,0.0);

	}
private:

	inline double cumulative(const double mean, const double sigma, const double x) const
	{
		static const double sqrt_pi_over_two = 1.2533141373155002512079;
		static const double sqrt_two         = 1.4142135623730950488017;

		return sigma*sqrt_pi_over_two*(1.0 + erf( (x-mean)/( sigma*sqrt_two ) ) );
	}
};


template<typename ArgType>
struct RngFormula< Gaussian<ArgType> >
{

	typedef ArgType value_type;
	__hydra_host__ __hydra_device__
	inline unsigned NCalls( Gaussian<ArgType>const&) const
	{
		return 1;
	}

	template< typename T>
	__hydra_host__ __hydra_device__
	inline unsigned NCalls( std::initializer_list<T>) const
	{
		return 1;
	}

	template<typename Engine>
	__hydra_host__ __hydra_device__
	inline value_type Generate(Engine& rng, Gaussian<ArgType>const& functor) const
	{
		double mean  = functor[0];
		double sigma = functor[1];

		double x = mean + sigma*RngBase::normal(rng);

		return static_cast<value_type>(x);
	}

	template<typename Engine, typename T>
	__hydra_host__ __hydra_device__
	inline value_type Generate(Engine& rng, std::initializer_list<T> pars) const
	{
		double mean  = pars.begin()[0];
		double sigma = pars.begin()[1];

		double x = mean + sigma*RngBase::normal(rng);

		return static_cast<value_type>(x);
	}



};

}  // namespace hydra


#endif /* GAUSSIAN_H_ */
