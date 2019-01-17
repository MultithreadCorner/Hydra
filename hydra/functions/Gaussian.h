
/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2019 Antonio Augusto Alves Junior
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
template<unsigned int ArgIndex=0>
class Gaussian: public BaseFunctor<Gaussian<ArgIndex>, double, 2>
{
	using BaseFunctor<Gaussian<ArgIndex>, double, 2>::_par;

public:

	Gaussian()=delete;

	Gaussian(Parameter const& mean, Parameter const& sigma ):
		BaseFunctor<Gaussian<ArgIndex>, double, 2>({mean, sigma})
		{}

	__hydra_host__ __hydra_device__
	Gaussian(Gaussian<ArgIndex> const& other ):
		BaseFunctor<Gaussian<ArgIndex>, double,2>(other)
		{}

	__hydra_host__ __hydra_device__
	Gaussian<ArgIndex>&
	operator=(Gaussian<ArgIndex> const& other ){
		if(this==&other) return  *this;
		BaseFunctor<Gaussian<ArgIndex>,double, 2>::operator=(other);
		return  *this;
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline double Evaluate(unsigned int, T*x)  const	{
		double m2 = (x[ArgIndex] - _par[0])*(x[ArgIndex] - _par[0] );
		double s2 = _par[1]*_par[1];
		return  CHECK_VALUE(::exp(-m2/(2.0 * s2 )), "par[0]=%f, par[1]=%f", _par[0], _par[1]);

	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline double Evaluate(T const& x)  const {
		double m2 = ( get<ArgIndex>(x) - _par[0])*(get<ArgIndex>(x) - _par[0] );
		double s2 = _par[1]*_par[1];
		return CHECK_VALUE( ::exp(-m2/(2.0 * s2 )), "par[0]=%f, par[1]=%f", _par[0], _par[1]);

	}

};

template<unsigned int ArgIndex>
class IntegrationFormula< Gaussian<ArgIndex>, 1>
{

protected:

	inline std::pair<GReal_t, GReal_t>
	EvalFormula(Gaussian<ArgIndex>const& functor, double LowerLimit, double UpperLimit )const
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


}  // namespace hydra


#endif /* GAUSSIAN_H_ */
