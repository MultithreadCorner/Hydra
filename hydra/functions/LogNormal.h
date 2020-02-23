
/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2020 Antonio Augusto Alves Junior
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
 * LogNormal.h
 *
 *  Created on: Aug 2, 2018
 *      Author: Marcos Romero Lamas
 *
 *  Updated on: Oct 30 2018
 *      Author: Antonio Augusto Alves Junior
 *         Log: Adding new analytical integration interface
 *
 *  Updated on: Feb 18 2020
 *      Author: Antonio Augusto Alves Junior
 *         Log: Implementing new call interface
 *
 *
 */

#ifndef LOGNORMAL_H_
#define LOGNORMAL_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/Pdf.h>
#include <hydra/Integrator.h>
#include <hydra/functions/detail/inverse_erf.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/Parameter.h>
#include <hydra/Distribution.h>
#include <hydra/Tuple.h>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <assert.h>
#include <utility>

namespace hydra {

/**
 * \ingroup common_functions
 * \class LogNormal
 *
 * In probability theory, a log-normal (or lognormal) distribution is a continuous probability distribution of a random
 * variable whose logarithm is normally distributed. Thus, if the random variable X is log-normally distributed, then Y = ln(X) has a normal distribution.
 */
template<typename ArgType, typename Signature=double(ArgType) >
class LogNormal: public BaseFunctor<LogNormal<ArgType>, Signature, 2>
{
	using BaseFunctor<LogNormal<ArgType>, Signature, 2>::_par;

public:

	LogNormal() = delete;

	LogNormal(Parameter const& mean, Parameter const& sigma ):
		BaseFunctor<LogNormal<ArgType>, Signature, 2>({mean, sigma})
		{}

	__hydra_host__ __hydra_device__
	LogNormal(LogNormal<ArgType> const& other ):
		BaseFunctor<LogNormal<ArgType>, Signature, 2>(other)
		{}

	__hydra_host__ __hydra_device__
	LogNormal<ArgType>&
	operator=(LogNormal<ArgType> const& other ){
		if(this==&other) return  *this;
		BaseFunctor<LogNormal<ArgType>, Signature, 2>::operator=(other);
		return  *this;
	}

	__hydra_host__ __hydra_device__
	inline double Evaluate(ArgType x)  const
	{
		double m2  = (::log(x) - _par[0])*(::log(x) - _par[0] );
		double s2  = _par[1]*_par[1];
		double val = (::exp(-m2/(2.0 * s2 ))) / x;
		return  CHECK_VALUE( (x>0 ? val : 0) , "par[0]=%f, par[1]=%f", _par[0], _par[1]);
	}


};

template<typename ArgType>
class IntegrationFormula< LogNormal<ArgType>, 1>
{

protected:

	inline std::pair<GReal_t, GReal_t>
	EvalFormula( LogNormal<ArgType>const& functor, double LowerLimit, double UpperLimit )const
	{


		double fraction = cumulative(functor[0], functor[1], UpperLimit)
						- cumulative(functor[0], functor[1], LowerLimit);

		return std::make_pair(
				CHECK_VALUE(fraction," par[0] = %f par[1] = %f LowerLimit = %f UpperLimit = %f",
						functor[0], functor[1], LowerLimit, UpperLimit ) , 0.0);


	}
private:

	inline double cumulative(const double mean, const double sigma, const double x) const
	{
		static const double sqrt_pi_over_two = 1.2533141373155002512079;
		static const double sqrt_two         = 1.4142135623730950488017;

		return sigma*sqrt_pi_over_two*( ::erf( (::log(x)-mean)/( sigma*sqrt_two ) ) );
	}

};

template<typename ArgType>
struct RngFormula< LogNormal<ArgType> >
{

	typedef ArgType value_type;

	template<typename Engine>
	__hydra_host__ __hydra_device__
	value_type Generate(Engine& rng, LogNormal<ArgType>const& functor) const
	{
		static const double sqrt_two  = 1.4142135623730950488017;
		double mean  = functor[0];
		double sigma = functor[1];

		double x = ::exp(RngBase::normal(rng));

		return static_cast<value_type>(x);
	}

	template<typename Engine, typename T>
	__hydra_host__ __hydra_device__
	value_type Generate(Engine& rng, std::initializer_list<T> pars) const
	{
		static const double sqrt_two  = 1.4142135623730950488017;
		double mean  = pars.begin()[0];
		double sigma = pars.begin()[1];

		double x = ::exp(RngBase::normal(rng));

		return static_cast<value_type>(x);
	}



};

}  // namespace hydra


#endif /* LOGNORMAL_H_ */
