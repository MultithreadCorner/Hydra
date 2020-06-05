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
 * JohnsonSuShape.h
 *
 *  Created on: 29/08/2018
 *      Author: Davide Brundu
 *              from RooJohnsonSU.cxx by Maurizio Martinelli
 *
 *  Updated on: Feb 18 2020
 *      Author: Antonio Augusto Alves Junior
 *         Log: Implementing new call interface
 *
 *  reference:
 *  Johnson, N. L. (1954),
 *  Systems of frequency curves derived from the first law of Laplace.,
 *  Trabajos de Estadistica, 5, 283-291.
 */

#ifndef JOHNSONSUSHAPE_H_
#define JOHNSONSUSHAPE_H_


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
#include <cassert>
#include <utility>


namespace hydra {


/**
 * \ingroup common_functions
 * \class JohnsonSUShape
 *
 * Implementation the JohnsonSU line shape.
 * https://en.wikipedia.org/wiki/Johnson%27s_SU-distribution
 *
 * @tparam ArgIndex : index of the argument when evaluating on multidimensional data. Default is 0.
 */
template<typename ArgType, typename Signature=double(ArgType)>
class JohnsonSU: public BaseFunctor<JohnsonSU<ArgType>, Signature, 4>
{
	using BaseFunctor<JohnsonSU<ArgType>, Signature, 4>::_par;

public:

	JohnsonSU()=delete;

	JohnsonSU(Parameter const& gamma, Parameter const& delta
			, Parameter const& xi, Parameter const& lambda):
			BaseFunctor<JohnsonSU<ArgType>, Signature, 4>({gamma, delta, xi, lambda})
			{}

	__hydra_host__ __hydra_device__
	JohnsonSU(JohnsonSU<ArgType> const& other ):
			BaseFunctor<JohnsonSU<ArgType>, Signature, 4>(other)
			{}

	__hydra_host__ __hydra_device__
	JohnsonSU<ArgType>&
	operator=(JohnsonSU<ArgType> const& other ){
		if(this==&other) return  *this;
		BaseFunctor<JohnsonSU<ArgType>, Signature, 4>::operator=(other);
		return  *this;
	}

	__hydra_host__ __hydra_device__
	inline double Evaluate(ArgType x)  const
	{
		//gathering parameters
		double gamma  = _par[0];
		double delta  = _par[1];
		double xi     = _par[2];
		//actually only 1/lambda is used
		double inverse_lambda = 1.0/_par[3];

		// z =  (x-xi)/lambda
		double z    = (x-xi)*inverse_lambda;

		// A = \frac{\delta}{ \lambda * \sqrt{2\pi} }
		double A = inverse_lambda*delta*hydra::math_constants::inverse_sqrt2Pi;

		//B = \frac{1}{\sqrt{1 + z^2}}
		double B = 1.0/::sqrt( 1 + z*z);

		// C = {(\gamma + \delta * \asinh(z) )}^{2}
		double C = gamma + delta*::asinh(z); C *=C;

		double result = A*B*::exp(-0.5*C);

		return CHECK_VALUE(result, "par[0]=%f, par[1]=%f, par[2]=%f, par[3]=%f", _par[0], _par[1], _par[2], _par[3]  );


	}


};

template<typename ArgType>
class IntegrationFormula< JohnsonSU<ArgType>, 1>
{

protected:

	inline std::pair<GReal_t, GReal_t>
	EvalFormula( JohnsonSU<ArgType>const& functor, double LowerLimit, double UpperLimit )const
	{
		double r = cumulative(functor[0], functor[1], functor[2], functor[3], UpperLimit)
				 - cumulative(functor[0], functor[1], functor[2], functor[3], LowerLimit);

				return std::make_pair(
				CHECK_VALUE(r," par[0] = %f par[1] = %f par[2] = %f par[3] = %f LowerLimit = %f UpperLimit = %f",\
						functor[0], functor[1],functor[2], functor[3], LowerLimit, UpperLimit ) ,0.0);

	}

private:

	inline double cumulative( const double gamma,  const double delta,  const double xi,  const double lambda, const double x) const
	{
		//actually only 1/lambda is used
		double inverse_lambda = 1.0/lambda;

		// z =  (x-xi)/lambda
		double z    = (x-xi)*inverse_lambda;

		// C = {(\gamma + \delta * \asinh(z) )}
		double C = gamma + delta*::asinh(z);

		return 0.5*(1.0 + ::erf(C*hydra::math_constants::inverse_sqrt2));
	}


};

template<typename ArgType>
struct RngFormula< JohnsonSU<ArgType> >
{

	typedef ArgType value_type;
	__hydra_host__ __hydra_device__
	inline unsigned NCalls( JohnsonSU<ArgType>const&) const
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
	inline value_type Generate(Engine& rng, JohnsonSU<ArgType>const& functor) const
	{
		double gamma  = functor[0];
		double delta  = functor[1];
		double xi     = functor[2];
		double lambda = functor[3];

		return static_cast<value_type>(::sinh((nci(RngBase::uniform(rng)) -gamma)/delta )*lambda + xi);
	}

	template<typename Engine, typename T>
	__hydra_host__ __hydra_device__
	inline 	value_type Generate(Engine& rng, std::initializer_list<T> pars) const
	{
		double gamma  = pars[0];
		double delta  = pars[1];
		double xi     = pars[2];
		double lambda = pars[3];

		return static_cast<value_type>(::sinh((nci(RngBase::uniform(rng)) -gamma)/delta )*lambda + xi);

	}
private:
	__hydra_host__ __hydra_device__
	inline double nci(double x) const
		{
			static const double sqrt_two         = 1.4142135623730950488017;

			return sqrt_two *(hydra::erfinv(2*x-1));
		}

};


}  // namespace hydra



#endif /* JOHNSONSUSHAPE_H_ */
