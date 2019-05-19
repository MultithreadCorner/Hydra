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
 * JohnsonSuShape.h
 *
 *  Created on: 29/08/2018
 *      Author: Davide Brundu
 *              from RooJohnsonSU.cxx by Maurizio Martinelli
 *
 *  reference: Johnson, N. L. (1954). Systems of frequency curves derived from the first law of Laplace., Trabajos de Estadistica, 5, 283-291.
 */

#ifndef JOHNSONSUSHAPE_H_
#define JOHNSONSUSHAPE_H_


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
template<unsigned int ArgIndex=0>
class JohnsonSUShape: public BaseFunctor<JohnsonSUShape<ArgIndex>, double, 4>
{
	using BaseFunctor<JohnsonSUShape<ArgIndex>, double, 4>::_par;

public:

	JohnsonSUShape()=delete;

	JohnsonSUShape(Parameter const& gamma, Parameter const& delta
			, Parameter const& xi, Parameter const& lambda):
			BaseFunctor<JohnsonSUShape<ArgIndex>, double, 4>({gamma, delta, xi, lambda})
			{}

	__hydra_host__ __hydra_device__
	JohnsonSUShape(JohnsonSUShape<ArgIndex> const& other ):
			BaseFunctor<JohnsonSUShape<ArgIndex>, double,4>(other)
			{}

	__hydra_host__ __hydra_device__
	JohnsonSUShape<ArgIndex>&
	operator=(JohnsonSUShape<ArgIndex> const& other ){
		if(this==&other) return  *this;
		BaseFunctor<JohnsonSUShape<ArgIndex>,double, 4>::operator=(other);
		return  *this;
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline double Evaluate(unsigned int , T* X)  const
	{
		double x      = X[ArgIndex];
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

	template<typename T>
	__hydra_host__ __hydra_device__
	inline	double Evaluate(T const& X)  const
	{
		double x     = hydra::get<ArgIndex>(X);
		//gathering parameters
		double gamma = _par[0];
		double delta = _par[1];
		double xi    = _par[2];
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

template<unsigned int ArgIndex>
class IntegrationFormula< JohnsonSUShape<ArgIndex>, 1>
{

protected:

	inline std::pair<GReal_t, GReal_t>
	EvalFormula( JohnsonSUShape<ArgIndex>const& functor, double LowerLimit, double UpperLimit )const
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


}  // namespace hydra



#endif /* JOHNSONSUSHAPE_H_ */
