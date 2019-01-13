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
 * GeneralizedGamma.h
 *
 *  Created on: 15/08/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GENERALIZEDGAMMA_H_
#define GENERALIZEDGAMMA_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/Pdf.h>
#include <hydra/Integrator.h>
#include <hydra/cpp/System.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/detail/utility/SafeCompare.h>
#include <hydra/detail/utility/MachineEpsilon.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <cassert>
#include <utility>

#include <gsl/gsl_sf_gamma.h>

namespace hydra {


/**
 * \ingroup common_functions
 * \class GeneralizedGamma
 *
 * The Gamma distribution is often used to describe variables bounded on one side. An even
more flexible version of this distribution is obtained by adding a third parameter giving
the so called generalized Gamma distribution :

\f[
f(x; a, b, c) = \frac{ac{ax}^{bc−1} e^{−ax}}{Γ(b)}
]\f

where a (a scale parameter) and b are the same real positive parameters as is used for the
Gamma distribution but a third parameter c has been added (c = 1 for the ordinary Gamma
distribution). This new parameter may in principle take any real value but normally we
consider the case where c > 0 or even c ≥ 1.
 */
template<unsigned int ArgIndex=0>
class GeneralizedGamma: public BaseFunctor<GeneralizedGamma<ArgIndex>, double, 4>
{
	using BaseFunctor<GeneralizedGamma<ArgIndex>, double, 4>::_par;

public:

	GeneralizedGamma()=delete;

	GeneralizedGamma(Parameter const& X0, Parameter const& A , Parameter const& B, Parameter const& C  ):
		BaseFunctor<GeneralizedGamma<ArgIndex>, double, 4>({X0,A , B,C})
		{}

	__hydra_host__ __hydra_device__
	GeneralizedGamma(GeneralizedGamma<ArgIndex> const& other ):
		BaseFunctor<GeneralizedGamma<ArgIndex>, double,4>(other)
		{}

	__hydra_host__ __hydra_device__
	GeneralizedGamma<ArgIndex>&
	operator=(GeneralizedGamma<ArgIndex> const& other ){
		if(this==&other) return  *this;
		BaseFunctor<GeneralizedGamma<ArgIndex>,double, 4>::operator=(other);
		return  *this;
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline double Evaluate(unsigned int, T*x)  const	{

		const double X = x[ArgIndex] - _par[0];

		const double A = _par[1];
		const double B = _par[2];
		const double C = _par[3];

		const double r = detail::SafeGreaterThan(X, 0.0, detail::machine_eps_f64() ) ?
				A*::fabs(C)*::pow(A*X, B*C-1.0)*::exp(-::pow(A*X, C))/::tgamma(B): 0.0;

		return  CHECK_VALUE( r, "par[0]=%f, par[1]=%f par[2]=%f, par[3]=%f", _par[0], _par[1], _par[2], _par[3]);

	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline double Evaluate(T x)  const {

		const double X = x[ArgIndex] - _par[0];

		const double A = _par[1];
		const double B = _par[2];
		const double C = _par[3];

		const double r = detail::SafeGreaterThan(X, 0.0, detail::machine_eps_f64() ) ?
						A*::fabs(C)*::pow(A*X, B*C-1.0)*::exp(-::pow(A*X, C))/::tgamma(B): 0.0;

		return  CHECK_VALUE( r, "par[0]=%f, par[1]=%f par[2]=%f, par[3]=%f", _par[0], _par[1], _par[2], _par[3]);


	}

};


template<unsigned int ArgIndex>
class IntegrationFormula< GeneralizedGamma<ArgIndex>, 1>
{

protected:

	inline std::pair<GReal_t, GReal_t>
	EvalFormula( GeneralizedGamma<ArgIndex>const& functor, double LowerLimit, double UpperLimit )const
	{

		double min = LowerLimit > functor[0] ? LowerLimit : functor[0];

		double fraction =
				cumulative(functor[0], functor[1], functor[2], functor[3], UpperLimit)	-
				cumulative(functor[0], functor[1], functor[2], functor[3], min);

		return std::make_pair(
				CHECK_VALUE(-fraction," par[0] = %f par[1] = %f par[2] = %f par[3] = %f LowerLimit = %f UpperLimit = %f",
						functor[0], functor[1], functor[2], functor[3], min, UpperLimit ) ,0.0);

	}
private:

	inline double cumulative(const double x0, const double A, const double B, const double C, const double x ) const
	{

		const double X = x - x0;
		const double r = detail::SafeGreaterThan(C, 0.0, ::fabs(C)*std::numeric_limits<double>::epsilon()) ?
				inc_gamma(B, ::pow(A*X,::fabs(C))) : 1.0-inc_gamma(B, ::pow(A*X,::fabs(C)));

		return r;
	}

	inline double inc_gamma( const double a, const double x) const {

		return gsl_sf_gamma_inc_Q(a, x);
	}


};



}  // namespace hydra



#endif /* GENERALIZEDGAMMA_H_ */
