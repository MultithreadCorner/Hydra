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
 * ArgusShape.h
 *
 *  Created on: 18/12/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef ARGUSSHAPE_H_
#define ARGUSSHAPE_H_


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
 *
 * \class ArgusShape
 *
 * Implementation describing the ARGUS background shape.
 *
 * \tparam ArgIndex : index of the argument when evaluating on multidimensional data. Default is 0.
 */
template<unsigned int ArgIndex=0>
class ArgusShape: public BaseFunctor<ArgusShape<ArgIndex>, double, 3>
{
	using BaseFunctor<ArgusShape<ArgIndex>, double, 3>::_par;

public:

	ArgusShape()=delete;

	/**
	 * ArgusShape constructor.
	 *
	 * @param m0 : resonance mass
	 * @param slope : slope parameter
	 * @param power : power
	 */
	ArgusShape(Parameter const& m0,	Parameter const& slope, Parameter const& power):
		BaseFunctor<ArgusShape<ArgIndex>, double, 3>({m0,slope, power})
		{}

	/**
	 * Copy constuctor.
	 *
	 * @param other
	 */
	__hydra_host__ __hydra_device__
	ArgusShape(ArgusShape<ArgIndex> const& other ):
		BaseFunctor<ArgusShape<ArgIndex>, double,3>(other)
		{}

	/**
	 * Assignment constructor.
	 *
	 * @param other
	 * @return
	 */
	__hydra_host__ __hydra_device__
	ArgusShape<ArgIndex>&
	operator=(ArgusShape<ArgIndex> const& other ){
		if(this==&other) return  *this;
		BaseFunctor<ArgusShape<ArgIndex>,double, 3>::operator=(other);
		return  *this;
	}

	template<typename T>
	__hydra_host__ __hydra_device__ inline
	double Evaluate(unsigned int, T*x)  const {

		double m  = x[ArgIndex]; //mass
		double m0 = _par[0]; //resonance mass
		double c  = _par[1]; //slope
		double p  = _par[2]; //power


		return  CHECK_VALUE( (m/m0)>=1.0 ? 0: m*pow((1 - (m/m0)*(m/m0)) ,p)*exp(c*(1 - (m/m0)*(m/m0))),\
				"par[0]=%f, par[1]=%f, _par[2]=%f", _par[0], _par[1], _par[2]) ;
	}

	template<typename T>
	__hydra_host__ __hydra_device__ inline
	double Evaluate(T x)  const {
		double m  = hydra::get<ArgIndex>(x); //mass
		double m0 = _par[0]; //resonance mass
		double c  = _par[1]; //slope
		double p  = _par[2]; //power

		return  CHECK_VALUE( (m/m0)>=1.0 ? 0: m*pow((1 - (m/m0)*(m/m0)) ,p)*exp(c*(1 - (m/m0)*(m/m0))),\
				"par[0]=%f, par[1]=%f, _par[2]=%f", _par[0], _par[1], _par[2]) ;

	}

};

/**
 * @class ArgusShapeAnalyticalIntegral
 * Implementation of analytical integral for the ARGUS background shape with power = 0.5.
 */

template<unsigned int ArgIndex>
class IntegrationFormula< ArgusShape<ArgIndex>, 1>
{

protected:

	inline std::pair<GReal_t, GReal_t>
	EvalFormula(ArgusShape<ArgIndex>const& functor, double LowerLimit, double UpperLimit ) const {

		if(functor[2]<0.5 || functor[2] > 0.5 ) {

			throw std::invalid_argument("ArgusShapeAnalyticalIntegral can not handle ArgusShape with power != 0.5. Return {nan, nan}");
		}

		double r = cumulative(functor[0], functor[1],UpperLimit)
							 - cumulative(functor[0], functor[1], LowerLimit);

		return std::make_pair( CHECK_VALUE(r, "par[0] = %f par[1] = %f par[2] = %f  LowerLimit = %f UpperLimit = %f",
						functor[0], functor[1], functor[2], LowerLimit, UpperLimit)	,0.0);
	}


private:

	inline double cumulative( const double m,  const double c,  const double x) const
	{
		static const double sqrt_pi = 1.7724538509055160272982;

		double f = x<m ? (1.0 -pow(x/m,2)) : 0.0;
		double r = -0.5*m*m*(exp(c*f)*sqrt(f)/c + 0.5/pow(-c,1.5)*sqrt_pi*erf(sqrt(-c*f)));
		return r;
	}

};


}  // namespace hydra



#endif /* ARGUSSHAPE_H_ */
