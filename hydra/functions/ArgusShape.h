/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2018 Antonio Augusto Alves Junior
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

#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/Pdf.h>
#include <hydra/detail/Integrator.h>
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
 * @class ArgusShape
 * Implementation describing the ARGUS background shape.
 * @tparam ArgIndex : index of the argument when evaluating on multidimensional data. Default is 0.
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
	__host__ __device__
	ArgusShape(ArgusShape<ArgIndex> const& other ):
		BaseFunctor<ArgusShape<ArgIndex>, double,3>(other)
		{}

	/**
	 * Assignment constructor.
	 *
	 * @param other
	 * @return
	 */
	__host__ __device__
	ArgusShape<ArgIndex>&
	operator=(ArgusShape<ArgIndex> const& other ){
		if(this==&other) return  *this;
		BaseFunctor<ArgusShape<ArgIndex>,double, 3>::operator=(other);
		return  *this;
	}

	template<typename T>
	__host__ __device__ inline
	double Evaluate(unsigned int, T*x)  const {

		double m  = x[ArgIndex]; //mass
		double m0 = _par[0]; //resonance mass
		double c  = _par[1]; //slope
		double p  = _par[2]; //power


		return  CHECK_VALUE( (m/m0)>=1.0 ? 0: m*pow((1 - (m/m0)*(m/m0)) ,p)*exp(c*(1 - (m/m0)*(m/m0))),\
				"par[0]=%f, par[1]=%f, _par[2]=%f", _par[0], _par[1], _par[2]) ;
	}

	template<typename T>
	__host__ __device__ inline
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
class ArgusShapeAnalyticalIntegral: public Integrator<ArgusShapeAnalyticalIntegral>
{

public:

	ArgusShapeAnalyticalIntegral(double min, double max):
		fLowerLimit(min),
		fUpperLimit(max)
	{
		assert(fLowerLimit < fUpperLimit && "hydra::ArgusShapeAnalyticalIntegral: MESSAGE << LowerLimit >= fUpperLimit >>");
	}

	inline ArgusShapeAnalyticalIntegral(ArgusShapeAnalyticalIntegral const& other):
		fLowerLimit(other.GetLowerLimit()),
		fUpperLimit(other.GetUpperLimit())
	{}

	inline ArgusShapeAnalyticalIntegral&
	operator=( ArgusShapeAnalyticalIntegral const& other)
	{
		if(this == &other) return *this;

		this->fLowerLimit = other.GetLowerLimit();
		this->fUpperLimit = other.GetUpperLimit();

		return *this;
	}

	double GetLowerLimit() const {
		return fLowerLimit;
	}

	void SetLowerLimit(double lowerLimit ) {
		fLowerLimit = lowerLimit;
	}

	double GetUpperLimit() const {
		return fUpperLimit;
	}

	void SetUpperLimit(double upperLimit) {
		fUpperLimit = upperLimit;
	}

	template<typename FUNCTOR>	inline
	std::pair<double, double> Integrate(FUNCTOR const& functor) const {

		if(functor[2]<0.5 || functor[2] > 0.5 ) {

			std::cout  <<  functor[2] << std::endl;
			throw std::invalid_argument("ArgusShapeAnalyticalIntegral can not handle ArgusShape with power != 0.5. Return {nan, nan}");
		}

		double r = cumulative(functor[0], functor[1],fUpperLimit)
						 - cumulative(functor[0], functor[1], fLowerLimit);


		return std::make_pair(
				CHECK_VALUE(r, "par[0] = %f par[1] = %f par[2] = %f  fLowerLimit = %f fUpperLimit = %f", functor[0], functor[1], functor[2], fLowerLimit,fUpperLimit)
			,0.0);
	}


private:

	inline double cumulative( const double m,  const double c,  const double x) const
	{
		static const double sqrt_pi = 1.7724538509055160272982;

		double f = x<m ? (1.0 -pow(x/m,2)) : 0.0;
		double r = -0.5*m*m*(exp(c*f)*sqrt(f)/c + 0.5/pow(-c,1.5)*sqrt_pi*erf(sqrt(-c*f)));
		return r;
	}

	double fLowerLimit;
	double fUpperLimit;

};



}  // namespace hydra



#endif /* ARGUSSHAPE_H_ */
