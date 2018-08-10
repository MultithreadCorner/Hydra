/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2017 Antonio Augusto Alves Junior
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
 * DeltaDMassBackground.h
 *
 *  Created on: Jul 31, 2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DELTADMASSBACKGROUND_H_
#define DELTADMASSBACKGROUND_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
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

#include <gsl/gsl_sf_gamma_inc.h>

namespace hydra {


template<unsigned int ArgIndex=0>
class DeltaDMassBackground: public BaseFunctor<DeltaDMassBackground<ArgIndex>, double, 4>
{
	using BaseFunctor<DeltaDMassBackground<ArgIndex>, double, 4>::_par;

public:

	DeltaDMassBackground() = delete;

	DeltaDMassBackground(Parameter const& threshold, Parameter const& A, Parameter const& B, Parameter const& C):
		BaseFunctor<DeltaDMassBackground<ArgIndex>, double, 4>({threshold, A, B, C})
		{}

	__hydra_host__ __hydra_device__
	DeltaDMassBackground(DeltaDMassBackground<ArgIndex> const& other ):
	BaseFunctor<DeltaDMassBackground<ArgIndex>, double,4>(other)
	{}

	__hydra_host__ __hydra_device__
	DeltaDMassBackground<ArgIndex>&
	operator=(DeltaDMassBackground<ArgIndex> const& other ){
		if(this==&other) return  *this;
		BaseFunctor<DeltaDMassBackground<ArgIndex>,double, 4>::operator=(other);
		return  *this;
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline double Evaluate(unsigned int, T* x)  const	{

		double arg   = (x[ArgIndex] - _par[0]);
		double ratio = (arg/_par[0]);
		double val   = arg>0 ? (1.0- ::exp(-arg/_par[3]))*::pow(ratio, _par[1]) + _par[2]*(ratio-1.0) : 0;

		return  CHECK_VALUE( (val>0 ? val : 0), "par[0]=%f, par[1]=%f, par[2]=%f, par[3]=%f ", _par[0], _par[1], _par[2], _par[3]);

	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline double Evaluate(T x)  const {

		double arg   = get<ArgIndex>(x) - _par[0];
		double ratio = (arg/_par[0]);
		//                     (1.0- exp(-x/c)))*pow(x/m, a) + b*(x/m-1.0)
		double val   = arg>0 ? (1.0- ::exp(-arg/_par[3]))*::pow(ratio, _par[1]) + _par[2]*(ratio-1.0) : 0;

		return  CHECK_VALUE( (val>0 ? val : 0), "par[0]=%f, par[1]=%f, par[2]=%f, par[3]=%f ", _par[0], _par[1], _par[2], _par[3]);

	}

};


/**
 * @class DeltaDMassBackgroundAnalyticalIntegral
 * Implementation of analytical integral for the ARGUS background shape with power = 0.5.
 */
class DeltaDMassBackgroundAnalyticalIntegral: public Integrator<DeltaDMassBackgroundAnalyticalIntegral>
{

public:

	DeltaDMassBackgroundAnalyticalIntegral(double min, double max):
		fLowerLimit(min),
		fUpperLimit(max)
	{
		assert(fLowerLimit < fUpperLimit && "hydra::DeltaDMassBackgroundAnalyticalIntegral: MESSAGE << LowerLimit >= fUpperLimit >>");
	}

	inline DeltaDMassBackgroundAnalyticalIntegral(DeltaDMassBackgroundAnalyticalIntegral const& other):
		fLowerLimit(other.GetLowerLimit()),
		fUpperLimit(other.GetUpperLimit())
	{}

	inline DeltaDMassBackgroundAnalyticalIntegral&
	operator=( DeltaDMassBackgroundAnalyticalIntegral const& other)
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



		double r = cumulative(functor[0], functor[1],functor[2], functor[3],fUpperLimit)
						 - cumulative(functor[0], functor[1], fLowerLimit);


		return std::make_pair(
				CHECK_VALUE(r, "par[0] = %f par[1] = %f par[2] = %f  fLowerLimit = %f fUpperLimit = %f", functor[0], functor[1], functor[2], fLowerLimit,fUpperLimit)
			,0.0);
	}


private:

	inline double cumulative( const double m0, const double a, const double b, const double c,  const double x) const
	{
		double arg = x - m0;

		double r = c*::pow(arg/c,-a)*::pow(arg/m0,a)*sf_gamma(a + 1, x/c)
				+ ::pow(arg*(arg/m),a)/(a + 1) + b*x*((0.5 *x)/m - 1);
	}

	inline double sf_gamma(const double a, const double x ) const
	{
		return  gsl_sf_gamma_inc(a, x);
	}

	double fLowerLimit;
	double fUpperLimit;

};
}// namespace hydra

#endif /* DELTADMASSBACKGROUND_H_ */
