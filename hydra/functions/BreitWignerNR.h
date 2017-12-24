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
 * BreitWignerNR.h
 *
 *  Created on: Dec 13, 2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef BREITWIGNERNR_H_
#define BREITWIGNERNR_H_

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

template<unsigned int ArgIndex=0>
class BreitWignerNR: public BaseFunctor<BreitWignerNR<ArgIndex>, double, 2>
{
	using BaseFunctor<BreitWignerNR<ArgIndex>, double, 2>::_par;

public:

	BreitWignerNR()=delete;

	BreitWignerNR(Parameter const& mean, Parameter const& lambda ):
		BaseFunctor<BreitWignerNR<ArgIndex>, double, 2>({mean, lambda})
		{}

	__host__ __device__
	BreitWignerNR(BreitWignerNR<ArgIndex> const& other ):
		BaseFunctor<BreitWignerNR<ArgIndex>, double,2>(other)
		{}

	__host__ __device__
	BreitWignerNR<ArgIndex>&
	operator=(BreitWignerNR<ArgIndex> const& other ){
		if(this==&other) return  *this;
		BaseFunctor<BreitWignerNR<ArgIndex>,double, 2>::operator=(other);
		return  *this;
	}

	template<typename T>
	__host__ __device__ inline
	double Evaluate(unsigned int n, T*x)  const
	{
		double mean  = _par[0];
		double width = _par[1];
		double m     = x[ArgIndex];

		double m2 = (m - mean)*(m - mean);
		double w2 = width*width;

		return CHECK_VALUE(1.0/(m2 + 0.25*w2), "par[0]=%f, par[1]=%f", _par[0], _par[1]) ;
	}

	template<typename T>
	__host__ __device__ inline
	double Evaluate(T x)  const
	{
		double mean  = _par[0];
		double width = _par[1];
		double m     = x[ArgIndex];

		double m2 = (m - mean)*(m - mean);
		double w2 = width*width;

		return  CHECK_VALUE(1.0/(m2 + 0.25*w2), "par[0]=%f, par[1]=%f", _par[0], _par[1]) ;
	}

};

class BreitWignerNRAnalyticalIntegral: public Integrator<BreitWignerNRAnalyticalIntegral>
{

public:

	BreitWignerNRAnalyticalIntegral(double min, double max):
		fLowerLimit(min),
		fUpperLimit(max)
	{
		assert(fLowerLimit < fUpperLimit
				&& "hydra::BreitWignerNRAnalyticalIntegral: MESSAGE << LowerLimit >= fUpperLimit >>");
	}

	inline BreitWignerNRAnalyticalIntegral(BreitWignerNRAnalyticalIntegral const& other):
		fLowerLimit(other.GetLowerLimit()),
		fUpperLimit(other.GetUpperLimit())
	{}

	inline BreitWignerNRAnalyticalIntegral&
	operator=( BreitWignerNRAnalyticalIntegral const& other)
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

		double r = cumulative(functor[0], functor[1], fUpperLimit)
						 - cumulative(functor[0], functor[1], fLowerLimit);

		return std::make_pair(
				CHECK_VALUE(r," par[0] = %f par[1] = %f fLowerLimit = %f fUpperLimit = %f", functor[0], functor[1], fLowerLimit,fUpperLimit ), 0.0);
	}


private:

	inline double cumulative(double mean, double width, double x) const
	{
		double c = 2.0/width;
		return c*( atan( c*( x - mean)));
	}

	double fLowerLimit;
	double fUpperLimit;

};




}  // namespace hydra



#endif /* BREITWIGNERNR_H_ */
