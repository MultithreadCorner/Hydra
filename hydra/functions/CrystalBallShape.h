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
 * CrystalBallShape.h
 *
 *  Created on: 19/12/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CRYSTALBALLSHAPE_H_
#define CRYSTALBALLSHAPE_H_

#include <hydra/Function.h>
#include <hydra/Pdf.h>
#include <hydra/detail/Integrator.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <cassert>
#include <utility>


namespace hydra {

/**
 * @class CrystalBallShape
 * Implementation the Crystal Ball line shape.
 *
 * @tparam ArgIndex : index of the argument when evaluating on multidimensional data. Default is 0.
 */
template<unsigned int ArgIndex=0>
class CrystalBallShape: public BaseFunctor<CrystalBallShape<ArgIndex>, double, 4>
{
	using BaseFunctor<CrystalBallShape<ArgIndex>, double, 4>::_par;

public:

	CrystalBallShape()=delete;

	CrystalBallShape(Parameter const& mean, Parameter const& sigma
			, Parameter const& alpha, Parameter const& n):
		BaseFunctor<CrystalBallShape<ArgIndex>, double, 4>({mean, sigma, alpha, n})
		{}

	__host__ __device__
	CrystalBallShape(CrystalBallShape<ArgIndex> const& other ):
		BaseFunctor<CrystalBallShape<ArgIndex>, double,4>(other)
		{}

	__host__ __device__
	CrystalBallShape<ArgIndex>&
	operator=(CrystalBallShape<ArgIndex> const& other ){
		if(this==&other) return  *this;
		BaseFunctor<CrystalBallShape<ArgIndex>,double, 4>::operator=(other);
		return  *this;
	}

	template<typename T>
	__host__ __device__ inline
	double Evaluate(unsigned int n, T*x)
	{
		double m     = x[ArgIndex]; //mass
		double mean  = _par[0];
		double sigma = _par[1];
		double alpha = _par[2];
		double N     = _par[3];

		double t = (alpha < 0) ? (m-mean)/sigma:(mean-m)/sigma;
		double abs_alpha = fabs(alpha);

		return t >= -abs_alpha ?
				exp(-0.5*t*t):
				pow(N/absAlpha,N)*exp(-0.5*absAlpha*absAlpha)/pow(N/absAlpha - absAlpha- t, N);
	}

	template<typename T>
	__host__ __device__ inline
	double Evaluate(T x)
	{
		double m     = hydra::get<ArgIndex>(x); //mass
		double mean  = _par[0];
		double sigma = _par[1];
		double alpha = _par[2];
		double n     = _par[3];

		double t = (alpha < 0) ? (m-mean)/sigma:(mean-m)/sigma;
		double abs_alpha = fabs(alpha);

		return t >= -abs_alpha ? exp(-0.5*t*t):
				pow(N/abs_alpha,N)*exp(-0.5*abs_alpha*abs_alpha)/pow(N/abs_alpha - abs_alpha- t, N);
	}

};


class CrystalBallShapeAnalyticalIntegral: public Integrator<CrystalBallShapeAnalyticalIntegral>
{

public:

	CrystalBallShapeAnalyticalIntegral(double min, double max):
		fLowerLimit(min),
		fUpperLimit(max)
	{
		std::assert(fLowerLimit >= fUpperLimit
				&& "hydra::ArgusShapeAnalyticalIntegral: MESSAGE << LowerLimit >= fUpperLimit >>");
	 }

	inline CrystalBallShapeAnalyticalIntegral(CrystalBallShapeAnalyticalIntegral const& other):
		fLowerLimit(other.GetLowerLimit()),
		fUpperLimit(other.GetUpperLimit())
	{}

	inline CrystalBallShapeAnalyticalIntegral&
	operator=( CrystalBallShapeAnalyticalIntegral const& other)
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
	std::pair<double, double> Integrate(FUNCTOR const& functor){

		double fraction = cumulative(functor[0], functor[1], fUpperLimit)
						 - cumulative(functor[0], functor[1], fLowerLimit);

		double scale = functor[1]*sqrt(2.0*PI);

		return std::make_pair(fraction*scale ,0.0);
	}


private:

	inline double cumulative(double mean, double sigma, double alpha, double N,  double x)
	{
		static const double sqrtPiOver2 = 1.2533141373;
		static const double sqrt2       = 1.4142135624;

		double t = (alpha < 0) ? (m-mean)/sigma:(mean-m)/sigma;
		double abs_alpha = fabs(alpha);


		return ( t >= -abs_alpha )
				? sigma*sqrtPiOver2*(1.0 + erf(t/sqrt2 ) ):
				( pow(-abs_alpha + N/abs_alpha - t, -N )*( -abs_alpha + N/abs_alpha - t))/(N-1);
	}

	double fLowerLimit;
	double fUpperLimit;

};
}  // namespace hydra



#endif /* CRYSTALBALLSHAPE_H_ */
