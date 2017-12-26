
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
 * Gaussian.h
 *
 *  Created on: Dec 11, 2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GAUSSIAN_H_
#define GAUSSIAN_H_

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
#include <assert.h>
#include <utility>

namespace hydra {

template<unsigned int ArgIndex=0>
class Gaussian: public BaseFunctor<Gaussian<ArgIndex>, double, 2>
{
	using BaseFunctor<Gaussian<ArgIndex>, double, 2>::_par;

public:

	Gaussian()=delete;

	Gaussian(Parameter const& mean, Parameter const& sigma ):
		BaseFunctor<Gaussian<ArgIndex>, double, 2>({mean, sigma})
		{}

	__host__ __device__
	Gaussian(Gaussian<ArgIndex> const& other ):
		BaseFunctor<Gaussian<ArgIndex>, double,2>(other)
		{}

	__host__ __device__
	Gaussian<ArgIndex>&
	operator=(Gaussian<ArgIndex> const& other ){
		if(this==&other) return  *this;
		BaseFunctor<Gaussian<ArgIndex>,double, 2>::operator=(other);
		return  *this;
	}

	template<typename T>
	__host__ __device__ inline
	double Evaluate(unsigned int n, T*x)  const	{
		double m2 = (x[ArgIndex] - _par[0])*(x[ArgIndex] - _par[0] );
		double s2 = _par[1]*_par[1];
		return  CHECK_VALUE(exp(-m2/(2.0 * s2 )), "par[0]=%f, par[1]=%f", _par[0], _par[1]);

	}

	template<typename T>
	__host__ __device__ inline
	double Evaluate(T x)  const {
		double m2 = ( get<ArgIndex>(x) - _par[0])*(get<ArgIndex>(x) - _par[0] );
		double s2 = _par[1]*_par[1];
		return CHECK_VALUE( exp(-m2/(2.0 * s2 )), "par[0]=%f, par[1]=%f", _par[0], _par[1]);

	}

};

class GaussianAnalyticalIntegral: public Integrator<GaussianAnalyticalIntegral>
{

public:

	GaussianAnalyticalIntegral(double min, double max):
		fLowerLimit(min),
		fUpperLimit(max)
	{
		assert( fLowerLimit < fUpperLimit && "hydra::ArgusShapeAnalyticalIntegral: MESSAGE << LowerLimit >= fUpperLimit >>");
	 }

	inline GaussianAnalyticalIntegral(GaussianAnalyticalIntegral const& other):
		fLowerLimit(other.GetLowerLimit()),
		fUpperLimit(other.GetUpperLimit())
	{}

	inline GaussianAnalyticalIntegral&
	operator=( GaussianAnalyticalIntegral const& other)
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

		double fraction = cumulative(functor[0], functor[1], fUpperLimit)
						- cumulative(functor[0], functor[1], fLowerLimit);

		return std::make_pair(
				CHECK_VALUE(fraction," par[0] = %f par[1] = %f fLowerLimit = %f fUpperLimit = %f", functor[0], functor[1], fLowerLimit,fUpperLimit ) ,0.0);
	}


private:

	inline double cumulative(const double mean, const double sigma, const double x) const
	{
		static const double sqrt_pi_over_two = 1.2533141373155002512079;
		static const double sqrt_two         = 1.4142135623730950488017;

		return sigma*sqrt_pi_over_two*(1.0 + erf( (x-mean)/( sigma*sqrt_two ) ) );
	}

	double fLowerLimit;
	double fUpperLimit;

};



}  // namespace hydra


#endif /* GAUSSIAN_H_ */
