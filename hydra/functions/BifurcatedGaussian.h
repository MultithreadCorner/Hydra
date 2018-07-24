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
 * BifurcatedGaussian.h
 *
 *  Created on: 11/04/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef BIFURCATEDGAUSSIAN_H_
#define BIFURCATEDGAUSSIAN_H_


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
#include <assert.h>
#include <utility>

namespace hydra {

template<unsigned int ArgIndex=0>
class BifurcatedGaussian: public BaseFunctor<BifurcatedGaussian<ArgIndex>, double, 3>
{
	using BaseFunctor<BifurcatedGaussian<ArgIndex>, double, 3>::_par;

public:

	BifurcatedGaussian()=delete;

	BifurcatedGaussian(Parameter const& mean, Parameter const& sigma_left , Parameter const& sigma_right ):
		BaseFunctor<BifurcatedGaussian<ArgIndex>, double, 3>({mean, sigma_left, sigma_right})
		{}

	__hydra_host__ __hydra_device__
	BifurcatedGaussian(BifurcatedGaussian<ArgIndex> const& other ):
		BaseFunctor<BifurcatedGaussian<ArgIndex>, double,3>(other)
		{}

	__hydra_host__ __hydra_device__
	BifurcatedGaussian<ArgIndex>&
	operator=(BifurcatedGaussian<ArgIndex> const& other ){
		if(this==&other) return  *this;
		BaseFunctor<BifurcatedGaussian<ArgIndex>,double, 3>::operator=(other);
		return  *this;
	}

	template<typename T>
	__hydra_host__ __hydra_device__ inline
	double Evaluate(unsigned int, T*x)  const	{

		double m2 = (x[ArgIndex] - _par[0])*(x[ArgIndex] - _par[0] );
		double sigmaL = _par[1];
		double sigmaR = _par[2];

		double coef = ( (x[ArgIndex] - _par[0]) <= 0.0)*(::fabs(sigmaL) > 1e-30)*( -0.5/(sigmaL*sigmaL))
		            + ( (x[ArgIndex] - _par[0])  > 0.0)*(::fabs(sigmaR) > 1e-30)*( -0.5/(sigmaR*sigmaR)) ;

		return  CHECK_VALUE(exp(coef*m2), "par[0]=%f, par[1]=%f, par[2]=%f", _par[0], _par[1], _par[2]);

	}

	template<typename T>
	__hydra_host__ __hydra_device__ inline
	double Evaluate(T x)  const {

		double m2 = ( get<ArgIndex>(x) - _par[0])*(get<ArgIndex>(x) - _par[0] );

		double sigmaL = _par[1];
		double sigmaR = _par[2];

		double coef = ( (x[ArgIndex] - _par[0]) <= 0.0)*(::fabs(sigmaL) > 1e-30)*( -0.5/(sigmaL*sigmaL))
				    + ( (x[ArgIndex] - _par[0])  > 0.0)*(::fabs(sigmaR) > 1e-30)*( -0.5/(sigmaR*sigmaR)) ;

		return  CHECK_VALUE(exp(coef*m2), "par[0]=%f, par[1]=%f, par[2]=%f", _par[0], _par[1], _par[2]);

	}

};

class BifurcatedGaussianAnalyticalIntegral: public Integrator<BifurcatedGaussianAnalyticalIntegral>
{

public:

	BifurcatedGaussianAnalyticalIntegral(double min, double max):
		fLowerLimit(min),
		fUpperLimit(max)
	{
		assert( fLowerLimit < fUpperLimit && "hydra::ArgusShapeAnalyticalIntegral: MESSAGE << LowerLimit >= fUpperLimit >>");
	 }

	inline BifurcatedGaussianAnalyticalIntegral(BifurcatedGaussianAnalyticalIntegral const& other):
		fLowerLimit(other.GetLowerLimit()),
		fUpperLimit(other.GetUpperLimit())
	{}

	inline BifurcatedGaussianAnalyticalIntegral&
	operator=( BifurcatedGaussianAnalyticalIntegral const& other)
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

		double fraction = cumulative(functor[0], functor[1], functor[2] );

		return std::make_pair(
				CHECK_VALUE(fraction," par[0] = %f par[1] = %f par[2] = %f fLowerLimit = %f fUpperLimit = %f", functor[0], functor[1], functor[2], fLowerLimit,fUpperLimit ) ,0.0);
	}


private:

	inline double cumulative(const double mean, const double sigma_left, const double sigma_right) const
	{
		static const double sqrt_pi_over_two = 1.2533141373155002512079;
		static const double sqrt_two         = 1.4142135623730950488017;


		double xscaleL = sqrt_two*sigma_left;
		double xscaleR = sqrt_two*sigma_right;

		double integral = 0.0;

		if(fUpperLimit < mean)
		{
			integral = sigma_left * ( ::erf((fUpperLimit - mean)/xscaleL) - ::erf((fLowerLimit - mean)/xscaleL) );
		}
		else if (fLowerLimit > mean)
		{
			integral = sigma_right * ( ::erf((fUpperLimit - mean)/xscaleR) - ::erf((fLowerLimit - mean)/xscaleR) );
		}
		else
		{
			integral =sigma_right*::erf((fUpperLimit - mean)/xscaleR) -  sigma_left*::erf((fLowerLimit - mean)/xscaleL);
		}

		return integral*sqrt_pi_over_two;

	}

	double fLowerLimit;
	double fUpperLimit;

};



}  // namespace hydra




#endif /* BIFURCATEDGAUSSIAN_H_ */
