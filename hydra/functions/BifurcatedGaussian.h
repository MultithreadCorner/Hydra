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
 * BifurcatedGaussian.h
 *
 *  Created on: 11/04/2018
 *      Author: Antonio Augusto Alves Junior
 *
 *  Updated on: Feb 18 2020
 *      Author: Davide Brundu
 *         Log: Update call interface
 */

#ifndef BIFURCATEDGAUSSIAN_H_
#define BIFURCATEDGAUSSIAN_H_


#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/Pdf.h>
#include <hydra/Integrator.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/Parameter.h>
#include <hydra/Distribution.h>
#include <hydra/Tuple.h>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <assert.h>
#include <utility>

namespace hydra {

/**
 * \ingroup common_functions
 * \class BifurcatedGaussian
 *
 *
 */
template<typename ArgType, typename Signature=double(ArgType) >
class BifurcatedGaussian: public BaseFunctor<BifurcatedGaussian<ArgType>, Signature, 3>
{
	using BaseFunctor<BifurcatedGaussian<ArgType>, Signature, 3>::_par;

public:

	BifurcatedGaussian()=delete;

	BifurcatedGaussian(Parameter const& mean, Parameter const& sigma_left , Parameter const& sigma_right ):
		BaseFunctor<BifurcatedGaussian<ArgType>, Signature, 3>({mean, sigma_left, sigma_right})
		{}

	__hydra_host__ __hydra_device__
	BifurcatedGaussian(BifurcatedGaussian<ArgType> const& other ):
		BaseFunctor<BifurcatedGaussian<ArgType>, Signature, 3>(other)
		{}

	__hydra_host__ __hydra_device__
	BifurcatedGaussian<ArgType>&
	operator=(BifurcatedGaussian<ArgType> const& other ){
		if(this==&other) return  *this;
		BaseFunctor<BifurcatedGaussian<ArgType>, Signature, 3>::operator=(other);
		return  *this;
	}

	__hydra_host__ __hydra_device__ inline
	double Evaluate(ArgType x)  const	{

		double m2 = (x - _par[0])*(x - _par[0] );
		double sigmaL = _par[1];
		double sigmaR = _par[2];

		double coef = ( (x - _par[0]) <= 0.0)*(::fabs(sigmaL) > 1e-30)*( -0.5/(sigmaL*sigmaL))
		            + ( (x - _par[0])  > 0.0)*(::fabs(sigmaR) > 1e-30)*( -0.5/(sigmaR*sigmaR)) ;

		return  CHECK_VALUE(exp(coef*m2), "par[0]=%f, par[1]=%f, par[2]=%f", _par[0], _par[1], _par[2]);

	}


};


template<typename ArgType>
class IntegrationFormula< BifurcatedGaussian<ArgType>, 1>
{

protected:

	inline std::pair<GReal_t, GReal_t>
	EvalFormula( BifurcatedGaussian<ArgType>const& functor, double LowerLimit, double UpperLimit )const
	{
		double fraction = cumulative(functor[0], functor[1], functor[2],  LowerLimit,  UpperLimit  );

		return std::make_pair(	CHECK_VALUE(fraction," par[0] = %f par[1] = %f par[2] = %f LowerLimit = %f UpperLimit = %f",
				functor[0], functor[1], functor[2], LowerLimit,UpperLimit ) ,0.0);

	}

private:

	inline double cumulative(const double mean, const double sigma_left, const double sigma_right,
			double LowerLimit, double UpperLimit ) const
	{
		static const double sqrt_pi_over_two = 1.2533141373155002512079;
		static const double sqrt_two         = 1.4142135623730950488017;


		double xscaleL = sqrt_two*sigma_left;
		double xscaleR = sqrt_two*sigma_right;

		double integral = 0.0;

		if(UpperLimit < mean)
		{
			integral = sigma_left * ( ::erf((UpperLimit - mean)/xscaleL) - ::erf((LowerLimit - mean)/xscaleL) );
		}
		else if (LowerLimit > mean)
		{
			integral = sigma_right * ( ::erf((UpperLimit - mean)/xscaleR) - ::erf((LowerLimit - mean)/xscaleR) );
		}
		else
		{
			integral =sigma_right*::erf((UpperLimit - mean)/xscaleR) -  sigma_left*::erf((LowerLimit - mean)/xscaleL);
		}

		return integral*sqrt_pi_over_two;

	}

};


template<typename ArgType>
struct RngFormula< BifurcatedGaussian<ArgType> >
{

	typedef ArgType value_type;

	__hydra_host__ __hydra_device__
	inline unsigned NCalls( BifurcatedGaussian<ArgType>const&) const
	{
		return 2;
	}

	template< typename T>
	__hydra_host__ __hydra_device__
	inline unsigned NCalls( std::initializer_list<T>) const
	{
		return 2;
	}

	template<typename Engine>
	__hydra_host__ __hydra_device__
	inline value_type Generate(Engine& rng, BifurcatedGaussian<ArgType>const& functor) const
	{
		double mean  = functor[0];
		double sigma_left  = functor[1];
		double sigma_right = functor[2];

		double forking_point = sigma_left/(sigma_left+sigma_right);

		double x = RngBase::normal(rng);
		if( RngBase::uniform(rng) < forking_point)
			x = -fabs(mean + sigma_left*x);
		else
			x = fabs(mean + sigma_right*x);

		return static_cast<value_type>(x);
	}

	template<typename Engine, typename T>
	__hydra_host__ __hydra_device__
	inline value_type Generate(Engine& rng, std::initializer_list<T> pars) const
	{
		double mean        = pars.begin()[0];
		double sigma_left  = pars.begin()[1];
		double sigma_right = pars.begin()[2];

		double forking_point = sigma_left/(sigma_left+sigma_right);

		double x = RngBase::normal(rng);
		if( RngBase::uniform(rng) < forking_point)
			x = -fabs(mean + sigma_left*x);
		else
			x = fabs(mean + sigma_right*x);

		return static_cast<value_type>(x);
	}

};

}  // namespace hydra




#endif /* BIFURCATEDGAUSSIAN_H_ */
