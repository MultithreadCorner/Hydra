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
 * CrystalBallShape.h
 *
 *  Created on: 19/12/2017
 *      Author: Antonio Augusto Alves Junior
 *
 *  Updated on: Feb 21 2020
 *      Author: Davide Brundu
 *         Log: Update call interface
 */

#ifndef CRYSTALBALLSHAPE_H_
#define CRYSTALBALLSHAPE_H_


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
 * @class CrystalBallShape
 * Implementation the Crystal Ball line shape.
 *
 */
template<typename ArgType, typename Signature=double(ArgType)>
class CrystalBallShape: public BaseFunctor<CrystalBallShape<ArgType>, Signature, 4>
{
	using BaseFunctor<CrystalBallShape<ArgType>, Signature, 4>::_par;

public:

	CrystalBallShape()=delete;

	CrystalBallShape(Parameter const& mean, Parameter const& sigma
			, Parameter const& alpha, Parameter const& n):
		BaseFunctor<CrystalBallShape<ArgType>, Signature, 4>({mean, sigma, alpha, n})
		{}

	__hydra_host__ __hydra_device__
	CrystalBallShape(CrystalBallShape<ArgType> const& other ):
		BaseFunctor<CrystalBallShape<ArgType>, Signature, 4>(other)
		{}

	__hydra_host__ __hydra_device__
	CrystalBallShape<ArgType>&
	operator=(CrystalBallShape<ArgType> const& other ){
		if(this==&other) return  *this;
		BaseFunctor<CrystalBallShape<ArgType>, Signature, 4>::operator=(other);
		return  *this;
	}

	__hydra_host__ __hydra_device__
	inline double Evaluate(ArgType x)  const
	{
		double m     = x; //mass
		double mean  = _par[0];
		double sigma = _par[1];
		double alpha = _par[2];
		double N     = _par[3];

		double t = (alpha < 0.0) ? (mean-m)/sigma:(mean-m)/sigma;
		double abs_alpha = fabs(alpha);


		double r = (t >= -abs_alpha) ? exp(-0.5*t*t):
				pow(N/abs_alpha, N)*exp(-0.5*abs_alpha*abs_alpha)/pow(N/abs_alpha - abs_alpha- t, N);

		return CHECK_VALUE(r, "par[0]=%f, par[1]=%f, par[2]=%f, par[3]=%f", _par[0], _par[1], _par[2], _par[3]  );
	}


};

template<typename ArgType>
class IntegrationFormula< CrystalBallShape<ArgType>, 1>
{

protected:

	inline std::pair<GReal_t, GReal_t>
	EvalFormula( CrystalBallShape<ArgType>const& functor, double LowerLimit, double UpperLimit )const
	{
		double r = integral(functor[0], functor[1], functor[2], functor[3], LowerLimit, UpperLimit  );

				return std::make_pair(
				CHECK_VALUE(r," par[0] = %f par[1] = %f par[2] = %f par[3] = %f LowerLimit = %f UpperLimit = %f",\
						functor[0], functor[1],functor[2], functor[3], LowerLimit, UpperLimit ) ,0.0);

	}
private:


	inline double integral( const double m0,  const double sigma,  const double alpha,  const double n,
			double LowerLimit, double UpperLimit) const
	{
		// borrowed from roofit
		static const double sqrtPiOver2 = 1.2533141373;
		static const double sqrt2 = 1.4142135624;

		double result = 0.0;
		bool   useLog = false;

		if( fabs(n-1.0) < 1.0e-05 )
			useLog = true;

		double sig = fabs(sigma);

		double tmin = (LowerLimit-m0)/sig;
		double tmax = (UpperLimit-m0)/sig;

		if(alpha < 0) {
			double tmp = tmin;
			tmin = -tmax;
			tmax = -tmp;
		}

		double absAlpha = fabs(alpha);

		if( tmin >= -absAlpha ) {
			result += sig*sqrtPiOver2*(   erf(tmax/sqrt2) - erf(tmin/sqrt2) );
		}
		else if( tmax <= -absAlpha ) {
			double a = pow(n/absAlpha,n)*exp(-0.5*absAlpha*absAlpha);
			double b = n/absAlpha - absAlpha;

			if(useLog) {
				result += a*sig*( log(b-tmin) - log(b-tmax) );
			}
			else {
				result += a*sig/(1.0-n)*(   1.0/(pow(b-tmin,n-1.0)) - 1.0/(pow(b-tmax,n-1.0)) );
			}
		}
		else {

			double a = pow(n/absAlpha,n)*exp(-0.5*absAlpha*absAlpha);
			double b = n/absAlpha - absAlpha;

			double term1 = 0.0;
			if(useLog) {
				term1 = a*sig*(  log(b-tmin) - log(n/absAlpha));
			}
			else {
				term1 = a*sig/(1.0-n)*( 1.0/(pow(b-tmin,n-1.0)) - 1.0/(pow(n/absAlpha,n-1.0)) );
			}

			double term2 = sig*sqrtPiOver2*(erf(tmax/sqrt2) - erf(-absAlpha/sqrt2) );


			result += term1 + term2;
		}

		return result;
	}

};


}  // namespace hydra



#endif /* CRYSTALBALLSHAPE_H_ */
