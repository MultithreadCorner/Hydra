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
 * JohnsonSuShape.h
 *
 *  Created on: 29/08/2018
 *      Author: Davide Brundu
 *              from RooJohnsonSU.cxx by Maurizio Martinelli
 *
 *  reference: Johnson, N. L. (1954). Systems of frequency curves derived from the first law of Laplace., Trabajos de Estadistica, 5, 283-291.
 */

#ifndef JOHNSONSUSHAPE_H_
#define JOHNSONSUSHAPE_H_


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


namespace hydra {


/**
 * @class JohnsonSUShape
 * Implementation the JohnsonSU line shape.
 *
 * @tparam ArgIndex : index of the argument when evaluating on multidimensional data. Default is 0.
 */
template<unsigned int ArgIndex=0>
class JohnsonSUShape: public BaseFunctor<JohnsonSUShape<ArgIndex>, double, 4>
{
	using BaseFunctor<JohnsonSUShape<ArgIndex>, double, 4>::_par;

public:

	JohnsonSUShape()=delete;

	JohnsonSUShape(Parameter const& mean, Parameter const& width
			, Parameter const& nu, Parameter const& tau):
			BaseFunctor<JohnsonSUShape<ArgIndex>, double, 4>({mean, width, nu, tau})
			{}

	__hydra_host__ __hydra_device__
	JohnsonSUShape(JohnsonSUShape<ArgIndex> const& other ):
			BaseFunctor<JohnsonSUShape<ArgIndex>, double,4>(other)
			{}

	__hydra_host__ __hydra_device__
	JohnsonSUShape<ArgIndex>&
	operator=(JohnsonSUShape<ArgIndex> const& other ){
		if(this==&other) return  *this;
		BaseFunctor<JohnsonSUShape<ArgIndex>,double, 4>::operator=(other);
		return  *this;
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline double Evaluate(unsigned int , T*x)  const
	{
		double m     = x[ArgIndex]; //mass
		double mean  = _par[0];
		double width = _par[1];
		double nu    = _par[2];
		double tau   = _par[3];

		double w       = exp( tau * tau);
		double omega   = - nu * tau;
		double c       = 0.5 * (w-1) * (w * cosh(2 * omega) + 1);
		c              = pow(c, -0.5);
		double z       = (m - (mean + c * width * sqrt(w) * sinh(omega) )) / c / width;
		double r       = -nu + asinh(z) / tau;

		double val     = 1. / (c * width * 2 * PI);
		val *= 1. / (tau * sqrt(z*z+1));
		val *= exp(-0.5 * r * r);

		return CHECK_VALUE(val, "par[0]=%f, par[1]=%f, par[2]=%f, par[3]=%f", _par[0], _par[1], _par[2], _par[3]  );
		

	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline	double Evaluate(T x)  const
	{
		double m     = hydra::get<ArgIndex>(x); //mass
		double mean  = _par[0];
		double width = _par[1];
		double nu    = _par[2];
		double tau   = _par[3];

		double w       = exp( tau * tau);
		double omega   = - nu * tau;
		double c       = 0.5 * (w-1) * (w * cosh(2 * omega) + 1);
		c              = pow(c, -0.5);
		double z       = (m - (mean + c * width * sqrt(w) * sinh(omega) )) / c / width;
		double r       = -nu + asinh(z) / tau;

		double val     = 1. / (c * width * 2 * PI);
		val *= 1. / (tau * sqrt(z*z+1));
		val *= exp(-0.5 * r * r);

		return CHECK_VALUE(val, "par[0]=%f, par[1]=%f, par[2]=%f, par[3]=%f", _par[0], _par[1], _par[2], _par[3]  );
	}
};



class JohnsonSUShapeAnalyticalIntegral: public Integrator<JohnsonSUShapeAnalyticalIntegral>
{

public:

	JohnsonSUShapeAnalyticalIntegral(double min, double max):
		fLowerLimit(min),
		fUpperLimit(max)
	{
		assert(fLowerLimit < fUpperLimit
				&& "hydra::JohnsonSUShapeAnalyticalIntegral: MESSAGE << LowerLimit >= fUpperLimit >>");
	 }

	inline JohnsonSUShapeAnalyticalIntegral(JohnsonSUShapeAnalyticalIntegral const& other):
		fLowerLimit(other.GetLowerLimit()),
		fUpperLimit(other.GetUpperLimit())
	{}

	inline JohnsonSUShapeAnalyticalIntegral&
	operator=( JohnsonSUShapeAnalyticalIntegral const& other)
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

		double r = integral(functor[0], functor[1], functor[2], functor[3] );

		return std::make_pair(
		CHECK_VALUE(r," par[0] = %f par[1] = %f par[2] = %f par[3] = %f fLowerLimit = %f fUpperLimit = %f",\
				functor[0], functor[1],functor[2], functor[3], fLowerLimit,fUpperLimit ) ,0.0);
	}


private:

	inline double integral( const double mean,  const double width,  const double nu,  const double tau) const
	{
		// calculate a few variables
		double w       = exp( tau * tau);
		double omega   = - nu * tau;
		double c       = 0.5 * (w-1) * (w * cosh(2 * omega) + 1);
		c              = pow(c, -0.5);
		double zmax    = (- fUpperLimit + (mean + c * width * sqrt(w) * sinh(omega) )) / c / width;
		double zmin    = (- fLowerLimit + (mean + c * width * sqrt(w) * sinh(omega) )) / c / width;
		static const double pi = atan2(0.0,-1.0);
		static const double PiBy2 = pi/2.0;
		static const double rootPiBy2 = sqrt(PiBy2);

		// the integral calculation
		double ret = 0;
 
		ret =  -0.25/rootPiBy2* ( erf( (nu*tau + asinh( zmax ) )/(sqrt(2)*tau) )-
				erf( (nu*tau + asinh( zmin ) )/(sqrt(2)*tau) ) );
		return ret;
	}

	double fLowerLimit;
	double fUpperLimit;

};


}  // namespace hydra



#endif /* JOHNSONSUSHAPE_H_ */
