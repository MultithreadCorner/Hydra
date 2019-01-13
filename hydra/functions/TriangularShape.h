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
 * TriangularShape.h
 *
 *  Created on: 04/09/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef TRIANGULARSHAPE_H_
#define TRIANGULARSHAPE_H_



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
 * \class TriangularShape
 * From: https://en.wikipedia.org/wiki/Triangular_distribution
 *
 * In probability theory and statistics, the triangular distribution is a continuous
 * probability distribution with lower limit a, upper limit b and mode c, where a < b and a ≤ c ≤ b.
 * The triangular distribution is typically used as a subjective description of a population for which there is only limited sample data,
 * and especially in cases where the relationship between variables is known but data is scarce (possibly because of the high cost of collection). It is based on a knowledge of the minimum and maximum and an "inspired guess"[3] as to the modal value.
 * For these reasons, the triangle distribution has been called a "lack of knowledge" distribution.
 *
 */
template<unsigned int ArgIndex=0>
class TriangularShape:public BaseFunctor<TriangularShape<ArgIndex>, double, 3>
{
	using BaseFunctor<TriangularShape<ArgIndex>, double, 3>::_par;

public:

	TriangularShape() = delete;

	TriangularShape(Parameter const& A, Parameter const& B, Parameter const& C ):
		BaseFunctor<TriangularShape<ArgIndex>, double, 3>({A,B,C}) {}

	__hydra_host__ __hydra_device__
	TriangularShape(TriangularShape<ArgIndex> const& other):
		BaseFunctor<TriangularShape<ArgIndex>, double, 3>(other) {}

	__hydra_host__ __hydra_device__
	inline TriangularShape<ArgIndex>&
	operator=( TriangularShape<ArgIndex> const& other)
	{
		if(this == &other) return *this;
		BaseFunctor<TriangularShape<ArgIndex>,double,3>::operator=(other);
		return *this;
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline double Evaluate(unsigned int, T* x)  const	{

		return  CHECK_VALUE( triangle(x[ ArgIndex], _par[0], _par[1],  _par[2] ),"par[0]=%f par[1]=%f par[2]=%f ", _par[0] , _par[1] , _par[2] ) ;
	}

	template<typename T>
	__hydra_host__ __hydra_device__ inline
	double Evaluate(T x)  const	{

		return CHECK_VALUE( triangle(get<ArgIndex>(x), _par[0], _par[1],  _par[2] ),"par[0]=%f par[1]=%f par[2]=%f ", _par[0] , _par[1] , _par[2]  );
	}
private:

	__hydra_host__ __hydra_device__
	inline double triangle(const double x, const double a, const double b, const double c ) const {

		double slope = x <= c ? 2.0*(x-a)/((b-a)*(c-a)) :  2.0*(b-x)/((b-a)*(b-c)) ;

		double filter = (x < b)&&(x>a)? 1.0: 0.0;

		return slope*filter;
	}
};


template<unsigned int ArgIndex>
class IntegrationFormula< TriangularShape<ArgIndex>, 1>
{

protected:

	inline std::pair<GReal_t, GReal_t>
	EvalFormula( TriangularShape<ArgIndex>const& functor, double LowerLimit, double UpperLimit )const
	{
		double a = functor[0];
		double b = functor[1];
		double c = functor[2];

		double r  =  (cdf(a, b, c, UpperLimit) - cdf(a, b, c, LowerLimit)) ;
		return std::make_pair( CHECK_VALUE(r, "par[0]=%f par[1]=%f par[2]=%f ", a, b, c ) , 0.0);

	}

private:

	double cdf( const double a, const double b, const double c, const double x ) const {

		if(x < a) return 0.0;
		else if(x >b ) return 1.0;
		else if((x > a)&&(x<=c)) {

			double delta = ( x - a);

			return delta*delta/((b-a)*(c-a));
		}
		else if((x > c)&&(x<=b)) {

			double delta = (b-x);

			return 1.0 - delta*delta/((b-a)*(b-c));
		}

		return 0.0;
	}


};


}
#endif /* TRIANGULARSHAPE_H_ */
