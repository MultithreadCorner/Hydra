/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016-2017 Antonio Augusto Alves Junior
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
 * UniformShape.h
 *
 *  Created on: 15/09/2018
 *      Author: Antonio Augusto Alves Junior
 *
 *  Updated on: Feb 21 2020
 *      Author: Davide Brundu
 *         Log: Update call interface
 */

#ifndef UNIFORMSHAPE_H_
#define UNIFORMSHAPE_H_




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
#include <cassert>
#include <utility>

namespace hydra {
/**
 * \ingroup common_functions
 * \class UniformShape
 * From: https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)
 *
 * In probability theory and statistics, the continuous uniform distribution or rectangular
 * distribution is a family of symmetric probability distributions such that for each member
 * of the family, all intervals of the same length on the distribution's support are equally probable.
 * The support is defined by the two parameters, a and b, which are its minimum and maximum values.
 * The distribution is often abbreviated U(a,b).
 *
 */
template<typename ArgType, typename Signature=double(ArgType)>
class UniformShape:public BaseFunctor<UniformShape<ArgType>, Signature, 2>
{
	using BaseFunctor<UniformShape<ArgType>, Signature, 2>::_par;

public:

	UniformShape() = delete;

	UniformShape(Parameter const& A, Parameter const& B ):
		BaseFunctor<UniformShape<ArgType>, Signature, 2>({A,B}) {}

	__hydra_host__ __hydra_device__
	UniformShape(UniformShape<ArgType> const& other):
		BaseFunctor<UniformShape<ArgType>, Signature, 2>(other) {}

	__hydra_host__ __hydra_device__
	inline UniformShape<ArgType>&
	operator=( UniformShape<ArgType> const& other)
	{
		if(this == &other) return *this;
		BaseFunctor<UniformShape<ArgType>, Signature, 2>::operator=(other);
		return *this;
	}


	__hydra_host__ __hydra_device__
	inline double Evaluate(ArgType x)  const	{

		return  CHECK_VALUE( uniform(x, _par[0], _par[1] ),"par[0]=%f par[1]=%f ", _par[0] , _par[1] ) ;
	}


private:

	__hydra_host__ __hydra_device__
	inline double uniform(const double x, const double a, const double b ) const {

		double slope = 1.0/(b-a) ;

		double filter = (x < b)&&(x>=a)? 1.0: 0.0;

		return slope*filter;
	}
};

template<typename ArgType>
class IntegrationFormula< UniformShape<ArgType>, 1>
{

protected:

	inline std::pair<GReal_t, GReal_t>
	EvalFormula( UniformShape<ArgType>const& functor, double LowerLimit, double UpperLimit )const
	{
		double a = functor[0];
		double b = functor[1];

		double r  =  (cdf(a, b, UpperLimit) - cdf(a, b, LowerLimit)) ;
		return std::make_pair( CHECK_VALUE(r, "par[0]=%f par[1]=%f ", a, b ) , 0.0);

	}

private:


	double cdf( const double a, const double b, const double x ) const {

		if(x <= a) return 0.0;

		else if(x >b ) return 1.0;

		else if((x > a)&&(x<=b))
		{
		  return (x-a)/(b-a);
		}

		return 0.0;
	}
};

template<typename ArgType>
struct RngFormula< UniformShape<ArgType> >
{

	typedef ArgType value_type;
	__hydra_host__ __hydra_device__
	inline unsigned NCalls( UniformShape<ArgType>const&) const
	{
		return 1;
	}

	template< typename T>
	__hydra_host__ __hydra_device__
	inline unsigned NCalls( std::initializer_list<T>) const
	{
		return 1;
	}

	template<typename Engine>
	__hydra_host__ __hydra_device__
	inline value_type Generate(Engine& rng, UniformShape<ArgType>const& functor) const
	{
		double HYDRA_PREVENT_MACRO_SUBSTITUTION min = functor[0];
		double HYDRA_PREVENT_MACRO_SUBSTITUTION max = functor[1];

		double x= RngBase::uniform(rng)*(max - min) + min;

		return static_cast<value_type>(x);
	}

	template<typename Engine, typename T>
	__hydra_host__ __hydra_device__
	inline value_type Generate(Engine& rng, std::initializer_list<T> pars) const
	{
		double HYDRA_PREVENT_MACRO_SUBSTITUTION min = pars.begin()[0];
		double HYDRA_PREVENT_MACRO_SUBSTITUTION max = pars.begin()[1];

		double x=   RngBase::uniform(rng)*(max - min) + min;

		return static_cast<value_type>(x);
	}



};


}  // namespace hydra


#endif /* UNIFORMSHAPE_H_ */
