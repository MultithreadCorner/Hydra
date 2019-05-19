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
 * Exponential.h
 *
 *  Created on: Dec 11, 2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef EXPONENTIAL_H_
#define EXPONENTIAL_H_

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
 * \class Exponential
 * https://en.wikipedia.org/wiki/Exponential_function
 *
 */
template<unsigned int ArgIndex=0>
class Exponential:public BaseFunctor<Exponential<ArgIndex>, double, 1>
{
	using BaseFunctor<Exponential<ArgIndex>, double, 1>::_par;

public:

	Exponential() = delete;

	Exponential(Parameter const& tau):
		BaseFunctor<Exponential<ArgIndex>, double, 1>({tau}) {}

	__hydra_host__ __hydra_device__
	Exponential(Exponential<ArgIndex> const& other):
		BaseFunctor<Exponential<ArgIndex>, double, 1>(other) {}

	__hydra_host__ __hydra_device__
	inline Exponential<ArgIndex>&
	operator=( Exponential<ArgIndex> const& other)
	{
		if(this == &other) return *this;
		BaseFunctor<Exponential<ArgIndex>,double,1>::operator=(other);
		return *this;
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline double Evaluate(unsigned int, T* x)  const	{

		return  CHECK_VALUE(::exp(x[ ArgIndex]*_par[0] ),"par[0]=%f ", _par[0] ) ;
	}

	template<typename T>
	__hydra_host__ __hydra_device__ inline
	double Evaluate(T x)  const	{

		return CHECK_VALUE(::exp(get<ArgIndex>(x)*_par[0] ),"par[0]=%f ", _par[0] );
	}

};

template<unsigned int ArgIndex>
class IntegrationFormula< Exponential<ArgIndex>, 1>
{

protected:

	inline std::pair<GReal_t, GReal_t>
	EvalFormula( Exponential<ArgIndex>const& functor, double LowerLimit, double UpperLimit )const
	{
		double tau = functor[0];
		double r   =  (::exp(UpperLimit*tau) - ::exp(LowerLimit*tau))/tau ;
		return std::make_pair( CHECK_VALUE(r, "par[0]=%f ", tau ) , 0.0);

	}

};



}  // namespace hydra

#endif /* EXPONENTIAL_H_ */
