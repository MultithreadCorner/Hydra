/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2021 Antonio Augusto Alves Junior
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
#include <hydra/Distribution.h>
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
template<typename ArgType, typename Signature=double(ArgType) >
class Exponential:public BaseFunctor<Exponential<ArgType>, Signature, 1>
{
	using BaseFunctor<Exponential<ArgType>, Signature, 1>::_par;


public:

	Exponential() = delete;

	Exponential(Parameter const& tau):
		BaseFunctor<Exponential<ArgType>, Signature, 1>({tau}) {}

	__hydra_host__ __hydra_device__
	Exponential(Exponential<ArgType> const& other):
		BaseFunctor<Exponential<ArgType>, Signature, 1>(other) {}

	__hydra_host__ __hydra_device__
	inline Exponential<ArgType>&
	operator=( Exponential<ArgType> const& other)
	{
		if(this == &other) return *this;
		BaseFunctor<Exponential<ArgType>, Signature,1>::operator=(other);
		return *this;
	}

	__hydra_host__ __hydra_device__
	inline double Evaluate(ArgType x)  const	{

		return  CHECK_VALUE(::exp(-x*_par[0] ),"par[0]=%f ", _par[0] ) ;
	}



};

template<typename ArgType>
class IntegrationFormula< Exponential<ArgType>, 1>
{

protected:

	inline std::pair<GReal_t, GReal_t>
	EvalFormula( Exponential<ArgType>const& functor, double LowerLimit, double UpperLimit )const
	{
		double tau = functor[0];
		double r   =  -(::exp(-UpperLimit*tau) - ::exp(-LowerLimit*tau))/tau ;
		return std::make_pair( CHECK_VALUE(r, "par[0]=%f ", tau ) , 0.0);

	}

};

template<typename ArgType>
struct RngFormula< Exponential<ArgType> >
{

	typedef ArgType value_type;

	__hydra_host__ __hydra_device__
	inline unsigned NCalls( Exponential<ArgType>const&) const
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
	inline value_type Generate(Engine& rng, Exponential<ArgType>const& functor) const
	{
		double tau  = functor[0];

		return static_cast<value_type>(-tau*log(RngBase::uniform(rng)));
	}

	template<typename Engine, typename T>
	__hydra_host__ __hydra_device__
	inline value_type Generate(Engine& rng,  std::initializer_list<T> pars) const
	{
		double tau  = pars.begin()[0];

		return static_cast<value_type>(-tau*log(RngBase::uniform(rng)));
	}

};

}  // namespace hydra

#endif /* EXPONENTIAL_H_ */
