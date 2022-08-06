/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2022 Antonio Augusto Alves Junior
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
 * DoubleExponential.h
 *
 *  Created on: Feb 19, 2020
 *      Author: Davide Brundu
 */

#ifndef DOUBLEEXPONENTIAL_H_
#define DOUBLEEXPONENTIAL_H_

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
 * \class DoubleExponential, a.k.a. Laplace distribution
 * https://en.wikipedia.org/wiki/Laplace_distribution
 *
 */
template<typename ArgType, typename Signature=double(ArgType)>
class DoubleExponential:public BaseFunctor<DoubleExponential<ArgType>, Signature, 2>
{
	using BaseFunctor<DoubleExponential<ArgType>, Signature, 2>::_par;


public:

	DoubleExponential() = delete;

	DoubleExponential(Parameter const& tau, Parameter const& mean):
		BaseFunctor<DoubleExponential<ArgType>, Signature, 2>({tau,mean}) {}

	__hydra_host__ __hydra_device__
	DoubleExponential(DoubleExponential<ArgType> const& other):
		BaseFunctor<DoubleExponential<ArgType>, Signature, 2>(other) {}

	__hydra_host__ __hydra_device__
	inline DoubleExponential<ArgType>&
	operator=( DoubleExponential<ArgType> const& other)
	{
		if(this == &other) return *this;
		BaseFunctor<DoubleExponential<ArgType>, double(ArgType), 2>::operator=(other);
		return *this;
	}

	__hydra_host__ __hydra_device__
	inline double Evaluate(ArgType x)  const  {

		return  CHECK_VALUE(::exp( ::fabs(x - _par[1]) *_par[0] ),"par[0]=%f, par[1]=%f ", _par[0], _par[1] ) ;
	}



};

template<typename ArgType>
class IntegrationFormula< DoubleExponential<ArgType>, 2>
{

protected:

	inline std::pair<GReal_t, GReal_t>
	EvalFormula( DoubleExponential<ArgType>const& functor, double LowerLimit, double UpperLimit ) const
	{
		double tau  = functor[0];
		double mean = functor[1];

		double cumulative_up    =  (UpperLimit<mean)?
		                           ( 0.5*::exp( (UpperLimit-mean)*tau) : 1 - 0.5*::exp((UpperLimit-mean)*tau)) ;
		double cumulative_low   =  (LowerLimit<mean)?
		                           ( 0.5*::exp( (LowerLimit-mean)*tau) : 1 - 0.5*::exp((LowerLimit-mean)*tau)) ;

        double r = cumulative_up - cumulative_low;

		return std::make_pair( CHECK_VALUE(r, "par[0]=%f, par[1]=%f", tau, mean ) , 0.0);

	}


};





}  // namespace hydra

#endif /* DOUBLEEXPONENTIAL_H_ */
