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
 * LogLikelihood.h
 *
 *  Created on: 06/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup fit
 */


#ifndef _LOGLIKELIHOOD1_H_
#define _LOGLIKELIHOOD1_H_


#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/TypeTraits.h>

#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/detail/external/thrust/functional.h>



namespace hydra{


namespace detail{


template<typename FUNCTOR>
struct LogLikelihood1
{
	LogLikelihood1(FUNCTOR const& functor):
		fFunctor(functor),
		fNorm(functor.GetNorm())
	{}

	__hydra_host__ __hydra_device__ inline
	LogLikelihood1( LogLikelihood1<FUNCTOR> const& other):
	fFunctor(other.fFunctor),
	fNorm(other.fNorm)
	{}

	template<typename Type>
	__hydra_host__ __hydra_device__ inline
	GReal_t operator()(Type& x) const
	{
		return ::log(fNorm*fFunctor( x ));
	}


	FUNCTOR  fFunctor;
	const GReal_t fNorm;
};


template<typename FUNCTOR>
struct LogLikelihood2
{
	LogLikelihood2(FUNCTOR const& functor):
		fFunctor(functor),
		fNorm(functor.GetNorm())
	{}

	__hydra_host__ __hydra_device__ inline
	LogLikelihood2( LogLikelihood1<FUNCTOR> const& other):
	  fFunctor(other.fFunctor),
		fNorm(other.fNorm)
	{}

	template<typename Args, typename Weights>
   	__hydra_host__ __hydra_device__ inline
   	GReal_t operator()(Args& x, Weights& w) const
   	{

		double weight = 1.0;
		multiply_tuple(weight, w );
		return weight*::log(fNorm*fFunctor( x ));
	}


    FUNCTOR fFunctor;
    const GReal_t fNorm;
};

}//namespace detail


}//namespace hydra


#endif /* LOGLIKELIHOOD1_H_*/
