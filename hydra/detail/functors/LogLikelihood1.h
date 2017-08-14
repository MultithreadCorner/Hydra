/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 Antonio Augusto Alves Junior
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

#include <thrust/tuple.h>
#include <thrust/functional.h>



namespace hydra{


namespace detail{


template<typename FUNCTOR>
struct LogLikelihood1
{
	LogLikelihood1(FUNCTOR const& functor):
		fFunctor(functor)
	{}

	__host__ __device__ inline
	LogLikelihood1( LogLikelihood1<FUNCTOR> const& other):
	  fFunctor(other.fFunctor)
	{}

	template<typename Type>
   	__host__ __device__ inline
   	GReal_t operator()(Type& x)
   	{
		return log(fFunctor( x ));
	}

private:

    FUNCTOR fFunctor;

};

}//namespace detail


}//namespace hydra


#endif /* LOGLIKELIHOOD1_H_*/
