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
 * ProcessGaussKronrodQuadrature.h
 *
 *  Created on: 02/02/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PROCESSGAUSSKRONRODQUADRATURE_H_
#define PROCESSGAUSSKRONRODQUADRATURE_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/experimental/Collection.h>

namespace hydra {

namespace experimental {

struct GaussKronrodCall
{
	GReal_t fGaussCall;
	GReal_t fGaussKronrodCall;

	//_DeclareStorable(GaussKronrodCall, fGaussCall, fGaussKronrodCall)
};

template <typename FUNCTOR>
struct GaussKronrodUnary
{
	GaussKronrodUnary()=delete;

	GaussKronrodUnary(FUNCTOR fFunctor):
	fFunctor(functor)
	{}

	__host__ __device__ inline
	GaussKronrodUnary(GaussKronrodUnary<FUNCTOR> const& other ):
	fFunctor(other.fFunctor)
	{}
	__host__ __device__ inline
	GaussKronrodUnary& operator=(GaussKronrodUnary<FUNCTOR> const& other )
	{
		if( this== &other) return *this;

		fFunctor=other.fFunctor;
		return *this;
	}

	template<typename ROW>
	__host__ __device__ inline
	operator()
	{

	}





	FUNCTOR fFunctor;

};


struct GaussKronrodBinary
{

};

}  // namespace experimental

}  // namespace hydra



#endif /* PROCESSGAUSSKRONRODQUADRATURE_H_ */
