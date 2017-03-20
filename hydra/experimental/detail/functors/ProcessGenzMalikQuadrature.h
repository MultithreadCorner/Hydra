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
 * ProcessGenzMalikQuadrature.h
 *
 *  Created on: 17/03/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PROCESSGENZMALIKQUADRATURE_H_
#define PROCESSGENZMALIKQUADRATURE_H_

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

namespace hydra {

namespace experimental {

template <size_t N>
struct GenzMalikBoxResult
{

	GReal_t fRule7;
	GReal_t fRule5;
	GReal_t fFourDifference[N];

};

template < typename FUNCTOR, typename AbscissaIterator, size_t N, unsigned int BACKEND=hydra::host>
struct ProcessGenzMalikBox
{
	typedef typename AbscissaIterator::value_type abscissa_t;

	ProcessGenzMalikBox()=delete;

	ProcessGenzMalikBox(FUNCTOR const& functor,
			AbscissaIterator begin,
			AbscissaIterator end):
		fFunctor(functor)
	{}

	__host__ __device__ inline
	ProcessGenzMalikBox(ProcessGenzMalikBox<FUNCTOR, AbscissaIterator, N, BACKEND> const& other ):
	fFunctor(other.fFunctor)
	{}

	__host__ __device__ inline
	ProcessGenzMalikBox<FUNCTOR, AbscissaIterator, N, BACKEND>&
	operator=(ProcessGenzMalikBox<FUNCTOR, AbscissaIterator, N, BACKEND> const& other )
	{
		if( this== &other) return *this;

		fFunctor=other.fFunctor;
		return *this;
	}

	template<typename T>
	__host__ __device__
	inline GenzMalikBoxResult<N>	operator()(size_t i)
	{

		typedef detail::BackendTraits<BACKEND> system_t;
		GenzMalikBoxResult<N> box_result =
				thrust::transform_reduce(system_t(),
				fBegin, fEnd, ProcessGenzMalikUnaryCall<N>(fFunctor),
				GenzMalikBoxResult<N>() ,
				ProcessGenzMalikBinaryCall<N>());

		return box_result;


	}

	AbscissaIterator fEnd;
	AbscissaIterator fBegin;
	FUNCTOR fFunctor;
};



}  // namespace experimental

} // namespace hydra
#endif /* PROCESSGENZMALIKQUADRATURE_H_ */
