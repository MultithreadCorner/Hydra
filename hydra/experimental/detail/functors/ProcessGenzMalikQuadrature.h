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
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/utility/Generic.h>

namespace hydra {

namespace experimental {

namespace detail {

template <size_t N>
struct GenzMalikBoxResult
{
	GenzMalikBoxResult():
		fRule7(0),
		fRule5(0)
	{
#pragma unroll N
		for(size_t i=0; i<N; i++)
			fFourDifference[i]=0.0;
	}

	GReal_t fRule7;
	GReal_t fRule5;
	GReal_t fFourDifference[N];

};

template< typename FUNCTOR, size_t N>
struct ProcessGenzMalikUnaryCall
{
	ProcessGenzMalikUnaryCall()=delete;

	ProcessGenzMalikUnaryCall(FUNCTOR const& functor):
			fFunctor(functor)
		{ }

	__host__ __device__
	ProcessGenzMalikUnaryCall(ProcessGenzMalikUnaryCall<FUNCTOR, N> const& other ):
	fFunctor(other.fFunctor)
	{}

	__host__ __device__ inline
	ProcessGenzMalikUnaryCall<FUNCTOR, N>&
	operator=(ProcessGenzMalikUnaryCall<FUNCTOR, N> const& other )
	{
		if( this== &other) return *this;

		fFunctor=other.fFunctor;
		return *this;
	}

	void set_four_difference_central(GReal_t value, GReal_t (&fdarray)[N])
	{

#pragma unroll N
		for(size_t i=0; i<N; i++)
			fdarray[i]=value;

	}

	void set_four_difference_unilateral(GChar_t index, GReal_t value, GReal_t (&fdarray)[N])
	{

#pragma unroll N
		for(size_t i=0; i<N; i++)
		fdarray[i]= (index==i)?value:0.0;

	}

	void set_four_difference_multilateral(GReal_t (&fdarray)[N])
		{

#pragma unroll N
			for(size_t i=0; i<N; i++)
			fdarray[i]= 0.0;

		}

	template<typename T>
	__host__ __device__
	inline GenzMalikBoxResult<N> operator()(T box)
	{
		GenzMalikBoxResult<N> box_result;

		GReal_t w5          = thrust::get<0>(box);
		GReal_t w7          = thrust::get<1>(box);
		GChar_t w_four_diff = thrust::get<2>(box);
		GChar_t index       = thrust::get<3>(box);

		auto args = hydra::detail::split_tuple<4>(box);

		GReal_t fval          = fFunctor(args.second);
		box_result.fRule7     = fval*w7;
		box_result.fRule5     = fval*w5;
		GReal_t fourdiff      = fval*w_four_diff;

		(index==N) ? set_four_difference_central(fourdiff, box_result.fFourDifference  ):0;
		(index>=0)&(index<N) ? set_four_difference_unilateral(index,fourdiff, box_result.fFourDifference  ):0;
		(index<0) ? set_four_difference_multilateral(index,fourdiff, box_result.fFourDifference  ):0;

		return box_result;
	}
	FUNCTOR fFunctor;
};

template< size_t N>
struct ProcessGenzMalikBinaryCall
{
	ProcessGenzMalikBinaryCall();

	__host__ __device__
	inline GenzMalikBoxResult<N> operator()(GenzMalikBoxResult<N> box1, GenzMalikBoxResult<N> box2)
	{
		GenzMalikBoxResult<N> box_result;

		box_result.fRule5       = box1.fRule5 + box2.fRule5;
		box_result.fRule7       = box1.fRule7 + box2.fRule7;

#pragma unroll N
			for(size_t i=0; i<N; i++)
				box_result.fFourDifference[i]= box1.fFourDifference[i] + box2.fFourDifference[i];

		return box_result;
	}

};




template < typename FUNCTOR, size_t N, unsigned int BACKEND=hydra::host>
struct ProcessGenzMalikBox
{

	ProcessGenzMalikBox()=delete;

	ProcessGenzMalikBox(FUNCTOR const& functor):
		fFunctor(functor)
	{}

	__host__ __device__
	ProcessGenzMalikBox(ProcessGenzMalikBox<FUNCTOR, N, BACKEND> const& other ):
	fFunctor(other.fFunctor)
	{}

	__host__ __device__ inline
	ProcessGenzMalikBox<FUNCTOR, N, BACKEND>&
	operator=(ProcessGenzMalikBox<FUNCTOR, N, BACKEND> const& other )
	{
		if( this== &other) return *this;

		fFunctor=other.fFunctor;
		return *this;
	}

	template<typename T>
	__host__ __device__
	inline GenzMalikBoxResult<N> operator()(T box)
	{
		typedef detail::BackendTraits<BACKEND> system_t;
		auto abscissa_begin = box.GetAbscissas().begin();
		auto abscissa_end   = box.GetAbscissas().end();

		GenzMalikBoxResult<N> box_result =
				thrust::transform_reduce(system_t(), abscissa_begin, abscissa_end,
				ProcessGenzMalikUnaryCall<N>(fFunctor),
				GenzMalikBoxResult<N>() ,
				ProcessGenzMalikBinaryCall<N>());

		return box_result;


	}

	FUNCTOR fFunctor;
};

}  // namespace detail

}  // namespace experimental

} // namespace hydra

#endif /* PROCESSGENZMALIKQUADRATURE_H_ */
