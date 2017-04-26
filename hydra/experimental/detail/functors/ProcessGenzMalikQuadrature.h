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
#include <hydra/experimental/detail/GenzMalikBox.h>
#include <hydra/experimental/GenzMalikQuadrature.h>




namespace hydra {

namespace experimental {

namespace detail {


template<  size_t N, typename FUNCTOR, typename RuleIterator>
struct ProcessGenzMalikUnaryCall
{

	typedef typename RuleIterator::value_type rule_abscissa_t;
	typedef typename hydra::detail::tuple_type<N,GReal_t >::type abscissa_t;

	ProcessGenzMalikUnaryCall()=delete;

	ProcessGenzMalikUnaryCall(GReal_t *lowerLimit, GReal_t *upperLimit, FUNCTOR const& functor):
			fFunctor(functor)

	{
#pragma unroll N
		for(size_t i=0; i<N; i++)
		{
			fA[i] = (upperLimit[i] - lowerLimit[i])/2.0;
		    fB[i] = (upperLimit[i] + lowerLimit[i])/2.0;

		}
	}

	__host__ __device__
	ProcessGenzMalikUnaryCall(ProcessGenzMalikUnaryCall< N, FUNCTOR, RuleIterator> const& other ):
	fFunctor(other.GetFunctor())
	{
#pragma unroll N
		for(size_t i=0; i<N; i++)
		{
			this->fA[i]=other.fA[i];
			this->fB[i]=other.fB[i];
		}
	}

	__host__ __device__
	ProcessGenzMalikUnaryCall< N, FUNCTOR, RuleIterator>&
	operator=(ProcessGenzMalikUnaryCall< N, FUNCTOR, RuleIterator> const& other )
	{
		if( this== &other) return *this;

		fFunctor=other.GetFunctor();
#pragma unroll N
		for(size_t i=0; i<N; i++)
		{
			this->fA[i]=other.fA[i];
			this->fB[i]=other.fB[i];
		}

		return *this;
	}

	template<typename T>
	__host__ __device__
	inline GenzMalikBoxResult<N> operator()(T rule_abscissa)
	{
		GenzMalikBoxResult<N> box_result;
/*
		GReal_t w5          = thrust::get<0>(rule_abscissa);
		GReal_t w7          = thrust::get<1>(rule_abscissa);
		GChar_t w_four_diff = thrust::get<3>(rule_abscissa);
		GChar_t index       = thrust::get<4>(rule_abscissa);

		abscissa_t args;
		get_transformed_abscissa( rule_abscissa, args  );


		GReal_t fval          = fFunctor(args);
		box_result.fRule7     = fval*w7;
		box_result.fRule5     = fval*w5;	*/
		//GReal_t fourdiff      = fval*w_four_diff;

	//	(index==N) ? set_four_difference_central(fourdiff, box_result.fFourDifference  ):0;
		//(index>=0)&(index<N) ? set_four_difference_unilateral(index,fourdiff, box_result.fFourDifference  ):0;
		//(index<0) ? set_four_difference_multilateral( box_result.fFourDifference  ):0;

		return box_result;
	}


	__host__ __device__
	FUNCTOR GetFunctor() const {
		return fFunctor;
	}

	__host__ __device__
	void SetFunctor(FUNCTOR functor) {
		fFunctor = functor;
	}

private:

	template<size_t I>
	typename std::enable_if< (I==N), void  >::type
	__host__ __device__
	get_transformed_abscissa( rule_abscissa_t const& original_abscissa, abscissa_t& transformed_abscissa )
	{	}

	template<size_t I=0>
	typename std::enable_if< (I<N), void  >::type
	__host__ __device__
	get_transformed_abscissa( rule_abscissa_t const& original_abscissa,
			abscissa_t& transformed_abscissa  )
	{

		thrust::get<I>(transformed_abscissa)  =
				fA[I]*thrust::get<2>(original_abscissa )*thrust::get<I+5>(original_abscissa )+ fB[I];

		get_transformed_abscissa<I+1>(original_abscissa, transformed_abscissa );
	}

	__host__ __device__
	GBool_t set_four_difference_central(GReal_t value, GReal_t (&fdarray)[N])
	{

#pragma unroll N
		for(size_t i=0; i<N; i++)
			fdarray[i]=value;
return 1;
	}

	__host__ __device__
	GBool_t set_four_difference_unilateral(GChar_t index, GReal_t value, GReal_t (&fdarray)[N])
	{

#pragma unroll N
		for(size_t i=0; i<N; i++)
		fdarray[i]= (index==i)?value:0.0;

		return 1;
	}

	__host__ __device__
	GBool_t set_four_difference_multilateral(GReal_t (&fdarray)[N])
		{

#pragma unroll N
			for(size_t i=0; i<N; i++)
			fdarray[i]= 0.0;
			return 1;
		}

	FUNCTOR fFunctor;
	//GenzMalikBox<N> fBox;
	GReal_t fA[N];
	GReal_t fB[N];

};

//-----------------------------------------------------
//
//
//-----------------------------------------------------

template< size_t N>
struct ProcessGenzMalikBinaryCall
{

	__host__ __device__
	inline GenzMalikBoxResult<N> operator()(GenzMalikBoxResult<N>const& box1, GenzMalikBoxResult<N>const& box2)
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




template <size_t N, typename FUNCTOR, typename RuleIterator>
struct ProcessGenzMalikBox
{

	ProcessGenzMalikBox(){};

	ProcessGenzMalikBox(FUNCTOR const& functor,
			RuleIterator begin, RuleIterator end):
		fFunctor(functor),
		fBegin(begin),
		fEnd(end)
	{}

	__host__ __device__
	ProcessGenzMalikBox(ProcessGenzMalikBox< N, FUNCTOR, RuleIterator> const& other ):
	fFunctor(other.fFunctor),
	fBegin(other.fBegin),
	fEnd(other.fEnd)
	{}

	__host__ __device__ inline
	ProcessGenzMalikBox< N, FUNCTOR, RuleIterator>&
	operator=(ProcessGenzMalikBox< N, FUNCTOR, RuleIterator> const& other )
	{
		if( this== &other) return *this;

		fFunctor=other.fFunctor;
		fBegin=other.fBegin;
		fEnd=other.fEnd;
		return *this;
	}

	template<typename T>
	__host__
	inline void operator()(T& box)
	{


		GenzMalikBoxResult<N> box_result =
				thrust::transform_reduce(thrust::device, fBegin,fEnd,
				ProcessGenzMalikUnaryCall<N, FUNCTOR, RuleIterator>(box.GetLowerLimit(), box.GetUpperLimit(), fFunctor),
				GenzMalikBoxResult<N>() ,
				ProcessGenzMalikBinaryCall<N>());


		box=box_result;

	}

	FUNCTOR fFunctor;
	RuleIterator fBegin;
	RuleIterator fEnd;
};



}  // namespace detail

}  // namespace experimental

} // namespace hydra

#endif /* PROCESSGENZMALIKQUADRATURE_H_ */
