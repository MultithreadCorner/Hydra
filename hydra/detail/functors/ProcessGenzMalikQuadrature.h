/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2018 Antonio Augusto Alves Junior
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

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Containers.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/utility/Generic.h>
#include <hydra/detail/GenzMalikBox.h>
#include <hydra/GenzMalikQuadrature.h>
#include <hydra/detail/TypeTraits.h>
#include <hydra/detail/utility/Arithmetic_Tuple.h>
#include <hydra/detail/Argument.h>
#include <hydra/detail/external/thrust/functional.h>
#include <hydra/detail/external/thrust/transform_reduce.h>
#include <hydra/detail/external/thrust/reduce.h>
#include <hydra/detail/external/thrust/transform.h>
#include <hydra/detail/external/thrust/functional.h>
#include <hydra/detail/external/thrust/copy.h>
#include <hydra/detail/external/thrust/execution_policy.h>





namespace hydra {


namespace detail {


template<  size_t N, typename FUNCTOR, typename RuleIterator>
struct ProcessGenzMalikUnaryCall
{

	typedef typename RuleIterator::value_type rule_abscissa_t;
	typedef typename hydra::detail::tuple_type<N,GReal_t >::type abscissa_t;
	typedef typename hydra::detail::tuple_type<N+2, GReal_t>::type data_type;

	ProcessGenzMalikUnaryCall()=delete;

	ProcessGenzMalikUnaryCall(GReal_t * __restrict__ lowerLimit, GReal_t * __restrict__ upperLimit, FUNCTOR const& functor):
			fFunctor(functor)

	{
//#pragma unroll N
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
//#pragma unroll N
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
//#pragma unroll N
		for(size_t i=0; i<N; i++)
		{
			this->fA[i]=other.fA[i];
			this->fB[i]=other.fB[i];
		}

		return *this;
	}

	template<typename T>
	__host__ __device__
	inline data_type operator()(T&& rule_abscissa)
	{

		GChar_t index       = HYDRA_EXTERNAL_NS::thrust::get<4>(rule_abscissa);

		abscissa_t args;
		get_transformed_abscissa( rule_abscissa, args  );


		GReal_t _temp[N+2]{0};
		GReal_t fval  = fFunctor(args);
		_temp[0]      = fval*HYDRA_EXTERNAL_NS::thrust::get<1>(rule_abscissa);//w7;
		_temp[1]      = fval*HYDRA_EXTERNAL_NS::thrust::get<0>(rule_abscissa);//w5;

		GReal_t fourdiff      = fval*HYDRA_EXTERNAL_NS::thrust::get<3>(rule_abscissa);//w_four_diff;

		((size_t)index==N) ? set_four_difference_central(fourdiff,  &_temp[2] ):0;
		(index>=0)&((size_t)index<N) ? set_four_difference_unilateral(index,fourdiff,  &_temp[2] ):0;
		(index<0) ? set_four_difference_multilateral( &_temp[2]):0;

       // hydra::detail::arrayToTuple<GReal_t, N+2>(&_temp[0]);

		return hydra::detail::arrayToTuple<GReal_t, N+2>(&_temp[0]);;
	}


	__host__ __device__ inline
	FUNCTOR GetFunctor() const {
		return fFunctor;
	}

	__host__ __device__ inline
	void SetFunctor(FUNCTOR functor) {
		fFunctor = functor;
	}


	template<size_t I>
	typename std::enable_if< (I==N), void  >::type
	__host__ __device__ inline
	get_transformed_abscissa( rule_abscissa_t const& , abscissa_t& ){}

	template<size_t I=0>
	typename std::enable_if< (I<N), void  >::type
	__host__ __device__ inline
	get_transformed_abscissa( rule_abscissa_t const& original_abscissa,
			abscissa_t& transformed_abscissa  )
	{

		HYDRA_EXTERNAL_NS::thrust::get<I>(transformed_abscissa)  =
				fA[I]*HYDRA_EXTERNAL_NS::thrust::get<2>(original_abscissa )*HYDRA_EXTERNAL_NS::thrust::get<I+5>(original_abscissa )+ fB[I];

		get_transformed_abscissa<I+1>(original_abscissa, transformed_abscissa );
	}

	__host__ __device__ inline
	GBool_t set_four_difference_central(GReal_t value,  GReal_t * const __restrict__ fdarray)
	{

//#pragma unroll N
		for(size_t i=0; i<N; i++)
			fdarray[i]=value;
return 1;
	}

	__host__ __device__ inline
	GBool_t set_four_difference_unilateral(GChar_t index, GReal_t value, GReal_t* const __restrict__  fdarray)
	{

//#pragma unroll N
		for(size_t i=0; i<N; i++)
		fdarray[i]= ((size_t)index==i)?value:0.0;

		return 1;
	}

	__host__ __device__ inline
	GBool_t set_four_difference_multilateral( GReal_t * const __restrict__ fdarray)
		{

//#pragma unroll N
			for(size_t i=0; i<N; i++)
			fdarray[i]= 0.0;
			return 1;
		}

	FUNCTOR fFunctor;
	GReal_t fA[N];
	GReal_t fB[N];

};

//-----------------------------------------------------
//
//
//-----------------------------------------------------

template< size_t N>
struct ProcessGenzMalikBinaryCall:
		public HYDRA_EXTERNAL_NS::thrust::binary_function< typename hydra::detail::tuple_type<N+2, GReal_t>::type ,
		                                typename hydra::detail::tuple_type<N+2, GReal_t>::type,
		                                typename hydra::detail::tuple_type<N+2, GReal_t>::type      >
{

	__host__ __device__
	inline typename hydra::detail::tuple_type<N+2, GReal_t>::type
	operator()(typename hydra::detail::tuple_type<N+2, GReal_t>::type box1,
			typename hydra::detail::tuple_type<N+2, GReal_t>::type box2)
	{

		return hydra::detail::addTuples(box1, box2 );

	}

};




template <size_t N, typename FUNCTOR, typename RuleIterator, typename BoxIterator>
struct ProcessGenzMalikBox
{

	ProcessGenzMalikBox(){};

	ProcessGenzMalikBox(FUNCTOR const& functor,
				RuleIterator begin, RuleIterator end, BoxIterator box_begin, BoxIterator box_end):
			fFunctor(functor),
			fBoxBegin(box_begin),
			fBoxEnd(box_end),
			fRuleBegin(begin),
			fRuleEnd(end)
		{}

	__host__ __device__
	ProcessGenzMalikBox(ProcessGenzMalikBox< N, FUNCTOR, RuleIterator,BoxIterator > const& other ):
	fFunctor(other.fFunctor),
	fBoxBegin(other.fBoxBegin),
	fBoxEnd(other.fBoxEnd),
	fRuleBegin(other.fRuleBegin),
	fRuleEnd(other.fRuleEnd)
	{}

	__host__ __device__ inline
	ProcessGenzMalikBox< N, FUNCTOR, RuleIterator,BoxIterator >&
	operator=(ProcessGenzMalikBox< N, FUNCTOR, RuleIterator,BoxIterator > const& other )
	{
		if( this== &other) return *this;

		fFunctor=other.fFunctor;
		fBoxBegin=other.fBoxBegin;
		fBoxEnd=other.fBoxEnd;
		fRuleBegin=other.fRuleBegin;
		fRuleEnd=other.fRuleEnd;

		return *this;
	}

	__host__
	inline void operator()(size_t index)
	{
		typedef typename hydra::detail::tuple_type<N+2, GReal_t>::type tuple_t;
		/*
		typedef hydra::mc_device_vector<tuple_t > device_super_t;
		typedef hydra::mc_host_vector<tuple_t > host_super_t;
		typedef multivector<device_super_t> device_rvector_t;
		typedef multivector<host_super_t> host_rvector_t;
*/
		auto box_result =
				HYDRA_EXTERNAL_NS::thrust::transform_reduce( fRuleBegin, fRuleEnd,
				ProcessGenzMalikUnaryCall<N, FUNCTOR, RuleIterator>(fBoxBegin[index].GetLowerLimit(), fBoxBegin[index].GetUpperLimit(), fFunctor),
				tuple_t() ,
				ProcessGenzMalikBinaryCall<N>());

		fBoxBegin[index]=box_result;

	}

	FUNCTOR fFunctor;
	BoxIterator  fBoxBegin;
	BoxIterator  fBoxEnd;
	RuleIterator fRuleBegin;
	RuleIterator fRuleEnd;


};



}  // namespace detail


} // namespace hydra

#endif /* PROCESSGENZMALIKQUADRATURE_H_ */
