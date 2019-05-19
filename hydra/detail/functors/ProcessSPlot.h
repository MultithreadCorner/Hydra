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
 * ProcessSPlot.h
 *
 *  Created on: 15/09/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PROCESSSPLOT_H_
#define PROCESSSPLOT_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/utility/Arithmetic_Tuple.h>
#include <hydra/detail/utility/Generic.h>
#include <hydra/detail/FunctorTraits.h>
#include <hydra/detail/external/thrust/functional.h>

namespace hydra {

namespace detail {

template<typename F1, typename F2, typename ...Fs >
struct CovMatrixUnary
{
	typedef hydra::tuple<F1, F2, Fs...> functors_tuple_type;
	constexpr static size_t nfunctors = sizeof...(Fs)+2;
	typedef typename detail::tuple_type<nfunctors*nfunctors,double>::type matrix_t;


	CovMatrixUnary( Parameter(&coeficients)[nfunctors], functors_tuple_type const& functors ):
		fFunctors(functors)
	{
		for(size_t i=0;i<nfunctors; i++)
				fCoeficients[i] = coeficients[i].GetValue();
	}

	__hydra_host__ __hydra_device__
	CovMatrixUnary( CovMatrixUnary<F1, F2, Fs...> const& other ):
		fFunctors( other.fFunctors )
	{
		for(size_t i=0;i<nfunctors; i++)
			fCoeficients[i] = other.fCoeficients[i];
	}

	__hydra_host__ __hydra_device__
	CovMatrixUnary<F1, F2, Fs...> &
	operator=( CovMatrixUnary<F1, F2, Fs...> const& other )
	{
			if(this==&other) return *this;
			fFunctors = other.fFunctors ;
			for(size_t i=0;i<nfunctors; i++)
				fCoeficients[i] = other.fCoeficients[i];
			 return *this;
	}


	template<size_t N, size_t I>
	struct index
	{
	 constexpr static size_t x= I/N;
	 constexpr static size_t y= I%N;
	};


	template<typename ...T, size_t ...I>
	__hydra_host__ __hydra_device__ inline
	matrix_t combiner_helper(GReal_t denominator,HYDRA_EXTERNAL_NS::thrust::tuple<T...>& tpl,
			hydra::detail::index_sequence<I...>)
	{
		constexpr size_t N = sizeof ...(T);
	    return HYDRA_EXTERNAL_NS::thrust::make_tuple(
	    ( HYDRA_EXTERNAL_NS::thrust::get< index<N, I>::x >(tpl) *
	    		HYDRA_EXTERNAL_NS::thrust::get< index<N, I>::y >(tpl) ) /
	    		(denominator*denominator)
	    		... );
	}

	template<typename ...T>
	__hydra_host__ __hydra_device__ inline
	matrix_t combiner(GReal_t denominator, HYDRA_EXTERNAL_NS::thrust::tuple<T...>& tpl)
   {
	    constexpr size_t N = sizeof ...(T);

	    return combiner_helper( denominator, tpl, detail::make_index_sequence<N*N>{});
	}



	template<typename Type>
	__hydra_host__ __hydra_device__ inline
	matrix_t operator()(Type& x)
	{
		auto fvalues  = detail::invoke_normalized(x, fFunctors);
		auto wfvalues = detail::multiply_array_tuple(fCoeficients, fvalues);
		GReal_t denominator   = 0;
		detail::add_tuple_values(denominator, wfvalues);

        matrix_t result = combiner(denominator,  fvalues);

      return result;
	}

	GReal_t    fCoeficients[nfunctors];
	functors_tuple_type fFunctors;
};

template<typename T>
struct CovMatrixBinary
		:public HYDRA_EXTERNAL_NS::thrust::binary_function< T const&, T const&, T >
{
	__hydra_host__ __hydra_device__ inline
	T operator()(T const& x, T const& y )
	{
		return x+y;
	}
};

template<typename F1, typename F2, typename ...Fs >
struct SWeights
{
	constexpr static size_t nfunctors = sizeof...(Fs)+2;
	typedef hydra::tuple<F1, F2, Fs...> functors_tuple_type;
	typedef HYDRA_EXTERNAL_NS::Eigen::Matrix<double, nfunctors, nfunctors> cmatrix_t;
	typedef typename hydra::detail::tuple_type<nfunctors, double>::type tuple_t;


	SWeights( Parameter(&coeficients)[nfunctors], functors_tuple_type const& functors,
			cmatrix_t icmatrix ):
				fFunctors(functors),
				fICovMatrix( icmatrix )
	{

		for(size_t i=0;i<nfunctors; i++)
			fCoeficients[i] = coeficients[i].GetValue();
	}

	__hydra_host__ __hydra_device__ inline
	SWeights(SWeights<F1, F2, Fs...> const& other ):
		fFunctors( other.fFunctors ),
		fICovMatrix( other.fICovMatrix )
	{

		for(size_t i=0;i<nfunctors; i++)
				fCoeficients[i] = other.fCoeficients[i];
	}


	template<typename Type>
	__hydra_host__ __hydra_device__ inline
	tuple_t operator()(Type& x)
	{
		auto fvalues  = detail::invoke_normalized(x, fFunctors);
		double values[nfunctors];
		detail::tupleToArray(fvalues, values);
		HYDRA_EXTERNAL_NS::Eigen::Matrix<double, nfunctors,1> values_vector = HYDRA_EXTERNAL_NS::Eigen::Map<HYDRA_EXTERNAL_NS::Eigen::Matrix<double, nfunctors,1> >(values);
		HYDRA_EXTERNAL_NS::Eigen::Matrix<double, nfunctors,1> sweights = fICovMatrix*values_vector;
		auto wfvalues = detail::multiply_array_tuple(fCoeficients, fvalues);
		GReal_t denominator   = 0;
		detail::add_tuple_values(denominator, wfvalues);
		sweights /=denominator;

		auto r = detail::arrayToTuple<double,nfunctors >(sweights.data());
		return r;
	}

	GReal_t    fCoeficients[nfunctors];
	functors_tuple_type fFunctors;
	cmatrix_t  fICovMatrix;
};

}  // namespace detail

}  // namespace hydra

#endif /* PROCESSSPLOT_H_ */
