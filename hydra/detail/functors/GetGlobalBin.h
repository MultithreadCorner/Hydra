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
 * GetGlobalBin.h
 *
 *  Created on: 23/09/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GETGLOBALBIN_H_
#define GETGLOBALBIN_H_

#include <hydra/detail/external/thrust/functional.h>
#include <hydra/Tuple.h>
#include <hydra/detail/utility/Utility_Tuple.h>

namespace hydra {

namespace detail {


template<size_t N, typename T>
struct GetGlobalBin: public HYDRA_EXTERNAL_NS::thrust::unary_function<typename tuple_type<N,T>::type ,size_t>
{
	typedef typename tuple_type<N,T>::type ArgType;

	GetGlobalBin( size_t (&grid)[N], T (&lowerlimits)[N], T (&upperlimits)[N])
	{
		fNGlobalBins=1;
		for( size_t i=0; i<N; i++){
			fNGlobalBins *=grid[i];
			fGrid[i]=grid[i];
			fLowerLimits[i]=lowerlimits[i];
			fDelta[i]= upperlimits[i] - lowerlimits[i];
		}
	}

	__hydra_host__ __hydra_device__
	GetGlobalBin( GetGlobalBin<N, T> const& other ):
	fNGlobalBins(other.fNGlobalBins)
	{
		for( size_t i=0; i<N; i++){
			fGrid[i] = other.fGrid[i];
			fDelta[i] = other.fDelta[i];
			fLowerLimits[i] = other.fLowerLimits[i];
		}
		fNGlobalBins =other.fNGlobalBins;
	}

	__hydra_host__ __hydra_device__
	GetGlobalBin<N, T>&
	operator=( GetGlobalBin<N, T> const& other )
	{
		if(this==&other) return *this;
		for( size_t i=0; i<N; i++){
			fGrid[i]= other.fGrid[i];
			fDelta[i] = other.fDelta[i];
			fLowerLimits[i] = other.fLowerLimits[i];

		}
		fNGlobalBins =other.fNGlobalBins;
		return *this;
	}

	//k = i_1*(dim_2*...*dim_n) + i_2*(dim_3*...*dim_n) + ... + i_{n-1}*dim_n + i_n

	template<size_t I>
	__hydra_host__ __hydra_device__
	typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if< I== N, void>::type
	get_global_bin(const size_t (&)[N], size_t& ){ }

	template<size_t I=0>
	__hydra_host__ __hydra_device__
	typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if< (I< N), void>::type
	get_global_bin(const size_t (&indexes)[N], size_t& index)
	{
	    size_t prod =1;
	    for(size_t i=N-1; i>I; i--)
	           prod *=fGrid[i];
	    index += prod*indexes[I];

	    get_global_bin<I+1>( indexes, index);
	}



	__hydra_host__ __hydra_device__
	size_t get_bin( T (&X)[N]){

		size_t indexes[N];
		size_t bin=0;
		for(size_t i=0; i<N; i++)
			indexes[i]=size_t(X[i]);

		get_global_bin(indexes,  bin);


		return bin;
	}

	__hydra_host__ __hydra_device__
	size_t operator()(ArgType value){

		T X[N];

		tupleToArray(value, X );


		bool is_underflow = true;
		bool is_overflow  = true;

		for(size_t i=0; i<N; i++){
			X[i]  = (X[i]-fLowerLimits[i])*fGrid[i]/fDelta[i];
			is_underflow = is_underflow && (X[i]<0.0);
			is_overflow  = is_overflow && (X[i]>fGrid[i]);
		}

		return is_underflow ? fNGlobalBins : (is_overflow ? fNGlobalBins+1 : get_bin(X) );

	}


	T fLowerLimits[N];
	T fDelta[N];
	size_t   fGrid[N];
	size_t   fNGlobalBins;



};

//---------------

template<typename T>
struct GetGlobalBin<1,T>: public HYDRA_EXTERNAL_NS::thrust::unary_function<T,size_t>
{

	GetGlobalBin( size_t grid, T lowerlimits, T upperlimits):
		fLowerLimits(lowerlimits),
		fDelta( upperlimits - lowerlimits),
		fGrid(grid),
		fNGlobalBins(grid)
	{ }

	__hydra_host__ __hydra_device__
	GetGlobalBin( GetGlobalBin<1, T> const& other ):
	fNGlobalBins(other.fNGlobalBins),
	fGrid(other.fGrid ),
	fDelta(other.fDelta ),
	fLowerLimits(other.fLowerLimits )
	{}

	__hydra_host__ __hydra_device__
	GetGlobalBin<1, T>&
	operator=( GetGlobalBin<1, T> const& other )
	{
		if(this==&other) return *this;

		fGrid  = other.fGrid;
		fDelta = other.fDelta;
		fLowerLimits = other.fLowerLimits;
		fNGlobalBins = other.fNGlobalBins;

		return *this;
	}

	__hydra_host__ __hydra_device__
	size_t get_bin(T X){

		return size_t(X) ;
	}

	__hydra_host__ __hydra_device__
 size_t	operator()(T& value){

		T X = value;

		bool is_underflow = true;
		bool is_overflow  = true;

		X  = (X-fLowerLimits)*fGrid/fDelta;
		is_underflow =(X<0.0);
		is_overflow  =(X>fGrid);


		return is_underflow ? fNGlobalBins  : (is_overflow ? fNGlobalBins+1 : get_bin(X) );

	}


	T fLowerLimits;
	T fDelta;
	size_t   fGrid;
	size_t   fNGlobalBins;



};

}//namespace detail

}//namespace hydra

#endif /* GETGLOBALBIN_H_ */
