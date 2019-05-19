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
 * GetBinCenter.h
 *
 *  Created on: 02/12/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GETBINCENTER_H_
#define GETBINCENTER_H_


#include <hydra/detail/external/thrust/functional.h>
#include <hydra/Tuple.h>
#include <hydra/detail/utility/Utility_Tuple.h>

namespace hydra {

namespace detail {

template<typename T, size_t N >
struct GetBinCenter;

template<typename T, size_t N>
struct GetBinCenter: public HYDRA_EXTERNAL_NS::thrust::unary_function<size_t, typename tuple_type<N,T>::type>
{
	GetBinCenter()=delete;
	GetBinCenter( size_t (&grid)[N], T (&lowerlimits)[N], T (&upperlimits)[N])
	{
		fNGlobalBins=1;
		for( size_t i=0; i<N; i++){
			fNGlobalBins *=grid[i];
			fGrid[i]=grid[i];
			fLowerLimits[i]=lowerlimits[i];
			fDelta[i]= upperlimits[i] - lowerlimits[i];
			fIncrement[i]=(upperlimits[i] - lowerlimits[i])/grid[i];
		}
	}

	__hydra_host__ __hydra_device__
	GetBinCenter( GetBinCenter<T, N> const& other ):
	fNGlobalBins(other.fNGlobalBins)
	{
		for( size_t i=0; i<N; i++){
			fGrid[i] = other.fGrid[i];
			fDelta[i] = other.fDelta[i];
			fLowerLimits[i] = other.fLowerLimits[i];
			fIncrement[i]=other.fIncrement[i];
		}
		fNGlobalBins =other.fNGlobalBins;
	}

	__hydra_host__ __hydra_device__
	GetBinCenter<T, N>&
	operator=( GetBinCenter<T,N> const& other )
	{
		if(this==&other) return *this;
		for( size_t i=0; i<N; i++){
			fGrid[i]= other.fGrid[i];
			fDelta[i] = other.fDelta[i];
			fLowerLimits[i] = other.fLowerLimits[i];
			fIncrement[i]=other.fIncrement[i];

		}
		fNGlobalBins =other.fNGlobalBins;
		return *this;
	}

	//----------------------------------------
	// multiply static array elements
	//----------------------------------------
	template< size_t I>
	__hydra_host__ __hydra_device__ inline typename std::enable_if< (I==N), void  >::type
	multiply( size_t (&)[N] , size_t&  )
	{ }

	template<size_t I=0>
	__hydra_host__ __hydra_device__ inline typename std::enable_if< (I<N), void  >::type
	multiply( size_t (&obj)[N], size_t& result )
	{
		result = I==0? 1.0: result;
		result *= obj[I];
		multiply<I+1>( obj, result );
	}

	//end of recursion
	template<size_t I>
	__hydra_host__ __hydra_device__ inline typename std::enable_if< (I==N), void  >::type
	get_indexes(size_t,  size_t (&)[N])
	{}

	//begin of the recursion
	template<size_t I=0>
	__hydra_host__ __hydra_device__ inline typename std::enable_if< (I<N), void  >::type
	get_indexes(size_t index,  size_t (&indexes)[N] )
	{
		size_t factor    =  1;
		multiply<I+1>(fGrid, factor );
		indexes[I]  =  index/factor;
		size_t next_index =  index%factor;
		get_indexes< I+1>(next_index, indexes );
	}




	__hydra_host__ __hydra_device__ inline
    typename tuple_type<N,T>::type operator()(size_t global_bin){

		size_t  indexes[N];
		get_indexes(global_bin,indexes);

		T X[N];

		for(size_t i=0; i<N; i++)
			X[i] = fLowerLimits[i] + (0.5 + indexes[i])*fIncrement[i];


		return arrayToTuple<T,N>(X);

	}


	T fLowerLimits[N];
	T fDelta[N];
	T fIncrement[N];
	size_t   fGrid[N];
	size_t   fNGlobalBins;



};

//---------------

template<typename T>
struct GetBinCenter<T,1>: public HYDRA_EXTERNAL_NS::thrust::unary_function<T,T>
{
	GetBinCenter()=delete;

	GetBinCenter( size_t grid, T lowerlimits, T upperlimits):
		fLowerLimits(lowerlimits),
		fDelta( upperlimits - lowerlimits),
		fGrid(grid),
		fNGlobalBins(grid),
		fIncrement((upperlimits - lowerlimits)/grid)
	{ }

	__hydra_host__ __hydra_device__
	GetBinCenter( GetBinCenter<T, 1> const& other ):
	fNGlobalBins(other.fNGlobalBins),
	fGrid(other.fGrid ),
	fDelta(other.fDelta ),
	fLowerLimits(other.fLowerLimits ),
	fIncrement(other.fIncrement)
	{}

	__hydra_host__ __hydra_device__
	GetBinCenter<T,1>&
	operator=( GetBinCenter<T,1> const& other )
	{
		if(this==&other) return *this;

		fGrid  = other.fGrid;
		fDelta = other.fDelta;
		fLowerLimits = other.fLowerLimits;
		fNGlobalBins = other.fNGlobalBins;
		fIncrement = other.fIncrement;
		return *this;
	}



	__hydra_host__ __hydra_device__ inline
  T	operator()(size_t global_bin){

		return fLowerLimits + (global_bin +0.5)*fIncrement;
	}

	T fIncrement;
	T fLowerLimits;
	T fDelta;
	size_t   fGrid;
	size_t   fNGlobalBins;



};

}//namespace detail

}  // namespace hydra

#endif /* GETBINCENTER_H_ */
