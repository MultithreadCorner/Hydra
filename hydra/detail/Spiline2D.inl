/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2023 Antonio Augusto Alves Junior
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
 * Spiline2D.inl
 *
 *  Created on: 16 de set. de 2023
 *      Author:  Antonio Augusto Alves Junior
 */

#ifndef SPILINE2D_INL_
#define SPILINE2D_INL_



#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/detail/utility/CheckValue.h>


namespace hydra {

/**
 * @fn  spiline2D(IteratorX, IteratorX, IteratorX, IteratorM, Type, Type)
 * @brief
 *
 * @param firstx
 * @param lastx
 * @param firsty
 * @param measurements
 * @param x
 * @param y
 * @return
 */
template<typename IteratorX, typename IteratorY, typename IteratorM, typename Type>
__hydra_host__ __hydra_device__
inline typename std::enable_if<
	std::is_floating_point<typename hydra::thrust::iterator_traits<IteratorX>::value_type >::value &&
	std::is_floating_point<typename hydra::thrust::iterator_traits<IteratorY>::value_type >::value &&
	std::is_floating_point<typename hydra::thrust::iterator_traits<IteratorM>::value_type >::value
, Type>::type
spiline2D(IteratorX firstx, IteratorX lastx, IteratorY firsty, IteratorM measurements, Type x, Type y)
{
	//get the neighbors on x and y-direction first
	using hydra::thrust::min;

	size_t N = hydra::thrust::distance(firstx, lastx);

	auto iterx = detail::spiline::lower_bound(firstx, lastx, x);
	size_t dist_x = hydra::thrust::distance(firstx, iterx);
	size_t ix = dist_x > 0 ? dist_x - 1: 0;

	const double X[4] = { firstx[ix-1], firstx[ix], firstx[ix+1], firstx[ix+2] };

	auto itery = detail::spiline::lower_bound(firsty, firsty + N, y);
	size_t dist_y = hydra::thrust::distance(firsty, itery);
	size_t iy = dist_y > 0 ? dist_y - 1: 0;

	const double Y[4] = { firsty[iy-1], firsty[iy], firsty[iy+1], firsty[iy+2] };

	double M[4][4] = {  };

	unsigned m=0;
	//get the relevant measurements
	for( auto j= iy-1; j < iy+3; ++j ){
		for( auto i=ix-1 ; i < ix+3; ++i){

			unsigned l = j*4 + i;

			M[ j-iy + 1 ][ i - ix +1  ] = measurements[l];
		}
	}

	double partial_spiline[4];

	for(unsigned j=0; j<4; ++j){
		partial_spiline[j] = detail::spiline::cubic_spiline(ix, N, X, M[j], x );
	}

	return detail::spiline::cubic_spiline(iy,N, Y, partial_spiline, y );



}

} // namespace hydra

#endif /* SPILINE2D_INL_ */
