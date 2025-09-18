/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2025 Antonio Augusto Alves Junior
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
 * @fn  spline2D(IteratorX, IteratorX, IteratorX, IteratorM, Type, Type)
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
template<typename IteratorX, typename IteratorY, typename IteratorM, typename TypeX, typename TypeY >
__hydra_host__ __hydra_device__
inline typename std::enable_if<
	std::is_convertible<typename hydra::thrust::iterator_traits<IteratorX>::value_type, double >::value &&
	std::is_convertible<typename hydra::thrust::iterator_traits<IteratorY>::value_type, double >::value &&
	std::is_convertible<typename hydra::thrust::iterator_traits<IteratorM>::value_type, double >::value &&
	std::is_convertible<TypeX, double >::value &&
	std::is_convertible<TypeY, double >::value, double>::type
spline2D(IteratorX firstx, IteratorX lastx, IteratorY firsty, IteratorY lasty, IteratorM measurements, TypeX x, TypeY y)
{
	//get the neighbors on x and y-direction first
	using hydra::thrust::min;

	size_t NX = hydra::thrust::distance(firstx, lastx);

	auto iterx = detail::spline::lower_bound(firstx, lastx, x);
	size_t dist_x = hydra::thrust::distance(firstx, iterx);
	size_t ix = dist_x > 0 ? dist_x - 1: 0;

	const double X[4] = { firstx[ ix-1], firstx[ix], firstx[(ix+1)], firstx[ix+2] };


	//----------------------

	size_t NY = hydra::thrust::distance(firsty, lasty);

	auto itery = detail::spline::lower_bound(firsty, firsty + NY, y);
	size_t dist_y = hydra::thrust::distance(firsty, itery);
	size_t iy = dist_y > 0 ? dist_y - 1: 0;

	const double Y[4] = { firsty[iy-1], firsty[iy], firsty[iy+1], firsty[iy+2] };

	double M[4][4] = {  };


	//get the relevant measurements
	for( unsigned j= iy>0?iy-1:0; j < iy+3; ++j ){

		for( unsigned i=ix>0? ix-1:0 ; i < ix+3; ++i){

			unsigned l = j*NX + i;

			M[ j-iy + 1 ][ i - ix +1  ] = measurements[l < NX*NY ? l : l-1];

		}
	}

	double partial_spline[4]= {  };;

	for(unsigned j=0; j<4; ++j){
		partial_spline[j] = detail::spline::cubic_spline<double>(ix, NX, X, M[j], x );

	}

	return detail::spline::cubic_spline<double>(iy,NY, Y, partial_spline, y );

}

template<typename IterableX, typename IterableY,typename IterableM, typename TypeX,typename TypeY >
__hydra_host__ __hydra_device__
inline typename std::enable_if<
					   hydra::detail::is_iterable<IterableX>::value &&
                       hydra::detail::is_iterable<IterableY>::value &&
                       hydra::detail::is_iterable<IterableM>::value &&
                       std::is_convertible<typename IterableX::value_type, double >::value &&
                       std::is_convertible<typename IterableY::value_type, double >::value &&
					   std::is_convertible<typename IterableM::value_type, double >::value &&
					   std::is_convertible<TypeX, double >::value &&
					   std::is_convertible<TypeY, double >::value ,
                       double >::type
spline2D(IterableX&& abscissa_x,  IterableY&& abscissa_y, IterableM measurements, TypeX x, TypeX y){


	return spline2D(
			std::forward<IterableX>(abscissa_x).begin(),
			std::forward<IterableX>(abscissa_x).end(),
			std::forward<IterableY>(abscissa_y).begin(),
			std::forward<IterableY>(abscissa_y).end(),
			std::forward<IterableM>(measurements).begin() , x,y);
}

} // namespace hydra

#endif /* SPILINE2D_INL_ */
