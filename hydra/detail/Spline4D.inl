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
 * Spiline4D.inl
 *
 *  Created on: 16 de set. de 2023
 *      Author:  Antonio Augusto Alves Junior
 */

#ifndef SPILINE4D_INL_
#define SPILINE4D_INL_



#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/detail/utility/CheckValue.h>


namespace hydra {


template<typename IteratorX, typename IteratorY, typename IteratorW, typename IteratorZ, typename IteratorM,
          typename TypeX, typename TypeY, typename TypeW, typename TypeZ >
__hydra_host__ __hydra_device__
inline typename std::enable_if<
	std::is_convertible<typename hydra::thrust::iterator_traits<IteratorX>::value_type, double >::value &&
	std::is_convertible<typename hydra::thrust::iterator_traits<IteratorY>::value_type, double >::value &&
	std::is_convertible<typename hydra::thrust::iterator_traits<IteratorW>::value_type, double >::value &&
	std::is_convertible<typename hydra::thrust::iterator_traits<IteratorZ>::value_type, double >::value &&
	std::is_convertible<typename hydra::thrust::iterator_traits<IteratorM>::value_type, double >::value &&
	std::is_convertible<TypeX, double >::value &&
	std::is_convertible<TypeY, double >::value &&
	std::is_convertible<TypeW, double >::value &&
	std::is_convertible<TypeZ, double >::value, double>::type
spline4D(IteratorX firstx, IteratorX lastx,
		 IteratorY firsty, IteratorY lasty,
		 IteratorW firstw, IteratorW lastw,
		 IteratorZ firstz, IteratorZ lastz,
		 IteratorM measurements, TypeX x, TypeY y, TypeW w, TypeZ z)
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

	//----------------------

	size_t NW = hydra::thrust::distance(firstw, lastw);

	auto iterw = detail::spline::lower_bound(firstw, firstw + NW, y);
	size_t dist_w = hydra::thrust::distance(firstw, iterw);
	size_t iw = dist_w > 0 ? dist_w - 1: 0;

	const double W[4] = { firstw[iw-1], firstw[iw], firstw[iw+1], firstw[iw+2] };

	//----------------------

	size_t NZ = hydra::thrust::distance(firstz, lastz);

	auto iterz = detail::spline::lower_bound(firstz, firstz + NZ, z);
	size_t dist_z = hydra::thrust::distance(firstz, iterz);
	size_t iz = dist_z > 0 ? dist_z - 1: 0;

	const double Z[4] = { firstz[iz-1], firstz[iz], firstz[iz+1], firstz[iz+2] };

	//----------------------

	double M[4][4][4][4] = {  };


	//get the relevant measurements
	for( unsigned k= iz>0?iz-1:0; k < (iz +3 ); ++k ){ //Z

		for( unsigned l= iw>0?iw-1:0; l < (iw+3); ++l ){ //W

			for( unsigned j= iy>0?iy-1:0; j < (iy+3); ++j ){ //Y

				for( unsigned i=ix>0? ix-1:0 ; i < (ix+3); ++i){ //X

					unsigned m = ( ( k*NW  + l)*NY + j )*NX + i;// k*NZ*NY*Nx + l*NY*NX + j*NX + i => ( ( k*NZ  + l)*NY + j )*NX + i

					M[k-iz + 1][ l-iw + 1 ][ j-iy + 1 ][ i - ix +1  ] = measurements[m ];
					//std::cout << "(i, j, l ) = (" << i <<"," << j <<", " << l << ") : " << M[k-iz + 1][ l-iw + 1 ][ j-iy + 1 ][ i - ix +1  ] << std::endl;

				}
			}
		}
	}

	double partial_spline[4]= {  };;

	for(unsigned k=0; k<4; ++k){
		double* slice = reinterpret_cast<double(&)[64]>(M[k]);

		partial_spline[k] = spline3D( X, X +4, Y, Y+4, W, W+4, slice, x, y, w );
        //std::cout << "l " << l << " " << partial_spline[l] << std::endl;

	}

	return detail::spline::cubic_spline<double>(iz, NZ, Z, partial_spline, z );

}

template<typename IterableX, typename IterableY,typename IterableW,typename IterableZ,typename IterableM, typename TypeX,typename TypeY, typename TypeW, typename TypeZ >
__hydra_host__ __hydra_device__
inline typename std::enable_if<
					   hydra::detail::is_iterable<IterableX>::value &&
                       hydra::detail::is_iterable<IterableY>::value &&
					   hydra::detail::is_iterable<IterableW>::value &&
					   hydra::detail::is_iterable<IterableZ>::value &&
                       hydra::detail::is_iterable<IterableM>::value &&
                       std::is_convertible<typename IterableX::value_type, double >::value &&
                       std::is_convertible<typename IterableY::value_type, double >::value &&
					   std::is_convertible<typename IterableW::value_type, double >::value &&
					   std::is_convertible<typename IterableZ::value_type, double >::value &&
					   std::is_convertible<typename IterableM::value_type, double >::value &&
					   std::is_convertible<TypeX, double >::value &&
					   std::is_convertible<TypeY, double >::value &&
					   std::is_convertible<TypeW, double >::value &&
					   std::is_convertible<TypeZ, double >::value ,
                       double >::type
spline3D(IterableX&& abscissa_x,  IterableY&& abscissa_y, IterableW&& abscissa_w, IterableZ&& abscissa_z, IterableM measurements, TypeX x, TypeX y, TypeW w, TypeZ z ){


	return spline4D(
			std::forward<IterableX>(abscissa_x).begin(), std::forward<IterableX>(abscissa_x).end(),
			std::forward<IterableY>(abscissa_y).begin(), std::forward<IterableY>(abscissa_y).end(),
			std::forward<IterableW>(abscissa_w).begin(), std::forward<IterableW>(abscissa_w).end(),
			std::forward<IterableZ>(abscissa_z).begin(), std::forward<IterableZ>(abscissa_z).end(),
			std::forward<IterableM>(measurements).begin() , x,y,w,z);
}

} // namespace hydra

#endif /* SPILINE4D_INL_ */
