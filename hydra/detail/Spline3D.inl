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
 * Spiline3D.inl
 *
 *  Created on: 16 de set. de 2023
 *      Author:  Antonio Augusto Alves Junior
 */

#ifndef SPILINE3D_INL_
#define SPILINE3D_INL_



#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/detail/utility/CheckValue.h>


namespace hydra {


template<typename IteratorX, typename IteratorY, typename IteratorZ, typename IteratorM, typename TypeX, typename TypeY, typename TypeZ >
__hydra_host__ __hydra_device__
inline typename std::enable_if<
	std::is_convertible<typename hydra::thrust::iterator_traits<IteratorX>::value_type, double >::value &&
	std::is_convertible<typename hydra::thrust::iterator_traits<IteratorY>::value_type, double >::value &&
	std::is_convertible<typename hydra::thrust::iterator_traits<IteratorZ>::value_type, double >::value &&
	std::is_convertible<typename hydra::thrust::iterator_traits<IteratorM>::value_type, double >::value &&
	std::is_convertible<TypeX, double >::value &&
	std::is_convertible<TypeY, double >::value &&
	std::is_convertible<TypeZ, double >::value, double>::type
spline3D(IteratorX firstx, IteratorX lastx,
		 IteratorY firsty, IteratorY lasty,
		 IteratorZ firstz, IteratorZ lastz,
		 IteratorM measurements, TypeX x, TypeY y, TypeZ z)
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

	size_t NZ = hydra::thrust::distance(firstz, lastz);

	auto iterz = detail::spline::lower_bound(firstz, firstz + NZ, z);
	size_t dist_z = hydra::thrust::distance(firstz, iterz);
	size_t iz = dist_z > 0 ? dist_z - 1: 0;

	const double Z[4] = { firstz[iz-1], firstz[iz], firstz[iz+1], firstz[iz+2] };

	double M[4][4][4] = {  };


	//get the relevant measurements
	for( unsigned l= iz>0?iz-1:0; l < iz+3; ++l ){

		for( unsigned j= iy>0?iy-1:0; j < iy+3; ++j ){

			for( unsigned i=ix>0? ix-1:0 ; i < ix+3; ++i){

				unsigned m = (l*NY + j)*NX + i;

				M[ l-iz + 1 ][ j-iy + 1 ][ i - ix +1  ] = measurements[m ];
				std::cout << "(i, j, l ) = (" << i <<"," << j <<", " << l << ") ; " << M[ l-iz + 1 ][ j-iy + 1 ][ i - ix +1  ] << std::endl;

			}
		}
	}

	double partial_spline[4]= {  };;

	for(unsigned l=0; l<4; ++l){
		double* slice = reinterpret_cast<double(&)[16]>(M[l]);

		partial_spline[l] = spline2D( X, X +4, Y, Y+4,  slice, x, y );
std::cout << "l " << l << " " << partial_spline[l] << std::endl;

	}

	return detail::spline::cubic_spline<double>(iz,NZ, Z, partial_spline, z );

}

template<typename IterableX, typename IterableY,typename IterableZ,typename IterableM, typename TypeX,typename TypeY, typename TypeZ >
__hydra_host__ __hydra_device__
inline typename std::enable_if<
					   hydra::detail::is_iterable<IterableX>::value &&
                       hydra::detail::is_iterable<IterableY>::value &&
					   hydra::detail::is_iterable<IterableZ>::value &&
                       hydra::detail::is_iterable<IterableM>::value &&
                       std::is_convertible<typename IterableX::value_type, double >::value &&
                       std::is_convertible<typename IterableY::value_type, double >::value &&
					   std::is_convertible<typename IterableZ::value_type, double >::value &&
					   std::is_convertible<typename IterableM::value_type, double >::value &&
					   std::is_convertible<TypeX, double >::value &&
					   std::is_convertible<TypeY, double >::value &&
					   std::is_convertible<TypeZ, double >::value ,
                       double >::type
spline3D(IterableX&& abscissa_x,  IterableY&& abscissa_y, IterableZ&& abscissa_z, IterableM measurements, TypeX x, TypeX y, TypeZ z ){


	return spline2D(
			std::forward<IterableX>(abscissa_x).begin(),
			std::forward<IterableX>(abscissa_x).end(),
			std::forward<IterableY>(abscissa_y).begin(),
			std::forward<IterableY>(abscissa_y).end(),
			std::forward<IterableZ>(abscissa_z).begin(),
			std::forward<IterableZ>(abscissa_z).end(),
			std::forward<IterableM>(measurements).begin() , x,y,z);
}

} // namespace hydra

#endif /* SPILINE3D_INL_ */
