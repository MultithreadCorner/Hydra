/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2017 Antonio Augusto Alves Junior
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
 * Utils.h
 *
 *  Created on: 12/12/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <hydra/Types.h>
#include <utility>


namespace hydra {

enum Wave{ SWave=0, PWave, DWave, FWave, GWave, HWave };

namespace detail {

	template<typename T, unsigned int N, unsigned int I>
	inline __host__ __device__
	typename std::enable_if<I==N, void >::type
	pow_helper(T const, T&){}


	template<typename T, unsigned int N, unsigned int I>
	inline __host__ __device__
	typename std::enable_if< (I< N), void >::type
	pow_helper(T const x, T& r){
		r *= x ;
		pow_helper<T,N,I+1>(x,r);
	}

}  // namespace detail

/**
 * Power with integer exponent.
 * @param x
 * @return
 */
	template<typename T, unsigned int N>
	inline __host__ __device__
	T pow(const T x){
		T r = 1;
		detail::pow_helper<T,N,0>(x,r);
		return r ;
	}
/**
 * Round to nearest integer. Rounds half integers to the nearest even integer.
 * @param x
 * @return
 */
	template<typename T>
	inline  __host__ __device__
	int nint(const T x)
	{
		// Round to nearest integer. Rounds half integers to the nearest
		// even integer.
		int i = (x > 0) ? int(x + 0.5) - ( int(x + 0.5) & 1 && (x + 0.5 == T( int(x + 0.5))))
				: ( int(x - 0.5) ) +  ( int(x - 0.5) & 1 && (x - 0.5 == T(int(x - 0.5) )) ) ;

		return i;
	}

/**
 * Momentum in mother frame of daughter particle in two-body-decay.
 *
 * @param mother_mass
 * @param daughter1_mass
 * @param daughter2_mass
 * @return
 */
	__host__  __device__
	inline double pmf( const double mother_mass, const double daughter1_mass, const double daughter2_mass) {

		double mother_mass_sq  = mother_mass*mother_mass;

		return  ::sqrt( ( mother_mass_sq - ( daughter1_mass + daughter2_mass)*( daughter1_mass + daughter2_mass))
				*( mother_mass_sq - ( daughter1_mass - daughter2_mass)*( daughter1_mass - daughter2_mass)) )/(2.0*mother_mass);
	}


	//
	__host__ __device__ inline
	double triangle_function( const double x, const double y, const double z)
	{

		return (x - ::pow( ::sqrt(y) + ::sqrt(z), 2))*(x - ::pow( ::sqrt(y) - ::sqrt(z), 2));
	}

	__host__ __device__ inline
	double cosine_helicity_angle_theta_ab( const double mab_sq, const double mbc_sq, const double mac_sq, const double ma_sq, const double mb_sq, const double mc_sq)
	{

		double s = mab_sq + mbc_sq + mac_sq - ma_sq - mb_sq - mc_sq;
		double numerator   = (s - mab_sq - mc_sq)*(mab_sq + ma_sq - mb_sq) + 2*mab_sq*(mc_sq+ma_sq - mbc_sq);
		double denominator = sqrt(lambda(s, mab_sq, mc_sq)*lambda(mab_sq, ma_sq, mb_sq));

		return numerator/denominator;

	}



}  // namespace hydra



#endif /* UTILS_H_ */
