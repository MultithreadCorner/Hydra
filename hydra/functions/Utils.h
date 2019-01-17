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
 * Utils.h
 *
 *  Created on: 12/12/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef UTILS_H_
#define UTILS_H_


#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <utility>
#include <ratio>

namespace hydra {

/**
 *  \ingroup common_functions
 * Orbital angular momentum tags.
 */
enum Wave{ SWave=0, PWave, DWave, FWave, GWave, HWave };



namespace detail {
	/**
	 * Round to nearest integer at compile time.
	 * Rounds half integers to the nearest even integer.
	 */
	template<int N, int D>
	struct nearest_int {
		private:
			typedef std::ratio_add<std::ratio<N,D>, std::ratio<1,2>> sum;
			typedef std::ratio_subtract<std::ratio<N,D>, std::ratio<1,2>> diff;

		public:

			static constexpr int value = std::ratio_greater<std::ratio<N,D>, std::ratio<0,D>>::value  ?
					int( sum::num/sum::den) - (  int( sum::num/sum::den)  & 1 ) :
					int( diff::num/diff::den) + (  int( diff::num/diff::den)  & 1 ) ;
	};



	template<typename T, unsigned int N, unsigned int I>
	__hydra_host__ __hydra_device__
	inline typename std::enable_if<I==N, void >::type
	pow_helper(T const, T&){}


	template<typename T, unsigned int N, unsigned int I>
	inline __hydra_host__ __hydra_device__
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
	inline __hydra_host__ __hydra_device__
	T pow(T x){
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
	inline  __hydra_host__ __hydra_device__
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
	__hydra_host__  __hydra_device__
	inline double pmf( const double mother_mass, const double daughter1_mass, const double daughter2_mass) {

		double mother_mass_sq  = mother_mass*mother_mass;

		return  ::sqrt( ( mother_mass_sq - ( daughter1_mass + daughter2_mass)*( daughter1_mass + daughter2_mass))
				*( mother_mass_sq - ( daughter1_mass - daughter2_mass)*( daughter1_mass - daughter2_mass)) )/(2.0*mother_mass);
	}



}  // namespace hydra



#endif /* UTILS_H_ */
