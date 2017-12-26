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

	template<typename T, unsigned int N>
	inline __host__ __device__
	T pow(const T x){
		T r = 1;
		detail::pow_helper<T,N,0>(x,r);
		return r ;
	}

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

}  // namespace hydra



#endif /* UTILS_H_ */
