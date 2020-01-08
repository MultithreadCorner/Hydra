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
 * LSB.h
 *
 *  Created on: 31/12/2019
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef LSB_H_
#define LSB_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/utility/Integer.h>
#include<type_traits>
#include<utility>

namespace hydra {

namespace detail {

/* lsb - returns the position of the
 * less significant bit in a 32/64 bit word in 0-base indexing
 * it takes as parameters unsigned integers only.
 *
 * lsb runs on host and device, and she will try to use compile
 * intrinsics for NVCC, GCC, CLANG, if not possible it will always
 * fall back into a generic code.
 *
 * 64 bit implementation
 */
template<typename Integer>
__hydra_host__ __hydra_device__
inline typename std::enable_if<
		std::is_integral<Integer>::value  &&
    	!(std::is_signed<Integer>::value) &&
    	(sizeof(Integer)==8)
    , unsigned>::type
lsb( Integer x){

	if(!x) return 64;

//host path will try to use
//GCC or CLANG intrinsics first
#ifndef __CUDA_ARCH__

	#if defined(__GNUC__) && !defined(__clang__)

		return __builtin_ffsl(x)-1;

	#elif  defined(__clang__) && !defined(__GNUC__)

		return __builtin_ffsl(x)-1;

	#else

		static const char table[256] = {
	     #define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
				64, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
				LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6),
				LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)
		};

		x = (x & -x);
		size_t p;
		unsigned r;

		if( (p = x >> 56) ) {  r = 56 + table[p]; }
		else if( (p = x >> 48) ) {  r = 48 + table[p]; }
		else if( (p = x >> 40) ) {  r = 40 + table[p]; }
		else if( (p = x >> 32) ) {  r = 32 + table[p]; }
		else if( (p = x >> 24) ) {  r = 24 + table[p]; }
		else if( (p = x >> 16) ) {  r = 16 + table[p]; }
		else if( (p = x >> 8)  ) {  r = 8  + table[p]; }
		else {  r = table[x]; }

		return r;

	#endif
//device path will try to use
//NVCC intrinsics, also available CLANG
#else

		return __ffsl(x);

#endif
}

/*
 * 32 bit implementation
 */
template<typename Integer>
__hydra_host__ __hydra_device__
inline typename std::enable_if<
		std::is_integral<Integer>::value  &&
    	!(std::is_signed<Integer>::value) &&
    	(sizeof(Integer)==4)
    , unsigned>::type
lsb( Integer x){

	if(!x) return 32;

//host path will try to use
//GCC and CLANG intrisics
//
#ifndef __CUDA_ARCH__

	#if defined(__GNUC__) && !defined(__clang__)

		return __builtin_ffs(x)-1;

	#elif  defined(__clang__) && !defined(__GNUC__)

		return __builtin_ffs(x)-1;

	#else

		static const char table[256] = {
	#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
				64, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
				LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6),
				LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)
		};

		x = (x & -x);
		size_t p;
		unsigned r;


		     if( (p = x >> 24) ) {  r = 24 + table[p]; }
		else if( (p = x >> 16) ) {  r = 16 + table[p]; }
		else if( (p = x >> 8)  ) {  r = 8  + table[p]; }
		else {  r = table[x]; }

		return r;
	#endif
#else
		return __ffs(x)-1;

#endif
}


}  // namespace detail

}  // namespace hydra


#endif /* LSB_H_ */
