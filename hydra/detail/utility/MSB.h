/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2018 Antonio Augusto Alves Junior
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
 * MSB.h
 *
 *  Created on: 05/01/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef MSB_H_
#define MSB_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/utility/Integer.h>
#include <type_traits>
#include<utility>

namespace hydra {

namespace detail {

/* msb - returns the position of the
 * more significant bit in a 32/64 bit word in 0-base indexing
 * it takes as parameters unsigned integers only.
 *
 * msbruns on host and device, and she will try to use compile
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
    	(std::numeric_limits<Integer>::digits==64)
    , unsigned>::type
msb( Integer  const x){

	if(!x) return 64;

//host path will try to use
//GCC or CLANG intrinsics first
#ifndef __CUDA_ARCH__

	#if defined(__GNUC__) && !defined(__clang__)

		return 63 - __builtin_clzl(x);

	#elif  defined(__clang__) && !defined(__GNUC__)

		return 63 - __builtin_clzl(x);

	#else

	   Integer _x
	   unsigned y;
	   int n = 64;

	   y = _x >>32;  if (y != 0) {n = n -32;  _x = y;}
	   y = _x >>16;  if (y != 0) {n = n -16;  _x = y;}
	   y = _x >> 8;  if (y != 0) {n = n - 8;  _x = y;}
	   y = _x >> 4;  if (y != 0) {n = n - 4;  _x = y;}
	   y = _x >> 2;  if (y != 0) {n = n - 2;  _x = y;}
	   y = _x >> 1;  if (y != 0) return n - 2;

	   return 63 - n - _x;

	#endif
//device path will try to use
//NVCC intrinsics, also available CLANG
#else

		return 63 -__clzl(x);

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
    	(std::numeric_limits<Integer>::digits<=32)
    , unsigned>::type
msb( Integer const x){

	if(!x) return 32;

//host path will try to use
//GCC and CLANG intrisics
//
#ifndef __CUDA_ARCH__

	#if defined(__GNUC__) && !defined(__clang__)

		return 31 - __builtin_clz(x);

	#elif  defined(__clang__) && !defined(__GNUC__)

		return 31 - __builtin_clz(x);

	#else

	   unsigned y;
	   int n = 32;
	   y = _x >>16;  if (y != 0) {n = n -16;  _x = y;}
	   y = _x >> 8;  if (y != 0) {n = n - 8;  _x = y;}
	   y = _x >> 4;  if (y != 0) {n = n - 4;  _x = y;}
	   y = _x >> 2;  if (y != 0) {n = n - 2;  _x = y;}
	   y = _x >> 1;  if (y != 0) return n - 2;
	   return 31 - n - _x;

	#endif
#else
		return 31 - __clz(x);

#endif
}

/*
 * 16 bit implementation
 */
/*
template<typename Integer>
__hydra_host__ __hydra_device__
inline typename std::enable_if<
		std::is_integral<Integer>::value  &&
    	!(std::is_signed<Integer>::value) &&
    	(std::numeric_limits<Integer>::digits==16)
    , unsigned>::type
msb( Integer const& x){

	if(!x) return 16;

//host path will try to use
//GCC and CLANG intrisics
//
#ifndef __CUDA_ARCH__

	#if defined(__GNUC__) && !defined(__clang__)

		return 15 - __builtin_clz(x);

	#elif  defined(__clang__) && !defined(__GNUC__)

		return 15 - __builtin_clz(x);

	#else

	   unsigned y;
	   int n = 16;

	   y = x >> 8;  if (y != 0) {n = n - 8;  x = y;}
	   y = x >> 4;  if (y != 0) {n = n - 4;  x = y;}
	   y = x >> 2;  if (y != 0) {n = n - 2;  x = y;}
	   y = x >> 1;  if (y != 0) return n - 2;
	   return 15 - n - x;

	#endif
#else
		return 15 - __clz(x);

#endif
}
*/

}  // namespace detail

}  // namespace hydra




#endif /* MSB_H_ */
