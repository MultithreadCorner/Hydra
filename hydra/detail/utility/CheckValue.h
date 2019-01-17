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
 * CheckValue.h
 *
 *  Created on: 20/12/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CHECKVALUE_H_
#define CHECKVALUE_H_

#include <hydra/detail/Config.h>
#include <stdio.h>
#include <cmath>
#include <utility>

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"


#define CHECK_VALUE(x, fmt, ...)  detail::CheckValue(x, fmt, __FILE__ ,__PRETTY_FUNCTION__ , __LINE__ , __VA_ARGS__)

namespace hydra {
namespace detail {


template<typename T, typename ...Ts>
__hydra_host__ __hydra_device__
inline T kill(T&& x){
	std::exit(1);
	return x;
}

template<typename T, typename ...Ts>
__hydra_host__ __hydra_device__
inline T CheckValue( T&& x, char const* fmt, char const* file, char const* function, unsigned int line, Ts&& ...par)
{
	/*
#ifndef __CUDA_ARCH__

	return std::isnan(std::forward<T>(x))?
			 printf(ANSI_COLOR_RED "\n HYDRA WARNING:" ANSI_COLOR_RESET "NaN found\n FILE:  %s \n FUNCTION: %s \n LINE: %d \n", file, function,line),
			 printf(fmt,std::forward<Ts>(par)... ),std::forward<T>(x) : std::forward<T>(x);
#else
*/


	return ::isnan(std::forward<T>(x))?
				 printf("\n HYDRA WARNING: NAN found on\n FILE:  %s \n FUNCTION: %s \n LINE: %d \n", file, function,line),\
				 printf(fmt, std::forward<Ts>(par)... ),std::forward<T>(x)/*, kill<T>(std::forward<T>(x))*/ : std::forward<T>(x);

//#endif
}

}  // namespace detail

}  // namespace hydra

#endif /* CHECKVALUE_H_ */
