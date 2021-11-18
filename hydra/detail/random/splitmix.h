/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2021 Antonio Augusto Alves Junior
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
 * splitmix.h
 *
 *  Created on: Sep 28, 2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SPLITMIX_H_
#define SPLITMIX_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/random/detail/squares_key.h>
#include <stdint.h>

namespace hydra {

namespace random {

template<typename UIntType>
__hydra_host__ __hydra_device__
inline UIntType splitmix( UIntType& );


template<>
__hydra_host__ __hydra_device__
inline uint32_t splitmix<uint32_t>(uint32_t& x) {
	uint32_t z = (x += 0x6D2B79F5UL);
	z = (z ^ (z >> 15)) * (z | 1UL);
	z ^= z + (z ^ (z >> 7)) * (z | 61UL);
	return z ^ (z >> 14);
}

template<>
__hydra_host__ __hydra_device__
inline uint64_t  splitmix<uint64_t>(uint64_t& x){
	uint64_t z = (x += 0x9e3779b97f4a7c15);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
	return z ^ (z >> 31);
}


}  // namespace random

}  // namespace hydra




#endif /* SPLITMIX_H_ */
