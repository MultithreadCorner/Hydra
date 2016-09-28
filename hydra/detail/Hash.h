/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 Antonio Augusto Alves Junior
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
 * Hash.h
 *  Obs: inspirated on the boost implementation
 *  Created on: 14/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef HASH_H_
#define HASH_H_

#include <functional>

namespace hydra {

	namespace detail {

		template<class T>
		inline void hash_combine(std::size_t& seed, T const& v) {
			std::hash<T> hasher;
			seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}

		template<class It>
		inline std::size_t hash_range(It first, It last) {
			std::size_t seed = 0;

			for (; first != last; ++first) {
				hash_combine(seed, *first);
			}

			return seed;
		}

		template<class It>
		inline void hash_range(std::size_t& seed, It first, It last) {
			for (; first != last; ++first) {
				hash_combine(seed, *first);
			}
		}

	}//namespace detail
}//namespace hydra
#endif /* HASH_H_ */
