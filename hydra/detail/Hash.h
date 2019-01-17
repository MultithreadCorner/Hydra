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
 * Hash.h
 *  Obs: inspirated on the boost implementation
 *  Created on: 14/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup generic
 */

#ifndef HASH_H_
#define HASH_H_

#include <hydra/Tuple.h>
#include <utility>
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

		template< class It>
		inline void hash_range(std::size_t& seed, It first, It last) {
			for (; first != last; ++first) {
				hash_combine(seed, *first);
			}
		}

		namespace tuple {

			template< typename T, unsigned int N, unsigned int I>
			inline typename std::enable_if< (I == N), void  >::type
			hash_tuple_helper(std::size_t&, T const&){ }

			template< typename T, unsigned int N, unsigned int I=0>
			inline typename std::enable_if< (I < N), void  >::type
			hash_tuple_helper(std::size_t& seed, T const& _tuple){

				hydra::detail::hash_combine(seed, hydra::get<I>(_tuple));

				tuple::hash_tuple_helper<T,N, I+1>(seed, _tuple  );
			}


		}  // namespace tuple

		template< typename ...T>
		inline void hash_tuple(std::size_t& seed, hydra::tuple<T...> const& _tuple){

			tuple::hash_tuple_helper<hydra::tuple<T...>, sizeof...(T) >(seed, _tuple );
		}

		template< typename ...T>
		inline std::size_t hash_tuple( hydra::tuple<T...> const& _tuple){

			std::size_t seed = 0;

			hash_tuple(seed, _tuple);

			return seed;

		}




	}//namespace detail
}//namespace hydra
#endif /* HASH_H_ */
