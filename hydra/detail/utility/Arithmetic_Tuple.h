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
 * Arithmetic_Tuple.h
 *
 *  Created on: 21/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef ARITHMETIC_TUPLE_H_
#define ARITHMETIC_TUPLE_H_


//hydra
#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/utility/Generic.h>

//thrust
#include <thrust/tuple.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/tuple_of_iterator_references.h>

namespace hydra {

		namespace detail {

		    // add two tuples element by element
			template<size_t I, typename ... T>
			__host__ __device__
			inline typename thrust::tuple_element<I, thrust::tuple<T...>>::type
			addTuplesHelper(const thrust::tuple<T...>&a, const thrust::tuple<T...>&b) {
				return thrust::get<I>(a) + thrust::get<I>(b);
			}

			template<typename ... T, size_t ... I>
			__host__ __device__
			inline thrust::tuple<T...>
			addTuples(const thrust::tuple<T...>&a, const thrust::tuple<T...>&b, index_sequence<I...>) {
				return thrust::make_tuple(addTuplesHelper<I>(a,b)...);
			}

			template<typename ...T>
			__host__ __device__
			inline thrust::tuple<T...>
			addTuples(const thrust::tuple<T...>&a, const thrust::tuple<T...>&b) {
				return addTuples<T...>(a, b, make_index_sequence<sizeof...(T)> {} );
			}

			//


	}//namespace detail

	template<typename ...T>
	__host__ __device__
	inline thrust::tuple<T...> operator+(const thrust::tuple<T...>&a,
			const thrust::tuple<T...>&b){

		return detail::addTuples(a,b);
	}



}//namespace hydra
#endif /* ARITHMETIC_TUPLE_H_ */
