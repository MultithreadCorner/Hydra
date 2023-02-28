/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2023 Antonio Augusto Alves Junior
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
 * TupleUtility.h
 *
 *  Created on: 27/05/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef TUPLEUTILITY_H_
#define TUPLEUTILITY_H_

#include <hydra/detail/Config.h>

#include <hydra/Tuple.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/type_traits/void_t.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <type_traits>
#include <utility>
#include <hydra/detail/TupleTraits.h>

namespace hydra {

namespace detail {

namespace tuple_utility {

	template<typename T>
	struct do_tuple : std::conditional< detail::is_tuple<T>::value, T, hydra::tuple<T> > {};

	template<typename ...T>
	struct flat_tuple: detail::merged_tuple< typename do_tuple<T>::type... > {};


}  // namespace tuple_utility

	template<typename T>
	__hydra_host__ __hydra_device__
	typename std::enable_if< detail::is_tuple<T>::value, T>::type
	tupler( T const& data ) { return data;}

	template<typename T>
	__hydra_host__ __hydra_device__
	typename std::enable_if<!detail::is_tuple<T>::value, hydra::tuple<T> >::type
	tupler( T const& data )	{ return hydra::make_tuple( data ); }

}  // namespace detail


	template<typename ...T>
	__hydra_host__ __hydra_device__
	typename detail::tuple_utility::flat_tuple<T...>::type
	get_flat_tuple(T const&... args)
	{
		return hydra_thrust::tuple_cat( detail::tupler(args)...);
	}

}  // namespace hydra

#endif /* TUPLEUTILITY_H_ */
