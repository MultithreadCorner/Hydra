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
 * multiarray.inc
 *
 *  Created on: 23/07/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef MULTIARRAY_INC_
#define MULTIARRAY_INC_

#include <hydra/detail/Config.h>

#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/utility/Generic.h>
#include <thrust/tuple.h>
#include <utility>
#include <array>

namespace hydra {

namespace detail {

//----------------------------------------------
//front
template<typename VECTOR, size_t ...Index>
inline auto get_front_tuple_helper(std::array<VECTOR, sizeof ...(Index)>& data,	index_sequence<Index...>)
-> decltype( thrust::make_tuple( data[Index].front()...) )
{
	return thrust::make_tuple( data[Index].front()... );
}

template<typename VECTOR, size_t N>
inline auto get_front_tuple(std::array<VECTOR, N>& data)
-> decltype( get_front_tuple_helper(data , make_index_sequence<N> { }) )
{
	return get_front_tuple_helper( data , make_index_sequence<N> { } );
}

//cfront
template<typename VECTOR, size_t ...Index>
inline auto get_cfront_tuple_helper(std::array<VECTOR, sizeof ...(Index)> const& data,	index_sequence<Index...>)
-> decltype( thrust::make_tuple( data[Index].front()...) )
{
	return thrust::make_tuple( data[Index].front()... );
}

template<typename VECTOR, size_t N>
inline auto get_cfront_tuple(std::array<VECTOR, N> const& data)
-> decltype( get_cfront_tuple_helper(data , make_index_sequence<N> { }) )
{
	return get_cfront_tuple_helper( data , make_index_sequence<N> { } );
}
//----------------------------------------------
//back
template<typename VECTOR, size_t ...Index>
inline auto get_back_tuple_helper(std::array<VECTOR, sizeof ...(Index)>& data,	index_sequence<Index...>)
-> decltype( thrust::make_tuple( data[Index].back()...) )
{
	return thrust::make_tuple( data[Index].back()... );
}

template<typename VECTOR, size_t N>
inline auto get_back_tuple(std::array<VECTOR, N>& data)
-> decltype( get_back_tuple_helper(data , make_index_sequence<N> { }) )
{
	return get_back_tuple_helper( data , make_index_sequence<N> { } );
}

//cback
template<typename VECTOR, size_t ...Index>
inline auto get_cback_tuple_helper(std::array<VECTOR, sizeof ...(Index)> const& data,	index_sequence<Index...>)
-> decltype( thrust::make_tuple( data[Index].back()...) )
{
	return thrust::make_tuple( data[Index].back()... );
}

template<typename VECTOR, size_t N>
inline auto get_cback_tuple(std::array<VECTOR, N>const& data)
-> decltype( get_cback_tuple_helper(data , make_index_sequence<N> { }) )
{
	return get_cback_tuple_helper( data , make_index_sequence<N> { } );
}
//----------------------------------------------
//data
template<typename VECTOR, size_t ...Index>
inline auto get_data_tuple_helper(std::array<VECTOR, sizeof ...(Index)>& data,	index_sequence<Index...>)
-> decltype( thrust::make_tuple( data[Index].data()...) )
{
	return thrust::make_tuple( data[Index].data()... );
}

template<typename VECTOR, size_t N>
inline auto get_data_tuple(std::array<VECTOR, N>& data)
-> decltype( get_data_tuple_helper(data , make_index_sequence<N> { }) )
{
	return get_data_tuple_helper( data , make_index_sequence<N> { } );
}

//cdata
template<typename VECTOR, size_t ...Index>
inline auto get_cdata_tuple_helper(std::array<VECTOR, sizeof ...(Index)>const& data,	index_sequence<Index...>)
-> decltype( thrust::make_tuple( data[Index].data()...) )
{
	return thrust::make_tuple( data[Index].data()... );
}

template<typename VECTOR, size_t N>
inline auto get_cdata_tuple(std::array<VECTOR, N>const& data)
-> decltype( get_cdata_tuple_helper(data , make_index_sequence<N> { }) )
{
	return get_cdata_tuple_helper( data , make_index_sequence<N> { } );
}


template<size_t I, typename Vector, size_t N, template<typename ...> class Tuple, typename ...Iterators>
inline typename thrust::detail::enable_if<(I == sizeof...(Iterators)), void >::type
do_insert(size_t dist, std::array<Vector, N>& data,
		Tuple<Iterators...> const& first_tuple, Tuple<Iterators...> const& last_tuple)
{}

template<size_t I = 0, typename Vector, size_t N, template<typename ...> class Tuple, typename ...Iterators>
inline typename thrust::detail::enable_if<(I < sizeof...(Iterators)), void >::type
do_insert(size_t dist, std::array<Vector, N>& data,
		Tuple<Iterators...> const& first_tuple, Tuple<Iterators...> const& last_tuple)
{
    	auto first = thrust::get<I>(first_tuple);
    	auto last  = thrust::get<I>(last_tuple);
    	data[I].insert(data[I].begin() + dist, first, last);

    	do_insert<I + 1, Vector, N, Tuple, Iterators... >(dist, data, first_tuple, last_tuple );
}


template<size_t I, typename Vector, size_t N, template<typename ...> class Tuple, typename ...T>
inline typename thrust::detail::enable_if<(I == sizeof...(T)), void >::type
do_push_back(std::array<Vector, N>& data, Tuple<T...> const& value)
{}

template<size_t I = 0, typename Vector, size_t N, template<typename ...> class Tuple, typename ...T>
inline typename thrust::detail::enable_if<(I < sizeof...(T)), void >::type
do_push_back(std::array<Vector, N>& data, Tuple<T...> const& value)
{
    	data[I].push_back(thrust::get<I>(value));

    	do_push_back<I + 1, Vector, N, Tuple, T... >(data, value );
}

}  // namespace detail

}  // namespace hydra

#endif /* MULTIARRAY_INC_ */
