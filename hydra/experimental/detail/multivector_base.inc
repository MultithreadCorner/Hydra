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
 * multivector.inc
 *
 *  Created on: 18/10/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef MULTIVECTOR_BASE_INC_
#define MULTIVECTOR_BASE_INC_

#include <thrust/tuple.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/utility/Generic.h>

namespace hydra {

namespace experimental {

namespace detail {


template< template<typename...> class Vector,
template<typename...> class Allocator,
typename ...Args, size_t ...index >
auto _vctor_helper(hydra::detail::index_sequence<index...>, const size_t n, thrust::tuple<Args...> const& args )
-> decltype( thrust::make_tuple(
		Vector<typename thrust::tuple_element<index, thrust::tuple<Args...>>::type
         , Allocator< typename thrust::tuple_element<index, thrust::tuple<Args...>>::type >>(n, thrust::get<index>(args))... ))
{
	return  thrust::make_tuple( Vector<typename thrust::tuple_element<index, thrust::tuple<Args...>>::type
			, Allocator<typename thrust::tuple_element<index, thrust::tuple<Args...>>::type >>(n, thrust::get<index>(args))... );
}

template<template<typename...> class Vector, template<typename...> class Allocator, typename ...Args>
auto _vctor( size_t n, thrust::tuple<Args...> const& args )
-> decltype( _vctor_helper<Vector, Allocator, Args...>( hydra::detail::make_index_sequence< sizeof ...(Args)> { }, n , args) )
{
	return _vctor_helper<Vector, Allocator, Args...>( hydra::detail::make_index_sequence< sizeof ...(Args) > { }, n, args);
}

/**
 * this macros expands with pattern:
 * void get<j>(storage).function(args...);
 * for j=0,1,2...N
 */
#define _GenerateVoidCallArgs(function) \
template<size_t I = 0, typename Tuple, typename ...Args> __host__  \
inline typename thrust::detail::enable_if<( I == thrust::tuple_size<Tuple>::value ), void>::type \
function##_call_args( Tuple&, Args const&... ){} \
 \
template<size_t I = 0, typename Tuple, typename ...Args> __host__  \
inline typename thrust::detail::enable_if<( I < thrust::tuple_size<Tuple>::value ),void >::type \
function##_call_args( Tuple& tpl, Args const&... args) \
{ \
thrust::get<I>(tpl).function( args...); \
function##_call_args<I + 1, Tuple>( tpl, args... ); \
}

/**
 * this macros expands with pattern:
 * void get<j>(storage).function( get<j>(args));
 * for j=0,1,2...N
 */
#define _GenerateVoidCallTuple(function) \
template<size_t I = 0, typename Tuple, typename ...Args> __host__  \
inline typename thrust::detail::enable_if<( (I == thrust::tuple_size<Tuple>::value) && (thrust::tuple_size<Tuple>::value==sizeof...(Args))), void>::type \
function##_call_tuple( Tuple&, thrust::tuple<Args...> const& ){} \
 \
template<size_t I = 0, typename Tuple, typename ...Args> __host__  \
inline typename thrust::detail::enable_if<( (I < thrust::tuple_size<Tuple>::value) && (thrust::tuple_size<Tuple>::value==sizeof...(Args)) ),void >::type \
function##_call_tuple( Tuple& tpl, thrust::tuple<Args...> const& args) \
{ \
thrust::get<I>(tpl).function( thrust::get<I>(args) ); \
function##_call_tuple<I + 1, Tuple, Args...>( tpl, args ); \
}

/**
 * this macros expands with pattern:
 * r<j> = get<j>(storage).function( args...);
 * return tuple< r<0>, r<1>, r<2>... >;
 * for j=0,1,2...N
*/
#define _GenerateNonVoidCallArgs(function) \
template< typename Tuple, typename ...Args, size_t ...index > \
__host__  inline auto \
function##_call_args_helper(hydra::detail::index_sequence<index...>, Tuple& tpl, Args& ...args ) \
-> thrust::tuple<decltype(thrust::get<index>(tpl).function(args...) )...> \
{ \
	return thrust::tuple<decltype(thrust::get<index>(tpl).function(args...)) ...>(thrust::get<index>(tpl).function(args...) ...); \
} \
  \
template< typename Tuple, typename ...Args > \
__host__   inline auto \
function##_call_args( Tuple& tpl, Args& ...args) \
-> decltype(function##_call_args_helper( hydra::detail::make_index_sequence< thrust::tuple_size<Tuple>::value> { }, tpl, args...)) \
{ \
	constexpr size_t Size = thrust::tuple_size<Tuple>::value; \
    \
	return function##_call_args_helper( hydra::detail::make_index_sequence<Size> { }, tpl, args...); \
}

/**
 * this macros expands with pattern:
 * r<j> = get<j>(storage).function( args...);
 * return tuple< r<0>, r<1>, r<2>... >;
 * for j=0,1,2...N
 */
#define _GenerateNonVoidCallTuple(function) \
template< typename Tuple, typename ...Args, size_t ...index > \
__host__  inline auto \
function##_call_tuple_helper(hydra::detail::index_sequence<index...>, Tuple& tpl, thrust::tuple<Args...> const& args ) \
-> thrust::tuple<decltype(thrust::get<index>(tpl).function(thrust::get<index>(args)) )...> \
{ \
	return thrust::tuple<decltype(thrust::get<index>(tpl).function(thrust::get<index>(args)) )...>(thrust::get<index>(tpl).function( thrust::get<index>(args) ) ...); \
} \
  \
template< typename Tuple, typename ...Args > \
__host__   inline auto \
function##_call_tuple( Tuple& tpl, thrust::tuple<Args...> const& args ) \
-> decltype(function##_call_tuple_helper( hydra::detail::make_index_sequence< thrust::tuple_size<Tuple>::value> { }, tpl, thrust::tuple<Args...> const& args)) \
{\
	constexpr size_t Size = thrust::tuple_size<Tuple>::value; \
 \
	return function##_call_tuple_helper( hydra::detail::make_index_sequence<Size> { }, tpl, args); \
}

/**
 * this macros expands with pattern:
 * r<j> = get<j>(storage).function( args...);
 * return tuple< r<0>, r<1>, r<2>... >;
 * for j=0,1,2...N
 */
#define _GenerateNonVoidCallArgsC(function) \
template< typename Tuple, typename ...Args, size_t ...index > \
__host__  inline auto \
function##_call_args_helper(hydra::detail::index_sequence<index...>, Tuple const& tpl, Args& ...args ) \
-> thrust::tuple<decltype(thrust::get<index>(tpl).function(args...) )...> \
{ \
	return thrust::tuple<decltype(thrust::get<index>(tpl).function(args...) )...>(thrust::get<index>(tpl).function(args...) ...); \
} \
  \
template< typename Tuple, typename ...Args > \
__host__   inline auto \
function##_call_args( Tuple const& tpl, Args& ...args) \
-> decltype(function##_call_args_helper( hydra::detail::make_index_sequence< thrust::tuple_size<Tuple>::value> { }, tpl, args...)) \
{ \
	constexpr size_t Size = thrust::tuple_size<Tuple>::value; \
    \
	return function##_call_args_helper( hydra::detail::make_index_sequence<Size> { }, tpl, args...); \
}

/**
 * this macros expands with pattern:
 * r<j> = get<j>(storage).function( args...);
 * return tuple< r<0>, r<1>, r<2>... >;
 * for j=0,1,2...N
 */
#define _GenerateNonVoidCallTupleC(function) \
template< typename Tuple, typename ...Args, size_t ...index > \
__host__ inline auto \
function##_call_tuple_helper(hydra::detail::index_sequence<index...>, Tuple const& tpl, thrust::tuple<Args...> const& args ) \
-> thrust::tuple<decltype(thrust::get<index>(tpl).function(thrust::get<index>(args)) )...> \
{ \
	return thrust::tuple<decltype(thrust::get<index>(tpl).function(thrust::get<index>(args)) )...>(thrust::get<index>(tpl).function( thrust::get<index>(args) ) ...); \
} \
  \
template< typename Tuple, typename ...Args > \
__host__   inline auto \
function##_call_tuple( Tuple const& tpl, thrust::tuple<Args...> const& args ) \
-> decltype(function##_call_tuple_helper( hydra::detail::make_index_sequence< thrust::tuple_size<Tuple>::value> { }, tpl, thrust::tuple<Args...> const& args)) \
{\
	constexpr size_t Size = thrust::tuple_size<Tuple>::value; \
 \
	return function##_call_tuple_helper( hydra::detail::make_index_sequence<Size> { }, tpl, args); \
}

}  // namespace detail

}  // namespace experimental

}  // namespace hydra



#endif /* MULTIVECTOR_BASE_INC_ */
