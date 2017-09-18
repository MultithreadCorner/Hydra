/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2017 Antonio Augusto Alves Junior
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

#include <utility>
#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/utility/Generic.h>

namespace hydra {


namespace detail {


template< template<typename...> class Vector,
template<typename...> class Allocator,
typename ...Args, size_t ...index >
auto _vctor_helper(hydra::detail::index_sequence<index...>, const size_t n, HYDRA_EXTERNAL_NS::thrust::tuple<Args...> const& args )
-> decltype( HYDRA_EXTERNAL_NS::thrust::make_tuple(
		Vector<typename HYDRA_EXTERNAL_NS::thrust::tuple_element<index, HYDRA_EXTERNAL_NS::thrust::tuple<Args...>>::type
         , Allocator< typename HYDRA_EXTERNAL_NS::thrust::tuple_element<index, HYDRA_EXTERNAL_NS::thrust::tuple<Args...>>::type >>(n, HYDRA_EXTERNAL_NS::thrust::get<index>(args))... ))
{
	return  HYDRA_EXTERNAL_NS::thrust::make_tuple( Vector<typename HYDRA_EXTERNAL_NS::thrust::tuple_element<index, HYDRA_EXTERNAL_NS::thrust::tuple<Args...>>::type
			, Allocator<typename HYDRA_EXTERNAL_NS::thrust::tuple_element<index, HYDRA_EXTERNAL_NS::thrust::tuple<Args...>>::type >>(n, HYDRA_EXTERNAL_NS::thrust::get<index>(args))... );
}

template<template<typename...> class Vector, template<typename...> class Allocator, typename ...Args>
auto _vctor( size_t n, HYDRA_EXTERNAL_NS::thrust::tuple<Args...> const& args )
-> decltype( _vctor_helper<Vector, Allocator, Args...>( hydra::detail::make_index_sequence< sizeof ...(Args)> { }, n , args) )
{
	return _vctor_helper<Vector, Allocator, Args...>( hydra::detail::make_index_sequence< sizeof ...(Args) > { }, n, args);
}


template<size_t I = 0,
template<typename...> class Vector, template<typename...> class Allocator,typename ...T>
inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if< (I == sizeof...(T)) ,void >::type
_move_storage(HYDRA_EXTERNAL_NS::thrust::tuple<Vector<T, Allocator<T>>...>& This,
		HYDRA_EXTERNAL_NS::thrust::tuple<Vector<T, Allocator<T>>...>&& Other )
{ }

template<size_t I = 0,
template<typename...> class Vector, template<typename...> class Allocator, typename ...T>
inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if< (I < sizeof...(T)) ,void >::type
_move_storage(HYDRA_EXTERNAL_NS::thrust::tuple<Vector<T, Allocator<T>>...>& This,
		HYDRA_EXTERNAL_NS::thrust::tuple<Vector<T, Allocator<T>>...>&& Other )
{
	HYDRA_EXTERNAL_NS::thrust::get<I>(This)=std::move(HYDRA_EXTERNAL_NS::thrust::get<I>(Other));
	detail::_move_storage<I + 1,Vector, Allocator, T... >( This, std::forward<HYDRA_EXTERNAL_NS::thrust::tuple<Vector<T, Allocator<T>>...>&&>(Other));
}

/**
 * this macros expands with pattern:
 * void get<j>(storage).function(args...);
 * for j=0,1,2...N
 */
#define _GenerateVoidCallArgs(function) \
template<size_t I = 0, typename Tuple, typename ...Args> __host__  \
inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<( I == HYDRA_EXTERNAL_NS::thrust::tuple_size<Tuple>::value ), void>::type \
function##_call_args( Tuple&, Args const&... ){} \
 \
template<size_t I = 0, typename Tuple, typename ...Args> __host__  \
inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<( I < HYDRA_EXTERNAL_NS::thrust::tuple_size<Tuple>::value ),void >::type \
function##_call_args( Tuple& tpl, Args const&... args) \
{ \
HYDRA_EXTERNAL_NS::thrust::get<I>(tpl).function( args...); \
function##_call_args<I + 1, Tuple>( tpl, args... ); \
}

/**
 * this macros expands with pattern:
 * void get<j>(storage).function( get<j>(args));
 * for j=0,1,2...N
 */
#define _GenerateVoidCallTuple(function) \
template<size_t I = 0, typename Tuple, typename ...Args> __host__  \
inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<( (I == HYDRA_EXTERNAL_NS::thrust::tuple_size<Tuple>::value) && (HYDRA_EXTERNAL_NS::thrust::tuple_size<Tuple>::value==sizeof...(Args))), void>::type \
function##_call_tuple( Tuple&, HYDRA_EXTERNAL_NS::thrust::tuple<Args...> const& ){} \
 \
template<size_t I = 0, typename Tuple, typename ...Args> __host__  \
inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<( (I < HYDRA_EXTERNAL_NS::thrust::tuple_size<Tuple>::value) && (HYDRA_EXTERNAL_NS::thrust::tuple_size<Tuple>::value==sizeof...(Args)) ),void >::type \
function##_call_tuple( Tuple& tpl, HYDRA_EXTERNAL_NS::thrust::tuple<Args...> const& args) \
{ \
HYDRA_EXTERNAL_NS::thrust::get<I>(tpl).function( HYDRA_EXTERNAL_NS::thrust::get<I>(args) ); \
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
-> HYDRA_EXTERNAL_NS::thrust::tuple<decltype(HYDRA_EXTERNAL_NS::thrust::get<index>(tpl).function(args...) )...> \
{ \
	return HYDRA_EXTERNAL_NS::thrust::tuple<decltype(HYDRA_EXTERNAL_NS::thrust::get<index>(tpl).function(args...)) ...>(HYDRA_EXTERNAL_NS::thrust::get<index>(tpl).function(args...) ...); \
} \
  \
template< typename Tuple, typename ...Args > \
__host__   inline auto \
function##_call_args( Tuple& tpl, Args& ...args) \
-> decltype(function##_call_args_helper( hydra::detail::make_index_sequence< HYDRA_EXTERNAL_NS::thrust::tuple_size<Tuple>::value> { }, tpl, args...)) \
{ \
	constexpr size_t Size = HYDRA_EXTERNAL_NS::thrust::tuple_size<Tuple>::value; \
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
function##_call_tuple_helper(hydra::detail::index_sequence<index...>, Tuple& tpl, HYDRA_EXTERNAL_NS::thrust::tuple<Args...> const& args ) \
-> HYDRA_EXTERNAL_NS::thrust::tuple<decltype(HYDRA_EXTERNAL_NS::thrust::get<index>(tpl).function(HYDRA_EXTERNAL_NS::thrust::get<index>(args)) )...> \
{ \
	return HYDRA_EXTERNAL_NS::thrust::tuple<decltype(HYDRA_EXTERNAL_NS::thrust::get<index>(tpl).function(HYDRA_EXTERNAL_NS::thrust::get<index>(args)) )...>(HYDRA_EXTERNAL_NS::thrust::get<index>(tpl).function( HYDRA_EXTERNAL_NS::thrust::get<index>(args) ) ...); \
} \
  \
template< typename Tuple, typename ...Args > \
__host__   inline auto \
function##_call_tuple( Tuple& tpl, HYDRA_EXTERNAL_NS::thrust::tuple<Args...> const& args ) \
-> decltype(function##_call_tuple_helper( hydra::detail::make_index_sequence< HYDRA_EXTERNAL_NS::thrust::tuple_size<Tuple>::value> { }, tpl, HYDRA_EXTERNAL_NS::thrust::tuple<Args...> const& args)) \
{\
	constexpr size_t Size = HYDRA_EXTERNAL_NS::thrust::tuple_size<Tuple>::value; \
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
-> HYDRA_EXTERNAL_NS::thrust::tuple<decltype(HYDRA_EXTERNAL_NS::thrust::get<index>(tpl).function(args...) )...> \
{ \
	return HYDRA_EXTERNAL_NS::thrust::tuple<decltype(HYDRA_EXTERNAL_NS::thrust::get<index>(tpl).function(args...) )...>(HYDRA_EXTERNAL_NS::thrust::get<index>(tpl).function(args...) ...); \
} \
  \
template< typename Tuple, typename ...Args > \
__host__   inline auto \
function##_call_args( Tuple const& tpl, Args& ...args) \
-> decltype(function##_call_args_helper( hydra::detail::make_index_sequence< HYDRA_EXTERNAL_NS::thrust::tuple_size<Tuple>::value> { }, tpl, args...)) \
{ \
	constexpr size_t Size = HYDRA_EXTERNAL_NS::thrust::tuple_size<Tuple>::value; \
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
function##_call_tuple_helper(hydra::detail::index_sequence<index...>, Tuple const& tpl, HYDRA_EXTERNAL_NS::thrust::tuple<Args...> const& args ) \
-> HYDRA_EXTERNAL_NS::thrust::tuple<decltype(HYDRA_EXTERNAL_NS::thrust::get<index>(tpl).function(HYDRA_EXTERNAL_NS::thrust::get<index>(args)) )...> \
{ \
	return HYDRA_EXTERNAL_NS::thrust::tuple<decltype(HYDRA_EXTERNAL_NS::thrust::get<index>(tpl).function(HYDRA_EXTERNAL_NS::thrust::get<index>(args)) )...>(HYDRA_EXTERNAL_NS::thrust::get<index>(tpl).function( HYDRA_EXTERNAL_NS::thrust::get<index>(args) ) ...); \
} \
  \
template< typename Tuple, typename ...Args > \
__host__   inline auto \
function##_call_tuple( Tuple const& tpl, HYDRA_EXTERNAL_NS::thrust::tuple<Args...> const& args ) \
-> decltype(function##_call_tuple_helper( hydra::detail::make_index_sequence< HYDRA_EXTERNAL_NS::thrust::tuple_size<Tuple>::value> { }, tpl, HYDRA_EXTERNAL_NS::thrust::tuple<Args...> const& args)) \
{\
	constexpr size_t Size = HYDRA_EXTERNAL_NS::thrust::tuple_size<Tuple>::value; \
 \
	return function##_call_tuple_helper( hydra::detail::make_index_sequence<Size> { }, tpl, args); \
}

}  // namespace detail


}  // namespace hydra



#endif /* MULTIVECTOR_BASE_INC_ */
