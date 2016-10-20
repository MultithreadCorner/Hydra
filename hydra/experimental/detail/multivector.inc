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

#ifndef MULTIVECTOR_INC_
#define MULTIVECTOR_INC_

namespace hydra {

namespace experimental {

namespace detail {


#define _GenerateVoidMember(function) \
template<size_t I = 0, typename Tuple, typename ...Args> __host__ __device__ \
inline typename thrust::detail::enable_if<I == thrust::tuple_size<Tuple>::value, void>::type \
function##_in_tuple( Tuple&, Args&... ){} \
 \
template<size_t I = 0, typename Tuple, typename ...Args> __host__ __device__ \
inline typename thrust::detail::enable_if<(I < thrust::tuple_size<Tuple>::value),void >::type \
function##_in_tuple( Tuple& tpl, Args&... args) \
{ \
thrust::get<I>(tpl).function(args...); \
function##_in_tuple<I + 1, Tuple>( tpl, args... ); \
}


#define _GenerateVoidMemberArgs(function) \
template<size_t I = 0, typename Tuple, typename ...Args> __host__ __device__ \
inline typename thrust::detail::enable_if<( (I == thrust::tuple_size<Tuple>::value) && (thrust::tuple_size<Tuple>::value==sizeof...(Args))), void>::type \
function##_invoke_with_args( Tuple&, Args const&... ){} \
 \
template<size_t I = 0, typename Tuple, typename ...Args> __host__ __device__ \
inline typename thrust::detail::enable_if<( (I < thrust::tuple_size<Tuple>::value) && (thrust::tuple_size<Tuple>::value==sizeof...(Args)) ),void >::type \
function##_invoke_with_args( Tuple& tpl, Args const&... args) \
{ \
thrust::get<I>(tpl).function( thrust::get<I>(thrust::make_tuple(args...)) ); \
function##_invoke_with_args<I + 1, Tuple>( tpl, args... ); \
}

#define _GenerateVoidMemberTuple(function) \
template<size_t I = 0, typename Tuple, typename ...Args> __host__ __device__ \
inline typename thrust::detail::enable_if<( (I == thrust::tuple_size<Tuple>::value) && (thrust::tuple_size<Tuple>::value==sizeof...(Args))), void>::type \
function##_invoke_with_tuple( Tuple&, thrust::tuple<Args...> const& ){} \
 \
template<size_t I = 0, typename Tuple, typename ...Args> __host__ __device__ \
inline typename thrust::detail::enable_if<( (I < thrust::tuple_size<Tuple>::value) && (thrust::tuple_size<Tuple>::value==sizeof...(Args)) ),void >::type \
function##_invoke_with_tuple( Tuple& tpl, thrust::tuple<Args...> const& args) \
{ \
thrust::get<I>(tpl).function( thrust::get<I>(args) ); \
function##_invoke_with_tuple<I + 1, Tuple, Args...>( tpl, args ); \
}


#define _GenerateNonVoidMember(function) \
template< typename Tuple, typename ...Args, size_t ...index > \
__host__ __device__ inline auto \
function##_invoke_helper(hydra::detail::index_sequence<index...>, Tuple& tpl, Args& ...args ) \
-> decltype( thrust::make_tuple(thrust::get<index>(tpl).function(args...)...) ) \
{ \
	return thrust::make_tuple(thrust::get<index>(tpl).function(args...) ...); \
} \
  \
template< typename Tuple, typename ...Args > \
__host__  __device__ inline auto \
function##_invoke( Tuple& tpl, Args& ...args) \
-> decltype(function##_invoke_helper( hydra::detail::make_index_sequence< thrust::tuple_size<Tuple>::value> { }, tpl, args...)) \
{ \
	constexpr size_t Size = thrust::tuple_size<Tuple>::value; \
    \
	return function##_invoke_helper( hydra::detail::make_index_sequence<Size> { }, tpl, args...); \
} \



}  // namespace detail

}  // namespace experimental

}  // namespace hydra



#endif /* MULTIVECTOR_INC_ */
