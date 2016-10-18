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


#define MultiVector_GEN_MEMBER_V(function) \
template<size_t I = 0, typename ...Tp, typename ...Args> __host__ __device__ \
inline typename thrust::detail::enable_if<I == sizeof...(Tp), void>::type \
function##_in_tuple(thrust::tuple<Tp...>&, Args&& ...args){} \
 \
template<size_t I = 0, typename ...Tp, typename ...Args> __host__ __device__ \
inline typename thrust::detail::enable_if<(I < sizeof...(Tp)),void >::type \
function##_in_tuple(thrust::tuple<Tp...>& t,  Args&& ...args) \
{ \
thrust::get<I>(t).function(args...); \
function##_in_tuple<I + 1,Tp...>( t, args... ); \
}

#define MULTIVECTOR_MEMBER(function) \
template< typename Tup, typename ArgType, size_t ... index>
__host__ __device__
inline auto invoke_helper( ArgType& x, Tup& tup, index_sequence<index...>)
-> decltype(thrust::make_tuple(thrust::get<index>(tup)(x)...))
{
	return thrust::make_tuple(thrust::get<index>(tup)(x)...);
}

template< typename Tup, typename ArgType>
__host__  __device__
inline auto invoke(ArgType& x, Tup& tup)
-> decltype(invoke_helper(x, tup, make_index_sequence< thrust::tuple_size<Tup>::value> { }))
{
	constexpr size_t Size = thrust::tuple_size<Tup>::value;
	return invoke_helper( x, tup, make_index_sequence<Size> { });
}


MULTIVECTOR_MEMBER(resize)

int main() {

   return;


}



#endif /* MULTIVECTOR_INC_ */
