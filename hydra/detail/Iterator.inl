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
 * Iterator.inl
 *
 *  Created on: 12/05/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef ITERATOR_INL_
#define ITERATOR_INL_

#include <utility>
#include <hydra/detail/external/thrust/iterator/reverse_iterator.h>

namespace hydra {

//directs iterators
template< class C >
auto begin(  const C& c ) -> decltype(c.begin()){
	return c.begin();
}

template< class C >
auto end(  const C& c ) -> decltype(c.end()){
	return c.end();
}

template< class C >
auto begin( C&& c ) -> decltype(std::forward<C>(c).begin()){
	return std::forward<C>(c).begin();
}

template< class C >
auto end( C&& c ) -> decltype(std::forward<C>(c).end()){
	return std::forward<C>(c).end();
}

template< class T, size_t N >
T* begin( T (&array)[N] ){
	return &array[0];
}

template< class T, size_t N >
T* end( T (&array)[N] ){
	return &array[N];
}

//reverse iterators
template< class C >
auto rbegin( C&& c ) -> decltype(std::forward<C>(c).rbegin()){
	return std::forward<C>(c).rbegin();
}

template< class C >
auto rend( C&& c ) -> decltype(std::forward<C>(c).rend()){
	return std::forward<C>(c).rend();
}

template< class C >
auto rbegin( const C& c ) -> decltype(c.rbegin()){
	return c.rbegin();
}

template< class C >
auto rend( const C& c ) -> decltype(c.rend()){
	return c.rend();
}

template< class T, size_t N >
T* rbegin( T (&array)[N] ){
	return HYDRA_EXTERNAL_NS::thrust::reverse_iterator<T*>(&array[0]);
}

template< class T, size_t N >
T* rend( T (&array)[N] ){
	return HYDRA_EXTERNAL_NS::thrust::reverse_iterator<T*>(&array[N]);
}

}  // namespace hydra



#endif /* ITERATOR_INL_ */
