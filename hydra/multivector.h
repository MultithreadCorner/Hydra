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
 * multivector.h
 *
 *  Created on: 29/10/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef MULTIVECTOR_H_
#define MULTIVECTOR_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/multivector_base.h>
#include <thrust/tuple.h>

namespace hydra {


template<typename T>
class multivector;

/**
 * @brief This class implements storage for an array of tuples using a SOA memory layout.
 * This container mimics the interface of std::vector.
 * @tparam Vector underlying model/layout for contiguous memory storage.
 * @tparam  Allocator memory allocator for contiguous memory storage.
 */
template< template<typename...> class Vector, template<typename...> class Allocator,  typename ...T>
class multivector<Vector<thrust::tuple<T...>, Allocator<thrust::tuple<T...>>>> : public multivector_base<Vector, Allocator, T...>
{
public:
	/**
	 * @brief default constructor
	 */
	explicit  multivector():
			multivector_base<Vector, Allocator, T...>()
	{ }

	/**
	 * @brief constructor size_t n
	 */
	explicit multivector(size_t n):
			multivector_base<Vector, Allocator, T...>( n)
	{}

	/**
	 * @brief constructor size_t n, ...values
	 */
	explicit multivector(size_t n, T... value):
			multivector_base<Vector, Allocator, T...>(n, value...)
	{}

	/**
	 * @brief constructor size_t n, ...values
	 */
	explicit multivector(size_t n, typename multivector_base<Vector, Allocator, T...>::value_tuple_type  value):
		multivector_base<Vector, Allocator, T...>(n, value)
	{}

	/**
	 * @brief copy constructor
	 */
	template< template<typename...> class Vector2,
	template<typename...> class Allocator2>
	multivector( multivector<Vector2<thrust::tuple<T...>, Allocator2<thrust::tuple<T...>>>>const&  other):
	multivector_base<Vector, Allocator, T...>(other)
	{}

	multivector( multivector<Vector<thrust::tuple<T...>, Allocator<thrust::tuple<T...>>>>const&  other):
	multivector_base<Vector, Allocator, T...>(other)
	{}


	/**
	 * @brief move constructor
	 */
	multivector( multivector<Vector<thrust::tuple<T...>,
			Allocator<thrust::tuple<T...>>>> &&  other):
	multivector_base<Vector, Allocator, T...>(std::move(other))
	{}


	/**
	 * @brief assignment operator=
	 */
	template< template<typename...> class Vector2,
	template<typename...> class Allocator2>
	multivector<Vector<thrust::tuple<T...>, Allocator<thrust::tuple<T...>>>>&
	operator=( multivector<Vector2<thrust::tuple<T...>, Allocator2<thrust::tuple<T...>>>> const&  v)
	{
		if(this==&v) return *this;
		multivector_base<Vector, Allocator, T...>::operator=(v);
		return *this;
	}


	multivector<Vector<thrust::tuple<T...>, Allocator<thrust::tuple<T...>>>>&
	operator=( multivector<Vector<thrust::tuple<T...>, Allocator<thrust::tuple<T...>>>> const&  v)
	{
		if(this==&v) return *this;
		multivector_base<Vector, Allocator, T...>::operator=(v);
		return *this;
	}

	multivector<Vector<thrust::tuple<T...>, Allocator<thrust::tuple<T...>>>>&
	operator=( multivector<Vector<thrust::tuple<T...>, Allocator<thrust::tuple<T...>>>> &&  v)
	{
		if(this==&v) return *this;
		multivector_base<Vector, Allocator, T...>::operator=(std::move(v));
		return *this;
	}



};

}  // namespace hydra

#endif /* MULTIVECTOR_H_ */
