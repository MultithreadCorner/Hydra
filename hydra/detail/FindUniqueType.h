/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2022 Antonio Augusto Alves Junior
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
 * FindUniqueType.h
 *
 *  Created on: 11/02/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef FINDUNIQUETYPE_H_
#define FINDUNIQUETYPE_H_

#include <type_traits>
#include <hydra/detail/utility/StaticAssert.h>

namespace hydra {

namespace detail {

template<size_t I, class T, class... Types>
struct find_unique_type_impl;


template<size_t I, class T, class U, class... Types>
struct find_unique_type_impl<I,T,U,Types...> : find_unique_type_impl<I+1, T, Types...> {};


template<size_t I, class T, class... Types>
struct find_unique_type_impl<I,T,T,Types...> : std::integral_constant<size_t, I>
{
  HYDRA_STATIC_ASSERT((find_unique_type_impl<I,T,Types...>::value==-1),
		  "Type not unique in type list")
};


template<size_t I, class T>
struct find_unique_type_impl<I,T>: std::integral_constant<int, -1> {};


template<class T, class... Types>
struct find_unique_type : find_unique_type_impl<0,T,Types...>
{
	 HYDRA_STATIC_ASSERT((int(find_unique_type::value) != -1),
		  "Type not found in type list")
};


}  // namespace detail

}  // namespace hydra

#endif /* FINDUNIQUETYPE_H_ */
