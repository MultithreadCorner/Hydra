/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2021 Antonio Augusto Alves Junior
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
 * GetTupleElement.h
 *
 *  Created on: 11/02/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GETTUPLEELEMENT_H_
#define GETTUPLEELEMENT_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/FindUniqueType.h>
#include <hydra/detail/TypeTraits.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/tuple_of_iterator_references.h>

#include <type_traits>

namespace hydra {

namespace detail {


//for tuples
template<typename T, typename... Types>
__hydra_host__ __hydra_device__
T& get_tuple_element(hydra_thrust::tuple<Types...>& t)
{
  return hydra_thrust::get<
		  find_unique_type<typename remove_device_reference<typename std::decay<T>::type>::type,
		  typename remove_device_reference<typename std::decay<Types>::type>::type...>::value>(t);
}


template<class T, class... Types>
__hydra_host__ __hydra_device__
const T& get_tuple_element(const hydra_thrust::tuple<Types...>& t)
{
  return hydra_thrust::get<
		  find_unique_type<typename remove_device_reference<typename std::decay<T>::type>::type,
		  typename remove_device_reference<typename std::decay<Types>::type>::type...>::value>(t);
}


template<class T, class... Types>
__hydra_host__ __hydra_device__
T&& get_tuple_element(hydra_thrust::tuple<Types...>&& t)
{
  return hydra_thrust::get<
		  find_unique_type<typename remove_device_reference<typename std::decay<T>::type>::type,
		  typename remove_device_reference<typename std::decay<Types>::type>::type...>::value>(std::move(t));
}

//for tuples of device_references
template<typename T, typename... Types>
__hydra_host__ __hydra_device__
hydra_thrust::device_reference<T>
get_tuple_element(hydra_thrust::tuple<hydra_thrust::device_reference<Types>...>& t)
{
  return hydra_thrust::get<
		  find_unique_type<typename remove_device_reference<typename std::decay<T>::type>::type,
		  typename remove_device_reference<typename std::decay<Types>::type>::type...>::value>(t);
}


template<class T, class... Types>
__hydra_host__ __hydra_device__
const hydra_thrust::device_reference<T>
get_tuple_element(const hydra_thrust::tuple<hydra_thrust::device_reference<Types>...>& t)
{
  return hydra_thrust::get<
		  find_unique_type<typename remove_device_reference<typename std::decay<T>::type>::type,
		  typename remove_device_reference<typename std::decay<Types>::type>::type...>::value>(t);
}


template<class T, class... Types>
__hydra_host__ __hydra_device__
hydra_thrust::device_reference<T>&&
get_tuple_element(hydra_thrust::tuple<hydra_thrust::device_reference<Types>...>&& t)
{
  return hydra_thrust::get<
		  find_unique_type<typename remove_device_reference<typename std::decay<T>::type>::type,
		  typename remove_device_reference<typename std::decay<Types>::type>::type...>::value>(std::move(t));
}


//hydra_thrust::detail::for tuple_of_iterator_references
template<typename T, typename... Types>
__hydra_host__ __hydra_device__
hydra_thrust::device_reference<T>
get_tuple_element(hydra_thrust::detail::tuple_of_iterator_references<
		hydra_thrust::device_reference<Types>...>& t)
{
  return hydra_thrust::get<
		  find_unique_type<typename remove_device_reference<typename std::decay<T>::type>::type,
		  typename remove_device_reference<typename std::decay<Types>::type>::type...>::value>(t);
}


template<class T, class... Types>
__hydra_host__ __hydra_device__
const hydra_thrust::device_reference<T>
get_tuple_element(const hydra_thrust::detail::tuple_of_iterator_references<
		hydra_thrust::device_reference<Types>...>& t)
{
  return hydra_thrust::get<
		  find_unique_type<typename remove_device_reference<typename std::decay<T>::type>::type,
		  typename remove_device_reference<typename std::decay<Types>::type>::type...>::value>(t);
}


template<class T, class... Types>
__hydra_host__ __hydra_device__
hydra_thrust::device_reference<T>
get_tuple_element(hydra_thrust::detail::tuple_of_iterator_references<
		hydra_thrust::device_reference<Types>...>&& t)
{
  return hydra_thrust::get<
		  find_unique_type<typename remove_device_reference<typename std::decay<T>::type>::type,
		  typename remove_device_reference<typename std::decay<Types>::type>::type...>::value>(std::move(t));
}


}  // namespace detail

}  // namespace hydra


#endif /* GETTUPLEELEMENT_H_ */
