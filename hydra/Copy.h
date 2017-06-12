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
 * Copy.h
 *
 *  Created on: 25/09/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup generic
 */

#ifndef COPY_H_
#define COPY_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Containers.h>
#include <hydra/detail/TypeTraits.h>
#include <hydra/multivector.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <type_traits>
#include <vector>
#include <utility>

namespace hydra {

namespace detail {

template<template<typename...> class CONTAINER, typename T, hydra::detail::Backend BACKEND>
struct copy_type{
	typedef hydra::detail::BackendPolicy<BACKEND> system_t;
	typedef typename system_t::template container<T> type;
};

}  // namespace detail

/**
 * Generic copy between backends, abstracting away details of the copied container.
 *
 * @param policy: corresponding to the destination back-end.
 * @param other : original container.
 * @return
 */
template<hydra::detail::Backend BACKEND, template<typename...> class CONTAINER, typename T, typename ...Ts >
auto get_copy(hydra::detail::BackendPolicy<BACKEND>const& policy , CONTAINER<T, Ts...>& other )
->typename  std::enable_if<
detail::is_specialization< CONTAINER<T, Ts...>, thrust::host_vector>::value ||
detail::is_specialization<CONTAINER<T, Ts...>, thrust::device_vector >::value ||
detail::is_specialization<CONTAINER<T, Ts...>, std::vector >::value,
typename detail::copy_type<CONTAINER, T, BACKEND>::type
>::type
{
	typedef typename detail::copy_type<CONTAINER, T, BACKEND>::type vector_t;
	return 	std::move(vector_t(other));
}

/**
 * Generic copy between backends, abstracting away details of the copied container.
 *
 * @param policy: corresponding to the destination back-end.
 * @param other : original container.
 * @return
 *
 * obs: overload for hydra::multivector
 */
template<hydra::detail::Backend BACKEND, template<typename...> class CONTAINER, typename T>
auto get_copy(hydra::detail::BackendPolicy<BACKEND> const& policy, CONTAINER<T>& other )
->typename  std::enable_if<
detail::is_specialization< CONTAINER<T> ,multivector>::value,
multivector<typename
hydra::detail::BackendPolicy<BACKEND>::template container<typename CONTAINER<T>::value_tuple_type> > >::type

{
	typedef hydra::detail::BackendPolicy<BACKEND> system_t;

	typedef typename  multivector<typename
			system_t::template container<typename CONTAINER<T>::value_tuple_type> > vector_t;
	return 	std::move(vector_t(other));
}



}  // namespace hydra


#endif /* COPY_H_ */
