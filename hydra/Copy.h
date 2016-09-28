/*
 * Copy.h
 *
 *  Created on: 25/09/2016
 *      Author: augalves
 */

#ifndef COPY_H_
#define COPY_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Containers.h>
#include <hydra/detail/TypeTraits.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <type_traits>
#include <vector>

namespace hydra {

namespace detail {

template<template<typename...> class CONTAINER, typename T,  unsigned int BACKEND>
struct copy_type{

	typedef detail::BackendTraits<BACKEND> system_t;
	typedef typename system_t::template container<T> type;
};

}  // namespace detail

template<unsigned int BACKEND, template<typename...> class CONTAINER, typename T, typename ...Ts >
auto get_copy(CONTAINER<T, Ts...>& other )
->typename  std::enable_if<
detail::is_specialization< CONTAINER<T, Ts...>, thrust::host_vector>::value ||
detail::is_specialization<CONTAINER<T, Ts...>, thrust::device_vector >::value ||
detail::is_specialization<CONTAINER<T, Ts...>, std::vector >::value,
typename detail::copy_type<CONTAINER, T, BACKEND>::type
>::type
{
	typedef typename detail::copy_type<CONTAINER, T, BACKEND>::type vector_t;
	return 	vector_t(other);
}

}  // namespace hydra


#endif /* COPY_H_ */
