/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2023 Antonio Augusto Alves Junior
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
 * HOST.h
 *
 *  Created on: 16/05/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef HOST_H_
#define HOST_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/detail/external/hydra_thrust/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/host_vector.h>
#if HYDRA_THRUST_DEVICE_SYSTEM==HYDRA_THRUST_DEVICE_SYSTEM_CUDA
#include <hydra/detail/external/hydra_thrust/system/cuda/memory_resource.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/memory.h>
#endif

namespace hydra {

namespace detail {

namespace host {

typedef hydra_thrust::detail::host_t	            host_t;
static const host_t   _host_;

}  // namespace host


template<>
struct BackendPolicy<Backend::Host>: hydra_thrust::host_execution_policy<host::host_t>
{
	typedef hydra_thrust::host_execution_policy<host::host_t> execution_policy_type;

	const host::host_t backend= host::_host_;

#if HYDRA_THRUST_DEVICE_SYSTEM==HYDRA_THRUST_DEVICE_SYSTEM_CUDA
	template<typename T>
	using   container = hydra_thrust::host_vector<T ,
			hydra_thrust::mr::stateless_resource_allocator<T, hydra_thrust::system::cuda::universal_host_pinned_memory_resource>>;
#else
	template<typename T>
	using   container = hydra_thrust::host_vector<T>;
#endif
};

}  // namespace detail

namespace host {


typedef hydra::detail::BackendPolicy<hydra::detail::Backend::Host> sys_t;

template<typename T>
using   vector = hydra::detail::BackendPolicy<hydra::detail::Backend::Host>::container<T>;

static const sys_t sys=sys_t();

}  // namespace host
}  // namespace hydra


#endif /* HOST_H_ */
