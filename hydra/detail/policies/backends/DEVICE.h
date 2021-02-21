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
 * DEVICE.h
 *
 *  Created on: 16/05/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DEVICE_H_
#define DEVICE_H_


#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/detail/external/hydra_thrust/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/device_vector.h>


namespace hydra {

namespace detail {

namespace device {

typedef hydra_thrust::detail::device_t	            device_t;
static const device_t   _device_;

}  // namespace device

template<>
struct BackendPolicy<Backend::Device>: hydra_thrust::device_execution_policy<device::device_t>
{
	typedef hydra_thrust::device_execution_policy<device::device_t> execution_policy_type;

	const device::device_t backend= device::_device_;

	template<typename T>
	using   container = hydra_thrust::device_vector<T>;

};


}  // namespace detail

namespace device {

typedef hydra::detail::BackendPolicy<hydra::detail::Backend::Device> sys_t;

template<typename T>
using   vector = hydra::detail::BackendPolicy<hydra::detail::Backend::Device>::container<T>;

static const sys_t sys=sys_t();

}  // namespace device

}  // namespace hydra



#endif /* DEVICE_H_ */
