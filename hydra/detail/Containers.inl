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
 * Containers.inl
 *
 *  Created on: 14/05/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CONTAINERS_INL_
#define CONTAINERS_INL_

#include <hydra/detail/Config.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#endif

namespace hydra
{


#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
	/*!
	 * Generic template typedef for thrust::host_vector. Use it instead of Thrust implementation
	 * in order to avoid problems to compile OpenMP based applications using gcc and without a cuda runtime installation.
	 */
	template <typename T>
		using  mc_device_vector = thrust::device_vector<T>;
	/*!
	 * Generic template typedef for thrust::host_vector. Use it instead of Thrust implementation
	 * in order to avoid problems to compile OpenMP based applications using gcc and without a cuda runtime installation.
	 * mc_host_vectot will always allocate page locked memory on CUDA backends in order to maximize speed in memory transfers
	 * to the device.
	 */
	template <typename T>
		using  mc_host_vector = thrust::host_vector<T, thrust::system::cuda::experimental::pinned_allocator<T>>;

#else
/*!
 * Generic template typedef for thrust::host_vector. Use it instead of Thrust implementation
 * in order to avoid problems to compile OpenMP based applications using gcc and without a cuda runtime installation.
 */
	template <typename T>
		using  mc_device_vector = thrust::device_vector<T>;
/*!
 * Generic template typedef for thrust::host_vector. Use it instead of Thrust implementation
 * in order to avoid problems to compile OpenMP based applications using gcc and without a cuda runtime installation.
 * mc_host_vectot will always allocate page locked memory on CUDA backends in order to maximize speed in memory transfers
 * to the device.
 */
	template <typename T>
		using  mc_host_vector   = thrust::host_vector<T>;
#endif


}  // namespace hydra

#endif /* CONTAINERS_INL_ */
