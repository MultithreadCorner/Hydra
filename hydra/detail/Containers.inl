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
 * Containers.inl
 *
 *  Created on: 14/05/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CONTAINERS_INL_
#define CONTAINERS_INL_

#include <hydra/detail/Config.h>
#include <hydra/detail/external/thrust/device_vector.h>
#include <hydra/detail/external/thrust/host_vector.h>
#include <hydra/detail/external/thrust/complex.h>
#if HYDRA_THRUST_DEVICE_SYSTEM==HYDRA_THRUST_DEVICE_SYSTEM_CUDA
#include <hydra/detail/external/thrust/system/cuda/experimental/pinned_allocator.h>
#endif

namespace hydra
{


#if HYDRA_THRUST_DEVICE_SYSTEM==HYDRA_THRUST_DEVICE_SYSTEM_CUDA
	/*!
	 * Generic template typedef for HYDRA_EXTERNAL_NS::thrust::host_vector. Use it instead of Thrust implementation
	 * in order to avoid problems to compile OpenMP based applications using gcc and without a cuda runtime installation.
	 */
	template <typename T>
		using  mc_device_vector = HYDRA_EXTERNAL_NS::thrust::device_vector<T>;
	/*!
	 * Generic template typedef for HYDRA_EXTERNAL_NS::thrust::host_vector. Use it instead of Thrust implementation
	 * in order to avoid problems to compile OpenMP based applications using gcc and without a cuda runtime installation.
	 * mc_host_vectot will always allocate page locked memory on CUDA backends in order to maximize speed in memory transfers
	 * to the device.
	 */
	template <typename T>
		using  mc_host_vector = HYDRA_EXTERNAL_NS::thrust::host_vector<T, HYDRA_EXTERNAL_NS::thrust::system::cuda::experimental::pinned_allocator<T>>;

#else
/*!
 * Generic template typedef for HYDRA_EXTERNAL_NS::thrust::host_vector. Use it instead of Thrust implementation
 * in order to avoid problems to compile OpenMP based applications using gcc and without a cuda runtime installation.
 */
	template <typename T>
		using  mc_device_vector = HYDRA_EXTERNAL_NS::thrust::device_vector<T>;
/*!
 * Generic template typedef for HYDRA_EXTERNAL_NS::thrust::host_vector. Use it instead of Thrust implementation
 * in order to avoid problems to compile OpenMP based applications using gcc and without a cuda runtime installation.
 * mc_host_vectot will always allocate page locked memory on CUDA backends in order to maximize speed in memory transfers
 * to the device.
 */
	template <typename T>
		using  mc_host_vector   = HYDRA_EXTERNAL_NS::thrust::host_vector<T>;
#endif


}  // namespace hydra

#endif /* CONTAINERS_INL_ */
