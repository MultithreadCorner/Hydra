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
 * CUDA.h
 *
 *  Created on: 16/05/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CUDA_H_
#define CUDA_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/detail/external/thrust/system/cuda/detail/par.h>
#include <hydra/detail/external/thrust/system/cuda/vector.h>

namespace hydra {

namespace detail {

namespace cuda {

typedef HYDRA_EXTERNAL_NS::thrust::system::cuda::detail::par_t   cuda_t;
static const cuda_t    _cuda_;

}  // namespace cuda

template<>
struct BackendPolicy<Backend::Cuda>: HYDRA_EXTERNAL_NS::thrust::execution_policy<cuda::cuda_t>
{
	//typedef HYDRA_EXTERNAL_NS::thrust::execution_policy<cuda::cuda_t> super_type;
	const cuda::cuda_t backend= cuda::_cuda_;

	template<typename T>
	using   container = HYDRA_EXTERNAL_NS::thrust::cuda::vector<T> ;

};


}  // namespace detail

namespace cuda {

typedef hydra::detail::BackendPolicy<hydra::detail::Backend::Cuda> sys_t;

template<typename T>
using   vector = hydra::detail::BackendPolicy<hydra::detail::Backend::Cuda>::container<T> ;

static const sys_t sys=sys_t();

}  // namespace cuda


}  // namespace hydra

#endif /* CUDA_H_ */
