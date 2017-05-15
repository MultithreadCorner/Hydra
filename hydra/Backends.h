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
 * Backends.h
 *
 *  Created on: 14/05/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef BACKENDS_H_
#define BACKENDS_H_

#include <thrust/detail/type_traits.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cpp/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/system/tbb/execution_policy.h>

namespace hydra {

//namespace experimental {

//namespace detail {

typedef thrust::system::cuda::detail::par_t CUDA;
typedef thrust::system::cpp::detail::par_t   CPP;
typedef thrust::system::omp::detail::par_t   OMP;
typedef thrust::system::tbb::detail::par_t   TBB;
typedef thrust::detail::device_t		  DEVICE;
typedef thrust::detail::host_t	            HOST;

//}  // namespace detail


static const CUDA   _cuda;
static const CPP    _cpp;
static const OMP    _omp;
static const TBB    _tbb;
static const DEVICE _device;
static const HOST   _host;

//}  // namespace experimental

}  // namespace hydra



#endif /* BACKENDS_H_ */
