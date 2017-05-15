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

#include <hydra/detail/Config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/execution_policy.h>

#if _ENABLE_TBB
#include <thrust/system/tbb/detail/par.h>
#endif

#if _ENABLE_OMP
#include <thrust/system/omp/detail/par.h>
#endif

#if _ENABLE_CPP
#include <thrust/system/cpp/detail/par.h>
#endif

#if _ENABLE_CUDA
#include <thrust/system/cuda/detail/par.h>
#endif

namespace hydra {

#if _ENABLE_CPP
 typedef thrust::system::cpp::detail::par_t   CPP;
 static const CPP    _cpp;
#endif

#if _ENABLE_CUDA
 	typedef thrust::system::cuda::detail::par_t CUDA;
 	static const CUDA   _cuda;
#endif

#if _ENABLE_OMP
	typedef thrust::system::omp::detail::par_t   OMP;
	static const OMP    _omp;
#endif

#if _ENABLE_TBB
	typedef thrust::system::tbb::detail::par_t   TBB;
	static const TBB    _tbb;
#endif


typedef thrust::detail::device_t		  DEVICE;
typedef thrust::detail::host_t	            HOST;
static const DEVICE _device;
static const HOST   _host;



}  // namespace hydra



#endif /* BACKENDS_H_ */
