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
 * Config.h
 *
 *  Created on: Feb 24, 2016
 *      Author: Antonio Augusto Alves Junior
 */



#ifndef CONFIG_H_
#define CONFIG_H_

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <thrust/detail/config.h>
#include <thrust/detail/config/host_device.h>



#if defined(__CUDACC__) && !(defined(__CUDA__) && defined(__clang__))

#define __hydra_exec_check_disable__ #pragma nv_exec_check_disable

#else

#define __hydra_exec_check_disable__

#endif

#if defined(__CUDACC__)
#define __hydra_align__(n) __align__(n)
#else
  #define       __hydra_align__(n) __attribute__((aligned(n)))
#endif

#ifdef __CUDACC__
 #include <cuda.h>
 #include <cuda_runtime.h>
 #include <cuda_runtime_api.h>
 #include <math_functions.h>
 #include <vector_functions.h>
#endif


#ifndef HYDRA_CERROR_LOG
#define HYDRA_OS std::cerr
#else
#define HYDRA_OS HYDRA_CERROR_LOG
#endif


 #define _ENABLE_CPP 1

#if defined(HYDRA_USING_TBB) || THRUST_HOST_SYSTEM == THRUST_HOST_TBB || THRUST_DEVICE_SYSTEM == THRUST_DEVICE_TBB
 #define _ENABLE_TBB 1
#else
 #define _ENABLE_TBB 0
#endif

#if defined(HYDRA_USING_CUDA) || defined(__CUDACC__)  || THRUST_DEVICE_SYSTEM == THRUST_DEVICE_CUDA
 #define _ENABLE_CUDA 1
#else
 #define _ENABLE_CUDA 0
#endif

#if defined(HYDRA_USING_OMP)  || THRUST_HOST_SYSTEM == THRUST_HOST_OMP || THRUST_DEVICE_SYSTEM == THRUST_DEVICE_OMP
 #define _ENABLE_OMP 1
#else
 #define _ENABLE_OMP 0
#endif





#endif /* CUDA_H_ */
