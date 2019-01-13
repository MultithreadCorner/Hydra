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
 * Config.h
 *
 *  Created on: Feb 24, 2016
 *      Author: Antonio Augusto Alves Junior
 */



#ifndef CONFIG_H_
#define CONFIG_H_

#define CUDA 1 // device
#define OMP  2 // device + host
#define TBB  3 // device + host
#define CPP  4 // device + host



#if(__cplusplus < 201103L)
#error "[Hydra]:  Hydra requires at least a C++11 compliant compiler."
#endif


#ifndef HYDRA_HOST_SYSTEM

#define HYDRA_THRUST_HOST_SYSTEM HYDRA_THRUST_HOST_SYSTEM_CPP

#elif (HYDRA_HOST_SYSTEM!=CPP && HYDRA_HOST_SYSTEM!=OMP && HYDRA_HOST_SYSTEM!=TBB)

#error "[Hydra]: Backend not supported. Supported host backends are {CPP, OMP, TBB}."

#endif


#ifndef HYDRA_DEVICE_SYSTEM

#define HYDRA_THRUST_DEVICE_SYSTEM HYDRA_THRUST_DEVICE_SYSTEM_CPP

#elif(HYDRA_DEVICE_SYSTEM!=CPP && HYDRA_DEVICE_SYSTEM!=OMP && HYDRA_DEVICE_SYSTEM!=TBB  && HYDRA_DEVICE_SYSTEM!=CUDA )

#error "[Hydra]: Backend not supported. Supported device backends are {CPP, OMP, TBB, CUDA}"

#endif



//host definition
#if (HYDRA_HOST_SYSTEM==CPP)
	#define HYDRA_THRUST_HOST_SYSTEM HYDRA_THRUST_HOST_SYSTEM_CPP
#elif (HYDRA_HOST_SYSTEM==OMP)
	#define HYDRA_THRUST_HOST_SYSTEM HYDRA_THRUST_HOST_SYSTEM_OMP
#elif (HYDRA_HOST_SYSTEM==TBB)
	#define HYDRA_THRUST_HOST_SYSTEM HYDRA_THRUST_HOST_SYSTEM_TBB
#endif


//device definition
#if   (HYDRA_DEVICE_SYSTEM==CPP)
	#define HYDRA_THRUST_DEVICE_SYSTEM HYDRA_THRUST_DEVICE_SYSTEM_CPP
#elif (HYDRA_DEVICE_SYSTEM==OMP)
	#define HYDRA_THRUST_DEVICE_SYSTEM HYDRA_THRUST_DEVICE_SYSTEM_OMP
#elif (HYDRA_DEVICE_SYSTEM==TBB)
	#define HYDRA_THRUST_DEVICE_SYSTEM HYDRA_THRUST_DEVICE_SYSTEM_TBB
#elif (HYDRA_DEVICE_SYSTEM==CUDA)
	#define HYDRA_THRUST_DEVICE_SYSTEM HYDRA_THRUST_DEVICE_SYSTEM_CUDA
	#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#endif



#include <hydra/detail/external/thrust/detail/config.h>
#include <hydra/detail/external/thrust/detail/config/host_device.h>



#define HYDRA_THRUST_VARIADIC_TUPLE

#define __hydra_exec_check_disable__  __thrust_exec_check_disable__


#if defined(__CUDACC__)
#define __hydra_align__(n) __align__(n)
#else
  #define  __hydra_align__(n) __attribute__((aligned(n)))
#endif

#ifdef __NVCC__
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif


#ifndef HYDRA_CERROR_LOG
#define HYDRA_OS std::cerr
#else
#define HYDRA_OS HYDRA_CERROR_LOG
#endif


#endif /* CUDA_H_ */
