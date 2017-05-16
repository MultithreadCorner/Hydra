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
 * BackendTraits.h
 *
 *  Created on: 14/05/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef BACKENDTRAITS_H_
#define BACKENDTRAITS_H_

/*
#include <hydra/detail/Config.h>
#include <hydra/Containers.h>
#include <hydra/Backends.h>

#if _ENABLE_TBB
#include <thrust/system/tbb/vector.h>
#endif

#if  _ENABLE_OMP
#include <thrust/system/omp/vector.h>
#endif

#if _ENABLE_CPP
#include <thrust/system/cpp/vector.h>
#endif

#if _ENABLE_CUDA
#include <thrust/system/cuda/vector.h>
#endif


namespace hydra {

//namespace experimental {

namespace detail {


template<typename BACKEND>
struct BackendTraits;

#if _ENABLE_TBB
template<>
struct BackendTraits<TBB>: thrust::execution_policy<TBB>
{
	const TBB backend= _tbb;
	template<typename T>
	using   container = thrust::tbb::vector<T> ;
};

#endif

#if  _ENABLE_OMP
template<>
struct BackendTraits<hydra::OMP>: thrust::execution_policy<hydra::OMP>
{
	const OMP backend= _omp;
	template<typename T>
	using   container = thrust::omp::vector<T> ;
};
#endif

#if _ENABLE_CPP
template<>
struct BackendTraits<hydra::CPP>: thrust::execution_policy<hydra::CPP>
{
	const CPP backend= _cpp;
	template<typename T>
	using   container = thrust::cpp::vector<T> ;
};
#endif

#if _ENABLE_CUDA
template<>
struct BackendTraits<CUDA>: thrust::execution_policy<CUDA>
{
	const  CUDA backend= _cuda;
	template<typename T>
	using   container = thrust::cuda::vector<T> ;
};
#endif



*/



/*
template<>
struct _BackendTraits<HOST>: thrust::execution_policy<HOST>
{
	const HOST backend= _host;
	template<typename T>
	using   container = hydra::mc_host_vector<T>;
};

template<>
struct _BackendTraits<DEVICE>: thrust::execution_policy<DEVICE>
{
	const DEVICE backend= _device;
	template<typename T>
	using   container = hydra::mc_device_vector<T>;
};
*/
/*
}  // namespace detail

//}  // namespace experimental

}  // namespace hydra*/

#endif /* BACKENDTRAITS_H_ */
