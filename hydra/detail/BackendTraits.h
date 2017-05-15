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


#include <hydra/detail/Config.h>
#include <hydra/Containers.h>
#include <hydra/Backends.h>

#include <thrust/system/cuda/vector.h>
#include <thrust/system/omp/vector.h>
#include <thrust/system/tbb/vector.h>
#include <thrust/system/cpp/vector.h>

namespace hydra {

//namespace experimental {

namespace detail {


template<typename BACKEND>
struct BackendTraits;

template<>
struct BackendTraits<CUDA>: thrust::execution_policy<CUDA>
{
	const  CUDA backend= _cuda;
	template<typename T>
	using   container = thrust::cuda::vector<T> ;
};

template<>
struct BackendTraits<OMP>: thrust::execution_policy<OMP>
{
	const OMP backend= _omp;
	template<typename T>
	using   container = thrust::omp::vector<T> ;
};

template<>
struct BackendTraits<TBB>: thrust::execution_policy<TBB>
{
	const TBB backend= _tbb;
	template<typename T>
	using   container = thrust::tbb::vector<T> ;
};

template<>
struct BackendTraits<CPP>: thrust::execution_policy<CPP>
{
	const CPP backend= _cpp;
	template<typename T>
	using   container = thrust::cpp::vector<T> ;
};

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

}  // namespace detail

//}  // namespace experimental

}  // namespace hydra

#endif /* BACKENDTRAITS_H_ */
