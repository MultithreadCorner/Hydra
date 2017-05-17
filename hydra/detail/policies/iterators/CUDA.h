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
 * CUDA.h
 *
 *  Created on: 16/05/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CUDA_H_
#define CUDA_H_


#include <hydra/detail/Config.h>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/system/cuda/vector.h>

namespace hydra {

namespace detail {

namespace cuda {

typedef thrust::system::cuda::detail::tag   cuda_tag;
static const cuda_tag    _cuda_tag_;

template<typename BACKEND>
struct IteratorPolicy;

template<>
struct IteratorPolicy<cuda_tag>: thrust::execution_policy<cuda_tag>
{
	const cuda_tag tag= _cuda_tag_;
	template<typename T>
	using   container = thrust::cuda::vector<T> ;
};

typedef IteratorPolicy<cuda_tag> tag_t;
static const tag_t tag;


}  // namespace cuda

}  // namespace detail

namespace cuda {

using hydra::detail::cuda::tag;
using hydra::detail::cuda::tag_t;

}  // namespace cuda

}  // namespace hydra

#endif /* CUDA_H_ */
