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
 * SystemTraits.h
 *
 *  Created on: Jul 20, 2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SYSTEMTRAITS_H_
#define SYSTEMTRAITS_H_

#include <hydra/cuda/System.h>
#include <hydra/cpp/System.h>
#include <hydra/omp/System.h>
#include <hydra/tbb/System.h>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/detail/type_traits.h>
#include <thrust/system/cpp/detail/par.h>
#include <thrust/system/cuda/detail/par.h>
#include <thrust/system/omp/detail/par.h>
#include <thrust/system/tbb/detail/par.h>

namespace hydra {

namespace detail {

template<typename ThrustSystem>
struct SystemTraits;

template<>
struct SystemTraits<thrust::system::cpp::detail::tag>
{ typedef hydra::cpp::sys_t policy; };

template<>
struct SystemTraits<thrust::system::omp::detail::tag>
{ typedef hydra::omp::sys_t policy; };

template<>
struct SystemTraits<thrust::system::tbb::detail::tag>
{ typedef hydra::tbb::sys_t policy; };

template<>
struct SystemTraits<thrust::system::cuda::detail::tag>
{ typedef hydra::cuda::sys_t policy; };



}  // namespace detail

}//namespace hydra

#endif /* SYSTEMTRAITS_H_ */
