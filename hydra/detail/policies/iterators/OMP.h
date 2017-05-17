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
 * OMP.h
 *
 *  Created on: 16/05/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef OMP_TAG_H_
#define OMP_TAG_H_

#include <hydra/detail/Config.h>
#include <thrust/system/omp/detail/execution_policy.h>
#include <thrust/system/omp/vector.h>

namespace hydra {

namespace detail {

namespace omp {

typedef thrust::system::omp::detail::tag   omp_tag;
static const omp_tag    _omp_tag_;

template<typename BACKEND>
struct IteratorPolicy;

template<>
struct IteratorPolicy<omp_tag>: thrust::execution_policy<omp_tag>
{
	const omp_tag tag= _omp_tag_;
	template<typename T>
	using   container = thrust::omp::vector<T> ;
};

typedef IteratorPolicy<omp_tag> tag_t;
static const tag_t tag;


}  // namespace omp

}  // namespace detail

namespace omp {

using hydra::detail::omp::tag;
using hydra::detail::omp::tag_t;

}  // namespace omp

}  // namespace hydra

#endif /* OMP_TAG_H_ */
