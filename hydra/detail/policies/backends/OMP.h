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
 * OMP.h
 *
 *  Created on: 16/05/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef OMP_H_
#define OMP_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/detail/external/thrust/system/omp/detail/par.h>
#include <hydra/detail/external/thrust/system/omp/vector.h>

namespace hydra {

namespace detail {

namespace omp {

typedef HYDRA_EXTERNAL_NS::thrust::system::omp::detail::par_t   omp_t;
static const omp_t    _omp_;

}  // namespace omp

template<>
struct BackendPolicy<Backend::Omp>: HYDRA_EXTERNAL_NS::thrust::execution_policy<omp::omp_t>
{
	//typedef  HYDRA_EXTERNAL_NS::thrust::execution_policy<omp::omp_t> super_type;
	const omp::omp_t backend= omp::_omp_;

	template<typename T>
	using   container = HYDRA_EXTERNAL_NS::thrust::omp::vector<T> ;

};

}  // namespace detail

namespace omp {

typedef hydra::detail::BackendPolicy<hydra::detail::Backend::Omp> sys_t;

template<typename T>
using   vector = hydra::detail::BackendPolicy<hydra::detail::Backend::Omp>::container<T> ;

static const sys_t sys=sys_t();

}  // namespace omp
}  // namespace hydra


#endif /* OMP_H_ */
