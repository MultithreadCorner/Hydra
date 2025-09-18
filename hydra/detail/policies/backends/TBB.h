/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2025 Antonio Augusto Alves Junior
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
 * TBB.h
 *
 *  Created on: 16/05/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef TBB_H_
#define TBB_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/detail/external/hydra_thrust/system/tbb/detail/par.h>
#include <hydra/detail/external/hydra_thrust/system/tbb/vector.h>

namespace hydra {

namespace detail {

namespace tbb {

typedef hydra::thrust::system::tbb::detail::par_t   tbb_t;
static const tbb_t    _tbb_;


}  // namespace tbb

template<>
struct BackendPolicy<Backend::Tbb>: hydra::thrust::execution_policy<tbb::tbb_t>
{
	typedef  hydra::thrust::execution_policy<tbb::tbb_t> execution_policy_type;

	const tbb::tbb_t backend= tbb::_tbb_;

	template<typename T>
	using   container = hydra::thrust::tbb::vector<T> ;

};

}  // namespace detail

namespace tbb {

typedef hydra::detail::BackendPolicy<hydra::detail::Backend::Tbb> sys_t;

template<typename T>
using   vector = hydra::detail::BackendPolicy<hydra::detail::Backend::Tbb>::container<T> ;

static const sys_t sys=sys_t();

}  // namespace tbb

}  // namespace hydra




#endif /* TBB_H_ */
