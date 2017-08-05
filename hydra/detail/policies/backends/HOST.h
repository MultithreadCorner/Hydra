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
 * HOST.h
 *
 *  Created on: 16/05/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef HOST_H_
#define HOST_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <thrust/execution_policy.h>
#include <hydra/Containers.h>

namespace hydra {

namespace detail {

namespace host {

typedef thrust::detail::host_t	            host_t;
static const host_t   _host_;

}  // namespace host


template<>
struct BackendPolicy<Backend::Host>: thrust::execution_policy<host::host_t>
{
	const host::host_t backend= host::_host_;

	template<typename T>
	using   container = hydra::mc_host_vector<T>;

};

}  // namespace detail

namespace host {


typedef hydra::detail::BackendPolicy<hydra::detail::Backend::Host> sys_t;

template<typename T>
using   vector = hydra::detail::BackendPolicy<hydra::detail::Backend::Host>::container<T>;

static const sys_t sys=sys_t();

}  // namespace host

}  // namespace hydra


#endif /* HOST_H_ */
