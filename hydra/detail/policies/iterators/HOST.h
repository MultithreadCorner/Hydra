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

#ifndef HOST_TAG_H_
#define HOST_TAG_H_

#include <hydra/detail/Config.h>
#include <thrust/iterator/detail/host_system_tag.h>
#include <hydra/Containers.h>

namespace hydra {

namespace detail {

namespace host {

typedef thrust::host_system_tag	       host_tag;
static const host_tag   _host_tag_;

template<typename BACKEND>
struct IteratorPolicy;

template<>
struct IteratorPolicy<host_t>: thrust::execution_policy<host_t>
{
	const host_tag tag= _host_tag_;
	template<typename T>
	using   container = hydra::mc_host_vector<T>;
};

typedef IteratorPolicy<host_t> tag_t;
static const tag_t tag;


}  // namespace host

}  // namespace detail

namespace host {

using hydra::detail::host::tag;
using hydra::detail::host::tag_t;

}  // namespace host

}  // namespace hydra


#endif /* HOST_TAG_H_ */
