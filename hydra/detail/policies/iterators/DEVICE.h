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
 * DEVICE.h
 *
 *  Created on: 16/05/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DEVICE_H_
#define DEVICE_H_


#include <hydra/detail/Config.h>
#include <thrust/iterator/detail/device_system_tag.h>
#include <hydra/Containers.h>

namespace hydra {

namespace detail {

namespace device {

typedef thrust::device_system_tag	       device_tag;
static const device_tag   _device_tag_;

template<typename BACKEND>
struct IteratorPolicy;

template<>
struct IteratorPolicy<device_t>: thrust::execution_policy<device_tag>
{
	const device_tag tag= _device_tag_;
	template<typename T>
	using   container = hydra::mc_device_vector<T>;
};

typedef IteratorPolicy<device_tag> tag_t;
static const tag_t tag;


}  // namespace device

}  // namespace detail

namespace device {

using hydra::detail::device::tag;
using hydra::detail::device::tag_t;

}  // namespace device

}  // namespace hydra



#endif /* DEVICE_H_ */
