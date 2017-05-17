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
 * TBB.h
 *
 *  Created on: 16/05/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef TBB_TAG_H_
#define TBB_TAG_H_

#include <hydra/detail/Config.h>
#include <thrust/system/tbb/detail/execution_policy.h>
#include <thrust/system/tbb/vector.h>

namespace hydra {

namespace detail {

namespace tbb {

typedef thrust::system::tbb::detail::tag   tbb_tag;
static const tbb_tag    _tbb_tag_;

template<typename BACKEND>
struct IteratorPolicy;

template<>
struct IteratorPolicy<tbb_tag>: thrust::execution_policy<tbb_tag>
{
	const tbb_tag tag= _tbb_tag_;
	template<typename T>
	using   container = thrust::tbb::vector<T> ;
};

typedef IteratorPolicy<tbb_tag> tag_t;
static const tag_t tag;


}  // namespace tbb

}  // namespace detail

namespace tbb {

using hydra::detail::tbb::tag;
using hydra::detail::tbb::tag_t;

}  // namespace tbb

}  // namespace hydra

#endif /* TBB_TAG_H_ */
