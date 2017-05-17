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
 * CPP.h
 *
 *  Created on: 16/05/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CPP_TAG_H_
#define CPP_TAG_H_

#include <hydra/detail/Config.h>
#include <thrust/system/cpp/detail/execution_policy.h>
#include <thrust/system/cpp/vector.h>

namespace hydra {

namespace detail {

namespace cpp {

typedef thrust::system::cpp::detail::tag   cpp_tag;
static const cpp_tag    _cpp_tag_;

template<typename BACKEND>
struct IteratorPolicy;

template<>
struct IteratorPolicy<cpp_tag>: thrust::execution_policy<cpp_tag>
{
	const cpp_tag tag= _cpp_tag_;
	template<typename T>
	using   container = thrust::cpp::vector<T> ;
};

typedef IteratorPolicy<cpp_tag> tag_t;
static const tag_t tag;


}  // namespace cpp


}  // namespace detail

namespace cpp {

using hydra::detail::cpp::tag;
using hydra::detail::cpp::tag_t;


}  // namespace cpp

}  // namespace hydra


#endif /* CPP_TAG_H_ */
