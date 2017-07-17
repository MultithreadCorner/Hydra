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

#ifndef CPP_H_
#define CPP_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <thrust/system/cpp/detail/par.h>
#include <thrust/system/cpp/vector.h>

namespace hydra {

namespace detail {

namespace cpp {

typedef thrust::system::cpp::detail::par_t   cpp_t;
static const cpp_t    _cpp_;

}  // namespace cpp

template<>
struct BackendPolicy<Backend::Cpp>: thrust::execution_policy<cpp::cpp_t>
{
	const cpp::cpp_t backend= cpp::_cpp_;

	template<typename T>
	using   container = thrust::cpp::vector<T> ;


};


}  // namespace detail

namespace cpp {

typedef hydra::detail::BackendPolicy<hydra::detail::Backend::Cpp> sys_t;

template<typename T>
using   vector = hydra::detail::BackendPolicy<hydra::detail::Backend::Cpp>::container<T> ;

static const sys_t sys= sys_t();

}  // namespace cpp

}  // namespace hydra


#endif /* CPP_H_ */
