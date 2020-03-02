/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2020 Antonio Augusto Alves Junior
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
 * FunctorTraits.h
 *
 *  Created on: 09/02/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef FUNCTORTRAITS_H_
#define FUNCTORTRAITS_H_

#include <hydra/detail/Config.h>
#include<hydra/detail/utility/StaticAssert.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include<type_traits>

namespace hydra {

namespace detail {

template<typename T>
struct signature_type_impl
{
	typedef T type;
};

template<typename R, typename ...A>
struct signature_type
{
 typedef typename signature_type_impl<R(A...)>::type type;
};

template<typename S>
struct signature_traits;

template<typename R, typename...Args>
struct signature_traits<R(Args...)>
{
typedef typename std::decay<R>::type return_type;
typedef std::tuple<typename std::decay<Args>::type...> argument_type;
enum {arity=sizeof...(Args)};
};


template <typename T>
struct functor_traits:
		public functor_traits<decltype(&T::Evaluate)>{};

template <typename T>
struct lambda_traits:
		public functor_traits<decltype(&T::operator())>{};

template <typename ClassType, typename ReturnType, typename... Args>
struct functor_traits<ReturnType(ClassType::*)(Args...) >
{
	HYDRA_STATIC_ASSERT(int(sizeof...(Args))==-1,
			"Non-const T::operator() or T::Evaluate() not allowed." )
};

template <typename ClassType, typename ReturnType, typename... Args>
struct functor_traits<ReturnType(ClassType::*)(Args...) const>
{
	enum { arity = sizeof...(Args) };

	typedef ReturnType return_type;

	typedef hydra_thrust::tuple<Args...> argument_type;

	typedef hydra_thrust::tuple<
			typename std::decay<Args>::type...> argument_rvalue_type;

};


}  // namespace detail

}  // namespace hydra


#endif /* FUNCTORTRAITS_H_ */
