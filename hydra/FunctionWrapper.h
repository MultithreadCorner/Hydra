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
 * FunctionWrapper.h
 *
 *  Created on: 12/07/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup functor
 */


#ifndef FUNCTIONWRAPPER_H_
#define FUNCTIONWRAPPER_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Parameter.h>
#include <type_traits>
#include <functional>
#include <hydra/Function.h>
#include <typeinfo>
#include <initializer_list>
#include <array>
namespace hydra {


template<typename Sig,typename L, size_t N,
typename=typename std::enable_if<std::is_constructible<std::function<Sig>, L>::value>::type>
struct LambdaWrapper;

template<typename ReturnType, typename ...ArgType, typename L, size_t  N>
struct LambdaWrapper<ReturnType(ArgType...), L, N>:
public BaseFunctor<LambdaWrapper<ReturnType(ArgType...),L, N>, ReturnType,N >
{
	LambdaWrapper()=delete;

	LambdaWrapper(L const& lambda):
			BaseFunctor<LambdaWrapper<ReturnType(ArgType...),L, 0>, ReturnType,0 >(),
			fLambda(lambda)
		{}

	LambdaWrapper(L const& lambda,
			std::array<Parameter, N> const& parameters):
		BaseFunctor<LambdaWrapper<ReturnType(ArgType...),L, N>, ReturnType,N >(parameters),
		fLambda(lambda)
	{}

	__host__ __device__	 inline
	LambdaWrapper(LambdaWrapper<ReturnType(ArgType...), L, N> const& other ):
	BaseFunctor<LambdaWrapper<ReturnType(ArgType...),L, N>, ReturnType,N>(other),
	fLambda( other.GetLambda())
	{	}

	__host__ __device__	 inline
	LambdaWrapper<ReturnType(ArgType...), L, N>
	operator=(LambdaWrapper<ReturnType(ArgType...), L, N> const& other )
	{
		if(this==&other) return *this;
		BaseFunctor<LambdaWrapper<ReturnType(ArgType...),L, N>, ReturnType,N>::operator=(other);

		this->fLambda = other.GetLambda();

		return *this;
	}


	__host__ __device__	 inline
	L GetLambda() const {return fLambda; }


	template<typename T, size_t M=N>
	__host__ __device__ inline
	typename std::enable_if< (M>0), ReturnType >::type
	Evaluate(T a) {

		return fLambda(this->GetNumberOfParameters(), this->GetParameters(), a);
	}


	template<typename T, size_t M=N>
	__host__ __device__ inline
	typename std::enable_if< (M==0), ReturnType >::type
	Evaluate(T a) {

		return fLambda( a);
	}



private:
	L fLambda;
};

namespace detail {


template<typename L, typename ReturnType, typename ...Args, size_t N>
auto wrap_lambda_helper(L const& f, ReturnType r, thrust::tuple<Args...>const& t,
		std::array<Parameter, N> const& parameters)
-> LambdaWrapper<ReturnType(Args...), L, N>
{
	return LambdaWrapper<ReturnType(Args...), L, N>(f, parameters);
}

template<typename L, typename ReturnType, typename ...Args>
auto wrap_lambda_helper(L const& f, ReturnType r, thrust::tuple<Args...>const& t)
-> LambdaWrapper<ReturnType(Args...), L, 0>
{
	return LambdaWrapper<ReturnType(Args...), L, 0>(f);
}

}  // namespace detail


template<typename L, typename ...T>
auto wrap_lambda(L const& f,  T ...pars)
-> decltype(detail::wrap_lambda_helper(f, typename detail::function_traits<L>::return_type() ,
		typename detail::function_traits<L>::args_type(),  std::array<Parameter, sizeof...(T)>{}))
{
	typedef detail::function_traits<L> traits;
	typename traits::return_type r = typename traits::return_type();
	typename traits::args_type t;
	//static_assert(traits::args_type::dummy , "<<<+++++++++++++++++++");
	std::array<Parameter, sizeof...(T)> parameters{ pars...};

	return detail::wrap_lambda_helper(f, r, t, parameters);
}


template<typename L>
auto wrap_lambda(L const& f)
-> decltype(detail::wrap_lambda_helper(f, typename detail::function_traits<L>::return_type() ,
		typename detail::function_traits<L>::args_type()))
{
	typedef detail::function_traits<L> traits;
	typename traits::return_type r = typename traits::return_type();
	typename traits::args_type t;
	//static_assert(traits::args_type::dummy , "<<<+++++++++++++++++++");

	return detail::wrap_lambda_helper(f, r, t);
}



}

#endif /* FUNCTIONWRAPPER_H_ */
