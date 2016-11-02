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
#include <type_traits>
#include <functional>
#include <hydra/Function.h>
#include <typeinfo>

namespace hydra {


template<typename Sig,typename L,
typename=typename std::enable_if<std::is_constructible<std::function<Sig>, L>::value>::type>
struct LambdaWrapper;

template<typename ReturnType, typename ArgType, typename L>
struct LambdaWrapper<ReturnType(ArgType), L>:
public BaseFunctor<LambdaWrapper<ReturnType(ArgType),L>, ReturnType,0 >
{
	LambdaWrapper();

	LambdaWrapper(L const& lambda):
		fLambda(lambda){}

	__host__ __device__	 inline
	LambdaWrapper(LambdaWrapper<ReturnType(ArgType), L> const& other ):
	BaseFunctor<LambdaWrapper<ReturnType(ArgType),L>, ReturnType,0>(other),
	fLambda( other.GetLambda())
	{	}

	__host__ __device__	 inline
	L GetLambda() const {return fLambda; }


	__host__ __device__ inline
	ReturnType  Evaluate(ArgType  a) { return fLambda(a); }

private:
	L fLambda;
};

namespace detail {


template<typename L, typename ReturnType, typename ...Args>
auto wrap_lambda_helper(L const& f, ReturnType r, thrust::tuple<Args...>const& t)
-> LambdaWrapper<ReturnType(Args...), L>
{
	return LambdaWrapper<ReturnType(Args...), L>(f);
}

}  // namespace detail


template<typename L>
auto wrap_lambda(L const& f)
-> decltype(detail::wrap_lambda_helper(f, typename detail::function_traits<L>::return_type() ,
		typename detail::function_traits<L>::args_type()) )
{
	typedef detail::function_traits<L> traits;
	typename traits::return_type r = typename traits::return_type();
	typename traits::args_type t;
	//static_assert(traits::args_type::dummy , "<<<+++++++++++++++++++");
	return detail::wrap_lambda_helper(f, r, t);
}



}

#endif /* FUNCTIONWRAPPER_H_ */
