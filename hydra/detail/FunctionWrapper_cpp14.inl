/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2018 Antonio Augusto Alves Junior
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
 * FunctionWrapper_cpp14.inl
 *
 *  Created on: 06/12/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef FUNCTIONWRAPPER_CPP14_INL_
#define FUNCTIONWRAPPER_CPP14_INL_


#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Parameter.h>
#include <hydra/detail/utility/Generic.h>
#include <type_traits>
#include <functional>
#include <hydra/Function.h>
#include <typeinfo>
#include <initializer_list>
#include <array>

namespace hydra {


template<typename Lambda, typename ReturnType, size_t N>
class LambdaWrapper:public BaseFunctor<LambdaWrapper<Lambda, ReturnType, N>, ReturnType,N >
{

public:
	LambdaWrapper()=delete;

	/**
	 * Constructor for parametrized lambdas
	 * @param lambda
	 * @param parameters
	 */
	LambdaWrapper(L const& lambda,	std::array<Parameter, N> const& parameters):
				BaseFunctor<LambdaWrapper<Lambda, ReturnType, N>, ReturnType,N >(parameters),
		fLambda(lambda)
	{}

	/**
	 * Copy constructor
	 */
	__hydra_host__ __hydra_device__
	inline	LambdaWrapper(LambdaWrapper<ReturnType(ArgType...), L, N> const& other ):
	BaseFunctor<LambdaWrapper<Lambda, ReturnType, N>, ReturnType,N>(other),
	fLambda( other.GetLambda())
	{	}

	/**
	 * Assignment operator
	 */
	__hydra_host__ __hydra_device__
	inline LambdaWrapper<Lambda, ReturnType, N>
	operator=(LambdaWrapper<Lambda, ReturnType, N> const& other )
	{
		if(this==&other) return *this;

		BaseFunctor<LambdaWrapper<Lambda, ReturnType, N>, ReturnType,N>::operator=(other);
		fLambda=other.GetLambda();

		return *this;
	}


	/**
	 * Get the underlying lambda
	 */
	__hydra_host__ __hydra_device__
	inline const Lambda& GetLambda() const {return fLambda; }


	template< typename ...T, size_t M=N >
	__hydra_host__ __hydra_device__
	inline typename std::enable_if< (M>0), ReturnType >::type
	Evaluate(T... a)   const {


		return fLambda(this->GetNumberOfParameters(), this->GetParameters(),a...);
	}



	template< typename ...T, size_t M=N >
	__hydra_host__ __hydra_device__
	inline typename std::enable_if< (M==0), ReturnType >::type
	Evaluate(T...a)   const {

		return fLambda(a...);
	}



private:
	L fLambda;
};

namespace detail {


template<typename L, typename ReturnType, typename ...Args, size_t N>
auto wrap_lambda_helper(L const& f, ReturnType, HYDRA_EXTERNAL_NS::thrust::tuple<Args...>const& ,
		std::array<Parameter, N> const& parameters)
-> LambdaWrapper<ReturnType(Args...), L, N>
{
	return LambdaWrapper<ReturnType(Args...), L, N>(f, parameters);
}

template<typename L, typename ReturnType, typename ...Args>
auto wrap_lambda_helper(L const& f, ReturnType&& , HYDRA_EXTERNAL_NS::thrust::tuple<Args...>&& )
-> LambdaWrapper<ReturnType(Args...), L, 0>
{
	return LambdaWrapper<ReturnType(Args...), L, 0>(f);
}

}  // namespace detail

/**
 * @ingroup functor
 * @brief Function template for wrap a C++11 lambda into a hydra lambda with a certain number of parameters.
 * @param f C++11 lambda implementing the operator()(n, params, args) where n is the number of parameters, params a pointer to the parameter array and args are the arguments.
 * @param pars parameters.
 * @return LambdaWrapper object.
 */
template<typename L, typename ...T>
auto wrap_lambda(L const& f,  T const& ...pars)
-> decltype(detail::wrap_lambda_helper( std::declval<L>(),
		std::declval<typename detail::function_traits<L>::return_type>() ,
		std::declval<typename detail::function_traits<L>::args_type>() ,
		std::declval<std::array<Parameter, sizeof...(T)>>()))
{
	typedef detail::function_traits<L> traits;
	typename traits::return_type r = typename traits::return_type();
	typename traits::args_type t;
	std::array<Parameter, sizeof...(T)> parameters{ pars...};

	return detail::wrap_lambda_helper(f, r, t, parameters);
}

/**
 * @ingroup functor
 * @brief Function template for wrap a C++11 lambda into a hydra lambda.
 * @param f C++11 lambda implementing the operator()(args)
 * @return
 */
template<typename L>
auto wrap_lambda(L const& f)
-> decltype(detail::wrap_lambda_helper(std::declval<L>(),
		std::declval<typename detail::function_traits<L>::return_type>() ,
		std::declval<typename detail::function_traits<L>::args_type>()))
{

	return detail::wrap_lambda_helper(f,
			typename detail::function_traits<L>::return_type{} ,
			typename detail::function_traits<L>::args_type{});
}



}



#endif /* FUNCTIONWRAPPER_CPP14_INL_ */
