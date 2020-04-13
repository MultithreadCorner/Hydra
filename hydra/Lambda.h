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
 * Lambda.h
 *
 *  Created on: 13/02/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef LAMBDA_H_
#define LAMBDA_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/utility/StaticAssert.h>
#include <hydra/detail/FunctorTraits.h>
#include <hydra/detail/ArgumentTraits.h>
#include <hydra/detail/Parameters.h>
#include <hydra/detail/FunctionArgument.h>
#include <hydra/detail/GetTupleElement.h>

#include <hydra/detail/external/hydra_thrust/iterator/detail/tuple_of_iterator_references.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

#include <array>
#include <initializer_list>
#include <type_traits>

namespace hydra {


template<typename LambdaType, size_t NPARAM=0>
class  Lambda;


template<typename LambdaType>
class  Lambda<LambdaType, 0>
{

	typedef typename detail::lambda_traits<LambdaType>::argument_rvalue_type argument_rvalue_type;


public:

	typedef void hydra_lambda_type;

	typedef typename detail::lambda_traits<LambdaType>::return_type   return_type;
	typedef typename detail::lambda_traits<LambdaType>::argument_type argument_type;

	enum {arity=detail::lambda_traits<LambdaType>::arity};


	explicit Lambda()=delete;

	Lambda(LambdaType const& lambda):
		fLambda(lambda),
		fNorm(1.0)
		{}


	__hydra_host__ __hydra_device__
	Lambda(Lambda<LambdaType, 0> const& other):
	fLambda(other.GetLambda()),
	fNorm(other.GetNorm())
	{ }


   template<typename T= LambdaType>
	__hydra_host__ __hydra_device__
	inline typename std::enable_if< std::is_copy_assignable<T>::value,
	Lambda<T, 0> &>::type
	operator=(Lambda<T, 0> const & other )
	{
		if(this == &other) return *this;

		fLambda  = other.GetLambda();
        fNorm = other.GetNorm();

		return *this;
	}

	__hydra_host__ __hydra_device__
	inline const LambdaType& GetLambda() const
	{
		return fLambda;
	}

	void PrintRegisteredParameters()
	{

		HYDRA_CALLER ;
		HYDRA_MSG <<HYDRA_ENDL;
		HYDRA_MSG <<"Normalization " << fNorm << HYDRA_ENDL;
		HYDRA_MSG <<HYDRA_ENDL;
		return;
	}

	__hydra_host__ __hydra_device__
	inline double GetNorm() const
	{
		return fNorm;
	}

	__hydra_host__ __hydra_device__
	inline void SetNorm(double norm)
	{
		fNorm = norm;
	}


	template<typename ...T>
	__hydra_host__  __hydra_device__
	inline typename std::enable_if<
	(!detail::is_valid_type_pack<argument_rvalue_type, T...>::value),
	return_type>::type
	operator()(T...x)  const
	{
		HYDRA_STATIC_ASSERT(int(sizeof...(T))==-1,
				"This Hydra lambda can not be called with these arguments.\n"
				"Possible functions arguments are:\n\n"
				"1) List of arguments matching or convertible to the lambda signature.\n"
				"2) One tuple containing the arguments in the lambda's signature.\n"
				"3) Two tuples that concatenated contain the arguments in the lambda's signature.\n"
				"4) One argument and one tuple that concatenated contain the arguments in the lambda's signature.\n"
		        "5) One tuple that is convertible to a tuple of the arguments in the lambda's signature.\n\n"
				"Please inspect the error messages issued above to find the line generating the error."
				)

				return return_type(0);
	}


	/**
	 * \brief Function call operator overload
	 * taking a pack of parameters convertible to
	 * the lambda signature
	 */
	template<typename ...T>
	__hydra_host__  __hydra_device__
	inline typename std::enable_if<
	detail::is_valid_type_pack<argument_rvalue_type, T...>::value,
	 return_type>::type
	operator()(T...x)  const
	{
		return fLambda(x...);
	}

	/**
	 * \brief Unary function call operator overload
	 * taking a tuple containing
	 * the lambda arguments in any other.
	 */
	template<typename T>
	__hydra_host__ __hydra_device__
	inline typename std::enable_if<
	( detail::is_tuple_type< typename std::decay<T>::type >::value )                 &&
	(!detail::is_tuple_of_function_arguments< typename std::decay<T>::type >::value) &&
	( std::is_convertible< typename std::decay<T>::type, argument_rvalue_type >::value ),
	return_type >::type
	operator()( T x )  const
	{
		return  raw_call(x);
	}

	/**
	 * \brief Unary function call operator overload
	 * taking a tuple containing
	 * the lambda arguments in any other.
	 */
	template<typename T>
	__hydra_host__ __hydra_device__
	inline typename std::enable_if<
	( detail::is_tuple_type<typename std::decay<T>::type>::value ) &&
	( detail::is_tuple_of_function_arguments< typename std::decay<T>::type >::value),
	return_type>::type
	operator()( T x )  const
	{
		return  call(x);
	}


	/**
	 * \brief Binary function call operator overload
	 * taking two tuples containing
	 * the lambda arguments in any other.
	 */
	template<typename T1, typename T2>
	__hydra_host__ __hydra_device__
	inline typename std::enable_if<
	( detail::is_tuple_type<typename std::decay<T1>::type>::value ) &&
	( detail::is_tuple_of_function_arguments< typename std::decay<T1>::type >::value ) &&
	( detail::is_tuple_type<typename std::decay<T2>::type>::value ) &&
	( detail::is_tuple_of_function_arguments< typename std::decay<T2>::type >::value ) ,
	return_type>::type
	operator()( T1 x, T2 y )  const
	{
		auto z = hydra_thrust::tuple_cat(x, y);

		return  call(z);
	}

	/**
	 * \brief Binary function call operator overload
	 * taking one tuple and a non-tuple, that
	 * containing put together would contain
	 * the lambda arguments in any other.
	 */
	template<typename T1, typename T2>
	__hydra_host__ __hydra_device__
	inline typename std::enable_if<
	(!detail::is_tuple_type< typename std::decay<T1>::type>::value ) &&
	( detail::is_function_argument< typename std::decay<T1>::type >::value ) &&
	( detail::is_tuple_type< typename std::decay<T2>::type>::value ) &&
	( detail::is_tuple_of_function_arguments< typename std::decay<T2>::type >::value ),
	return_type>::type
	operator()( T1 x, T2 y )  const
	{
		auto z = hydra_thrust::tuple_cat(hydra_thrust::make_tuple(x), y);
		return  call(z);
	}

	/**
	 * \brief Binary function call operator overload
	 * taking one tuple and a non-tuple, that
	 * containing put together would contain
	 * the lambda arguments in any other.
	 */
	template<typename T1, typename T2>
	__hydra_host__ __hydra_device__
	inline typename std::enable_if<
	(!detail::is_tuple_type< typename std::decay<T1>::type>::value ) &&
	( detail::is_function_argument< typename std::decay<T1>::type >::value ) &&
	( detail::is_tuple_type< typename std::decay<T2>::type>::value ) &&
	( detail::is_tuple_of_function_arguments< typename std::decay<T2>::type >::value ),
	return_type>::type
	operator()(  T2 y, T1 x )  const
	{
		auto z = hydra_thrust::tuple_cat(hydra_thrust::make_tuple(x), y);
		return  call(z);
	}

private:

	template<typename T, size_t ...I>
	__hydra_host__ __hydra_device__
	inline  return_type call_helper(T x, detail::index_sequence<I...> ) const
	{
		return fLambda( detail::get_tuple_element<
				typename hydra_thrust::tuple_element<I,argument_rvalue_type>::type >(x)...);
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline return_type call(T x) const
	{
		return call_helper(x, detail::make_index_sequence<arity>{});
	}

	template<typename T, size_t ...I>
	__hydra_host__ __hydra_device__
	inline  return_type raw_call_helper(T x, detail::index_sequence<I...> ) const
	{
		return fLambda(static_cast<typename hydra_thrust::tuple_element<I,argument_rvalue_type>::type>(
				hydra_thrust::get<I>(x))...);
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline return_type raw_call(T x) const
	{
		return raw_call_helper(x, detail::make_index_sequence<arity>{});
	}


	double fNorm;
	LambdaType fLambda;

};

template<typename LambdaType, size_t NPARAM>
class  Lambda : public detail::Parameters<NPARAM>
{

	typedef typename detail::lambda_traits<LambdaType>::argument_rvalue_type argument_rvalue_type;

public:

	typedef void hydra_lambda_type;

	typedef typename detail::lambda_traits<LambdaType>::return_type   return_type;
	typedef typename detail::lambda_traits<LambdaType>::argument_type argument_type;

	enum {arity=detail::lambda_traits<LambdaType>::arity};


	explicit Lambda()=delete;

	Lambda(LambdaType const& lambda, std::initializer_list<Parameter> init_parameters):
		detail::Parameters<NPARAM>( init_parameters ),
		fLambda(lambda),
		fNorm(1.0)
		{}

	Lambda(LambdaType const& lambda, std::array<Parameter,NPARAM> const& init_parameters):
		detail::Parameters<NPARAM>( init_parameters ),
		fLambda(lambda),
		fNorm(1.0)
		{ }


	__hydra_host__ __hydra_device__
	Lambda(Lambda<LambdaType, NPARAM> const& other):
	detail::Parameters<NPARAM>( other),
	fLambda(other.GetLambda()),
	fNorm(other.GetNorm())
	{ }

	__hydra_host__ __hydra_device__
	inline Lambda<LambdaType, NPARAM>&
	operator=(Lambda<LambdaType, NPARAM> const & other )
	{
		if(this != &other)
		{
			detail::Parameters<NPARAM>::operator=( other );
			fLambda     = other.GetLambda();
			this->fNorm = other.GetNorm();
		}

		return *this;
	}

	__hydra_host__ __hydra_device__
	inline LambdaType const& GetLambda() const
	{
		return fLambda;
	}

	void PrintRegisteredParameters()
	{

		HYDRA_CALLER ;
		HYDRA_MSG <<HYDRA_ENDL;
		HYDRA_MSG << "Registered parameters begin:" << HYDRA_ENDL;
		this->PrintParameters();
		HYDRA_MSG <<"Normalization " << fNorm << HYDRA_ENDL;
		HYDRA_MSG <<"Registered parameters end." << HYDRA_ENDL;
		HYDRA_MSG <<HYDRA_ENDL;
		return;
	}


	__hydra_host__ __hydra_device__
	inline double GetNorm() const
	{
		return fNorm;
	}


	__hydra_host__ __hydra_device__
	inline void SetNorm(double norm)
	{
		fNorm = norm;
	}

	template<typename ...T>
	__hydra_host__  __hydra_device__
	inline typename std::enable_if<
	(!detail::is_valid_type_pack<argument_type, size_t,
			const hydra::Parameter*, T...>::value),
	return_type>::type
	operator()(T...x)  const
	{
		HYDRA_STATIC_ASSERT(int(sizeof...(T))==-1,
				"This Hydra lambda can not be called with these arguments.\n"
				"Possible functions arguments are:\n\n"
				"1) List of arguments matching or convertible to the lambda signature.\n"
				"2) One tuple containing the arguments in the lambda's signature.\n"
				"3) Two tuples that concatenated contain the arguments in the lambda's signature.\n"
				"4) One argument and one tuple that concatenated contain the arguments in the lambda's signature.\n"
				"5) One tuple that is convertible to a tuple of the arguments in the lambda's signature.\n\n"
				"Please inspect the error messages issued above to find the line generating the error."	)


				return return_type(0);
	}

	template<typename ...T>
	__hydra_host__  __hydra_device__
	inline typename  std::enable_if<
	(detail::is_valid_type_pack<argument_type, size_t,
			const hydra::Parameter*, T...>::value),
	return_type>::type
	operator()(T...x)  const
	{
		return fLambda(this->GetNumberOfParameters(), this->GetParameters(), x...);
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline typename std::enable_if<
	( detail::is_tuple_type< typename std::decay<T>::type >::value )                 &&
	(!detail::is_tuple_of_function_arguments< typename std::decay<T>::type >::value) &&
	( std::is_convertible<
			typename detail::tuple_cat_type<
			         hydra::tuple<size_t, const Parameter*>,
			         typename std::decay<T>::type
			    >::type,
			argument_rvalue_type
			>::value ),
	return_type >::type
	operator()( T x )  const
	{
		return  raw_call(x);
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline typename std::enable_if<
	( detail::is_tuple_type<typename std::decay<T>::type>::value ) &&
	( detail::is_tuple_of_function_arguments< typename std::decay<T>::type >::value),
	return_type>::type
	operator()( T x )  const {
		return  call(x);
	}

	template<typename T1, typename T2>
	__hydra_host__ __hydra_device__
	inline typename std::enable_if<
	( detail::is_tuple_type<typename std::decay<T1>::type>::value ) &&
	( detail::is_tuple_of_function_arguments< typename std::decay<T1>::type >::value ) &&
	( detail::is_tuple_type<typename std::decay<T2>::type>::value ) &&
	( detail::is_tuple_of_function_arguments< typename std::decay<T2>::type >::value ) ,
	return_type>::type
	operator()( T1 x, T2 y )  const
	{
		auto z = hydra_thrust::tuple_cat(x, y);
		return  call(z);
	}

	template<typename T1, typename T2>
	__hydra_host__ __hydra_device__
	inline typename std::enable_if<
	(!detail::is_tuple_type< typename std::decay<T1>::type>::value ) &&
	( detail::is_function_argument< typename std::decay<T1>::type >::value ) &&
	( detail::is_tuple_type< typename std::decay<T2>::type>::value ) &&
	( detail::is_tuple_of_function_arguments< typename std::decay<T2>::type >::value ),
	return_type>::type
	operator()( T1 x, T2 y )  const
	{
		auto z = hydra_thrust::tuple_cat(hydra_thrust::make_tuple(x), y);
		return  call(z);
	}

	template<typename T1, typename T2>
	__hydra_host__ __hydra_device__
	inline typename std::enable_if<
	(!detail::is_tuple_type< typename std::decay<T1>::type>::value ) &&
	( detail::is_function_argument< typename std::decay<T1>::type >::value ) &&
	( detail::is_tuple_type< typename std::decay<T2>::type>::value ) &&
	( detail::is_tuple_of_function_arguments< typename std::decay<T2>::type >::value ),
	return_type>::type
	operator()(  T2 y, T1 x )  const
	{
		auto z = hydra_thrust::tuple_cat(hydra_thrust::make_tuple(x), y);
		return  call(z);
	}

private:

	template<typename T, size_t ...I>
	__hydra_host__ __hydra_device__
	inline  return_type call_helper(T x, detail::index_sequence<I...> ) const
	{

		return fLambda(this->GetNumberOfParameters(), this->GetParameters(),
			detail::get_tuple_element<
			typename hydra_thrust::tuple_element<I+2,argument_rvalue_type>::type >(x)...);
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline return_type call(T x) const
	{
		return call_helper(x, detail::make_index_sequence<arity-2>{});
	}

	template<typename T, size_t ...I>
	__hydra_host__ __hydra_device__
	inline  return_type raw_call_helper(T x, detail::index_sequence<I...> ) const
	{
		return fLambda(this->GetNumberOfParameters(), this->GetParameters(),
				static_cast<typename hydra_thrust::tuple_element<I+2,argument_rvalue_type>::type>(
				hydra_thrust::get<I>(x))...);
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline return_type raw_call(T x) const
	{
		return raw_call_helper(x, detail::make_index_sequence<arity-2>{});
	}


	double fNorm;
	LambdaType fLambda;

};

template<typename LambdaType>
hydra::Lambda<LambdaType, 0> wrap_lambda(LambdaType const& lambda)
{
	return hydra::Lambda<LambdaType, 0>(lambda);
}

template<typename LambdaType, typename ...T>
typename std::enable_if<
detail::all_true<std::is_same<T, hydra::Parameter>::value...>::value,
hydra::Lambda<LambdaType, sizeof...(T)>>::type
wrap_lambda(LambdaType const& lambda, T const&...parameters)
{
	return hydra::Lambda<LambdaType, sizeof...(T)>(lambda, {parameters...});
}

}  // namespace hydra



#endif /* LAMBDA_H_ */
