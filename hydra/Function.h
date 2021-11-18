/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2021 Antonio Augusto Alves Junior
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
 * Function.h
 *
 *  Created on: 05/05/2016
 *      Author: Antonio Augusto Alves Junior
 */


#ifndef FUNCTION_H_
#define FUNCTION_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/utility/StaticAssert.h>
#include <hydra/detail/Print.h>
#include <hydra/Parameter.h>
#include <hydra/detail/utility/Utility_Tuple.h>
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

namespace hydra
{

/**
 * @ingroup functor
 * @brief Base class for all functors in hydra.
 * @tparam Functor is "bare" c++ functor implementing the template<typename T> Evaluate(T x) method.
 * @tparam NPARAM number of parameters of the functor.
 */
template<typename Functor, typename Signature, std::size_t NPARAM>
class  BaseFunctor : public detail::Parameters<NPARAM>
{

public:

	typedef void hydra_functor_type;
	typedef typename detail::signature_traits<Signature>::return_type     return_type;
	typedef typename detail::signature_traits<Signature>::argument_type argument_type;

	enum { arity=detail::signature_traits<Signature>::arity };

	/**
	 * Default constructor
	 */
	__hydra_host__  __hydra_device__
	explicit BaseFunctor():
		detail::Parameters<NPARAM>(),
		fNorm(1.0),
		_par(*this)
	{}


    /**
     * @brief Constructor taking a list of parameters
     * @param init_parameters std::initializer_list<Parameter> with the parameters of the functor.
     */
	BaseFunctor(std::initializer_list<Parameter> init_parameters):
	detail::Parameters<NPARAM>( init_parameters ),
	fNorm(1.0),
	_par(*this)
	{}

	/**
	 * @brief Constructor taking std::array of parameters
	 * @param init_parameters std::array<Parameter,NPARAM> with the parameters of the functor.
	 */
	BaseFunctor(std::array<Parameter,NPARAM> const& init_parameters):
		detail::Parameters<NPARAM>( init_parameters ),
		fNorm(1.0),
		_par(*this)
		{ }


	/**
	 * @brief Copy constructor
	 */
	__hydra_host__ __hydra_device__
	BaseFunctor(BaseFunctor<Functor, Signature, NPARAM> const& other):
	detail::Parameters<NPARAM>( other),
	fNorm(other.GetNorm()),
	_par(*this)
	{ }

	/**
	 * @brief Assignment operator
	 */
	__hydra_host__ __hydra_device__
	inline BaseFunctor<Functor, Signature, NPARAM>&
	operator=(BaseFunctor<Functor, Signature, NPARAM> const & other )
	{
		if(this != &other)
		{
			detail::Parameters<NPARAM>::operator=( other );
			this->fNorm = other.GetNorm();
			_par=*this;
		}
		return *this;
	}


	__hydra_host__ __hydra_device__
	inline Functor& GetFunctor() {return *static_cast<Functor*>(this);}

	/**
	 * @brief Print registered parameters.
	 */
	void PrintRegisteredParameters()
	{

		HYDRA_CALLER ;
		HYDRA_MSG <<HYDRA_ENDL;

		this->PrintParameters();

		HYDRA_MSG <<"Normalization " << fNorm << HYDRA_ENDL;
		HYDRA_MSG <<"Registered parameters end." << HYDRA_ENDL;
		HYDRA_MSG <<HYDRA_ENDL;
		return;
	}


	__hydra_host__ __hydra_device__
	inline GReal_t GetNorm() const {
		return fNorm;
	}


	__hydra_host__ __hydra_device__
	inline void SetNorm(GReal_t norm) {
		fNorm = norm;
	}


	template<typename ...T>
	__hydra_host__  __hydra_device__
	inline typename std::enable_if<
	(!detail::is_valid_type_pack< argument_type, T...>::value),
	return_type>::type
	operator()(T...x)  const
	{
		//typename hydra::tuple<T...>::dummy a;
		HYDRA_STATIC_ASSERT(int(sizeof...(T))==-1,
				"This Hydra lambda can not be called with these arguments.\n"
				"Possible functions arguments are:\n\n"
				"1) List of arguments matching or convertible to the functor signature.\n"
				"2) One tuple containing the arguments in the functor's signature.\n"
				"3) Two tuples that concatenated contain the arguments in the functor's signature.\n"
				"4) One argument and one tuple that concatenated contain the arguments in the functor's signature.\n"
		        "5) One tuple that is convertible to a tuple of the arguments in the functor's signature.\n\n"
				"Please inspect the error messages issued above to find the line generating the error."
				)

				return typename Functor::return_type{};
	}


	/**
	 * \brief Function call operator overload
	 * taking a pack of parameters convertible to
	 * the lambda signature
	 */
	template<typename ...T>
	__hydra_host__  __hydra_device__
	inline typename std::enable_if<
	detail::is_valid_type_pack< argument_type, T...>::value,
	return_type>::type
	operator()(T...x)  const
	{
		return static_cast<const Functor*>(this)->Evaluate(x...);
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
	( hydra_thrust::detail::is_convertible< typename std::decay<T>::type,  argument_type >::value ),
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

	template<typename T, std::size_t ...I>
	__hydra_host__ __hydra_device__
	inline  return_type call_helper(T x, detail::index_sequence<I...> ) const
	{

		return static_cast<const Functor*>(this)->Evaluate(
				detail::get_tuple_element<
				typename hydra_thrust::tuple_element<I, argument_type>::type >(x)...);
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline return_type call(T x) const
    {
		return call_helper(x, detail::make_index_sequence<Functor::arity>{});
	}



	template<typename T, std::size_t ...I>
	__hydra_host__ __hydra_device__
	inline  return_type raw_call_helper(T x, detail::index_sequence<I...> ) const
	{
		return static_cast<const Functor*>(this)->Evaluate(
				static_cast<typename hydra_thrust::tuple_element<I, argument_type>::type>(
				hydra_thrust::get<I>(x))...);
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline return_type raw_call(T x) const
	{
		return raw_call_helper(x, detail::make_index_sequence<Functor::arity>{});
	}

    GReal_t fNorm;

protected:

    BaseFunctor<Functor, Signature,  NPARAM>& _par;

};



}//namespace hydra


#endif /* FUNCTION_H_ */
