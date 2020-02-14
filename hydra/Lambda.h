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
#include <hydra/detail/Parameters.h>
#include <hydra/detail/FunctionArgument.h>
#include <hydra/detail/GetTupleElement.h>

namespace hydra {

template<typename LambdaType, size_t NPARAM=0>
class  Lambda;



template<typename LambdaType>
class  Lambda<LambdaType, 0>
{

	typedef typename detail::lambda_traits<LambdaType>::argument_rvalue_type argument_rvalue_type;

public:

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

	__hydra_host__ __hydra_device__
	inline Lambda<Lambda, 0>&
	operator=(Lambda<LambdaType, 0> const & other )
	{
		if(this != &other)
		{
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
	(!std::is_convertible<std::tuple<T...>, argument_rvalue_type>::value),
	return_type>::type
	operator()(T...x)  const
	{
		HYDRA_STATIC_ASSERT(sizeof...(T)==-1,
				"This Hydra functor can not be called with these arguments." )

				return return_type{};
	}

	template<typename ...T>
	__hydra_host__  __hydra_device__
	inline typename std::enable_if<
	std::is_convertible<std::tuple<T...>,
	argument_rvalue_type>::value, return_type>::type
	operator()(T...x)  const
	{
		return fLambda(x...);
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline typename std::enable_if<
	(( detail::is_instantiation_of<
			hydra_thrust::detail::tuple_of_iterator_references,
			typename std::decay<T>::type>::value)) ||
			(( detail::is_instantiation_of<
					hydra_thrust::tuple,
					typename std::decay<T>::type>::value)),
					return_type>::type
					operator()( T x )  const { return  call(x); }





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


	double fNorm;
	LambdaType fLambda;

};

template<typename LambdaType, size_t NPARAM>
class  Lambda : public detail::Parameters<NPARAM>
{

	typedef typename detail::lambda_traits<LambdaType>::argument_rvalue_type argument_rvalue_type;

public:

	typedef typename detail::lambda_traits<LambdaType>::return_type   return_type;
	typedef typename detail::lambda_traits<LambdaType>::argument_type argument_type;

	enum {arity=detail::lambda_traits<LambdaType>::arity+NPARAM};


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
	inline Lambda<Lambda, NPARAM>&
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
	inline LambdaType const& GetLambda()
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
	(!std::is_convertible<std::tuple<T...>, argument_rvalue_type>::value),
	return_type>::type
	operator()(T...x)  const
	{
		HYDRA_STATIC_ASSERT(sizeof...(T)==-1,
				"This Hydra functor can not be called with these arguments." )

				return return_type{};
	}

	template<typename ...T>
	__hydra_host__  __hydra_device__
	inline typename std::enable_if<
	std::is_convertible<std::tuple<T...>,
	argument_rvalue_type>::value, return_type>::type
	operator()(T...x)  const
	{
		return fLambda(x...);
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline typename std::enable_if<
	(( detail::is_instantiation_of<
			hydra_thrust::detail::tuple_of_iterator_references,
			typename std::decay<T>::type>::value)) ||
			(( detail::is_instantiation_of<
					hydra_thrust::tuple,
					typename std::decay<T>::type>::value)),
					return_type>::type
					operator()( T x )  const { return  call(x); }





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


	double fNorm;
	LambdaType fLambda;

};


}  // namespace hydra



#endif /* LAMBDA_H_ */
