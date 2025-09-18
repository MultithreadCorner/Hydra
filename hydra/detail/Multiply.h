/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2025 Antonio Augusto Alves Junior
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
 * Multiply.h
 *
 *  Created on: 05/05/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup functor
 */


#ifndef MULTIPLY_H_
#define MULTIPLY_H_


#include <type_traits>

#include <hydra/detail/Config.h>
#include <hydra/Types.h>

#include <hydra/detail/TypeTraits.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/Constant.h>
#include <hydra/detail/Argument.h>
#include <hydra/Parameter.h>
#include <hydra/detail/CompositeTraits.h>
#include <hydra/detail/FunctorTraits.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>
#include <hydra/detail/BaseCompositeFunctor.h>
#include <hydra/detail/TupleUtility.h>
#include <hydra/detail/TupleTraits.h>

namespace hydra {


template<typename F1, typename F2, typename ...Fs>
class Multiply:  public BaseCompositeFunctor<
Multiply<F1, F2,Fs...>,
hydra::thrust::tuple<F1, F2, Fs...>,
 typename detail::merged_tuple<
 	 hydra::thrust::tuple< typename std::common_type<
 	 	 	 	 	 	 	 	 	 	 typename F1::return_type,
 	 	 	 	 	 	 	 	 	 	 typename F2::return_type,
 	 	 	 	 	 	 	 	 	 	 typename Fs::return_type...
 	 	 	 	 	 	 	 	 	>::type >,
 	 typename detail::stripped_tuple<
 	 	 typename detail::merged_tuple<
 	 	 	 typename F1::argument_type,
 	 	 	 typename F2::argument_type,
 	 	 	 typename Fs::argument_type ...
 	 	 >::type
 	 >::type
 >::type
>
{

	typedef BaseCompositeFunctor<
			Multiply<F1, F2,Fs...>,
			hydra::thrust::tuple<F1, F2, Fs...>,
			 typename detail::merged_tuple<
			 	 hydra::thrust::tuple< typename std::common_type<
			 	 	 	 	 	 	 	 	 	 	 typename F1::return_type,
			 	 	 	 	 	 	 	 	 	 	 typename F2::return_type,
			 	 	 	 	 	 	 	 	 	 	 typename Fs::return_type...
			 	 	 	 	 	 	 	 	 	>::type >,
			 	 typename detail::stripped_tuple<
			 	 	 typename detail::merged_tuple<
			 	 	 	 typename F1::argument_type,
			 	 	 	 typename F2::argument_type,
			 	 	 	 typename Fs::argument_type ...
			 	 	 >::type
			 	 >::type
			 >::type
			> super_type;

	public:

	Multiply()=delete;

	Multiply(F1 const& f1, F2 const& f2, Fs const&... fs ):
		super_type( f1, f2,fs...)
	{ }


	__hydra_host__ __hydra_device__
	Multiply(const Multiply<F1,F2, Fs...>& other):
	super_type( other)
	{ }

	__hydra_host__ __hydra_device__
	Multiply<F1,F2, Fs...>& operator=(Multiply<F1,F2, Fs...> const & other)
	{
		if(this==&other) return *this;
		super_type::operator=( other);
		return *this;
	}

  	template<typename ...T>
  	__hydra_host__ __hydra_device__ inline
  	typename super_type::return_type Evaluate(T... x ) const
  	{

  		return detail::product<typename super_type::return_type, hydra::tuple<T...>,F1,F2,Fs...>( hydra::tie(x...), this->GetFunctors() );

  	}


};

// multiplication: * operator two functors
template<typename T1, typename T2>
inline typename std::enable_if<
(detail::is_hydra_functor<T1>::value || detail::is_hydra_lambda<T1>::value ) &&
(detail::is_hydra_functor<T2>::value || detail::is_hydra_lambda<T2>::value ),
Multiply<T1, T2> >::type
operator*(T1 const& F1, T2 const& F2)
{
	return  Multiply<T1,T2>(F1, F2);
}

template <typename T, typename U>
inline typename std::enable_if<
(detail::is_hydra_functor<T>::value || detail::is_hydra_lambda<T>::value ) &&
(std::is_arithmetic<U>::value),
Multiply< Constant<U>, T> >::type
operator*(U const cte, T const& F)
{
	return  Constant<U>(cte)*F;
}

template <typename T, typename U>
inline typename std::enable_if<
(detail::is_hydra_functor<T>::value || detail::is_hydra_lambda<T>::value ) &&
(std::is_arithmetic<U>::value),
Multiply< Constant<U>, T> >::type
operator*( T const& F, U cte)
{
	return  Constant<U>(cte)*F;
}

template <typename T, typename U>
inline typename std::enable_if<
(detail::is_hydra_functor<T>::value || detail::is_hydra_lambda<T>::value ) &&
(std::is_arithmetic<U>::value),
Multiply< Constant<hydra::complex<U>>, T> >::type
operator*(hydra::complex<U> const& cte, T const& F)
{
	return  Constant<hydra::complex<U> >(cte)*F;
}

template <typename T, typename U>
inline typename std::enable_if<
(detail::is_hydra_functor<T>::value || detail::is_hydra_lambda<T>::value ) &&
(std::is_arithmetic<U>::value),
Multiply< Constant<U>, T> >::type
operator*( T const& F, hydra::complex<U> const& cte)
{
	return  Constant<hydra::complex<U> >(cte)*F;
}

// Convenience function
// Convenience function
template <typename F1, typename F2, typename ...Fs>
inline typename std::enable_if<
(detail::is_hydra_functor<F1>::value || detail::is_hydra_lambda<F1>::value ) &&
(detail::is_hydra_functor<F2>::value || detail::is_hydra_lambda<F2>::value ) &&
detail::all_true<
(detail::is_hydra_functor<Fs>::value || detail::is_hydra_lambda<Fs>::value )...>::value,
Multiply<F1, F2,Fs...>>::type
multiply(F1 const& f1, F2 const& f2, Fs const&... functors )
{
	return  Multiply<F1, F2,Fs... >(f1,f2, functors ... );
}

}//namespace hydra

#endif /* MULTIPLY_H_ */
