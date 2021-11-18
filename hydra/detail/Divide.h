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
 * Divide.h
 *
 *  Created on: 10/06/2016
 *      Author: Antonio Augusto Alves Junior
 */
/**
 * \file
 * \ingroup functor
 */

#ifndef DIVIDE_H_
#define DIVIDE_H_


#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/TypeTraits.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/base_functor.h>
#include <hydra/detail/Constant.h>
#include <hydra/Complex.h>
#include <hydra/detail/FunctorTraits.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>
#include <hydra/detail/BaseCompositeFunctor.h>
#include <hydra/detail/TupleUtility.h>
#include <hydra/detail/TupleTraits.h>

namespace hydra {


template<typename F1, typename F2 >
class Divide : public BaseCompositeFunctor<
Divide<F1, F2>,
hydra_thrust::tuple<F1, F2>,
 typename detail::merged_tuple<
 	 hydra_thrust::tuple< decltype( std::declval<typename F1::return_type>()/std::declval<typename F2::return_type>() )>,
 	 typename detail::stripped_tuple<
 	 	 typename detail::merged_tuple<
 	 	 	 typename F1::argument_type,
 	 	 	 typename F2::argument_type
 	 	 >::type
 	 >::type
 >::type
>
{

	typedef  BaseCompositeFunctor<
			Divide<F1, F2>,
			hydra_thrust::tuple<F1, F2>,
			 typename detail::merged_tuple<
			 	 hydra_thrust::tuple< decltype( std::declval<typename F1::return_type>()/std::declval<typename F2::return_type>() )>,
			 	 typename detail::stripped_tuple<
			 	 	 typename detail::merged_tuple<
			 	 	 	 typename F1::argument_type,
			 	 	 	 typename F2::argument_type
			 	 	 >::type
			 	 >::type
			 >::type
			>  super_type;

public:


	Divide() = delete;

	Divide(F1 const& f1, F2 const& f2):
		super_type( f1, f2)
	{ }

	__hydra_host__ __hydra_device__
	Divide(Divide<F1,F2> const& other):
	super_type( other )
	{ }

	__hydra_host__ __hydra_device__
	Divide<F1,F2>& operator=(Divide<F1,F2> const& other)
	{
		if(this==&other) return *this;
		super_type::operator=( other);
		return *this;
	}

	template<typename ...T>
	__hydra_host__ __hydra_device__ inline
	typename super_type::return_type Evaluate(T... x ) const
	{
		return hydra_thrust::get<0>(this->GetFunctors())( hydra::tie(x...) )/hydra_thrust::get<1>(this->GetFunctors())( hydra::tie(x...) );
	}


};

// divide: / operator two functors
template<typename T1, typename T2>
inline typename std::enable_if<
(detail::is_hydra_functor<T1>::value || detail::is_hydra_lambda<T1>::value) &&
(detail::is_hydra_functor<T2>::value || detail::is_hydra_lambda<T2>::value),
Divide<T1, T2> >::type
operator/(T1 const& F1, T2 const& F2)
{
	return Divide<T1,T2>(F1, F2);
}

template <typename T, typename U>
inline typename std::enable_if<
(detail::is_hydra_functor<T>::value || detail::is_hydra_lambda<T>::value ) &&
(std::is_arithmetic<U>::value),
Divide< Constant<U>, T> >::type
operator/(U const cte, T const& F)
{
	return Constant<U>(cte)/F;
}

template <typename T, typename U>
inline typename std::enable_if<
(detail::is_hydra_functor<T>::value || detail::is_hydra_lambda<T>::value ) &&
(std::is_arithmetic<U>::value),
Divide< Constant<U>, T> >::type
operator/( T const& F, U cte)
{
	return F/Constant<U>(cte);
}

template <typename T, typename U>
inline typename std::enable_if<
(detail::is_hydra_functor<T>::value || detail::is_hydra_lambda<T>::value ) &&
(std::is_arithmetic<U>::value),
Divide< Constant<hydra::complex<U>>, T> >::type
operator/(hydra::complex<U> const& cte, T const& F)
{
	return  Constant<hydra::complex<U> >(cte)/F;
}

template <typename T, typename U>
inline typename std::enable_if<
(detail::is_hydra_functor<T>::value || detail::is_hydra_lambda<T>::value ) &&
(std::is_arithmetic<U>::value),
Divide< Constant<hydra::complex<U>>, T> >::type
operator/( T const& F, hydra::complex<U> const& cte)
{
	return  F/Constant<hydra::complex<U> >(cte);
}


// Convenience function
template <typename F1, typename F2, typename ...Fs>
inline typename std::enable_if<
(detail::is_hydra_functor<F1>::value || detail::is_hydra_lambda<F1>::value ) &&
(detail::is_hydra_functor<F2>::value || detail::is_hydra_lambda<F2>::value ),
Divide<F1, F2>>::type
divide(F1 const& f1, F2 const& f2)
{
	return  Divide<F1, F2>(f1,f2);
}



}//namespace hydra


#endif /* DIVIDE_H_ */
