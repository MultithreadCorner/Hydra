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
 * Minus.h
 *
 *  Created on: 10/06/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup functor
 */


#ifndef MINUS_H_
#define MINUS_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/TypeTraits.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/base_functor.h>
#include <hydra/Parameter.h>
#include <hydra/detail/CompositeBase.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>

namespace hydra {



template<typename F1, typename F2 >
class  Minus: public detail::CompositeBase<F1, F2>
{
	public:

	    typedef typename detail::minus_result<
	    		        typename F1::return_type,
	    		        typename F2::return_type>::type  return_type;

		Minus()=delete;

		__hydra_host__
		Minus(F1 const& f1, F2 const& f2):
		detail::CompositeBase<F1, F2>( f1, f2)
	  	{ }

		__hydra_host__ __hydra_device__
		Minus(Minus<F1,F2> const& other):
		detail::CompositeBase<F1, F2>( other )
		{ }

		__hydra_host__ __hydra_device__
		Minus<F1,F2>& operator=(Minus<F1,F2> const& other)
		{
			if(this==&other) return *this;
			detail::CompositeBase< F1, F2>::operator=( other);
			return *this;
		}

	  	template<typename T1>
	  	__hydra_host__ __hydra_device__ inline
	  	return_type operator()(T1& t ) const
	  	{
	  		return hydra::get<0>(this->fFtorTuple)(t)-hydra::get<1>(this->fFtorTuple)(t);
	  	}


};

// + operator two functors
template<typename T1, typename T2>
inline typename std::enable_if<
(detail::is_hydra_functor<T1>::value || detail::is_hydra_lambda<T1>::value) &&
(detail::is_hydra_functor<T2>::value || detail::is_hydra_lambda<T2>::value),
Minus<T1, T2> >::type
operator-(T1 const& F1, T2 const& F2)
{
	return  Minus<T1,T2>(F1, F2);
}

template <typename T, typename U>
inline typename std::enable_if<
(detail::is_hydra_functor<T>::value || detail::is_hydra_lambda<T>::value) &&
(std::is_arithmetic<U>::value),
Minus< Constant<U>, T> >::type
operator-(U const cte, T const& F)
{
	return  Constant<U>(cte)-F;
}

template <typename T, typename U>
inline typename std::enable_if<
(detail::is_hydra_functor<T>::value || detail::is_hydra_lambda<T>::value) &&
(std::is_arithmetic<U>::value),
Minus< Constant<U>, T> >::type
operator-( T const& F, U cte)
{
	return  F-Constant<U>(cte);
}

template <typename T, typename U>
inline typename std::enable_if<
(detail::is_hydra_functor<T>::value || detail::is_hydra_lambda<T>::value) &&
(std::is_arithmetic<U>::value),
Minus< Constant<hydra::complex<U>>, T> >::type
operator-(hydra::complex<U> const& cte, T const& F)
{
	return  Constant<hydra::complex<U> >(cte)-F;
}

template <typename T, typename U>
inline typename std::enable_if<
(detail::is_hydra_functor<T>::value || detail::is_hydra_lambda<T>::value) &&
(std::is_arithmetic<U>::value),
Minus< Constant<U>, T> >::type
operator-( T const& F, hydra::complex<U> const& cte)
{
	return  F-Constant<hydra::complex<U> >(cte);
}

// Convenience function
template <typename F1, typename F2>
inline typename std::enable_if<
(detail::is_hydra_functor<F1>::value || detail::is_hydra_lambda<F1>::value) &&
(detail::is_hydra_functor<F2>::value || detail::is_hydra_lambda<F2>::value),
Minus<F1, F2>>::type
minus(F1 const& f1, F2 const& f2)
{
	return  Minus<F1, F2>(f1,f2);
}

}//namespace hydra



#endif /* MINUS_H_ */
