/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2019 Antonio Augusto Alves Junior
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
 * Sum.h
 *
 *  Created on: 05/05/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup functor
 */


#ifndef SUM_H_
#define SUM_H_


#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/base_functor.h>
#include <hydra/detail/Constant.h>
#include <hydra/detail/external/thrust/tuple.h>
#include <type_traits>
#include <hydra/Parameter.h>
#include <hydra/detail/CompositeBase.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>

namespace hydra {


template<typename F1, typename F2, typename ...Fs>
class Sum: public detail::CompositeBase< F1,F2, Fs...>
{
public:
	//tag
	typedef void hydra_functor_tag;

	typedef   std::true_type is_functor;
	typedef typename detail::sum_result<typename F1::return_type ,typename  F2::return_type,typename  Fs::return_type...>::type  return_type;

	Sum()=delete;


    Sum(F1 const& f1, F2 const& f2, Fs const&... fs ):
		detail::CompositeBase<F1,F2, Fs...>( f1, f2,fs...)
  	{ }

	__hydra_host__ __hydra_device__
	Sum(const Sum<F1,F2, Fs...>& other):
		detail::CompositeBase<F1,F2, Fs...>(  other)
	{ };

	__hydra_host__ __hydra_device__ inline
	Sum<F1,F2, Fs...>& operator=(Sum<F1,F2, Fs...> const& other)
	{
		if(this==&other) return *this;
		detail::CompositeBase<F1,F2, Fs...>::operator=( other);

		return *this;

	}

  	template<typename T1>
  	__hydra_host__ __hydra_device__ inline
  	return_type operator()(T1&& t ) const
  	{
  		return detail::accumulate<return_type,T1,F1,F2,Fs...>(std::forward<T1&>(t),this->fFtorTuple );

  	}

  	template<typename T1, typename T2>
  	__hydra_host__ __hydra_device__  inline
  	return_type operator()( T1&& t, T2&& cache) const
  	{

  		return this->IsCached() ? detail::extract<return_type,T2>(this->GetIndex(), std::forward<T2&>(cache)):\
  				detail::accumulate2<return_type,T1,T2,F1,F2,Fs...>(t,cache,this->fFtorTuple );
  	}


};

// + operator two functors
template<typename T1, typename T2,
typename=typename std::enable_if< T1::is_functor::value && T2::is_functor::value> >
__hydra_host__  inline
Sum<T1, T2> operator+(T1 const& F1, T2 const& F2){ return  Sum<T1,T2>(F1, F2); }

template <typename T1, typename T2,
typename=typename std::enable_if< (std::is_convertible<T1, double>::value ||\
		std::is_constructible<HYDRA_EXTERNAL_NS::thrust::complex<double>,T1>::value) && T2::is_functor::value>::type >
__hydra_host__  inline
Sum<Constant<T1>, T2>
operator+(T1 const cte, T2 const& F2){ return  Constant<T1>(cte)+F2; }


template <typename T1, typename T2,
typename=typename std::enable_if< (std::is_convertible<T1, double>::value ||\
		std::is_constructible<HYDRA_EXTERNAL_NS::thrust::complex<double>,T1>::value) && T2::is_functor::value>::type >
__hydra_host__  inline
Sum<Constant<T1>, T2>
operator+(T2 const& F2, T1 const cte ){	return  Constant<T1>(cte)+F2; }

// Convenience function
template <typename F1, typename F2, typename ...Fs>
__hydra_host__  inline
Sum<F1, F2,Fs...>
sum(F1 const& f1, F2 const& f2, Fs const&... functors )
{ return  Sum<F1, F2,Fs... >(f1,f2, functors ... ); }


}


#endif /* SUM_H_ */
