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
#include <thrust/tuple.h>
#include <type_traits>
#include <hydra/Parameter.h>

namespace hydra {


template<typename F1, typename F2, typename ...Fs>
struct  Sum
{

	//tag
	typedef void hydra_functor_tag;

	typedef   std::true_type is_functor;
	typedef typename detail::sum_result<typename F1::return_type ,typename  F2::return_type,typename  Fs::return_type...>::type  return_type;
	typedef typename thrust::tuple<F1, F2, Fs...> functors_type;

	__host__
    Sum():
    fIndex(-1),
	fCached(0)
	{};

	__host__
    Sum(F1 const& f1, F2 const& f2, Fs const&... functors ):
    fIndex(-1),
	fCached(0),
  	fFtorTuple(thrust::make_tuple(f1, f2, functors ...))
  	{  	}

	__host__ __device__ Sum(const Sum<F1,F2, Fs...>& other):
				fIndex( other.GetIndex() ),
				fCached( other.IsCached() ),
				fFtorTuple( other.GetFunctors() )
	{ };

	__host__ __device__
	Sum<F1,F2, Fs...>& operator=(Sum<F1,F2, Fs...> const& other)
	{
		this->fFtorTuple = other.GetFunctors() ;
		this->fIndex = other.GetIndex() ;
		this->fCached = other.IsCached() ;
		return *this;
	}

	__host__ inline
	void AddUserParameters(std::vector<hydra::Parameter*>& user_parameters )
	{
		detail::add_parameters_in_tuple(user_parameters, fFtorTuple );
	}


	__host__ inline
	void SetParameters(const std::vector<double>& parameters){

		detail::set_functors_in_tuple(fFtorTuple, parameters);
	}

	__host__ inline
	void PrintRegisteredParameters()
		{
		    HYDRA_CALLER ;
			HYDRA_MSG << "Registered parameters begin:" << HYDRA_ENDL;
			detail::print_parameters_in_tuple(fFtorTuple);
			HYDRA_MSG <<"Registered parameters end." << HYDRA_ENDL;
			return;
		}


	__host__ __device__ inline
	functors_type GetFunctors() const {return this->fFtorTuple;}

	__host__ __device__ inline
	int GetIndex() const { return this->fIndex; }

	__host__ __device__ inline
	void SetIndex(int index) {this->fIndex = index;}

	__host__ __device__ inline
	bool IsCached() const
	{ return this->fCached;}

	__host__ __device__ inline
	void SetCached(bool cached=1)
	{ this->fCached = cached; }



  	template<typename T1>
  	__host__ __device__ inline
  	return_type operator()(T1&& t )
  	{
  		return detail::accumulate<return_type,T1,F1,F2,Fs...>(std::forward<T1&>(t),fFtorTuple );

  	}

  	template<typename T1, typename T2>
  	__host__ __device__  inline
  	return_type operator()( T1&& t, T2&& cache)
  	{

  		return fCached ? detail::extract<return_type,T2>(fIndex, std::forward<T2&>(cache)):\
  				detail::accumulate2<return_type,T1,T2,F1,F2,Fs...>(t,cache,fFtorTuple );
  	}

  	template<typename T>
  	  	__host__ __device__ inline
  	  	return_type operator()( T* x, T* p)
  	  	{

  	  		return detail::accumulate2<return_type,T*,T*,F1,F2,Fs...>(x,p,fFtorTuple );
  	  	}



private:

	int  fIndex;
	bool fCached;
functors_type fFtorTuple;
};

// + operator two functors
template<typename T1, typename T2,
typename=typename std::enable_if< T1::is_functor::value && T2::is_functor::value> >
__host__  inline
Sum<T1, T2> operator+(T1 const& F1, T2 const& F2){ return  Sum<T1,T2>(F1, F2); }

template <typename T1, typename T2,
typename=typename std::enable_if< (std::is_convertible<T1, double>::value ||\
		std::is_constructible<thrust::complex<double>,T1>::value) && T2::is_functor::value>::type >
__host__  inline
Sum<Constant<T1>, T2>
operator+(T1 const cte, T2 const& F2){ return  Constant<T1>(cte)+F2; }


template <typename T1, typename T2,
typename=typename std::enable_if< (std::is_convertible<T1, double>::value ||\
		std::is_constructible<thrust::complex<double>,T1>::value) && T2::is_functor::value>::type >
__host__  inline
Sum<Constant<T1>, T2>
operator+(T2 const& F2, T1 const cte ){	return  Constant<T1>(cte)+F2; }

// Convenience function
template <typename F1, typename F2, typename ...Fs>
__host__  inline
Sum<F1, F2,Fs...>
sum(F1 const& f1, F2 const& f2, Fs const&... functors )
{ return  Sum<F1, F2,Fs... >(f1,f2, functors ... ); }


}



#endif /* SUM_H_ */
