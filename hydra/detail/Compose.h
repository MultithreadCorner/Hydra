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
 * Compose.h
 *
 *  Created on: 11/07/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef COMPOSE_H_
#define COMPOSE_H_



#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/TypeTraits.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/base_functor.h>
#include <hydra/detail/Constant.h>
#include <hydra/Parameter.h>

namespace hydra {


template<typename F0, typename F1, typename... Fs >
//struct  Compose :public detail::compose_base_functor<F0,F1,Fs...>::type
struct  Compose
{
	    //tag
	    typedef void hydra_functor_tag;
	    //typedef typename detail::compose_base_functor<F0, F1,Fs...>::type base_type;
		typedef typename F0::return_type  return_type;
		typedef typename thrust::tuple<typename F1::return_type, typename Fs::return_type...> argument_type;
		typedef typename thrust::tuple<F1, Fs...> functors_type;

		__host__
		Compose():
		fIndex(-1),
		fCached(0)
		{};

		__host__
		Compose(F0 const& f0, F1 const& f1,  Fs const& ...fs):
		fIndex(-1),
		fCached(0),
		fF0(f0),
	  	fFtorTuple(thrust::make_tuple(f1, fs...))
	  	{ }

		__host__ __device__ inline
		Compose(Compose<F0,F1,Fs...> const& other):
		    fF0( other.GetF0()),
			fFtorTuple( other.GetFunctors() ),
			fIndex( other.GetIndex() ),
			fCached( other.IsCached() )
		{ };

		__host__ __device__ inline
		Compose<F0,F1,Fs...>& operator=(Compose<F0,F1,Fs...> const& other)
		{
			this->fF0= other.GetF0();
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

		__host__ __device__ inline
		functors_type GetFunctors() const {return this->fFtorTuple;}

		__host__ __device__ inline
		F0 GetF0() const {return this->fF0;}

		__host__ __device__ inline
		int GetIndex() const { return this->fIndex; }

		__host__ __device__ inline
		void SetIndex(int index) {this->fIndex = index;}

		__host__ __device__ inline
		bool IsCached() const
		{ return this->fCached;}

		__host__ __device__ inline
		void SetCached(bool cached=true)
		{ this->fCached = cached; }


	  	template<typename T1>
	  	__host__ __device__ inline
	  	return_type operator()(T1& t )
	  	{
	  		return fF0.Evaluate(detail::invoke<argument_type, functors_type, T1>(std::forward<T1&>(t), fFtorTuple));
	  	}

	  	template<typename T1, typename T2>
	  	__host__ __device__  inline
	  	return_type operator()( T1& t, T2& cache)
	  	{
	  		return fCached ? detail::extract<return_type,T2>(fIndex, std::forward<T2&>(cache)):\
	  				fF0.Evaluate(detail::invoke<argument_type, functors_type, T1, T2>(std::forward<T1&>(t),
	  						std::forward<T2&>(cache),fFtorTuple));
	  	}

	private:
	  	F0 fF0;
		functors_type fFtorTuple;
		int  fIndex;
		bool fCached;

};



// Conveniency function
template < typename T0, typename T1, typename ...Ts,
typename=typename std::enable_if< detail::all_true<T0::is_functor::value, T1::is_functor::value,Ts::is_functor::value...>::value >::type >
__host__ inline
Compose<T0,T1,Ts...>
compose(T0 const& F0, T1 const& F1, Ts const&...Fs)
{
	return  Compose<T0,T1,Ts...>(F0, F1, Fs...);
}



}

#endif /* COMPOSE_H_ */
