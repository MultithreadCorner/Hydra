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
#include <hydra/detail/CompositeBase.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>

namespace hydra {


template<typename F0, typename F1, typename... Fs >
class Compose: public detail::CompositeBase<F0, F1, Fs...>
{
public:
	    //tag
	    typedef void hydra_functor_tag;
	    typedef typename F0::return_type  return_type;
		typedef typename HYDRA_EXTERNAL_NS::thrust::tuple<typename F1::return_type, typename Fs::return_type...> argument_type;



		Compose()=delete;

		Compose(F0 const& f0, F1 const& f1,  Fs const& ...fs):
			detail::CompositeBase<F0, F1, Fs...>( f0, f1,fs...)
		{ }

		__hydra_host__ __hydra_device__
		inline Compose(Compose<F0,F1,Fs...> const& other):
		detail::CompositeBase<F0, F1, Fs...>( other)
		{ }

		__hydra_host__ __hydra_device__
		inline Compose<F0,F1,Fs...>& operator=(Compose<F0,F1,Fs...> const& other)
		{
			if(this==&other) return *this;
			detail::CompositeBase<F0, F1, Fs...>::operator=( other);

			return *this;
		}

	  	template<typename T1>
	  	__hydra_host__ __hydra_device__
	  	inline return_type operator()(T1& x ) const
	  	{

	  		//evaluating f(g_1(x), g_2(x), ..., g_n(x))

	  		auto g = detail::dropFirst(this->fFtorTuple);

	  		auto f =  hydra::get<0>(this->fFtorTuple);

	  		typedef decltype(g) G_tuple ;

	  		return f(detail::invoke<G_tuple, T1>(x,g ));
	  	}


	  	template<typename T1, typename T2>
	  	__hydra_host__ __hydra_device__
	  	inline 	return_type operator()( T1& x, T2& cache) const
	  	{
	  		//evaluating f(g_1(x), g_2(x), ..., g_n(x))

	  		auto& g = detail::dropFirst(this->fFtorTuple);

	  		auto& f =  hydra::get<0>(this->fFtorTuple);

	  		typedef decltype(g) G_tuple ;

	  		return this->IsCached() ?
	  				detail::extract<return_type,T2>(this->GetIndex(), std::forward<T2&>(cache)):
	  				f(detail::invoke< G_tuple, T1, T2>(std::forward<T1&>(x), std::forward<T2&>(cache), g));
	  	}

};


// Conveniency function
template < typename T0, typename T1, typename ...Ts,
typename=typename std::enable_if<T0::is_functor::value &&
								 T1::is_functor::value &&
								 detail::all_true<Ts::is_functor::value...>::value >::type >

inline Compose<T0,T1,Ts...>
compose(T0 const& F0, T1 const& F1, Ts const&...Fs){

	return  Compose<T0,T1,Ts...>(F0, F1, Fs...);
}



}

#endif /* COMPOSE_H_ */
