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
//#include <hydra/detail/CompositeBase.h>
#include <hydra/detail/FunctorTraits.h>
#include <hydra/detail/CompositeTraits.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>
#include <hydra/detail/BaseCompositeFunctor.h>
#include <hydra/detail/TupleUtility.h>


namespace hydra {


namespace detail {

namespace compose_signature {



}  // namespace compose_signature

}  // namespace detail


template<typename F0, typename F1, typename... Fs >
class Compose: public BaseCompositeFunctor<
						Compose<F0, F1,Fs...>,
						hydra_thrust::tuple<F0, F1, Fs...>,
						typename detail::merged_tuple<
						hydra_thrust::tuple<typename F0::return_type>,
						typename F1::argument_type, typename Fs::argument_type ... >::type >
{

	typedef  BaseCompositeFunctor<
			    Compose<F0, F1,Fs...>,
			    hydra_thrust::tuple<F0, F1, Fs...>,
			    typename detail::merged_tuple< hydra_thrust::tuple<typename F0::return_type>,
			    typename F1::argument_type, typename Fs::argument_type ... >::type > super_type;



public:


		Compose()=delete;

		Compose(F0 const& f0, F1 const& f1,  Fs const& ...fs):
		 super_type( f0, f1,fs...)
		{ }

		__hydra_host__ __hydra_device__
		inline Compose(Compose<F0,F1,Fs...> const& other):
	      super_type( other)
		{ }

		__hydra_host__ __hydra_device__
		inline Compose<F0,F1,Fs...>& operator=(Compose<F0,F1,Fs...> const& other)
		{
			if(this==&other) return *this;
			super_type::operator=( other);

			return *this;
		}

	  	template<typename ...T>
	  	__hydra_host__ __hydra_device__
	  	inline typename  super_type::return_type Evaluate(T... x ) const
	  	{

	  		auto g = detail::dropFirst(this->GetFunctors());

	  		auto f =  hydra::get<0>(this->GetFunctors());

	  		typedef decltype(g) G_tuple ;

	  		return f(detail::invoke<G_tuple, hydra_thrust::tuple<T...>>( hydra_thrust::tie(x...),g ));
	  	}


};


// Conveniency function
template < typename T0, typename T1, typename ...Ts >
inline typename std::enable_if<
(detail::is_hydra_functor<T0>::value || detail::is_hydra_lambda<T0>::value || detail::is_hydra_composite_functor<T0>::value) &&
(detail::is_hydra_functor<T1>::value || detail::is_hydra_lambda<T1>::value || detail::is_hydra_composite_functor<T1>::value) &&
detail::all_true<
(detail::is_hydra_functor<Ts>::value || detail::is_hydra_lambda<Ts>::value || detail::is_hydra_composite_functor<Ts>::value)...>::value,
Compose<T0,T1,Ts...>>::type
compose(T0 const& F0, T1 const& F1, Ts const&...Fs){

	return  Compose<T0,T1,Ts...>(F0, F1, Fs...);
}



}

#endif /* COMPOSE_H_ */
