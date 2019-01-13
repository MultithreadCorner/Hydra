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
 * FunctorTraits.h
 *
 *  Created on: 25/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup functor
 */


#ifndef FUNCTORTRAITS_H_
#define FUNCTORTRAITS_H_


#include <type_traits>
#include <hydra/Types.h>


namespace hydra {

namespace detail {

//tags to identify hydra pdf and functors

template<class T, class R = void>
struct tag_type { typedef R type; };


//sum of pdfs
template<class T, class Enable = void>
struct is_hydra_sum_pdf: std::false_type {};

template<class T>
struct is_hydra_sum_pdf<T, typename tag_type< typename T::hydra_sum_pdf_tag>::type>: std::true_type {};

//pdf
template<class T, class Enable = void>
struct is_hydra_pdf: std::false_type {};

template<class T>
struct is_hydra_pdf<T, typename tag_type< typename T::hydra_pdf_tag>::type>: std::true_type {};

//functor
template<class T, class Enable = void>
struct is_hydra_functor: std::false_type {};

template<class T>
struct is_hydra_functor<T, typename tag_type< typename T::hydra_functor_tag>::type>: std::true_type {};

//integrator
template<class T, class Enable = void>
struct is_hydra_integrator: std::false_type {};

template<class T>
struct is_hydra_integrator<T, typename tag_type< typename T::hydra_integrator_tag>::type>: std::true_type {};

//storable
template<class T, class Enable = void>
struct is_hydra_convertible_to_tuple: std::false_type {};

template<class T>
struct is_hydra_convertible_to_tuple<T, typename tag_type< typename T::hydra_convertible_to_tuple_tag>::type>: std::true_type {};



// hint from http://stackoverflow.com/a/16824239/6147498

// Primary template with a static assertion
// for a meaningful error message
// if it ever gets instantiated.
// We could leave it undefined if we didn't care.

template<typename, typename T>
struct has_analytical_integral {
	static_assert(
			std::integral_constant<T, false>::value,
			"Second template parameter needs to be of function type.");
};

// specialization that does the checking

template<typename C, typename Ret, typename... Args>
struct has_analytical_integral<C, Ret(Args...)> {
private:
	template<typename T>
	static constexpr auto check(T*)
	-> typename
	std::is_same<
	decltype( std::declval<T>().AnalyticalIntegral( std::declval<Args>()... ) ),
	Ret     // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	>::type;// attempt to call it and see if the return type is correct

	template<typename>
	static constexpr std::false_type check(...);

	typedef decltype(check<C>(0)) type;

public:
	static constexpr bool value = type::value;
};

template<typename Functor>
bool HasAnalyticalIntegral( Functor&){
	return has_analytical_integral<Functor, GReal_t(const GReal_t*,  const GReal_t*)>::value;
}

}  // namespace detail
}// namespace hydra

#endif /* FUNCTORTRAITS_H_ */
