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
 * ArgumentTraits.h
 *
 *  Created on: 15/02/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef ARGUMENTTRAITS_H_
#define ARGUMENTTRAITS_H_

#include <hydra/detail/Config.h>
#include<hydra/detail/utility/StaticAssert.h>
#include <hydra/detail/utility/Generic.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/tuple_of_iterator_references.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <type_traits>
#include <tuple>

//this->GetNumberOfParameters(), this->GetParameters(),

namespace hydra {

namespace detail {

template<typename Derived, typename Type>
struct FunctionArgument;

template<typename RefT, typename ...T>
struct is_valid_type_pack;

template<typename ...RefT, typename ...T>
struct is_valid_type_pack<hydra_thrust::tuple<RefT...>, T... >:
std::is_convertible<std::tuple<T...>, std::tuple<RefT...> > {};

template<typename ...RefT, typename ...T>
struct is_valid_type_pack<hydra_thrust::tuple<RefT...>, hydra_thrust::device_reference<T>...>:
       hydra_thrust::detail::is_convertible<hydra_thrust::tuple<T...>,  hydra_thrust::tuple<RefT...> > {};


template<typename ArgType>
struct is_tuple_type: std::false_type {};

template<typename... ArgTypes>
struct is_tuple_type<hydra_thrust::detail::tuple_of_iterator_references<ArgTypes...>>:
    std::true_type {};

template<typename... ArgTypes>
struct is_tuple_type<hydra_thrust::tuple<ArgTypes...>>:
    std::true_type {};

namespace fa_impl {

template<typename T>
struct void_tupe{ typedef void type; };

template<typename T, typename U=void>
struct has_value_type: std::false_type{};

template<typename T>
struct has_value_type<T,
    typename void_tupe< typename T::value_type>::type>:
std::true_type{};


}  // namespace function_argument_impl


template<typename Arg, bool B=fa_impl::has_value_type<Arg>::value>
struct is_function_argument;

template<typename Arg>
struct is_function_argument<Arg, false>:std::false_type{} ;


template<typename Arg>
struct is_function_argument<Arg, true>:
std::is_base_of<detail::FunctionArgument<Arg, typename Arg::value_type>, Arg>{} ;


template<typename Arg>
struct is_function_argument<hydra_thrust::device_reference<Arg>, true>:
std::is_base_of<detail::FunctionArgument<Arg, typename Arg::value_type>, Arg>{} ;

//----------------

template<typename... ArgTypes>
struct is_function_argument_pack:
		detail::all_true<is_function_argument<ArgTypes>::value...>{};

//----------------

template<typename ArgType>
struct is_tuple_of_function_arguments: std::false_type {};



template<typename... ArgTypes>
struct is_tuple_of_function_arguments<hydra_thrust::detail::tuple_of_iterator_references<ArgTypes&...> >:
is_function_argument_pack<typename std::decay<ArgTypes>::type...>{} ;

template<typename... ArgTypes>
struct is_tuple_of_function_arguments<
hydra_thrust::detail::tuple_of_iterator_references<hydra_thrust::device_reference<ArgTypes>...> >:
is_function_argument_pack<typename std::decay<ArgTypes>::type...>{} ;


template<typename... ArgTypes>
struct is_tuple_of_function_arguments<hydra_thrust::tuple<ArgTypes...>>:
is_function_argument_pack<typename std::decay<ArgTypes>::type...> {};

}  // namespace detail

}  // namespace hydra

#endif /* ARGUMENTTRAITS_H_ */
