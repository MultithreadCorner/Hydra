/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2023 Antonio Augusto Alves Junior
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
 * TupleTraits.h
 *
 *  Created on: 27/05/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef TUPLETRAITS_H_
#define TUPLETRAITS_H_


#include <hydra/detail/Config.h>

#include <hydra/Tuple.h>
#include <hydra/detail/FindUniqueType.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/type_traits/void_t.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <type_traits>
#include <utility>

namespace hydra {

namespace detail {

template<typename T>
struct is_tuple : std::false_type{ };

template<typename ...T>
struct is_tuple<hydra::tuple<T...>>: std::true_type {};

//-----------------------------------------------

template<typename T, typename Tuple>
struct is_intuple : std::false_type{ };

template<typename T>
struct is_intuple<T, hydra::tuple<>>:std::false_type{};

template<typename T, typename Head, typename ...Tail>
struct is_intuple<T, hydra::tuple<Head, Tail...>>:
	std::conditional< std::is_same<T,Head>::value,
		std::true_type,  is_intuple<T, hydra::tuple<Tail...> > >::type{};

//----------------------------------------
template<typename ...T>
struct merged_tuple;

template<typename ...T>
struct merged_tuple<hydra::tuple<T...>>
{
    typedef hydra::tuple<T...> type;
};

template<typename ...T, typename ...U>
struct merged_tuple<hydra::tuple<T...>, hydra::tuple<U...>>:
       merged_tuple<hydra::tuple<T..., U...> > {};

template<typename ...Z, typename ...T, typename ...U>
struct merged_tuple<hydra::tuple<T...>, hydra::tuple<U...>, Z...>:
       merged_tuple<hydra::tuple<T..., U...>, Z... > {};

//----------------------------------------

template<template<typename Type> class Selector, typename TypeList>
struct selected_tuple;

template<template<typename T> class Selector, typename Type>
struct selected_tuple<Selector, hydra::tuple<Type>>
{
    typedef typename std::conditional<
                          Selector<Type>::value,
                             hydra::tuple<Type>,
                             hydra::tuple<> >::type type;
};

template<template<typename T> class Selector, typename Head, typename ...Tail>
struct selected_tuple<Selector, hydra::tuple<Head,Tail...> >
{
    typedef typename std::conditional<
               Selector<Head>::value,
               typename merged_tuple<
                            typename selected_tuple<Selector, hydra::tuple<Head>   >::type,
                            typename selected_tuple<Selector, hydra::tuple<Tail...>>::type
                        >::type,
               typename selected_tuple<Selector, hydra::tuple<Tail...>>::type
            >::type type;
};

//================================================

namespace stuple_impl {

template<typename T, typename U>
struct stripped_tuple_impl;

template<typename ...T>
struct stripped_tuple_impl< hydra::tuple<T...>,  hydra::tuple< > >
{
  typedef  hydra::tuple<T...> type;
};

template< typename Tuple, typename Head, typename ...Tail>
struct stripped_tuple_impl< Tuple, hydra::tuple<Head, Tail... > > :
       stripped_tuple_impl< typename std::conditional< (detail::is_intuple<Head, Tuple>::value),
                    Tuple,  typename detail::merged_tuple<Tuple, hydra::tuple<Head> >::type >::type, hydra::tuple<Tail...> >{};

}  // namespace stuple_impl


template<typename T>
struct stripped_tuple;

template<typename ...T>
struct stripped_tuple<hydra::tuple<T...>>: stuple_impl::stripped_tuple_impl<  hydra::tuple< >,  hydra::tuple<T...> >{};


//------------------------------------------


template<typename Type, typename Tuple>
struct index_in_tuple;


template<typename Type, typename Head, typename ...Tail>
struct index_in_tuple<Type, hydra::thrust::tuple<Head, Tail...> >
: find_unique_type<Type, Head, Tail...>{};

}  // namespace detail

}  // namespace hydra


#endif /* TUPLETRAITS_H_ */
