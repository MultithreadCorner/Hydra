/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2025 Antonio Augusto Alves Junior
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
 * ZipIteratorUtility.h
 *
 *  Created on: 21/05/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef ZIPITERATORUTILITY_H_
#define ZIPITERATORUTILITY_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/IteratorTraits.h>
#include <hydra/detail/ZipIteratorTraits.h>
#include <hydra/detail/external/hydra_thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/hydra_thrust/type_traits/void_t.h>
#include <type_traits>

/*
#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

//----------------------------------------
template<typename ...T>
struct merged_tuple;

template<typename ...T>
struct merged_tuple<std::tuple<T...>>
{
    typedef std::tuple<T...> type;
};

template<typename ...T, typename ...U>
struct merged_tuple<std::tuple<T...>, std::tuple<U...>>:
       merged_tuple<std::tuple<T..., U...> > {};

template<typename ...Z, typename ...T, typename ...U>
struct merged_tuple<std::tuple<T...>, std::tuple<U...>, Z...>:
       merged_tuple<std::tuple<T..., U...>, Z... > {};

//----------------------------------------

template<template<typename Type> class Selector, typename TypeList>
struct select_tuple;

template<template<typename T> class Selector, typename Type>
struct select_tuple<Selector, std::tuple<Type>>
{
    typedef typename std::conditional<
                          Selector<Type>::value,
                          std::tuple<Type>,
                          std::tuple<>
                    >::type type;
};


template<template<typename T> class Selector, typename Head, typename ...Tail>
struct select_tuple<Selector, std::tuple<Head,Tail...> >
{
    typedef typename std::conditional<
               Selector<Head>::value,
               typename merged_tuple<
                            typename select_tuple<Selector, std::tuple<Head>   >::type,
                            typename select_tuple<Selector, std::tuple<Tail...>>::type
                        >::type,
               typename select_tuple<Selector, std::tuple<Tail...>>::type
            >::type type;
};



//------------------------------------------

template<template<typename Type> class Selector, typename TypeList, unsigned I=0>
struct filter_tuple;

template<template<typename T> class Selector, typename Type, unsigned I >
struct filter_tuple<Selector, std::tuple<Type>, I>
{
    typedef typename std::conditional<
                          Selector<Type>::value,
                          std::tuple<std::integral_constant<unsigned, I>>,
                          std::tuple<>
                    >::type type;
};


template<template<typename T> class Selector, typename Head, typename ...Tail, unsigned I >
struct filter_tuple<Selector, std::tuple<Head,Tail...>, I>
{
    typedef typename std::conditional<
               Selector<Head>::value,
               typename merged_tuple<
                            typename filter_tuple<Selector, std::tuple<Head>, I>::type,
                            typename filter_tuple<Selector, std::tuple<Tail...>, I+1>::type
                        >::type,
               typename filter_tuple<Selector, std::tuple<Tail...>, I+1>::type
            >::type type;
};

//-----------------------------------------

template<typename EntryList, typename ReturnType, typename T, size_t ...I>
ReturnType get_filtered_tuple_helper(std::index_sequence<I...>, T const& data )
{

    return std::make_tuple( std::get<std::tuple_element<I,EntryList>::type::value>(data)... );
}

template<template<typename T> class Selector, typename Head, typename ...Tail>
typename select_tuple<Selector, std::tuple<Head, Tail...>>::type
get_filtered_tuple( std::tuple<Head, Tail...> const& data )
{
    typedef typename filter_tuple<Selector, std::tuple<Head, Tail...>>::type list_type;
    typedef typename select_tuple<Selector, std::tuple<Head, Tail...>>::type return_type;

    return get_filtered_tuple_helper<list_type, return_type>( std::make_index_sequence<std::tuple_size<list_type>::value>{}, data );
}

//-----------------------------------

template<typename T>
struct is_tuple : std::false_type{ };

template<typename ...T>
struct is_tuple<std::tuple<T...>>: std::true_type {};

template<typename T>
struct do_tuple : std::conditional< is_tuple<T>::value, T, std::tuple<T> > {};

template<typename ...T>
struct flat_tuple: merged_tuple< typename do_tuple<T>::type... > {};

template<typename T>
typename std::enable_if<is_tuple<T>::value, T>::type
tupler( T const& data )
{
    return data;
}

template<typename T>
typename std::enable_if< !is_tuple<T>::value , std::tuple<T> >::type
tupler( T const& data )
{
    return std::make_tuple( data );
}

template<typename ...T>
typename  flat_tuple<T...>::type
get_flat_tuple(T const&... args)
{
    return std::tuple_cat( tupler(args)...);
}

int main()
{

  //typedef std::tuple<float, float, long, double, char, short, long, double> tuple_t;

  //auto data = get_filtered_tuple<std::is_floating_point>(tuple_t{0.0, 1.0, 2, 3.0, 4, 5 , 6, 7.0});
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^~~^^^~~~~~^^^~~~~~~~~~~~~^^^~~

  //std::cout << std::get<0>(data) << ", "<< std::get<1>(data) << ", "<< std::get<2>(data) << ", "<< std::get<3>(data) << std::endl;


  //flat_tuple<double, std::tuple<short, int>, long, std::tuple<float> >::type a;

  auto a = get_flat_tuple( 0.0, std::make_tuple(1, 2.0), 3.0 );



  a.printme;

  return 0;

}
 */

namespace hydra {

namespace detail {


template<typename ...T>
typename std::enable_if< all_true<is_zip_iterator<T>::value...>::value,
                 typename detail::merged_zip_iterator<T...>::type >::type
zip_iterator_cat( T const&... zip_iterators)
{
	return hydra::thrust::make_zip_iterator(
			 hydra::thrust::tuple_cat( zip_iterators.get_iterator_tuple()...) );
}

namespace meld_iterators_ns {

template<typename T>
auto convert_to_tuple(T&& iterator)
-> typename std::enable_if< detail::is_iterator<T>::value &&
                            detail::is_zip_iterator<T>::value,
    decltype( std::declval<T>().get_iterator_tuple() ) >::type
{
	return std::forward<T>(iterator).get_iterator_tuple();
}

template<typename T>
auto convert_to_tuple(T&& iterator)
-> typename std::enable_if< detail::is_iterator<T>::value &&
                          (!detail::is_zip_iterator<T>::value),
hydra::thrust::tuple<T> >::type
{
	return hydra::thrust::make_tuple(std::forward<T>(iterator));
}


}  // namespace meld_iterators_ns

template<typename ...Iterators>
auto meld_iterators(Iterators&&... iterators)
-> decltype( hydra::thrust::make_zip_iterator(
		hydra::thrust::tuple_cat( meld_iterators_ns::convert_to_tuple(std::forward<Iterators>(iterators))... )  ))
{
  return hydra::thrust::make_zip_iterator(
		  hydra::thrust::tuple_cat( meld_iterators_ns::convert_to_tuple(std::forward<Iterators>(iterators))... ));
}

}  // namespace detail

}  // namespace hydra


#endif /* ZIPITERATORUTILITY_H_ */
