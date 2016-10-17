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
 * EvaluateTuple.h
 *
 *  Created on: 19/06/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup generic
 */

#ifndef EVALUATE_H_
#define EVALUATE_H_

#include <array>
#include <type_traits>



#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Containers.h>
#include <hydra/Function.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/Range.h>
#include <thrust/tuple.h>
#include <hydra/detail/Evaluate.inl>

namespace hydra {


template<template<class, class...> class V=mc_device_vector, typename ...T>
struct EvalReturnType{ typedef V<thrust::tuple<T...>> type; };




//--------------------------------------
// Non cached functions
//--------------------------------------

template< typename Iterator,typename ...Iterators, typename Functor,
typename thrust::detail::enable_if<
hydra::detail::are_all_same<typename Range<Iterator>::system ,
typename Range<Iterators>::system...>::value, int>::type=0>
auto Eval(Functor const& t,Range<Iterator>const& range, Range<Iterators>const&...  ranges) ->
typename detail::if_then_else<std::is_same<thrust::device_system_tag,typename Range<Iterator>::system>::value,
mc_device_vector<typename Functor::return_type>,
mc_host_vector<typename Functor::return_type>>::type
{
	typedef typename detail::if_then_else<
			std::is_same<thrust::device_system_tag,typename  Range<Iterator>::system>::value,
			mc_device_vector<typename Functor::return_type>,
			mc_host_vector<typename Functor::return_type>>::type container;

	auto begin = thrust::make_zip_iterator(thrust::make_tuple(range.begin(), ranges.begin()...) );
	auto end   = thrust::make_zip_iterator(thrust::make_tuple(range.end(), ranges.end()...) );

	container Table( thrust::distance(begin, end) );

	thrust::transform(begin, end ,  Table.begin(),t );

	return Table;
}


template< typename Iterator,typename ...Iterators, typename ...Functors,
typename thrust::detail::enable_if<
hydra::detail::are_all_same<typename Range<Iterator>::system ,
typename Range<Iterators>::system...>::value, int>::type=0>
auto Eval(thrust::tuple<Functors...> const& t,Range<Iterator>const& range, Range<Iterators>const&...  ranges) ->
typename detail::if_then_else<std::is_same<thrust::device_system_tag,typename Range<Iterator>::system>::value,
typename EvalReturnType<mc_device_vector,typename Functors::return_type ...>::type,
typename EvalReturnType<mc_host_vector,typename Functors::return_type ...>::type>::type
{
	typedef typename detail::if_then_else<
			std::is_same<thrust::device_system_tag,typename  Range<Iterator>::system>::value,
			mc_device_vector< thrust::tuple<typename Functors::return_type ...> >,
			mc_host_vector< thrust::tuple<typename Functors::return_type ...> > >::type container;

	auto begin = thrust::make_zip_iterator(thrust::make_tuple(range.begin(), ranges.begin()...) );
	auto end   = thrust::make_zip_iterator(thrust::make_tuple(range.end(), ranges.end()...) );

	container Table( thrust::distance(begin, end) );

	thrust::transform(begin, end ,  Table.begin(),
			detail::process< thrust::tuple<typename Functors::return_type ...>, thrust::tuple<Functors...>>(t) );

	return Table;
}


//--------------------------------------
// Cached functions
//--------------------------------------
template< typename Tuple, typename Iterator,typename ...Iterators, typename Functor,
typename thrust::detail::enable_if<
hydra::detail::are_all_same<typename Range<Iterator>::system ,
typename Range<Iterators>::system...>::value, int>::type=0>
auto Eval(Functor const& t,
		typename detail::if_then_else<std::is_same<thrust::device_system_tag,typename Range<Iterator>::system>::value,
		mc_device_vector<Tuple>, mc_host_vector<Tuple>> const& Cache, Range<Iterator>const& range, Range<Iterators>const&...  ranges)
-> typename detail::if_then_else<std::is_same<thrust::device_system_tag,typename Range<Iterator>::system>::value,
typename EvalReturnType<mc_device_vector,typename Functor::return_type >::type,
typename EvalReturnType<mc_host_vector,typename Functor::return_type >::type>::type
{
	typedef typename detail::if_then_else<
			std::is_same<thrust::device_system_tag,typename  Range<Iterator>::system>::value,
			mc_device_vector< thrust::tuple<typename Functor::return_type > >,
			mc_host_vector< thrust::tuple<typename Functor::return_type > > >::type container;

	auto begin = thrust::make_zip_iterator(thrust::make_tuple(range.begin(), ranges.begin()...) );
	auto end   = thrust::make_zip_iterator(thrust::make_tuple(range.end(), ranges.end()...) );

	container Table( thrust::distance(begin, end) );

	thrust::transform(begin, end , Cache.begin(), Table.begin(), t );

	return Table;
}



template< typename Tuple, typename Iterator,typename ...Iterators, typename ...Functors,
typename thrust::detail::enable_if<
hydra::detail::are_all_same<typename Range<Iterator>::system ,
typename Range<Iterators>::system...>::value, int>::type=0>
auto Eval(thrust::tuple<Functors...> const& t,
		typename detail::if_then_else<std::is_same<thrust::device_system_tag,typename Range<Iterator>::system>::value,
		mc_device_vector<Tuple>, mc_host_vector<Tuple>> const& Cache, Range<Iterator>const& range, Range<Iterators>const&...  ranges)
-> typename detail::if_then_else<std::is_same<thrust::device_system_tag,typename Range<Iterator>::system>::value,
typename EvalReturnType<mc_device_vector,typename Functors::return_type ...>::type,
typename EvalReturnType<mc_host_vector,typename Functors::return_type ...>::type>::type
{
	typedef typename detail::if_then_else<
			std::is_same<thrust::device_system_tag,typename  Range<Iterator>::system>::value,
			mc_device_vector< thrust::tuple<typename Functors::return_type ...> >,
			mc_host_vector< thrust::tuple<typename Functors::return_type ...> > >::type container;

	auto begin = thrust::make_zip_iterator(thrust::make_tuple(range.begin(), ranges.begin()...) );
	auto end   = thrust::make_zip_iterator(thrust::make_tuple(range.end(), ranges.end()...) );

	container Table( thrust::distance(begin, end) );

	thrust::transform(begin, end , Cache.begin(), Table.begin(),
			detail::process< thrust::tuple<typename Functors::return_type ...>, thrust::tuple<Functors...>>(t) );

	return Table;
}


}/* namespace hydra */



#endif /* EVALUATE */
