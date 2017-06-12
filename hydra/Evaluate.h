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
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Containers.h>
#include <hydra/Function.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/Range.h>
#include <thrust/tuple.h>
#include <hydra/detail/Evaluate.inl>
#include <hydra/multivector.h>

namespace hydra {

//--------------------------------------
// Non cached functions
//--------------------------------------

/**
 * Evaluate a hydra functor on a range using the parallel policy
 *
 * @param policy : parallel policy
 * @param functor : hydra functor to be evaluated
 * @param begin : interator pointing to be begin of the range
 * @param end : interator pointing to be begin of the range
 * @return a vector with the results
 */
template< hydra::detail::Backend BACKEND, typename Iterator, typename Functor >
auto eval(hydra::detail::BackendPolicy<BACKEND>const& policy, Functor const& functor, Iterator begin, Iterator end)
-> typename hydra::detail::BackendPolicy<BACKEND>::template container<typename Functor::return_type>
{

	typedef	typename hydra::detail::BackendPolicy<BACKEND>::template
			container<typename Functor::return_type> container;
	size_t size = thrust::distance(begin, end) ;
	container Table( size );

	//auto fBegin = thrust::make_zip_iterator(thrust::make_tuple(begin) );
	//auto fEnd   = thrust::make_zip_iterator(thrust::make_tuple(end)   );

	thrust::transform(begin, end ,  Table.begin(), functor );

	return std::move(Table);
}

/**
 * Evaluate a tuple of hydra functors on a range using the parallel policy
 *
 * @param policy : parallel policy
 * @param functor : hydra functor to be evaluated
 * @param begin : interator pointing to be begin of the range
 * @param end : interator pointing to be begin of the range
 * @return a multivectors with the results
 */
template<hydra::detail::Backend BACKEND, typename Iterator, typename ...Functors>
auto eval(hydra::detail::BackendPolicy<BACKEND>const&  policy,thrust::tuple<Functors...> const& functors, Iterator begin, Iterator end)
-> multivector< typename hydra::detail::BackendPolicy<BACKEND>::template
container<thrust::tuple<typename Functors::return_type ...> >>
{
	typedef multivector<typename hydra::detail::BackendPolicy<BACKEND>::template container<
			       thrust::tuple<typename Functors::return_type ...> >> container;

	size_t size = thrust::distance(begin, end) ;
	container Table( size );

	//auto fBegin = thrust::make_zip_iterator(thrust::make_tuple(begin) );
	//auto fEnd   = thrust::make_zip_iterator(thrust::make_tuple(end)   );


	thrust::transform(begin, end ,  Table.begin(),
			detail::process< thrust::tuple<typename Functors::return_type ...>,
			thrust::tuple<Functors...>>(functors) );

	return std::move(Table);
}

/**
 * Evaluate a functor over a list of ranges
 *
 * @param policy : parallel policy
 * @param functor : hydra functor to be evaluated
 * @param begin : interator pointing to be begin of the range
 * @param end : interator pointing to be begin of the range
 * @param begins : interator pointing to be begin of the range
 * @return a multivectors with the results
 */
template<hydra::detail::Backend BACKEND, typename Functor, typename Iterator, typename ...Iterators>
auto eval(hydra::detail::BackendPolicy<BACKEND>const&  policy,Functor const& functor, Iterator begin, Iterator end, Iterators... begins)
-> typename hydra::detail::BackendPolicy<BACKEND>::template
container<typename Functor::return_type>
{
	typedef typename hydra::detail::BackendPolicy<BACKEND>::template
			container<typename Functor::return_type> container;


	size_t size = thrust::distance(begin, end) ;
	container Table( size );

	auto fBegin = thrust::make_zip_iterator(thrust::make_tuple(begin, begins...) );
	auto fEnd   = thrust::make_zip_iterator(thrust::make_tuple(end  , (begins+size)...) );

	thrust::transform(begin, end,  Table.begin(), functor );

	return std::move(Table);
}


/**
 * Evaluate a tuple of functors over a list of ranges
 *
 * @param policy : parallel policy
 * @param functor : hydra functor to be evaluated
 * @param begin : interator pointing to be begin of the range
 * @param end : interator pointing to be begin of the range
 * @param begins : interator pointing to be begin of the range
 * @return a multivectors with the results
 */
template<hydra::detail::Backend BACKEND, typename Iterator,  typename ...Iterators, typename ...Functors>
auto eval(hydra::detail::BackendPolicy<BACKEND>const&  policy, thrust::tuple<Functors...> const& functors,
		Iterator begin, Iterator end, Iterators... begins)
-> multivector< typename hydra::detail::BackendPolicy<BACKEND>::template
container<thrust::tuple<typename Functors::return_type ...> >>
{

	typedef multivector<	typename hydra::detail::BackendPolicy<BACKEND>::template container<
			            thrust::tuple<typename Functors::return_type ...> >> container;

	size_t size = thrust::distance(begin, end) ;
	container Table( size );

	auto fBegin = thrust::make_zip_iterator(thrust::make_tuple(begin, begins...) );
	auto fEnd   = thrust::make_zip_iterator(thrust::make_tuple(end  , (begins+size)...) );

	thrust::transform(fBegin, fEnd ,  Table.begin(),
			detail::process< thrust::tuple<typename Functors::return_type ...>,
			thrust::tuple<Functors...>>(functors) );

	return std::move(Table);
}

}/* namespace hydra */



#endif /* EVALUATE */
