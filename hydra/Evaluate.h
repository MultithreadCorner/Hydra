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
 * EvaluateTuple.h
 *
 *  Created on: 19/06/2016
 *      Author: Antonio Augusto Alves Junior
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
#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/detail/Evaluate.inc>
#include <hydra/multivector.h>

namespace hydra {

//--------------------------------------
// Non cached functions
//--------------------------------------

/**
 * @ingroup generic
 * Evaluate a hydra functor on a range using the parallel policy
 *
 * @param policy : parallel policy
 * @param functor : hydra functor to be evaluated
 * @param begin : interator pointing to be begin of the range
 * @param end : interator pointing to be begin of the range
 * @return a vector with the results
 */
template< hydra::detail::Backend BACKEND, typename Iterator, typename Functor >
auto eval(hydra::detail::BackendPolicy<BACKEND>, Functor const& functor, Iterator begin, Iterator end)
-> typename hydra::detail::BackendPolicy<BACKEND>::template container<typename Functor::return_type> ;

/**
 * @ingroup generic
 * Evaluate a tuple of hydra functors on a range using the parallel policy
 *
 * @param policy : parallel policy
 * @param functor : hydra functor to be evaluated
 * @param begin : interator pointing to be begin of the range
 * @param end : interator pointing to be begin of the range
 * @return a multivectors with the results
 */
template<hydra::detail::Backend BACKEND, typename Iterator, typename ...Functors>
auto eval(hydra::detail::BackendPolicy<BACKEND> ,HYDRA_EXTERNAL_NS::thrust::tuple<Functors...> const& functors, Iterator begin, Iterator end)
-> multivector<HYDRA_EXTERNAL_NS::thrust::tuple<typename Functors::return_type ...> , hydra::detail::BackendPolicy<BACKEND>>;
//-> multivector< typename hydra::detail::BackendPolicy<BACKEND>::template
//container<HYDRA_EXTERNAL_NS::thrust::tuple<typename Functors::return_type ...> >>;


/**
 * @ingroup generic
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
auto eval(hydra::detail::BackendPolicy<BACKEND>, Functor const& functor, Iterator begin, Iterator end, Iterators... begins)
-> typename hydra::detail::BackendPolicy<BACKEND>::template container<typename Functor::return_type>;

/**
 * @ingroup generic
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
auto eval(hydra::detail::BackendPolicy<BACKEND>, HYDRA_EXTERNAL_NS::thrust::tuple<Functors...> const& functors,
		Iterator begin, Iterator end, Iterators... begins)
-> multivector<HYDRA_EXTERNAL_NS::thrust::tuple<typename Functors::return_type ...> , hydra::detail::BackendPolicy<BACKEND> >;

//-> multivector< typename hydra::detail::BackendPolicy<BACKEND>::template container<HYDRA_EXTERNAL_NS::thrust::tuple<typename Functors::return_type ...> >>;


}/* namespace hydra */

#include <hydra/detail/Evaluate.inl>

#endif /* EVALUATE */
