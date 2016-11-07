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
 * Chain.h
 *
 *  Created on: 05/11/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CHAIN_H_
#define CHAIN_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/experimental/Events.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/functors/FlagAcceptReject.h>
//thrust
#include <thrust/tuple.h>
#include <thrust/zip_iterator.h>
#include <thrust/distance.h>

namespace hydra {

namespace experimental {

template<typename ...Decays>
struct Chain;

template<size_t ...N,  unsigned int BACKEND>
struct Chain< hydra::experimental::Events<N,BACKEND >...>{

	typedef thrust::tuple<typename
				hydra::experimental::Events<N,BACKEND >...> event_tuple;

	typedef thrust::tuple<typename
			hydra::experimental::Events<N,BACKEND >::iterator...> iterator_tuple;
	typedef thrust::tuple<typename
			hydra::experimental::Events<N,BACKEND >::const_iterator...> const_iterator_tuple;

	typedef typename system_t::template container<GReal_t>  vector_real;
	typedef typename system_t::template container<GBool_t>  vector_bool;
	//zipped iterators
	typedef thrust::zip_iterator<iterator_tuple>        iterator;
	typedef thrust::zip_iterator<const_iterator_tuple>  const_iterator;

	typedef   typename iterator::value_type value_type;
	typedef   typename iterator::reference  reference_type;

	template<typename BACKEND2>
	Chain(hydra::experimental::Chain<N...,BACKEND2 >const& other):
		fStorage(std::move(thrust::make_tuple( std::move(events)...)))
	{
			fBegin = thrust::make_zip_iterator( detail::begin_call_args(fStorage) );
			fConstBegin = thrust::make_zip_iterator( detail::cbegin_call_args(fStorage) );
			fEnd = thrust::make_zip_iterator( detail::end_call_args(fStorage) );
			fConstEnd = thrust::make_zip_iterator( detail::cend_call_args(fStorage) );
			fSize=(size_t) thrust::distance(fBegin, fEnd);

			fWeights( fSize ,1.0);
			fFlags( fSize, 1.0 );


	}

	Chain(hydra::experimental::Events<N,BACKEND >&& ...events):
		fStorage(std::move(thrust::make_tuple( std::move(events)...)))
	{

		auto begin  = thrust::make_zip_iterator( detail::begin_call_args(fStorage) );
		auto end    = thrust::make_zip_iterator( detail::end_call_args(fStorage) );

		fSize=(size_t) thrust::distance(begin, end);
		fWeights( fSize ,1.0);
		fFlags( fSize, 1.0 );

		fBegin = thrust::make_zip_iterator(thrust:: tuple_cat(fWeights.begin(),
				detail::begin_call_args(fStorage)) );
		fConstBegin = thrust::make_zip_iterator( thrust:: tuple_cat(fWeights.cbegin(),
				detail::cbegin_call_args(fStorage) ));
		fEnd = thrust::make_zip_iterator( detail::end_call_args(fStorage) );
		fConstEnd = thrust::make_zip_iterator( detail::cend_call_args(fStorage) );
		fSize=(size_t) thrust::distance(fBegin, fEnd);


	}


	iterator begin(){ return fBegin; }
	iterator  end(){ return fEnd; }
	const_iterator begin() const{ return fConstBegin; }
	const_iterator  end() const{ return fConstEnd; }

private:

	event_tuple fStorage;
	vector_bool fFlags; ///< Vector of flags. Accepted events are flagged 1 and rejected 0.
	vector_real fWeights; ///< Vector of event weights.
	iterator fBegin;
	iterator fEnd;
	const_iterator fConstBegin;
	const_iterator fConstEnd;
	size_t   fSize;


};

}  // namespace experimental

}  // namespace hydra


#endif /* CHAIN_H_ */
