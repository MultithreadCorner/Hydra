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

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <cassert>

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Containers.h>
#include <hydra/experimental/Events.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/utility/Generic.h>
#include <hydra/detail/functors/FlagAcceptReject.h>
//thrust
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/distance.h>

namespace hydra {

namespace experimental {

template<typename ...Decays>
struct Chain;

template<size_t ...N,  unsigned int BACKEND>
struct Chain< hydra::experimental::Events<N,BACKEND >...>{

	typedef hydra::detail::BackendTraits<BACKEND> system_t;

	typedef thrust::tuple<typename
				hydra::experimental::Events<N,BACKEND >...> event_tuple;

	typedef thrust::tuple<typename
			hydra::experimental::Events<N,BACKEND >::iterator...> iterator_tuple;
	typedef thrust::tuple<typename
			hydra::experimental::Events<N,BACKEND >::const_iterator...> const_iterator_tuple;

	typedef typename system_t::template container<GReal_t>  vector_real;
	typedef typename system_t::template container<GReal_t>::iterator vector_real_iterator;
	typedef typename system_t::template container<GReal_t>::const_iterator vector_real_const_iterator;

	typedef typename system_t::template container<GBool_t>  vector_bool;
	typedef typename system_t::template container<GBool_t>::iterator  vector_bool_iterator;
	typedef typename system_t::template container<GBool_t>::const_iterator  vector_bool_const_iterator;

	//zipped iterators
	typedef thrust::zip_iterator<
			decltype(thrust:: tuple_cat(thrust::tuple<vector_real_iterator>(), iterator_tuple()))>  iterator;
	typedef thrust::zip_iterator<
			decltype(thrust:: tuple_cat(thrust::tuple<vector_real_const_iterator>(), const_iterator_tuple()))>  const_iterator;

	typedef   typename iterator::value_type value_type;
	typedef   typename iterator::reference  reference_type;

	typedef decltype(hydra::detail::make_index_sequence<sizeof...(N)> { }) indexing_type;

	Chain(){};

	Chain(size_t nevents):
	fStorage(thrust::make_tuple(hydra::experimental::Events<N,BACKEND >(nevents)...) ),
	fSize(nevents)
	{

		fWeights.resize(fSize);
		fFlags.resize(fSize);

		fBegin = thrust::make_zip_iterator(thrust::tuple_cat(thrust::make_tuple(fWeights.begin()),
				detail::begin_call_args(fStorage)) );
		fEnd = thrust::make_zip_iterator( thrust::tuple_cat(thrust::make_tuple(fWeights.end()),
				detail::end_call_args(fStorage)) );

		fConstBegin = thrust::make_zip_iterator(thrust::tuple_cat(thrust::make_tuple(fWeights.cbegin()),
				detail::cbegin_call_args(fStorage) ));
		fConstEnd = thrust::make_zip_iterator(thrust::tuple_cat(thrust::make_tuple(fWeights.cend()),
				detail::cend_call_args(fStorage) ) );

	}


	Chain(hydra::experimental::Chain<hydra::experimental::Events<N,BACKEND >...>const& other):
		fStorage(CopyOtherStorage(other) ),
		fSize (other.GetNEvents())
		{



			fWeights = vector_real(fSize , 1.0);
			fFlags = vector_bool( fSize, 1.0 );

			fBegin = thrust::make_zip_iterator(thrust:: tuple_cat(thrust::make_tuple(fWeights.begin()),
					detail::begin_call_args(fStorage)) );
			fEnd = thrust::make_zip_iterator( thrust:: tuple_cat(thrust::make_tuple(fWeights.end()),
					detail::end_call_args(fStorage)) );

			fConstBegin = thrust::make_zip_iterator( thrust:: tuple_cat(thrust::make_tuple(fWeights.cbegin()),
					detail::cbegin_call_args(fStorage) ));
			fConstEnd = thrust::make_zip_iterator(thrust:: tuple_cat(thrust::make_tuple(fWeights.cend()),
					detail::cend_call_args(fStorage) ) );


		}

	template<unsigned int BACKEND2>
	Chain(hydra::experimental::Chain<hydra::experimental::Events<N,BACKEND2 >...>const& other):
	fStorage(CopyOtherStorage(other) ),
	fSize (other.GetNEvents())
	{



		fWeights = vector_real(fSize , 1.0);
		fFlags = vector_bool( fSize, 1.0 );

		fBegin = thrust::make_zip_iterator(thrust:: tuple_cat(thrust::make_tuple(fWeights.begin()),
				detail::begin_call_args(fStorage)) );
		fEnd = thrust::make_zip_iterator( thrust:: tuple_cat(thrust::make_tuple(fWeights.end()),
				detail::end_call_args(fStorage)) );

		fConstBegin = thrust::make_zip_iterator( thrust:: tuple_cat(thrust::make_tuple(fWeights.cbegin()),
				detail::cbegin_call_args(fStorage) ));
		fConstEnd = thrust::make_zip_iterator(thrust:: tuple_cat(thrust::make_tuple(fWeights.cend()),
				detail::cend_call_args(fStorage) ) );


	}



	Chain(hydra::experimental::Chain<hydra::experimental::Events<N,BACKEND >...>&& other):
		fStorage(std::move(other.MoveStorage())),
		fSize (other.GetNEvents())
	{


		other.resize(0);
		fWeights = vector_real(fSize , 1.0);
		fFlags = vector_bool( fSize, 1.0 );

		fBegin = thrust::make_zip_iterator(thrust:: tuple_cat(thrust::make_tuple(fWeights.begin()),
				detail::begin_call_args(fStorage)) );
		fEnd = thrust::make_zip_iterator( thrust:: tuple_cat(thrust::make_tuple(fWeights.end()),
				detail::end_call_args(fStorage)) );

		fConstBegin = thrust::make_zip_iterator( thrust:: tuple_cat(thrust::make_tuple(fWeights.cbegin()),
				detail::cbegin_call_args(fStorage) ));
		fConstEnd = thrust::make_zip_iterator(thrust:: tuple_cat(thrust::make_tuple(fWeights.cend()),
				detail::cend_call_args(fStorage) ) );


	}


	//************************

	hydra::experimental::Chain<hydra::experimental::Events<N,BACKEND >...>&
	operator=(hydra::experimental::Chain<hydra::experimental::Events<N,BACKEND >...> const& other)
	{
		if(this == &other) return *this;
		this->fStorage = CopyOtherStorage(other) ;
		this->fSize = other.GetNEvents();

		this->fWeights = vector_real(this->fSize , 1.0);
		this->fFlags = vector_bool( this->fSize, 1.0 );

		this->fBegin = thrust::make_zip_iterator(thrust:: tuple_cat(thrust::make_tuple(this->fWeights.begin()),
				detail::begin_call_args(this->fStorage)) );
		this->fEnd = thrust::make_zip_iterator( thrust:: tuple_cat(thrust::make_tuple(this->fWeights.end()),
				detail::end_call_args(this->fStorage)) );

		this->fConstBegin = thrust::make_zip_iterator( thrust:: tuple_cat(thrust::make_tuple(this->fWeights.cbegin()),
				detail::cbegin_call_args(this->fStorage) ));
		this->fConstEnd = thrust::make_zip_iterator(thrust:: tuple_cat(thrust::make_tuple(this->fWeights.cend()),
				detail::cend_call_args(this->fStorage) ) );

		return *this;

	}

	template<unsigned int BACKEND2>
	hydra::experimental::Chain<hydra::experimental::Events<N,BACKEND >...>&
	operator=(hydra::experimental::Chain<hydra::experimental::Events<N,BACKEND2 >...>const& other)
	{
		if(this == &other) return *this;
		this->fStorage=CopyOtherStorage(other, other.indexes);
		this->fSize = other.GetNEvents();

		this->fWeights = vector_real(this->fSize , 1.0);
		this->fFlags = vector_bool( this->fSize, 1.0 );

		this->fBegin = thrust::make_zip_iterator(thrust:: tuple_cat(thrust::make_tuple(this->fWeights.begin()),
				detail::begin_call_args(this->fStorage)) );
		this->fEnd = thrust::make_zip_iterator( thrust:: tuple_cat(thrust::make_tuple(this->fWeights.end()),
				detail::end_call_args(this->fStorage)) );

		this->fConstBegin = thrust::make_zip_iterator( thrust:: tuple_cat(thrust::make_tuple(this->fWeights.cbegin()),
				detail::cbegin_call_args(this->fStorage) ));
		this->fConstEnd = thrust::make_zip_iterator(thrust:: tuple_cat(thrust::make_tuple(this->fWeights.cend()),
				detail::cend_call_args(this->fStorage) ) );

		return *this;
	}


	hydra::experimental::Chain<hydra::experimental::Events<N,BACKEND >...>&
	operator=(hydra::experimental::Chain<hydra::experimental::Events<N,BACKEND >...>&& other)
	{
		if(this == &other) return *this;
		this->fStorage = std::move(other.MoveStorage());
		this->fSize = other.GetNEvents();

		other= hydra::experimental::Chain<hydra::experimental::Events<N,BACKEND >...>();

		this->fWeights = vector_real(this->fSize , 1.0);
		this->fFlags = vector_bool( this->fSize, 1.0 );

		this->fBegin = thrust::make_zip_iterator(thrust:: tuple_cat(thrust::make_tuple(this->fWeights.begin()),
				detail::begin_call_args(this->fStorage)) );
		this->fEnd = thrust::make_zip_iterator( thrust:: tuple_cat(thrust::make_tuple(this->fWeights.end()),
				detail::end_call_args(this->fStorage)) );

		this->fConstBegin = thrust::make_zip_iterator( thrust:: tuple_cat(thrust::make_tuple(fWeights.cbegin()),
				detail::cbegin_call_args(this->fStorage) ));
		this->fConstEnd = thrust::make_zip_iterator(thrust:: tuple_cat(thrust::make_tuple(this->fWeights.cend()),
				detail::cend_call_args(this->fStorage) ) );

		return *this;
	}



	Chain(hydra::experimental::Events<N,BACKEND >& ...events):
		fStorage(std::move(thrust::make_tuple( std::move(events)...))),
		fSize ( CheckSizes({events.GetNEvents()...}))
	{

		fWeights = vector_real(fSize , 1.0);
		fFlags = vector_bool( fSize, 1.0 );

		fBegin = thrust::make_zip_iterator(thrust:: tuple_cat(thrust::make_tuple(fWeights.begin()),
				detail::begin_call_args(fStorage)) );
		fEnd = thrust::make_zip_iterator( thrust:: tuple_cat(thrust::make_tuple(fWeights.end()),
				detail::end_call_args(fStorage)) );

		fConstBegin = thrust::make_zip_iterator( thrust:: tuple_cat(thrust::make_tuple(fWeights.cbegin()),
				detail::cbegin_call_args(fStorage) ));
		fConstEnd = thrust::make_zip_iterator(thrust:: tuple_cat(thrust::make_tuple(fWeights.cend()),
				detail::cend_call_args(fStorage) ) );

	}


	template<unsigned int I>
	auto GetDecay() const
	-> typename thrust::tuple_element<I, event_tuple>::type const&
	{
		return thrust::get<I>(fStorage);
	}



	size_t GetNEvents() const {
			return fSize;
	}

	vector_bool_iterator FlagsBegin() {
		return fFlags.begin();
	}

	vector_bool_iterator FlagsEnd() {
		return fFlags.end();
	}

	vector_real_iterator WeightsBegin() {
		return fWeights.begin();
	}

	vector_real_iterator WeightsEnd() {
		return fWeights.end();
	}

	vector_bool_const_iterator FlagsBegin() const{
		return fFlags.cbegin();
	}

	vector_bool_const_iterator FlagsEnd() const{
		return fFlags.cend();
	}

	vector_real_const_iterator WeightsBegin() const{
		return fWeights.cbegin();
	}

	vector_real_const_iterator WeightsEnd() const{
		return fWeights.cend();
	}

	iterator begin(){ return fBegin; }
	iterator  end(){ return fEnd; }
	const_iterator begin() const{ return fConstBegin; }
	const_iterator  end() const{ return fConstEnd; }
	const_iterator cbegin() const{ return fConstBegin; }
	const_iterator  cend() const{ return fConstEnd; }

	template<unsigned int I>
	iterator begin(){ return thrust::get<I>(fStorage).begin(); }

	template<unsigned int I>
	iterator  end(){ return thrust::get<I>(fStorage).end() ; }

	template<unsigned int I>
	const_iterator begin() const{ return thrust::get<I>(fStorage).cbegin() ; }

	template<unsigned int I>
	const_iterator  end() const{ return thrust::get<I>(fStorage).cend() ; }

	template<unsigned int I>
	const_iterator cbegin() const{ return thrust::get<I>(fStorage).cbegin() ; }

	template<unsigned int I>
	const_iterator  cend() const{ return thrust::get<I>(fStorage).cend() ; }

	size_t capacity() const  {
		return fFlags.capacity();
	}

	void resize(size_t n){

		fSize=n;
		fWeights.resize(n);
		fFlags.resize(n);
		detail::resize_call_args(fStorage, n);

		fBegin = thrust::make_zip_iterator(thrust:: tuple_cat(thrust::make_tuple(fWeights.begin()),
				detail::begin_call_args(fStorage)) );
		fEnd = thrust::make_zip_iterator( thrust:: tuple_cat(thrust::make_tuple(fWeights.end()),
				detail::end_call_args(fStorage)) );

		fConstBegin = thrust::make_zip_iterator( thrust:: tuple_cat(thrust::make_tuple(fWeights.cbegin()),
				detail::cbegin_call_args(fStorage) ));
		fConstEnd = thrust::make_zip_iterator(thrust:: tuple_cat(thrust::make_tuple(fWeights.cend()),
				detail::cend_call_args(fStorage) ) );

	}

	reference_type operator[](size_t i)
	{

		return fBegin[i];
	}

	reference_type operator[](size_t i) const
	{

		return fConstBegin[i];
	}
private:

	template<unsigned int  BACKEND2, size_t ...index>
	event_tuple _CopyStorage(hydra::experimental::Chain<hydra::experimental::Events<N, BACKEND2>...> const& other,
			hydra::detail::index_sequence<index...> indexes)
	{
		return thrust::make_tuple(hydra::experimental::Events<N,BACKEND >(other.template GetDecay<index>())...);
	}

	template<unsigned int  BACKEND2>
	event_tuple CopyOtherStorage(hydra::experimental::Chain<hydra::experimental::Events<N, BACKEND2>...> const& other)
	{
		return _CopyStorage(other, typename hydra::experimental::Chain<hydra::experimental::Events<N, BACKEND2>...>::indexing_type() );
	}

	event_tuple MoveStorage(){
		return std::move(fStorage);
	}

	size_t	CheckSizes(std::initializer_list<size_t> sizes)
	{
		assert(std::adjacent_find( sizes.begin(), sizes.end(),
				std::not_equal_to<size_t>() ) == sizes.end());
		size_t s=*sizes.end();
	 return	s;
	}

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
