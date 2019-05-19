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
 * Chain.inl
 *
 *  Created on: 27/06/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef CHAIN_INL_
#define CHAIN_INL_

namespace hydra {

template<size_t ...N, hydra::detail::Backend BACKEND>
Chain< Events<N,hydra::detail::BackendPolicy<BACKEND> >...>::Chain(size_t nevents):
fStorage(HYDRA_EXTERNAL_NS::thrust::make_tuple(Events<N,hydra::detail::BackendPolicy<BACKEND> >(nevents)...) ),
fSize(nevents)
{

	fWeights.resize(fSize);
	fFlags.resize(fSize);

	fBegin = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.begin()),
			detail::begin_call_args(fStorage)) );
	fEnd = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator( HYDRA_EXTERNAL_NS::thrust::tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.end()),
			detail::end_call_args(fStorage)) );

	fConstBegin = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.cbegin()),
			detail::cbegin_call_args(fStorage) ));
	fConstEnd = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.cend()),
			detail::cend_call_args(fStorage) ) );

}

template<size_t ...N, hydra::detail::Backend BACKEND>
Chain< Events<N,hydra::detail::BackendPolicy<BACKEND> >...>::Chain(Chain<Events<N,hydra::detail::BackendPolicy<BACKEND> >...>const& other):
fStorage(std::move(other.CopyStorage()) ),
fSize (other.GetNEvents())
{

	fWeights = vector_real(fSize , 1.0);
	fFlags = vector_bool( fSize, 1.0 );

	fBegin = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.begin()),
			detail::begin_call_args(fStorage)) );
	fEnd = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator( HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.end()),
			detail::end_call_args(fStorage)) );

	fConstBegin = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator( HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.cbegin()),
			detail::cbegin_call_args(fStorage) ));
	fConstEnd = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.cend()),
			detail::cend_call_args(fStorage) ) );

}

template<size_t ...N, hydra::detail::Backend BACKEND>
template<hydra::detail::Backend BACKEND2>
Chain< Events<N,hydra::detail::BackendPolicy<BACKEND> >...>::Chain(Chain<Events<N,hydra::detail::BackendPolicy<BACKEND2> >...>const& other):
fStorage(std::move(other.CopyStorage()) ),
fSize (other.GetNEvents())
{

	fWeights = vector_real(fSize , 1.0);
	fFlags = vector_bool( fSize, 1.0 );

	fBegin = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.begin()),
			detail::begin_call_args(fStorage)) );
	fEnd = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator( HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.end()),
			detail::end_call_args(fStorage)) );

	fConstBegin = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator( HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.cbegin()),
			detail::cbegin_call_args(fStorage) ));
	fConstEnd = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.cend()),
			detail::cend_call_args(fStorage) ) );

}

template<size_t ...N, hydra::detail::Backend BACKEND>
Chain< Events<N,hydra::detail::BackendPolicy<BACKEND> >...>::Chain(Chain<Events<N,hydra::detail::BackendPolicy<BACKEND> >...>&& other):
fStorage(std::move(other.MoveStorage())),
fSize (other.GetNEvents())
{

other.resize(0);
fWeights = vector_real(fSize , 1.0);
fFlags = vector_bool( fSize, 1.0 );

fBegin = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.begin()),
		detail::begin_call_args(fStorage)) );
fEnd = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator( HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.end()),
		detail::end_call_args(fStorage)) );

fConstBegin = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator( HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.cbegin()),
		detail::cbegin_call_args(fStorage) ));
fConstEnd = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.cend()),
		detail::cend_call_args(fStorage) ) );
}

template<size_t ...N, hydra::detail::Backend BACKEND>
Chain<Events<N,hydra::detail::BackendPolicy<BACKEND> >...>&
Chain< Events<N,hydra::detail::BackendPolicy<BACKEND> >...>::operator=(Chain<Events<N,hydra::detail::BackendPolicy<BACKEND> >...> const& other)
{
	if(this == &other) return *this;
	this->fStorage = std::move(other.CopyStorage()) ;
	this->fSize = other.GetNEvents();

	this->fWeights = vector_real(this->fSize , 1.0);
	this->fFlags = vector_bool( this->fSize, 1.0 );

	this->fBegin = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(this->fWeights.begin()),
			detail::begin_call_args(this->fStorage)) );
	this->fEnd = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator( HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(this->fWeights.end()),
			detail::end_call_args(this->fStorage)) );

	this->fConstBegin = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator( HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(this->fWeights.cbegin()),
			detail::cbegin_call_args(this->fStorage) ));
	this->fConstEnd = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(this->fWeights.cend()),
			detail::cend_call_args(this->fStorage) ) );

	return *this;

}

template<size_t ...N, hydra::detail::Backend BACKEND>
template<hydra::detail::Backend BACKEND2>
Chain<Events<N,hydra::detail::BackendPolicy<BACKEND> >...>&
Chain< Events<N,hydra::detail::BackendPolicy<BACKEND> >...>::operator=(Chain<Events<N,hydra::detail::BackendPolicy<BACKEND2> >...>const& other)
	{
		//if(this == &other) return *this;
		this->fStorage=std::move(other.CopyStorage());
		this->fSize = other.GetNEvents();

		this->fWeights = vector_real(this->fSize , 1.0);
		this->fFlags = vector_bool( this->fSize, 1.0 );

		this->fBegin = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(this->fWeights.begin()),
				detail::begin_call_args(this->fStorage)) );
		this->fEnd = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator( HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(this->fWeights.end()),
				detail::end_call_args(this->fStorage)) );

		this->fConstBegin = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator( HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(this->fWeights.cbegin()),
				detail::cbegin_call_args(this->fStorage) ));
		this->fConstEnd = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(this->fWeights.cend()),
				detail::cend_call_args(this->fStorage) ) );

		return *this;
	}

template<size_t ...N, hydra::detail::Backend BACKEND>
Chain<Events<N,hydra::detail::BackendPolicy<BACKEND> >...>&
Chain< Events<N,hydra::detail::BackendPolicy<BACKEND> >...>::operator=(Chain<Events<N,hydra::detail::BackendPolicy<BACKEND> >...>&& other)
	{
		if(this == &other) return *this;
		this->fStorage = std::move(other.MoveStorage());
		this->fSize = other.GetNEvents();

		other= Chain<Events<N,hydra::detail::BackendPolicy<BACKEND> >...>();

		this->fWeights = vector_real(this->fSize , 1.0);
		this->fFlags = vector_bool( this->fSize, 1.0 );

		this->fBegin = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(this->fWeights.begin()),
				detail::begin_call_args(this->fStorage)) );
		this->fEnd = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator( HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(this->fWeights.end()),
				detail::end_call_args(this->fStorage)) );

		this->fConstBegin = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator( HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.cbegin()),
				detail::cbegin_call_args(this->fStorage) ));
		this->fConstEnd = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(this->fWeights.cend()),
				detail::cend_call_args(this->fStorage) ) );

		return *this;
	}

template<size_t ...N, hydra::detail::Backend BACKEND>
Chain< Events<N,hydra::detail::BackendPolicy<BACKEND> >...>::Chain(Events<N,hydra::detail::BackendPolicy<BACKEND> > const& ...events):
		fSize ( CheckSizes({events.GetNEvents()...}) ),
		fStorage( HYDRA_EXTERNAL_NS::thrust::make_tuple( events... ))
	{

		fWeights = vector_real(fSize , 1.0);
		fFlags = vector_bool( fSize, 1.0 );

		fBegin = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.begin()),
				detail::begin_call_args(fStorage)) );
		fEnd = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator( HYDRA_EXTERNAL_NS::thrust::tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.end()),
				detail::end_call_args(fStorage)) );

		fConstBegin = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator( HYDRA_EXTERNAL_NS::thrust::tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.cbegin()),
				detail::cbegin_call_args(fStorage) ));
		fConstEnd = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust::tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.cend()),
				detail::cend_call_args(fStorage) ) );

	}

template<size_t ...N, hydra::detail::Backend BACKEND>
void Chain< Events<N,hydra::detail::BackendPolicy<BACKEND> >...>::resize(size_t n){

	fSize=n;
	fWeights.resize(n);
	fFlags.resize(n);
	detail::resize_call_args(fStorage, n);

	fBegin = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.begin()),
			detail::begin_call_args(fStorage)) );
	fEnd = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator( HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.end()),
			detail::end_call_args(fStorage)) );

	fConstBegin = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator( HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.cbegin()),
			detail::cbegin_call_args(fStorage) ));
	fConstEnd = HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(HYDRA_EXTERNAL_NS::thrust:: tuple_cat(HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.cend()),
			detail::cend_call_args(fStorage) ) );

}


}  // namespace hydra



#endif /* CHAIN_INL_ */
