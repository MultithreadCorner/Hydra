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
 * SparseHistogram.inl
 *
 *  Created on: 01/10/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SPARSEHISTOGRAM_INL_
#define SPARSEHISTOGRAM_INL_


#include <hydra/detail/external/hydra_thrust/reduce.h>
#include <hydra/detail/external/hydra_thrust/gather.h>
#include <hydra/detail/external/hydra_thrust/scatter.h>
#include <hydra/detail/functors/GetGlobalBin.h>
#include <hydra/Distance.h>
#include <hydra/detail/external/hydra_thrust/iterator/constant_iterator.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/memory.h>

#include<utility>

namespace hydra {

template<typename T,size_t N,  hydra::detail::Backend BACKEND >
template<typename Iterator1, typename Iterator2>
SparseHistogram<T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>&
SparseHistogram<T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>::Fill(Iterator1 begin, Iterator1 end, Iterator2 wbegin )
{
	using hydra::thrust::system::detail::generic::select_system;
	typedef  typename hydra::thrust::iterator_system<Iterator1>::type system1_t;
	typedef  typename hydra::thrust::iterator_system<Iterator2>::type system2_t;
	system1_t system1;
	system2_t system2;

	typedef  typename hydra::thrust::detail::remove_reference<
			decltype(select_system(fSystem,system1, system2 ))>::type common_system_t;
	//----------------

	size_t data_size = hydra::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<N,double>(fGrid, fLowerLimits, fUpperLimits);

	auto weights  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	hydra::thrust::copy(wbegin, wbegin+data_size, weights.first);

	auto keys_begin = hydra::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = hydra::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	hydra::thrust::copy(common_system_t(), keys_begin, keys_end, key_buffer.first);
	hydra::thrust::sort_by_key( common_system_t(), key_buffer.first, key_buffer.first+data_size, weights.first);

	//bins content
	auto reduced_values  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	auto reduced_keys    = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	auto reduced_end = hydra::thrust::reduce_by_key(common_system_t(),
			key_buffer.first, key_buffer.first +  key_buffer.second,
			weights.first, reduced_keys.first, reduced_values.first);

	hydra::thrust::return_temporary_buffer(common_system_t(), key_buffer.first, key_buffer.second );

	size_t histogram_size = hydra::thrust::distance(reduced_keys.first, reduced_end.first);

	fContents.resize(histogram_size);
	fBins.resize(histogram_size);
	fNBins = histogram_size ;

	hydra::thrust::copy(common_system_t(),reduced_keys.first, reduced_end.first,  fBins.begin());
	hydra::thrust::copy(common_system_t(),reduced_values.first, reduced_end.second,  fContents.begin());

	// deallocate storage with hydra::thrust::return_temporary_buffer

	hydra::thrust::return_temporary_buffer(common_system_t(), weights.first,  weights.second );
	hydra::thrust::return_temporary_buffer(common_system_t(), reduced_values.first, reduced_values.second);
	hydra::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first, reduced_keys.second );

	return *this;

}


template< typename T,size_t N, hydra::detail::Backend BACKEND >
template<hydra::detail::Backend BACKEND2, typename Iterator1, typename Iterator2>
SparseHistogram<T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>&
SparseHistogram<T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>::Fill(detail::BackendPolicy<BACKEND2> const& exec_policy,
		Iterator1 begin, Iterator1 end, Iterator2 wbegin )
{
	using hydra::thrust::system::detail::generic::select_system;
	typedef  typename hydra::thrust::iterator_system<Iterator1>::type system1_t;
	typedef  typename hydra::thrust::iterator_system<Iterator2>::type system2_t;
	system1_t system1;
	system2_t system2;

	typedef  typename hydra::thrust::detail::remove_reference<
			decltype(select_system(exec_policy,fSystem, system1, system2 ))>::type common_system_t;
	//----------------

	size_t data_size = hydra::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<N,double>(fGrid, fLowerLimits, fUpperLimits);

	auto weights  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	hydra::thrust::copy(wbegin, wbegin+data_size, weights.first);

	auto keys_begin = hydra::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = hydra::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	hydra::thrust::copy(common_system_t(), keys_begin, keys_end, key_buffer.first);
	hydra::thrust::sort_by_key( common_system_t(), key_buffer.first, key_buffer.first+data_size, weights.first);

	//bins content
	auto reduced_values  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	auto reduced_keys    = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	auto reduced_end = hydra::thrust::reduce_by_key(common_system_t(),
			key_buffer.first, key_buffer.first +  key_buffer.second,
			weights.first, reduced_keys.first, reduced_values.first);

	hydra::thrust::return_temporary_buffer(common_system_t(), key_buffer.first, key_buffer.second);

	size_t histogram_size = hydra::thrust::distance(reduced_keys.first, reduced_end.first);

	fContents.resize(histogram_size);
	fBins.resize(histogram_size);
	fNBins = histogram_size ;

	hydra::thrust::copy(common_system_t(),reduced_keys.first, reduced_end.first,  fBins.begin());
	hydra::thrust::copy(common_system_t(),reduced_values.first, reduced_end.second,  fContents.begin());

	// deallocate storage with hydra::thrust::return_temporary_buffer
	hydra::thrust::return_temporary_buffer(common_system_t(), weights.first, weights.second  );
	hydra::thrust::return_temporary_buffer(common_system_t(), reduced_values.first, reduced_values.second );
	hydra::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first, reduced_keys.second );

	return *this;
}

template<typename T, size_t N,  hydra::detail::Backend BACKEND >
template<typename Iterator>
SparseHistogram<T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>&
SparseHistogram<T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>::Fill(Iterator begin, Iterator end )
{
	using hydra::thrust::system::detail::generic::select_system;
	typedef  typename hydra::thrust::iterator_system<Iterator>::type system1_t;
	system1_t system1;

	typedef  typename hydra::thrust::detail::remove_reference<
			decltype(select_system(fSystem, system1 ))>::type common_system_t;


	//----------------

	size_t data_size = hydra::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<N,double>(fGrid, fLowerLimits, fUpperLimits);


	auto keys_begin = hydra::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = hydra::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	hydra::thrust::copy( common_system_t(),keys_begin, keys_end, key_buffer.first);
	hydra::thrust::sort( common_system_t(),key_buffer.first, key_buffer.first+data_size );


	//bins content
	auto reduced_values  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	auto reduced_keys    = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	//reduction_by_key
	auto  weights    = hydra::thrust::constant_iterator<double>(1.0);

	auto reduced_end = hydra::thrust::reduce_by_key(common_system_t(),
			key_buffer.first, key_buffer.first+data_size,
			weights, reduced_keys.first, reduced_values.first);

	hydra::thrust::return_temporary_buffer(common_system_t(), key_buffer.first, key_buffer.second);

	size_t histogram_size = hydra::thrust::distance(reduced_keys.first, reduced_end.first);

	fContents.resize(histogram_size);
	fBins.resize(histogram_size);
	fNBins = histogram_size ;

	hydra::thrust::copy(reduced_keys.first, reduced_end.first,  fBins.begin());
	hydra::thrust::copy(reduced_values.first, reduced_end.second,  fContents.begin());

	// deallocate storage with hydra::thrust::return_temporary_buffer
	hydra::thrust::return_temporary_buffer(common_system_t(), reduced_values.first, reduced_values.second);
	hydra::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first, reduced_keys.second);

	return *this;

}

template<typename T,size_t N,  hydra::detail::Backend BACKEND >
template<hydra::detail::Backend BACKEND2,typename Iterator>
SparseHistogram<T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>&
SparseHistogram<T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>::Fill(detail::BackendPolicy<BACKEND2> const& exec_policy,
		Iterator begin, Iterator end )
{
	typedef  typename hydra::thrust::iterator_system<Iterator>::type system1_t;
	system1_t system1;

	typedef  typename hydra::thrust::detail::remove_reference<
				decltype(select_system(exec_policy,fSystem, system1))>::type common_system_t;
	//----------------

	size_t data_size = hydra::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<N,double>(fGrid, fLowerLimits, fUpperLimits);


	auto keys_begin = hydra::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = hydra::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	hydra::thrust::copy( common_system_t(),keys_begin, keys_end, key_buffer.first);
	hydra::thrust::sort( common_system_t(),key_buffer.first, key_buffer.first+data_size );


	//bins content
	auto reduced_values  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	auto reduced_keys    = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	//reduction_by_key
	auto  weights    = hydra::thrust::constant_iterator<double>(1.0);

	auto reduced_end = hydra::thrust::reduce_by_key(common_system_t(),
			key_buffer.first, key_buffer.first+data_size,
			weights, reduced_keys.first, reduced_values.first);

	hydra::thrust::return_temporary_buffer(common_system_t(), key_buffer.first);

	size_t histogram_size = hydra::thrust::distance(reduced_keys.first, reduced_end.first);

	fContents.resize(histogram_size);
	fBins.resize(histogram_size);
	fNBins = histogram_size ;

	hydra::thrust::copy(reduced_keys.first, reduced_end.first,  fBins.begin());
	hydra::thrust::copy(reduced_values.first, reduced_end.second,  fContents.begin());

	// deallocate storage with hydra::thrust::return_temporary_buffer
	hydra::thrust::return_temporary_buffer(common_system_t(), reduced_values.first, reduced_values.second );
	hydra::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first, reduced_keys.second );

	return *this;

}

template<typename T, hydra::detail::Backend BACKEND >
template<typename Iterator>
SparseHistogram<T, 1,  detail::BackendPolicy<BACKEND>, detail::unidimensional>&
SparseHistogram<T, 1,  detail::BackendPolicy<BACKEND>, detail::unidimensional>::Fill(Iterator begin, Iterator end )
{
	using hydra::thrust::system::detail::generic::select_system;
	typedef  typename hydra::thrust::iterator_system<Iterator>::type system1_t;
	system1_t system1;

	typedef  typename hydra::thrust::detail::remove_reference<
			decltype(select_system(fSystem, system1 ))>::type common_system_t;


	size_t data_size = hydra::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<1,double>(fGrid, fLowerLimits, fUpperLimits);

	auto keys_begin = hydra::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = hydra::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	hydra::thrust::copy(common_system_t(), keys_begin, keys_end, key_buffer.first);
	hydra::thrust::sort(common_system_t(),key_buffer.first, key_buffer.first+data_size);

	//bins content
	auto reduced_values  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	auto reduced_keys    = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);
	auto weights         = hydra::thrust::constant_iterator<double>(1.0);

	//reduction_by_key
	auto reduced_end = hydra::thrust::reduce_by_key(common_system_t(),
			key_buffer.first, key_buffer.first+key_buffer.second,
			weights, reduced_keys.first, reduced_values.first);

	hydra::thrust::return_temporary_buffer(common_system_t(), key_buffer.first);

    size_t histogram_size = hydra::thrust::distance(common_system_t(),reduced_keys.first, reduced_end.first);

	fContents.resize(histogram_size);
	fBins.resize(histogram_size);
	fNBins = histogram_size ;

	hydra::thrust::copy(common_system_t(), reduced_keys.first, reduced_end.first,  fBins.begin());
	hydra::thrust::copy(common_system_t(), reduced_values.first, reduced_end.second,  fContents.begin());

    // deallocate storage with hydra::thrust::return_temporary_buffer
    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_values.first, reduced_values.second);
    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first, reduced_keys.second );

	return *this;

}


template<typename T, hydra::detail::Backend BACKEND >
template<hydra::detail::Backend BACKEND2, typename Iterator>
SparseHistogram<T, 1,  detail::BackendPolicy<BACKEND>, detail::unidimensional>&
SparseHistogram<T, 1,  detail::BackendPolicy<BACKEND>, detail::unidimensional>::Fill(detail::BackendPolicy<BACKEND2> const& exec_policy,
		Iterator begin, Iterator end )
{
	typedef  typename hydra::thrust::iterator_system<Iterator>::type system1_t;
	system1_t system1;

	typedef  typename hydra::thrust::detail::remove_reference<
			decltype(select_system(exec_policy,fSystem, system1))>::type common_system_t;

	size_t data_size = hydra::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<1,double>(fGrid, fLowerLimits, fUpperLimits);

	auto keys_begin = hydra::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = hydra::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	hydra::thrust::copy(common_system_t(), keys_begin, keys_end, key_buffer.first);
	hydra::thrust::sort(common_system_t(),key_buffer.first, key_buffer.first+data_size);

	//bins content
	auto reduced_values  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	auto reduced_keys    = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);
	auto weights         = hydra::thrust::constant_iterator<double>(1.0);

	//reduction_by_key
	auto reduced_end = hydra::thrust::reduce_by_key(common_system_t(),
			key_buffer.first, key_buffer.first+key_buffer.second,
			weights, reduced_keys.first, reduced_values.first);

	hydra::thrust::return_temporary_buffer(common_system_t(), key_buffer.first);

	size_t histogram_size = hydra::thrust::distance(common_system_t(),reduced_keys.first, reduced_end.first);

	fContents.resize(histogram_size);
	fBins.resize(histogram_size);
	fNBins = histogram_size ;

	hydra::thrust::copy(common_system_t(), reduced_keys.first, reduced_end.first,  fBins.begin());
	hydra::thrust::copy(common_system_t(), reduced_values.first, reduced_end.second,  fContents.begin());

    // deallocate storage with hydra::thrust::return_temporary_buffer
    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_values.first, reduced_values.second);
    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first, reduced_keys.second );


	return *this;
}


template<typename T, hydra::detail::Backend BACKEND >
template<typename Iterator1, typename Iterator2>
SparseHistogram<T, 1,  detail::BackendPolicy<BACKEND>, detail::unidimensional >&
SparseHistogram<T, 1,  detail::BackendPolicy<BACKEND>, detail::unidimensional >::Fill(Iterator1 begin, Iterator1 end, Iterator2 wbegin )
{
	using hydra::thrust::system::detail::generic::select_system;
	typedef  typename hydra::thrust::iterator_system<Iterator1>::type system1_t;
	typedef  typename hydra::thrust::iterator_system<Iterator2>::type system2_t;
	system1_t system1;
	system2_t system2;

	typedef  typename hydra::thrust::detail::remove_reference<
			decltype(select_system(fSystem,system1, system2 ))>::type common_system_t;

	size_t data_size = hydra::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<1,double>(fGrid, fLowerLimits, fUpperLimits);

	//work on local copy of data
	auto weights  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	hydra::thrust::copy(common_system_t(),wbegin, wbegin+data_size, weights.first);

	auto keys_begin = hydra::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = hydra::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	hydra::thrust::copy(common_system_t(), keys_begin, keys_end, key_buffer.first);
	hydra::thrust::sort_by_key(common_system_t(),key_buffer.first, key_buffer.first+data_size, weights.first);

	//bins content
	auto reduced_values  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	auto reduced_keys    = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	//reduction_by_key
	auto reduced_end = hydra::thrust::reduce_by_key(common_system_t(),
			key_buffer.first, key_buffer.first+data_size,
			weights.first, reduced_keys.first, reduced_values.first);

	hydra::thrust::return_temporary_buffer(common_system_t(), key_buffer.first);

	size_t histogram_size = hydra::thrust::distance(reduced_keys.first, reduced_end.first);

	fContents.resize(histogram_size);
	fBins.resize(histogram_size);
	fNBins = histogram_size ;


	hydra::thrust::copy(common_system_t(),reduced_keys.first, reduced_end.first,  fBins.begin());
	hydra::thrust::copy(common_system_t(),reduced_values.first, reduced_end.second,  fContents.begin());


    // deallocate storage with hydra::thrust::return_temporary_buffer
	hydra::thrust::return_temporary_buffer(common_system_t(), weights.first, weights.second );
    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_values.first,reduced_values.second );
    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first, reduced_keys.second );

	return *this;
}


template<typename T, hydra::detail::Backend BACKEND >
template<hydra::detail::Backend BACKEND2,typename Iterator1, typename Iterator2>
SparseHistogram<T, 1,  detail::BackendPolicy<BACKEND>,detail::unidimensional >&
SparseHistogram<T, 1,  detail::BackendPolicy<BACKEND>,detail::unidimensional >::Fill(detail::BackendPolicy<BACKEND2> const& exec_policy,
		Iterator1 begin, Iterator1 end, Iterator2 wbegin )
{
	using hydra::thrust::system::detail::generic::select_system;
	typedef  typename hydra::thrust::iterator_system<Iterator1>::type system1_t;
	typedef  typename hydra::thrust::iterator_system<Iterator2>::type system2_t;
	system1_t system1;
	system2_t system2;

	typedef  typename hydra::thrust::detail::remove_reference<
			decltype(select_system(exec_policy,fSystem,system1, system2 ))>::type common_system_t;

	size_t data_size = hydra::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<1,double>(fGrid, fLowerLimits, fUpperLimits);

	//work on local copy of data
	auto weights  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	hydra::thrust::copy(common_system_t(),wbegin, wbegin+data_size, weights.first);

	auto keys_begin = hydra::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = hydra::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	hydra::thrust::copy(common_system_t(), keys_begin, keys_end, key_buffer.first);
	hydra::thrust::sort_by_key(common_system_t(),key_buffer.first, key_buffer.first+data_size, weights.first);

	//bins content
	auto reduced_values  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	auto reduced_keys    = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	//reduction_by_key
	auto reduced_end = hydra::thrust::reduce_by_key(common_system_t(),
			key_buffer.first, key_buffer.first+data_size,
			weights.first, reduced_keys.first, reduced_values.first);

	hydra::thrust::return_temporary_buffer(common_system_t(), key_buffer.first);

	size_t histogram_size = hydra::thrust::distance(reduced_keys.first, reduced_end.first);

	fContents.resize(histogram_size);
	fBins.resize(histogram_size);
	fNBins = histogram_size ;


	hydra::thrust::copy(common_system_t(),reduced_keys.first, reduced_end.first,  fBins.begin());
	hydra::thrust::copy(common_system_t(),reduced_values.first, reduced_end.second,  fContents.begin());


    // deallocate storage with hydra::thrust::return_temporary_buffer
	hydra::thrust::return_temporary_buffer(common_system_t(), weights.first, weights.second );
    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_values.first, reduced_values.second );
    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first, reduced_keys.second );

	return *this;
}

/*
 * multidimensional specializations
 */

template<typename T, typename Iterator, size_t N , hydra::detail::Backend BACKEND>
inline typename std::enable_if< N==2, T >::type
SparseHistogram< T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>::Interpolate( std::array<size_t,N> const&  point){

	return spline2D(this->GetBinsCenters(placeholders::_0).begin(), this->GetBinsCenters(placeholders::_0).end(),
			 this->GetBinsCenters(placeholders::_1).begin(), this->GetBinsCenters(placeholders::_1).end(),
			 this->GetBinsContents().begin(), point[0],  point[1]);
}

template<typename T, typename Iterator, size_t N , hydra::detail::Backend BACKEND>
inline typename std::enable_if< N==3, T >::type
SparseHistogram< T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>::Interpolate( std::array<size_t,N> const&  point){

	return spline3D(this->GetBinsCenters(placeholders::_0).begin(), this->GetBinsCenters(placeholders::_0).end(),
				 this->GetBinsCenters(placeholders::_1).begin(), this->GetBinsCenters(placeholders::_1).end(),
				 this->GetBinsCenters(placeholders::_2).begin(), this->GetBinsCenters(placeholders::_2).end(),
				 this->GetBinsContents().begin(), point[0],  point[1],  point[2]);
}



//iterator based
template< typename T, typename Iterator,size_t N , hydra::detail::Backend BACKEND>
SparseHistogram< T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>
make_sparse_histogram( detail::BackendPolicy<BACKEND>, std::array<size_t, N> grid,
		std::array<double, N> const& lowerlimits,   std::array<double, N> const& upperlimits,
		Iterator first, Iterator end){

	hydra::SparseHistogram< T, N, detail::BackendPolicy<BACKEND>> _Hist( grid, lowerlimits, upperlimits);
	_Hist.Fill(first, end);

	return _Hist;
}

template< typename T,typename Iterator1,typename Iterator2, size_t N , hydra::detail::Backend BACKEND>
SparseHistogram< T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>
make_sparse_histogram( detail::BackendPolicy<BACKEND>, std::array<size_t, N> grid,
		std::array<double, N> const& lowerlimits,   std::array<double, N> const& upperlimits,
		Iterator1 first, Iterator1 end, Iterator2 wfirst){

	hydra::SparseHistogram< T, N, detail::BackendPolicy<BACKEND>> _Hist( grid, lowerlimits, upperlimits);
	_Hist.Fill(first, end, wfirst);

	return _Hist;
}


//iterable based
template< typename T, size_t N , hydra::detail::Backend BACKEND, typename Iterable>
inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value,
SparseHistogram< T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>>::type
make_sparse_histogram( detail::BackendPolicy<BACKEND> backend, std::array<size_t, N> grid,
		std::array<double, N>lowerlimits,   std::array<double, N> upperlimits,	Iterable&& data){

	return make_sparse_histogram<T>(backend,grid, lowerlimits, upperlimits,
			std::forward<Iterable>(data).begin(), std::forward<Iterable>(data).end());

}

template< typename T, size_t N , hydra::detail::Backend BACKEND, typename Iterable1,typename Iterable2 >
inline typename std::enable_if< hydra::detail::is_iterable<Iterable1>::value&&
hydra::detail::is_iterable<Iterable2>::value,
SparseHistogram< T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>>::type
make_sparse_histogram( detail::BackendPolicy<BACKEND> backend, std::array<size_t, N> grid,
		std::array<double, N>lowerlimits,   std::array<double, N> upperlimits,
		Iterable1&& data,
		Iterable2&& weights){

	return make_sparse_histogram<T>(backend,grid, lowerlimits, upperlimits,
			std::forward<Iterable1>(data).begin(),
			std::forward<Iterable1>(data).end(),
			std::forward<Iterable2>(weights).begin());

}

/*
 * unidimensional specializations
 */
//iterator based
template< typename T,typename Iterator, hydra::detail::Backend BACKEND>
SparseHistogram< T, 1,  detail::BackendPolicy<BACKEND>, detail::unidimensional>
make_sparse_histogram( detail::BackendPolicy<BACKEND>, size_t grid, double lowerlimits,  double upperlimits,
		Iterator first, Iterator end){

	hydra::SparseHistogram< T, 1, detail::BackendPolicy<BACKEND>> _Hist( grid, lowerlimits, upperlimits);
	_Hist.Fill(first, end);

	return _Hist;

}

template< typename T,typename Iterator1, typename Iterator2, hydra::detail::Backend BACKEND>
SparseHistogram< T, 1,  detail::BackendPolicy<BACKEND>, detail::unidimensional>
make_sparse_histogram( detail::BackendPolicy<BACKEND>, size_t grid, double lowerlimits,  double upperlimits,
		Iterator1 first, Iterator1 end, Iterator2 wfirst){

	hydra::SparseHistogram< T, 1, detail::BackendPolicy<BACKEND>> _Hist( grid, lowerlimits, upperlimits);
	_Hist.Fill(first, end, wfirst);

	return _Hist;

}

//iterable based
template< typename T, hydra::detail::Backend BACKEND, typename Iterable>
inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value,
SparseHistogram< T, 1,  detail::BackendPolicy<BACKEND>, detail::unidimensional>>::type
make_sparse_histogram( detail::BackendPolicy<BACKEND> backend, size_t grid,
		double lowerlimits,  double upperlimits,	Iterable&& data){

	return make_sparse_histogram<T>(backend,grid, lowerlimits, upperlimits,
			std::forward<Iterable>(data).begin(), std::forward<Iterable>(data).end());

}

template< typename T, hydra::detail::Backend BACKEND, typename Iterable1,typename Iterable2 >
inline typename std::enable_if< hydra::detail::is_iterable<Iterable1>::value&&
hydra::detail::is_iterable<Iterable2>::value,
SparseHistogram< T, 1,  detail::BackendPolicy<BACKEND>, detail::unidimensional>>::type
make_sparse_histogram( detail::BackendPolicy<BACKEND> backend, size_t grid,
		double lowerlimits, double upperlimits,
		Iterable1&& data,
		Iterable2&& weights){

	return make_sparse_histogram<T>(backend, grid, lowerlimits, upperlimits,
			std::forward<Iterable1>(data).begin(),
			std::forward<Iterable1>(data).end(),
			std::forward<Iterable2>(weights).begin());

}




}  // namespace hydra



#endif /* SPARSEHISTOGRAM_INL_ */
