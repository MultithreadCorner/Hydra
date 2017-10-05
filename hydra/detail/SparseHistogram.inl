/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016-2017 Antonio Augusto Alves Junior
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


#include <hydra/detail/external/thrust/reduce.h>
#include <hydra/detail/external/thrust/gather.h>
#include <hydra/detail/external/thrust/scatter.h>
#include <hydra/detail/functors/GetGlobalBin.h>
#include <hydra/Distance.h>
#include <hydra/detail/external/thrust/iterator/constant_iterator.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>

namespace hydra {

template<size_t N, typename T>
template<typename Iterator1, typename Iterator2>
void SparseHistogram<N, T, detail::multidimensional>::Fill(Iterator1 begin, Iterator1 end, Iterator2 wbegin )
{
	using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
	typedef  typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator1>::type system1_t;
	typedef  typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator2>::type system2_t;
	system1_t system1;
	system2_t system2;

	typedef  typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<
			decltype(select_system(system1, system2 ))>::type system_t;
	//----------------

	size_t data_size = HYDRA_EXTERNAL_NS::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<N,T>(fGrid, fLowerLimits, fUpperLimits);

	auto weights  = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(system_t(), data_size);
	hydra::copy(wbegin, wbegin+data_size, weights.first);

	auto keys_begin = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(system_t(), data_size);

	HYDRA_EXTERNAL_NS::thrust::copy(system_t(), keys_begin, keys_end, key_buffer.first);
	HYDRA_EXTERNAL_NS::thrust::sort_by_key( system_t(), key_buffer.first, key_buffer.first+data_size, weights.first);
	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system_t(), key_buffer.first);

	//bins content
	auto reduced_values  = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(system_t(), data_size);
	auto reduced_keys    = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(system_t(), data_size);

	auto reduced_end = HYDRA_EXTERNAL_NS::thrust::reduce_by_key(system_t(),
			key_buffer.first, key_buffer.first +  key_buffer.second,
			weights.first, reduced_keys.first, reduced_values.first);

	size_t histogram_size = HYDRA_EXTERNAL_NS::thrust::distance(reduced_keys.first, reduced_end.first);

	fContents.resize(histogram_size);
	fBins.resize(histogram_size);
	fNBins = histogram_size ;

	HYDRA_EXTERNAL_NS::thrust::copy(system_t(),reduced_keys.first, reduced_end.first,  fBins.begin());
	HYDRA_EXTERNAL_NS::thrust::copy(system_t(),reduced_values.first, reduced_end.second,  fContents.begin());

	// deallocate storage with HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer
	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system_t(), weights.first  );
	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system_t(), reduced_values.first);
	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system_t(), reduced_keys.first);

}


template<size_t N, typename T>
template<typename Iterator>
void SparseHistogram<N, T, detail::multidimensional>::Fill(Iterator begin, Iterator end )
{
	typedef  typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator>::type system_t;

	//----------------

	size_t data_size = HYDRA_EXTERNAL_NS::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<N,T>(fGrid, fLowerLimits, fUpperLimits);


	auto keys_begin = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(system_t(), data_size);

	HYDRA_EXTERNAL_NS::thrust::copy( system_t(),keys_begin, keys_end, key_buffer.first);
	HYDRA_EXTERNAL_NS::thrust::sort( system_t(),key_buffer.first, key_buffer.first+data_size );


	//bins content
	auto reduced_values  = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(system_t(), data_size);
	auto reduced_keys    = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(system_t(), data_size);

	//reduction_by_key
	auto  weights    = HYDRA_EXTERNAL_NS::thrust::constant_iterator<double>(1.0);

	auto reduced_end = HYDRA_EXTERNAL_NS::thrust::reduce_by_key(system_t(),
			key_buffer.first, key_buffer.first+data_size,
			weights, reduced_keys.first, reduced_values.first);

	size_t histogram_size = HYDRA_EXTERNAL_NS::thrust::distance(reduced_keys.first, reduced_end.first);

	fContents.resize(histogram_size);
	fBins.resize(histogram_size);
	fNBins = histogram_size ;

	HYDRA_EXTERNAL_NS::thrust::copy(system_t(),reduced_keys.first, reduced_end.first,  fBins.begin());
	HYDRA_EXTERNAL_NS::thrust::copy(system_t(),reduced_values.first, reduced_end.second,  fContents.begin());

	// deallocate storage with HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer
	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system_t(), reduced_values.first);
	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system_t(), reduced_keys.first);
	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system_t(), key_buffer.first);

}

template<typename T>
template<typename Iterator>
void SparseHistogram<1, T, detail::unidimensional>::Fill(Iterator begin, Iterator end )
{
	typedef  typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator>::type system_t;

	size_t data_size = HYDRA_EXTERNAL_NS::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<1,T>(fGrid, fLowerLimits, fUpperLimits);

	auto keys_begin = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(system_t(), data_size);

	HYDRA_EXTERNAL_NS::thrust::copy(system_t(), keys_begin, keys_end, key_buffer.first);
	HYDRA_EXTERNAL_NS::thrust::sort(system_t(),key_buffer.first, key_buffer.first+data_size);

	//bins content
	auto reduced_values  = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(system_t(), data_size);
	auto reduced_keys    = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(system_t(), data_size);
	auto weights         = HYDRA_EXTERNAL_NS::thrust::constant_iterator<double>(1.0);

	//reduction_by_key
	auto reduced_end = HYDRA_EXTERNAL_NS::thrust::reduce_by_key(system_t(),
			key_buffer.first, key_buffer.first+key_buffer.second,
			weights, reduced_keys.first, reduced_values.first);

	size_t histogram_size = HYDRA_EXTERNAL_NS::thrust::distance(system_t(),reduced_keys.first, reduced_end.first);

	fContents.resize(histogram_size);
	fBins.resize(histogram_size);
	fNBins = histogram_size ;

	HYDRA_EXTERNAL_NS::thrust::copy(system_t(), reduced_keys.first, reduced_end.first,  fBins.begin());
	HYDRA_EXTERNAL_NS::thrust::copy(system_t(), reduced_values.first, reduced_end.second,  fContents.begin());

    // deallocate storage with HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system_t(), reduced_values.first);
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system_t(), reduced_keys.first);
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system_t(), key_buffer.first);

}

template<typename T>
template<typename Iterator1, typename Iterator2>
void SparseHistogram<1, T,detail::unidimensional >::Fill(Iterator1 begin, Iterator1 end, Iterator2 wbegin )
{
	using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
	typedef  typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator1>::type system1_t;
	typedef  typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator2>::type system2_t;
	system1_t system1;
	system2_t system2;

	typedef  typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<
			decltype(select_system(system1, system2 ))>::type system_t;

	size_t data_size = HYDRA_EXTERNAL_NS::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<1,T>(fGrid, fLowerLimits, fUpperLimits);

	//work on local copy of data
	auto weights  = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(system_t(), data_size);
	hydra::copy(system_t(),wbegin, wbegin+data_size, weights.first);

	auto keys_begin = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(system_t(), data_size);

	HYDRA_EXTERNAL_NS::thrust::copy(system_t(), keys_begin, keys_end, key_buffer.first);
	HYDRA_EXTERNAL_NS::thrust::sort_by_key(system_t(),key_buffer.first, key_buffer.first+data_size, weights.first);
	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system_t(), key_buffer.first);

	//bins content
	auto reduced_values  = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(system_t(), data_size);
	auto reduced_keys    = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(system_t(), data_size);

	//reduction_by_key
	auto reduced_end = HYDRA_EXTERNAL_NS::thrust::reduce_by_key(system_t(),
			key_buffer.first, key_buffer.first+data_size,
			weights.first, reduced_keys.first, reduced_values.first);

	size_t histogram_size = HYDRA_EXTERNAL_NS::thrust::distance(reduced_keys.first, reduced_end.first);

	fContents.resize(histogram_size);
	fBins.resize(histogram_size);
	fNBins = histogram_size ;


	HYDRA_EXTERNAL_NS::thrust::copy(system_t(),reduced_keys.first, reduced_end.first,  fBins.begin());
	HYDRA_EXTERNAL_NS::thrust::copy(system_t(),reduced_values.first, reduced_end.second,  fContents.begin());


    // deallocate storage with HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer
	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system_t(), weights.first  );
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system_t(), reduced_values.first);
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system_t(), reduced_keys.first);

}


}  // namespace hydra



#endif /* SPARSEHISTOGRAM_INL_ */
