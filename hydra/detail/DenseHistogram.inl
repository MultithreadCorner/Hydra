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
 * DenseHistogram.inl
 *
 *  Created on: 23/09/2017
 *      Author: Antonio Augusto Alves Junior
 */

//#ifndef DENSEHISTOGRAM_INL_
//#define DENSEHISTOGRAM_INL_

#include <hydra/detail/external/thrust/memory.h>
#include <hydra/detail/external/thrust/reduce.h>
#include <hydra/detail/external/thrust/gather.h>
#include <hydra/detail/external/thrust/scatter.h>
#include <hydra/detail/functors/GetGlobalBin.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/Distance.h>
#include <hydra/detail/external/thrust/iterator/constant_iterator.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>

namespace hydra {

template< typename T, size_t N, hydra::detail::Backend BACKEND>
template<typename Iterator1, typename Iterator2>
DenseHistogram<T, N, detail::BackendPolicy<BACKEND>, detail::multidimensional>&
DenseHistogram<T, N, detail::BackendPolicy<BACKEND>, detail::multidimensional>::Fill(Iterator1 begin, Iterator1 end, Iterator2 wbegin )
{
	using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
	typedef  typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator1>::type system1_t;
	typedef  typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator2>::type system2_t;
	system1_t system1;
	system2_t system2;

	typedef  typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<
			decltype(select_system(fSystem, system1, system2 ))>::type common_system_t;


    //----------------

	size_t data_size = HYDRA_EXTERNAL_NS::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<N,T>(fGrid, fLowerLimits, fUpperLimits);

	//work on local copy of weights

	auto weights  = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	HYDRA_EXTERNAL_NS::thrust::copy(wbegin, wbegin+data_size, weights.first);

	auto keys_begin = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	HYDRA_EXTERNAL_NS::thrust::copy( keys_begin, keys_end, key_buffer.first);

	HYDRA_EXTERNAL_NS::thrust::sort_by_key(key_buffer.first, key_buffer.first + key_buffer.second, weights.first );


	//bins content
	auto bin_contents    = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(common_system_t(), fContents.size());
	auto reduced_values  = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	auto reduced_keys    = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	HYDRA_EXTERNAL_NS::thrust::fill(bin_contents.first, bin_contents.first+bin_contents.second, 0.0);

	auto reduced_end = HYDRA_EXTERNAL_NS::thrust::reduce_by_key(common_system_t(), key_buffer.first,
			key_buffer.first + key_buffer.second, weights.first, reduced_keys.first, reduced_values.first);

	HYDRA_EXTERNAL_NS::thrust::scatter( common_system_t(),  reduced_values.first, reduced_end.second,
			reduced_keys.first,bin_contents.first );

	HYDRA_EXTERNAL_NS::thrust::copy(bin_contents.first ,
			bin_contents.first+ bin_contents.second,  fContents.begin());

    // deallocate storage with HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer
	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), bin_contents.first );
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), reduced_values.first);
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first);
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), key_buffer.first);
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), weights.first  );

    return *this;
}


template<typename T, size_t N, hydra::detail::Backend BACKEND>
template<hydra::detail::Backend BACKEND2, typename Iterator1, typename Iterator2>
DenseHistogram<T, N, detail::BackendPolicy<BACKEND>, detail::multidimensional>&
DenseHistogram<T, N, detail::BackendPolicy<BACKEND>, detail::multidimensional>::Fill(detail::BackendPolicy<BACKEND2> const& exec_policy,
		Iterator1 begin, Iterator1 end, Iterator2 wbegin )
{
	using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
	typedef  typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator1>::type system1_t;
	typedef  typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator2>::type system2_t;
	system1_t system1;
	system2_t system2;

	typedef  typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<
			decltype(select_system(exec_policy,fSystem, system1, system2 ))>::type common_system_t;


    //----------------

	size_t data_size = HYDRA_EXTERNAL_NS::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<N,T>(fGrid, fLowerLimits, fUpperLimits);

	//work on local copy of weights

	auto weights  = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	HYDRA_EXTERNAL_NS::thrust::copy(wbegin, wbegin+data_size, weights.first);

	auto keys_begin = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	HYDRA_EXTERNAL_NS::thrust::copy( keys_begin, keys_end, key_buffer.first);

	HYDRA_EXTERNAL_NS::thrust::sort_by_key(key_buffer.first, key_buffer.first + key_buffer.second, weights.first );


	//bins content
	auto bin_contents    = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(common_system_t(), fContents.size());
	auto reduced_values  = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	auto reduced_keys    = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	HYDRA_EXTERNAL_NS::thrust::fill(bin_contents.first, bin_contents.first+bin_contents.second, 0.0);

	auto reduced_end = HYDRA_EXTERNAL_NS::thrust::reduce_by_key(common_system_t(), keys_begin, keys_end, weights.first,
    		reduced_keys.first, reduced_values.first);

	HYDRA_EXTERNAL_NS::thrust::scatter( common_system_t(),  reduced_values.first, reduced_end.second,
			reduced_keys.first,bin_contents.first );

	HYDRA_EXTERNAL_NS::thrust::copy(bin_contents.first ,
			bin_contents.first+ bin_contents.second,  fContents.begin());

    // deallocate storage with HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer
	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), bin_contents.first );
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), reduced_values.first);
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first);
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), key_buffer.first);
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), weights.first  );

    return *this;
}


template<typename T, size_t N, hydra::detail::Backend BACKEND>
template<typename Iterator>
DenseHistogram<T, N, detail::BackendPolicy<BACKEND>, detail::multidimensional>&
DenseHistogram<T, N,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional>::Fill(Iterator begin, Iterator end )
{
	using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
	typedef  typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator>::type system1_t;
	system1_t system1;

	typedef  typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<
			decltype(select_system(fSystem, system1 ))>::type common_system_t;


		typedef HYDRA_EXTERNAL_NS::thrust::pointer<T, common_system_t> buffer_t;

		size_t data_size = HYDRA_EXTERNAL_NS::thrust::distance(begin, end);

		auto key_functor = detail::GetGlobalBin<N,T>(fGrid, fLowerLimits, fUpperLimits);

		auto keys_begin = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(begin, key_functor );
		auto keys_end   = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(end, key_functor);
		auto key_buffer = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);


		HYDRA_EXTERNAL_NS::thrust::copy( keys_begin, keys_end, key_buffer.first);
		HYDRA_EXTERNAL_NS::thrust::sort(key_buffer.first, key_buffer.first+data_size);


		//bins content
		auto bin_contents    = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(common_system_t(), fContents.size());
		auto reduced_values  = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
		auto reduced_keys    = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);
		auto weights         = HYDRA_EXTERNAL_NS::thrust::constant_iterator<double>(1.0);

		auto reduced_end = HYDRA_EXTERNAL_NS::thrust::reduce_by_key(common_system_t(),
				key_buffer.first, key_buffer.first+data_size,
				weights, reduced_keys.first, reduced_values.first);

		HYDRA_EXTERNAL_NS::thrust::fill(bin_contents.first, bin_contents.first+bin_contents.second, 0.0);

		HYDRA_EXTERNAL_NS::thrust::scatter( common_system_t(),  reduced_values.first, reduced_end.second,
			  reduced_keys.first, bin_contents.first );

		std::cout<< "TEST => contents "
				<<  HYDRA_EXTERNAL_NS::thrust::distance(reduced_values.first, reduced_end.second )
		        << " =? " <<  fContents.size() << std::endl;
		HYDRA_EXTERNAL_NS::thrust::copy(bin_contents.first ,
				bin_contents.first+ bin_contents.second,  fContents.begin());

	    // deallocate storage with HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer
		HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), key_buffer.first);
		HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), bin_contents.first );
	    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), reduced_values.first);
	    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first);

	    return *this;
}

template<typename T, size_t N, hydra::detail::Backend BACKEND>
template<hydra::detail::Backend BACKEND2, typename Iterator>
DenseHistogram<T, N, detail::BackendPolicy<BACKEND>, detail::multidimensional>&
DenseHistogram<T, N,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional>::Fill(detail::BackendPolicy<BACKEND2> const& exec_policy, Iterator begin, Iterator end )
{
	typedef  typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator>::type system1_t;
		system1_t system1;

		typedef  typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<
					decltype(select_system(exec_policy,fSystem, system1))>::type common_system_t;

		typedef HYDRA_EXTERNAL_NS::thrust::pointer<T, common_system_t> buffer_t;

		size_t data_size = HYDRA_EXTERNAL_NS::thrust::distance(begin, end);

		auto key_functor = detail::GetGlobalBin<N,T>(fGrid, fLowerLimits, fUpperLimits);

		auto keys_begin = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(begin, key_functor );
		auto keys_end   = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(end, key_functor);
		auto key_buffer = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);


		HYDRA_EXTERNAL_NS::thrust::copy( keys_begin, keys_end, key_buffer.first);
		HYDRA_EXTERNAL_NS::thrust::sort(key_buffer.first, key_buffer.first+data_size);


		//bins content
		auto bin_contents    = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(common_system_t(), fContents.size());
		auto reduced_values  = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
		auto reduced_keys    = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);
		auto weights         = HYDRA_EXTERNAL_NS::thrust::constant_iterator<double>(1.0);

		auto reduced_end = HYDRA_EXTERNAL_NS::thrust::reduce_by_key(common_system_t(),
				key_buffer.first, key_buffer.first+data_size,
				weights, reduced_keys.first, reduced_values.first);

		HYDRA_EXTERNAL_NS::thrust::fill(bin_contents.first, bin_contents.first+bin_contents.second, 0.0);

		HYDRA_EXTERNAL_NS::thrust::scatter( common_system_t(),  reduced_values.first, reduced_end.second,
			  reduced_keys.first, bin_contents.first );

		std::cout<< "TEST => contents "
				<<  HYDRA_EXTERNAL_NS::thrust::distance(reduced_values.first, reduced_end.second )
		        << " =? " <<  fContents.size() << std::endl;
		HYDRA_EXTERNAL_NS::thrust::copy(bin_contents.first ,
				bin_contents.first+ bin_contents.second,  fContents.begin());

	    // deallocate storage with HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer
		HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), key_buffer.first);
		HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), bin_contents.first );
	    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), reduced_values.first);
	    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first);

	    return *this;
}


template<typename T, hydra::detail::Backend BACKEND>
template<typename Iterator>
DenseHistogram< T,1, detail::BackendPolicy<BACKEND>, detail::unidimensional>&
DenseHistogram< T,1, detail::BackendPolicy<BACKEND>, detail::unidimensional>::Fill(Iterator begin, Iterator end )
{
	using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
	typedef  typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator>::type system1_t;
	system1_t system1;

	typedef  typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<
			decltype(select_system(fSystem, system1 ))>::type common_system_t;

	size_t data_size = HYDRA_EXTERNAL_NS::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<1,T>(fGrid, fLowerLimits, fUpperLimits);

	//work on local copy of data

	auto keys_begin = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	HYDRA_EXTERNAL_NS::thrust::copy( keys_begin, keys_end, key_buffer.first);
	HYDRA_EXTERNAL_NS::thrust::sort(key_buffer.first, key_buffer.first+data_size );


	//bins content
	auto bin_contents    = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(common_system_t(), fContents.size());
	auto reduced_values  = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	auto reduced_keys    = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);
	auto  weights    = HYDRA_EXTERNAL_NS::thrust::constant_iterator<size_t>(1.0);

	auto reduced_end = HYDRA_EXTERNAL_NS::thrust::reduce_by_key(common_system_t(),
			key_buffer.first, key_buffer.first+data_size,
			weights, reduced_keys.first, reduced_values.first);

	HYDRA_EXTERNAL_NS::thrust::fill(bin_contents.first, bin_contents.first+bin_contents.second, 0.0);

	HYDRA_EXTERNAL_NS::thrust::scatter( common_system_t(), reduced_values.first, reduced_end.second,
		  reduced_keys.first, bin_contents.first);

	HYDRA_EXTERNAL_NS::thrust::copy(bin_contents.first ,
			bin_contents.first+ bin_contents.second,  fContents.begin());


    // deallocate storage with HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer
	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), bin_contents.first );
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), reduced_values.first);
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first);
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), key_buffer.first);

    return *this;
}




template<typename T, hydra::detail::Backend BACKEND>
template<hydra::detail::Backend BACKEND2, typename Iterator>
DenseHistogram< T,1, detail::BackendPolicy<BACKEND>, detail::unidimensional>&
DenseHistogram< T,1, detail::BackendPolicy<BACKEND>, detail::unidimensional>::Fill(detail::BackendPolicy<BACKEND2> const& exec_policy,
		Iterator begin, Iterator end )
{
	typedef  typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator>::type system1_t;
	system1_t system1;

	typedef  typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<
			decltype(select_system(exec_policy, fSystem,system1))>::type common_system_t;

	size_t data_size = HYDRA_EXTERNAL_NS::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<1,T>(fGrid, fLowerLimits, fUpperLimits);

	//work on local copy of data

	auto keys_begin = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	HYDRA_EXTERNAL_NS::thrust::copy( keys_begin, keys_end, key_buffer.first);
	HYDRA_EXTERNAL_NS::thrust::sort(key_buffer.first, key_buffer.first+data_size );


	//bins content
	auto bin_contents    = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(common_system_t(), fContents.size());
	auto reduced_values  = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	auto reduced_keys    = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);
	auto  weights    = HYDRA_EXTERNAL_NS::thrust::constant_iterator<size_t>(1.0);

	auto reduced_end = HYDRA_EXTERNAL_NS::thrust::reduce_by_key(common_system_t(),
			key_buffer.first, key_buffer.first+data_size,
			weights, reduced_keys.first, reduced_values.first);

	HYDRA_EXTERNAL_NS::thrust::fill(bin_contents.first, bin_contents.first+bin_contents.second, 0.0);

	HYDRA_EXTERNAL_NS::thrust::scatter( common_system_t(), reduced_values.first, reduced_end.second,
		  reduced_keys.first, bin_contents.first);

	HYDRA_EXTERNAL_NS::thrust::copy(bin_contents.first ,
			bin_contents.first+ bin_contents.second,  fContents.begin());


    // deallocate storage with HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer
	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), bin_contents.first );
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), reduced_values.first);
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first);
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), key_buffer.first);

    return *this;
}



template<typename T, hydra::detail::Backend BACKEND>
template<typename Iterator1, typename Iterator2>
DenseHistogram<T,1, detail::BackendPolicy<BACKEND>, detail::unidimensional>&
DenseHistogram<T,1, detail::BackendPolicy<BACKEND>, detail::unidimensional>::Fill(Iterator1 begin, Iterator1 end, Iterator2 wbegin )
{
	using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
	typedef  typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator1>::type system1_t;
	typedef  typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator2>::type system2_t;
	system1_t system1;
	system2_t system2;

	typedef  typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<
			decltype(select_system(fSystem,system1, system2 ))>::type common_system_t;

	size_t data_size = HYDRA_EXTERNAL_NS::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<1,T>(fGrid, fLowerLimits, fUpperLimits);

	//work on local copy of data
	auto weights  = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	HYDRA_EXTERNAL_NS::thrust::copy(wbegin, wbegin+data_size, weights.first);

	auto keys_begin = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	HYDRA_EXTERNAL_NS::thrust::copy(common_system_t(),  keys_begin, keys_end, key_buffer.first);
	HYDRA_EXTERNAL_NS::thrust::sort_by_key(common_system_t(), key_buffer.first, key_buffer.first+data_size, weights.first);

	//bins content
	auto bin_contents    = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(common_system_t(), fContents.size());
	auto reduced_values  = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	auto reduced_keys    = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);


	auto reduced_end = HYDRA_EXTERNAL_NS::thrust::reduce_by_key(common_system_t(),
			key_buffer.first, key_buffer.first+data_size,
			weights.first, reduced_keys.first, reduced_values.first);

	HYDRA_EXTERNAL_NS::thrust::fill( common_system_t(), bin_contents.first, bin_contents.first+bin_contents.second, 0.0);

	HYDRA_EXTERNAL_NS::thrust::scatter( common_system_t(), reduced_values.first, reduced_end.second,
		  reduced_keys.first, bin_contents.first);

	HYDRA_EXTERNAL_NS::thrust::copy( bin_contents.first ,
			bin_contents.first+ bin_contents.second,  fContents.begin());


    // deallocate storage with HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer
	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), bin_contents.first );
	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), weights.first  );
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), reduced_values.first);
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first);
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), key_buffer.first);

    return *this;
}




template<typename T, hydra::detail::Backend BACKEND>
template<hydra::detail::Backend BACKEND2, typename Iterator1, typename Iterator2>
DenseHistogram<T,1, detail::BackendPolicy<BACKEND>, detail::unidimensional >&
DenseHistogram<T,1, detail::BackendPolicy<BACKEND>, detail::unidimensional >::Fill(detail::BackendPolicy<BACKEND2> const& exec_policy,
		Iterator1 begin, Iterator1 end, Iterator2 wbegin )
{
	using HYDRA_EXTERNAL_NS::thrust::system::detail::generic::select_system;
	typedef  typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator1>::type system1_t;
	typedef  typename HYDRA_EXTERNAL_NS::thrust::iterator_system<Iterator2>::type system2_t;
	system1_t system1;
	system2_t system2;

	typedef  typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<
			decltype(select_system(exec_policy, fSystem,system1, system2 ))>::type common_system_t;

	size_t data_size = HYDRA_EXTERNAL_NS::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<1,T>(fGrid, fLowerLimits, fUpperLimits);

	//work on local copy of data
	auto weights  = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	HYDRA_EXTERNAL_NS::thrust::copy(wbegin, wbegin+data_size, weights.first);

	auto keys_begin = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	HYDRA_EXTERNAL_NS::thrust::copy(common_system_t(),  keys_begin, keys_end, key_buffer.first);
	HYDRA_EXTERNAL_NS::thrust::sort_by_key(common_system_t(), key_buffer.first, key_buffer.first+data_size, weights.first);

	//bins content
	auto bin_contents    = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(common_system_t(), fContents.size());
	auto reduced_values  = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	auto reduced_keys    = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);


	auto reduced_end = HYDRA_EXTERNAL_NS::thrust::reduce_by_key(common_system_t(),
			key_buffer.first, key_buffer.first+data_size,
			weights.first, reduced_keys.first, reduced_values.first);

	HYDRA_EXTERNAL_NS::thrust::fill( common_system_t(), bin_contents.first, bin_contents.first+bin_contents.second, 0.0);

	HYDRA_EXTERNAL_NS::thrust::scatter( common_system_t(), reduced_values.first, reduced_end.second,
		  reduced_keys.first, bin_contents.first);

	HYDRA_EXTERNAL_NS::thrust::copy( common_system_t(), bin_contents.first ,
			bin_contents.first+ bin_contents.second,  fContents.begin());


    // deallocate storage with HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer
	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), bin_contents.first );
	HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), weights.first  );
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), reduced_values.first);
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first);
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(common_system_t(), key_buffer.first);

    return *this;
}

template<typename Iterator, typename T, size_t N , hydra::detail::Backend BACKEND>
DenseHistogram< T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>
make_dense_histogram( detail::BackendPolicy<BACKEND>, std::array<size_t, N> grid,
		std::array<T, N> const& lowerlimits,   std::array<T, N> const& upperlimits,
		Iterator first, Iterator end){

	hydra::DenseHistogram< T, N, detail::BackendPolicy<BACKEND>> _Hist( grid, lowerlimits, upperlimits);
	_Hist.Fill(first, end);

	return _Hist;
}

template< typename T, size_t N , hydra::detail::Backend BACKEND, typename Iterable>
inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value,
DenseHistogram< T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>>::type
make_dense_histogram( detail::BackendPolicy<BACKEND> backend, std::array<size_t, N> grid,
		std::array<T, N>lowerlimits,   std::array<T, N> upperlimits,	Iterable&& data){

	return make_dense_histogram(backend,grid, lowerlimits, upperlimits,
			std::forward<Iterable>(data).begin(), std::forward<Iterable>(data).end());

}


template<typename Iterator, typename T, hydra::detail::Backend BACKEND>
DenseHistogram< T, 1,  detail::BackendPolicy<BACKEND>, detail::multidimensional>
make_dense_histogram( detail::BackendPolicy<BACKEND>, size_t grid, T lowerlimits,  T upperlimits,
		Iterator first, Iterator end){

	hydra::DenseHistogram< T, 1, detail::BackendPolicy<BACKEND>> _Hist( grid, lowerlimits, upperlimits);
	_Hist.Fill(first, end);

	return _Hist;

}

}  // namespace hydra

//#endif /* DENSEHISTOGRAM_INL_ */
