/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2020 Antonio Augusto Alves Junior
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

#include <hydra/detail/external/hydra_thrust/memory.h>
#include <hydra/detail/external/hydra_thrust/reduce.h>
#include <hydra/detail/external/hydra_thrust/gather.h>
#include <hydra/detail/external/hydra_thrust/scatter.h>
#include <hydra/detail/functors/GetGlobalBin.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/Distance.h>
#include <hydra/detail/external/hydra_thrust/iterator/constant_iterator.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>

namespace hydra {

template< typename T, size_t N, hydra::detail::Backend BACKEND>
template<typename Iterator1, typename Iterator2>
DenseHistogram<T, N, detail::BackendPolicy<BACKEND>, detail::multidimensional>&
DenseHistogram<T, N, detail::BackendPolicy<BACKEND>, detail::multidimensional>::Fill(Iterator1 begin, Iterator1 end, Iterator2 wbegin )
{
	using hydra::thrust::system::detail::generic::select_system;
	typedef  typename hydra::thrust::iterator_system<Iterator1>::type system1_t;
	typedef  typename hydra::thrust::iterator_system<Iterator2>::type system2_t;
	system1_t system1;
	system2_t system2;

	typedef  typename hydra::thrust::detail::remove_reference<
			decltype(select_system(fSystem, system1, system2 ))>::type common_system_t;


    //----------------

	size_t data_size = hydra::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<N,double>(fGrid, fLowerLimits, fUpperLimits);

	//work on local copy of weights

	auto weights  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	hydra::thrust::copy(wbegin, wbegin+data_size, weights.first);

	auto keys_begin = hydra::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = hydra::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	hydra::thrust::copy( keys_begin, keys_end, key_buffer.first);

	hydra::thrust::sort_by_key(key_buffer.first, key_buffer.first + key_buffer.second, weights.first );


	//bins content
	auto bin_contents    = hydra::thrust::get_temporary_buffer<double>(common_system_t(), fContents.size());
	auto reduced_values  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	auto reduced_keys    = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	hydra::thrust::fill(bin_contents.first, bin_contents.first+bin_contents.second, 0.0);

	auto reduced_end = hydra::thrust::reduce_by_key(common_system_t(), key_buffer.first,
			key_buffer.first + key_buffer.second, weights.first, reduced_keys.first, reduced_values.first);

	hydra::thrust::scatter( common_system_t(),  reduced_values.first, reduced_end.second,
			reduced_keys.first,bin_contents.first );

	hydra::thrust::copy(bin_contents.first ,
			bin_contents.first+ bin_contents.second,  fContents.begin());

    // deallocate storage with hydra::thrust::return_temporary_buffer
	hydra::thrust::return_temporary_buffer(common_system_t(), bin_contents.first, bin_contents.second );
    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_values.first, reduced_values.second );
    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first, reduced_keys.second);
    hydra::thrust::return_temporary_buffer(common_system_t(), key_buffer.first, key_buffer.second );
    hydra::thrust::return_temporary_buffer(common_system_t(), weights.first,  weights.second );

    return *this;
}


template<typename T, size_t N, hydra::detail::Backend BACKEND>
template<hydra::detail::Backend BACKEND2, typename Iterator1, typename Iterator2>
DenseHistogram<T, N, detail::BackendPolicy<BACKEND>, detail::multidimensional>&
DenseHistogram<T, N, detail::BackendPolicy<BACKEND>, detail::multidimensional>::Fill(detail::BackendPolicy<BACKEND2> const& exec_policy,
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

	//work on local copy of weights

	auto weights  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	hydra::thrust::copy(wbegin, wbegin+data_size, weights.first);

	auto keys_begin = hydra::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = hydra::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	hydra::thrust::copy( keys_begin, keys_end, key_buffer.first);

	hydra::thrust::sort_by_key(key_buffer.first, key_buffer.first + key_buffer.second, weights.first );


	//bins content
	auto bin_contents    = hydra::thrust::get_temporary_buffer<double>(common_system_t(), fContents.size());
	auto reduced_values  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	auto reduced_keys    = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	hydra::thrust::fill(bin_contents.first, bin_contents.first+bin_contents.second, 0.0);

	auto reduced_end = hydra::thrust::reduce_by_key(common_system_t(), keys_begin, keys_end, weights.first,
    		reduced_keys.first, reduced_values.first);

	hydra::thrust::scatter( common_system_t(),  reduced_values.first, reduced_end.second,
			reduced_keys.first,bin_contents.first );

	hydra::thrust::copy(bin_contents.first ,
			bin_contents.first+ bin_contents.second,  fContents.begin());

    // deallocate storage with hydra::thrust::return_temporary_buffer
	hydra::thrust::return_temporary_buffer(common_system_t(), bin_contents.first, bin_contents.second );
    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_values.first, reduced_values.second );
    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first,reduced_keys.second );
    hydra::thrust::return_temporary_buffer(common_system_t(), key_buffer.first, key_buffer.second );
    hydra::thrust::return_temporary_buffer(common_system_t(), weights.first, weights.second  );

    return *this;
}


template<typename T, size_t N, hydra::detail::Backend BACKEND>
template<typename Iterator>
DenseHistogram<T, N, detail::BackendPolicy<BACKEND>, detail::multidimensional>&
DenseHistogram<T, N,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional>::Fill(Iterator begin, Iterator end )
{
	using hydra::thrust::system::detail::generic::select_system;
	typedef  typename hydra::thrust::iterator_system<Iterator>::type system1_t;
	system1_t system1;

	typedef  typename hydra::thrust::detail::remove_reference<
			decltype(select_system(fSystem, system1 ))>::type common_system_t;


		typedef hydra::thrust::pointer<T, common_system_t> buffer_t;

		size_t data_size = hydra::thrust::distance(begin, end);

		auto key_functor = detail::GetGlobalBin<N,double>(fGrid, fLowerLimits, fUpperLimits);

		auto keys_begin = hydra::thrust::make_transform_iterator(begin, key_functor );
		auto keys_end   = hydra::thrust::make_transform_iterator(end, key_functor);
		auto key_buffer = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);


		hydra::thrust::copy( keys_begin, keys_end, key_buffer.first);
		hydra::thrust::sort(key_buffer.first, key_buffer.first+data_size);


		//bins content
		auto bin_contents    = hydra::thrust::get_temporary_buffer<double>(common_system_t(), fContents.size());
		auto reduced_values  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
		auto reduced_keys    = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);
		auto weights         = hydra::thrust::constant_iterator<double>(1.0);

		auto reduced_end = hydra::thrust::reduce_by_key(common_system_t(),
				key_buffer.first, key_buffer.first+data_size,
				weights, reduced_keys.first, reduced_values.first);

		hydra::thrust::fill(bin_contents.first, bin_contents.first+bin_contents.second, 0.0);

		hydra::thrust::scatter( common_system_t(),  reduced_values.first, reduced_end.second,
			  reduced_keys.first, bin_contents.first );

		hydra::thrust::copy(bin_contents.first ,
				bin_contents.first+ bin_contents.second,  fContents.begin());

	    // deallocate storage with hydra::thrust::return_temporary_buffer
		hydra::thrust::return_temporary_buffer(common_system_t(), key_buffer.first, key_buffer.second );
		hydra::thrust::return_temporary_buffer(common_system_t(), bin_contents.first, bin_contents.second );
	    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_values.first, reduced_values.second);
	    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first, reduced_keys.second);

	    return *this;
}

template<typename T, size_t N, hydra::detail::Backend BACKEND>
template<hydra::detail::Backend BACKEND2, typename Iterator>
DenseHistogram<T, N, detail::BackendPolicy<BACKEND>, detail::multidimensional>&
DenseHistogram<T, N,  hydra::detail::BackendPolicy<BACKEND>, detail::multidimensional>::Fill(detail::BackendPolicy<BACKEND2> const& exec_policy, Iterator begin, Iterator end )
{
	typedef  typename hydra::thrust::iterator_system<Iterator>::type system1_t;
		system1_t system1;

		typedef  typename hydra::thrust::detail::remove_reference<
					decltype(select_system(exec_policy,fSystem, system1))>::type common_system_t;

		typedef hydra::thrust::pointer<T, common_system_t> buffer_t;

		size_t data_size = hydra::thrust::distance(begin, end);

		auto key_functor = detail::GetGlobalBin<N,double>(fGrid, fLowerLimits, fUpperLimits);

		auto keys_begin = hydra::thrust::make_transform_iterator(begin, key_functor );
		auto keys_end   = hydra::thrust::make_transform_iterator(end, key_functor);
		auto key_buffer = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);


		hydra::thrust::copy( keys_begin, keys_end, key_buffer.first);
		hydra::thrust::sort(key_buffer.first, key_buffer.first+data_size);


		//bins content
		auto bin_contents    = hydra::thrust::get_temporary_buffer<double>(common_system_t(), fContents.size());
		auto reduced_values  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
		auto reduced_keys    = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);
		auto weights         = hydra::thrust::constant_iterator<double>(1.0);

		auto reduced_end = hydra::thrust::reduce_by_key(common_system_t(),
				key_buffer.first, key_buffer.first+data_size,
				weights, reduced_keys.first, reduced_values.first);

		hydra::thrust::fill(bin_contents.first, bin_contents.first+bin_contents.second, 0.0);

		hydra::thrust::scatter( common_system_t(),  reduced_values.first, reduced_end.second,
			  reduced_keys.first, bin_contents.first );


		hydra::thrust::copy(bin_contents.first ,
				bin_contents.first+ bin_contents.second,  fContents.begin());

	    // deallocate storage with hydra::thrust::return_temporary_buffer
		hydra::thrust::return_temporary_buffer(common_system_t(), key_buffer.first, key_buffer.second );
		hydra::thrust::return_temporary_buffer(common_system_t(), bin_contents.first, bin_contents.second );
	    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_values.first, reduced_values.second );
	    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first, reduced_keys.second );

	    return *this;
}


template<typename T, hydra::detail::Backend BACKEND>
template<typename Iterator>
DenseHistogram< T,1, detail::BackendPolicy<BACKEND>, detail::unidimensional>&
DenseHistogram< T,1, detail::BackendPolicy<BACKEND>, detail::unidimensional>::Fill(Iterator begin, Iterator end )
{
	using hydra::thrust::system::detail::generic::select_system;
	typedef  typename hydra::thrust::iterator_system<Iterator>::type system1_t;
	system1_t system1;

	typedef  typename hydra::thrust::detail::remove_reference<
			decltype(select_system(fSystem, system1 ))>::type common_system_t;

	size_t data_size = hydra::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<1,double>(fGrid, fLowerLimits, fUpperLimits);

	//work on local copy of data

	auto keys_begin = hydra::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = hydra::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	hydra::thrust::copy( keys_begin, keys_end, key_buffer.first);
	hydra::thrust::sort(key_buffer.first, key_buffer.first+data_size );


	//bins content
	auto bin_contents    = hydra::thrust::get_temporary_buffer<double>(common_system_t(), fContents.size());
	auto reduced_values  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	auto reduced_keys    = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);
	auto  weights    = hydra::thrust::constant_iterator<size_t>(1.0);

	auto reduced_end = hydra::thrust::reduce_by_key(common_system_t(),
			key_buffer.first, key_buffer.first+data_size,
			weights, reduced_keys.first, reduced_values.first);

	hydra::thrust::fill(bin_contents.first, bin_contents.first+bin_contents.second, 0.0);

	hydra::thrust::scatter( common_system_t(), reduced_values.first, reduced_end.second,
		  reduced_keys.first, bin_contents.first);

	hydra::thrust::copy(bin_contents.first ,
			bin_contents.first+ bin_contents.second,  fContents.begin());


    // deallocate storage with hydra::thrust::return_temporary_buffer
	hydra::thrust::return_temporary_buffer(common_system_t(), bin_contents.first, bin_contents.second );
    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_values.first, reduced_values.second);
    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first,reduced_keys.second );
    hydra::thrust::return_temporary_buffer(common_system_t(), key_buffer.first,  key_buffer.second);

    return *this;
}




template<typename T, hydra::detail::Backend BACKEND>
template<hydra::detail::Backend BACKEND2, typename Iterator>
DenseHistogram< T,1, detail::BackendPolicy<BACKEND>, detail::unidimensional>&
DenseHistogram< T,1, detail::BackendPolicy<BACKEND>, detail::unidimensional>::Fill(detail::BackendPolicy<BACKEND2> const& exec_policy,
		Iterator begin, Iterator end )
{
	typedef  typename hydra::thrust::iterator_system<Iterator>::type system1_t;
	system1_t system1;

	typedef  typename hydra::thrust::detail::remove_reference<
			decltype(select_system(exec_policy, fSystem,system1))>::type common_system_t;

	size_t data_size = hydra::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<1,double>(fGrid, fLowerLimits, fUpperLimits);

	//work on local copy of data

	auto keys_begin = hydra::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = hydra::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	hydra::thrust::copy( keys_begin, keys_end, key_buffer.first);
	hydra::thrust::sort(key_buffer.first, key_buffer.first+data_size );


	//bins content
	auto bin_contents    = hydra::thrust::get_temporary_buffer<double>(common_system_t(), fContents.size());
	auto reduced_values  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	auto reduced_keys    = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);
	auto  weights    = hydra::thrust::constant_iterator<size_t>(1.0);

	auto reduced_end = hydra::thrust::reduce_by_key(common_system_t(),
			key_buffer.first, key_buffer.first+data_size,
			weights, reduced_keys.first, reduced_values.first);

	hydra::thrust::fill(bin_contents.first, bin_contents.first+bin_contents.second, 0.0);

	hydra::thrust::scatter( common_system_t(), reduced_values.first, reduced_end.second,
		  reduced_keys.first, bin_contents.first);

	hydra::thrust::copy(bin_contents.first ,
			bin_contents.first+ bin_contents.second,  fContents.begin());


    // deallocate storage with hydra::thrust::return_temporary_buffer
	hydra::thrust::return_temporary_buffer(common_system_t(), bin_contents.first,bin_contents.second  );
    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_values.first, reduced_values.second );
    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first, reduced_keys.second );
    hydra::thrust::return_temporary_buffer(common_system_t(), key_buffer.first, key_buffer.ssecond );

    return *this;
}



template<typename T, hydra::detail::Backend BACKEND>
template<typename Iterator1, typename Iterator2>
DenseHistogram<T,1, detail::BackendPolicy<BACKEND>, detail::unidimensional>&
DenseHistogram<T,1, detail::BackendPolicy<BACKEND>, detail::unidimensional>::Fill(Iterator1 begin, Iterator1 end, Iterator2 wbegin )
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
	hydra::thrust::copy(wbegin, wbegin+data_size, weights.first);

	auto keys_begin = hydra::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = hydra::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	hydra::thrust::copy(common_system_t(),  keys_begin, keys_end, key_buffer.first);
	hydra::thrust::sort_by_key(common_system_t(), key_buffer.first, key_buffer.first+data_size, weights.first);

	//bins content
	auto bin_contents    = hydra::thrust::get_temporary_buffer<double>(common_system_t(), fContents.size());
	auto reduced_values  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	auto reduced_keys    = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);


	auto reduced_end = hydra::thrust::reduce_by_key(common_system_t(),
			key_buffer.first, key_buffer.first+data_size,
			weights.first, reduced_keys.first, reduced_values.first);

	hydra::thrust::fill( common_system_t(), bin_contents.first, bin_contents.first+bin_contents.second, 0.0);

	hydra::thrust::scatter( common_system_t(), reduced_values.first, reduced_end.second,
		  reduced_keys.first, bin_contents.first);

	hydra::thrust::copy( bin_contents.first ,
			bin_contents.first+ bin_contents.second,  fContents.begin());


    // deallocate storage with hydra::thrust::return_temporary_buffer
	hydra::thrust::return_temporary_buffer(common_system_t(), bin_contents.first, bin_contents.second);
	hydra::thrust::return_temporary_buffer(common_system_t(), weights.first, weights.second );
    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_values.first, reduced_values.second);
    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first, reduced_keys.second);
    hydra::thrust::return_temporary_buffer(common_system_t(), key_buffer.first, key_buffer.second);

    return *this;
}




template<typename T, hydra::detail::Backend BACKEND>
template<hydra::detail::Backend BACKEND2, typename Iterator1, typename Iterator2>
DenseHistogram<T,1, detail::BackendPolicy<BACKEND>, detail::unidimensional >&
DenseHistogram<T,1, detail::BackendPolicy<BACKEND>, detail::unidimensional >::Fill(detail::BackendPolicy<BACKEND2> const& exec_policy,
		Iterator1 begin, Iterator1 end, Iterator2 wbegin )
{
	using hydra::thrust::system::detail::generic::select_system;
	typedef  typename hydra::thrust::iterator_system<Iterator1>::type system1_t;
	typedef  typename hydra::thrust::iterator_system<Iterator2>::type system2_t;
	system1_t system1;
	system2_t system2;

	typedef  typename hydra::thrust::detail::remove_reference<
			decltype(select_system(exec_policy, fSystem,system1, system2 ))>::type common_system_t;

	size_t data_size = hydra::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<1,double>(fGrid, fLowerLimits, fUpperLimits);

	//work on local copy of data
	auto weights  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	hydra::thrust::copy(wbegin, wbegin+data_size, weights.first);

	auto keys_begin = hydra::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end   = hydra::thrust::make_transform_iterator(end, key_functor);
	auto key_buffer = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);

	hydra::thrust::copy(common_system_t(),  keys_begin, keys_end, key_buffer.first);
	hydra::thrust::sort_by_key(common_system_t(), key_buffer.first, key_buffer.first+data_size, weights.first);

	//bins content
	auto bin_contents    = hydra::thrust::get_temporary_buffer<double>(common_system_t(), fContents.size());
	auto reduced_values  = hydra::thrust::get_temporary_buffer<double>(common_system_t(), data_size);
	auto reduced_keys    = hydra::thrust::get_temporary_buffer<size_t>(common_system_t(), data_size);


	auto reduced_end = hydra::thrust::reduce_by_key(common_system_t(),
			key_buffer.first, key_buffer.first+data_size,
			weights.first, reduced_keys.first, reduced_values.first);

	hydra::thrust::fill( common_system_t(), bin_contents.first, bin_contents.first+bin_contents.second, 0.0);

	hydra::thrust::scatter( common_system_t(), reduced_values.first, reduced_end.second,
		  reduced_keys.first, bin_contents.first);

	hydra::thrust::copy( common_system_t(), bin_contents.first ,
			bin_contents.first+ bin_contents.second,  fContents.begin());


    // deallocate storage with hydra::thrust::return_temporary_buffer
	hydra::thrust::return_temporary_buffer(common_system_t(), bin_contents.first , bin_contents.second);
	hydra::thrust::return_temporary_buffer(common_system_t(), weights.first , weights.second );
    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_values.first , reduced_values.second);
    hydra::thrust::return_temporary_buffer(common_system_t(), reduced_keys.first , reduced_keys.second);
    hydra::thrust::return_temporary_buffer(common_system_t(), key_buffer.first , key_buffer.second);

    return *this;
}



/*
 * multidimensional specializations
 */

template<typename T,  size_t N , hydra::detail::Backend BACKEND>
template<size_t M>
inline typename std::enable_if< M==2, T >::type
DenseHistogram< T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>::Interpolate( std::array<size_t,2> const&  point){

	return spline2D(this->GetBinsCenters(placeholders::_0).begin(), this->GetBinsCenters(placeholders::_0).end(),
			 this->GetBinsCenters(placeholders::_1).begin(), this->GetBinsCenters(placeholders::_1).end(),
			 this->GetBinsContents().begin(), point[0],  point[1]);
}

template<typename T,  size_t N , hydra::detail::Backend BACKEND>
template<size_t M>
inline typename std::enable_if< M==3, T >::type
DenseHistogram< T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>::Interpolate( std::array<size_t,3> const&  point){

	return spline3D(this->GetBinsCenters(placeholders::_0).begin(), this->GetBinsCenters(placeholders::_0).end(),
				 this->GetBinsCenters(placeholders::_1).begin(), this->GetBinsCenters(placeholders::_1).end(),
				 this->GetBinsCenters(placeholders::_2).begin(), this->GetBinsCenters(placeholders::_2).end(),
				 this->GetBinsContents().begin(), point[0],  point[1],  point[2]);
}


template<typename T, size_t N , hydra::detail::Backend BACKEND>
template<size_t M>
inline typename std::enable_if< M==4, T >::type
DenseHistogram< T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>::Interpolate( std::array<size_t,4> const&  point){

	return spline4D(
			     this->GetBinsCenters(placeholders::_0).begin(), this->GetBinsCenters(placeholders::_0).end(),
				 this->GetBinsCenters(placeholders::_1).begin(), this->GetBinsCenters(placeholders::_1).end(),
				 this->GetBinsCenters(placeholders::_2).begin(), this->GetBinsCenters(placeholders::_2).end(),
				 this->GetBinsCenters(placeholders::_3).begin(), this->GetBinsCenters(placeholders::_3).end(),
				 this->GetBinsContents().begin(), point[0],  point[1],  point[2],  point[3]);
}

//iterator based
template<typename T, typename Iterator, size_t N , hydra::detail::Backend BACKEND>
DenseHistogram< T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>
make_dense_histogram( detail::BackendPolicy<BACKEND>, std::array<size_t, N> const& grid,
		std::array<double, N> const& lowerlimits,   std::array<double, N> const& upperlimits,
		Iterator first, Iterator end){

	hydra::DenseHistogram< T, N, detail::BackendPolicy<BACKEND>> _Hist( grid, lowerlimits, upperlimits);
	_Hist.Fill(first, end);

	return _Hist;
}

template< typename T,typename Iterator1,typename Iterator2, size_t N , hydra::detail::Backend BACKEND>
DenseHistogram< T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>
make_dense_histogram( detail::BackendPolicy<BACKEND>, std::array<size_t, N> const& grid,
		std::array<double, N> const& lowerlimits,   std::array<double, N> const& upperlimits,
		Iterator1 first, Iterator1 end, Iterator2 wfirst){

	hydra::DenseHistogram< T, N, detail::BackendPolicy<BACKEND>> _Hist( grid, lowerlimits, upperlimits);
	_Hist.Fill(first, end, wfirst);

	return _Hist;
}


//iterable based
template< typename T, size_t N , hydra::detail::Backend BACKEND, typename Iterable>
inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value,
DenseHistogram< T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>>::type
make_dense_histogram( detail::BackendPolicy<BACKEND> backend, std::array<size_t, N> const& grid,
		std::array<double, N> const& lowerlimits,   std::array<double, N> const& upperlimits,	Iterable&& data){

	return make_dense_histogram<T>(backend,grid, lowerlimits, upperlimits,
			std::forward<Iterable>(data).begin(), std::forward<Iterable>(data).end());

}

template< typename T, size_t N , hydra::detail::Backend BACKEND, typename Iterable1,typename Iterable2 >
inline typename std::enable_if< hydra::detail::is_iterable<Iterable1>::value&&
                                hydra::detail::is_iterable<Iterable2>::value,
DenseHistogram< T, N,  detail::BackendPolicy<BACKEND>, detail::multidimensional>>::type
make_dense_histogram( detail::BackendPolicy<BACKEND> backend, std::array<size_t, N>const&  grid,
		std::array<double, N>const& lowerlimits,   std::array<double, N>const&  upperlimits,
		Iterable1&& data,
		Iterable2&& weights){

	return make_dense_histogram<T>(backend,grid, lowerlimits, upperlimits,
			std::forward<Iterable1>(data).begin(),
			std::forward<Iterable1>(data).end(),
			std::forward<Iterable2>(weights).begin());

}

/*
 * unidimensional specializations
 */
//iterator based
template<typename T,typename Iterator,  hydra::detail::Backend BACKEND>
DenseHistogram< T, 1,  detail::BackendPolicy<BACKEND>, detail::unidimensional>
make_dense_histogram( detail::BackendPolicy<BACKEND>, size_t grid, double lowerlimits,  double upperlimits,
		Iterator first, Iterator end){

	hydra::DenseHistogram< T, 1, detail::BackendPolicy<BACKEND>> _Hist( grid, lowerlimits, upperlimits);
	_Hist.Fill(first, end);

	return _Hist;

}

template< typename T,typename Iterator1, typename Iterator2, hydra::detail::Backend BACKEND>
DenseHistogram< T, 1,  detail::BackendPolicy<BACKEND>, detail::unidimensional>
make_dense_histogram( detail::BackendPolicy<BACKEND>, size_t grid, double lowerlimits, double upperlimits,
		Iterator1 first, Iterator1 end, Iterator2 wfirst){

	hydra::DenseHistogram< T, 1, detail::BackendPolicy<BACKEND>> _Hist( grid, lowerlimits, upperlimits);
	_Hist.Fill(first, end, wfirst);

	return _Hist;

}

//iterable based
template< typename T, hydra::detail::Backend BACKEND, typename Iterable>
inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value,
DenseHistogram< T, 1,  detail::BackendPolicy<BACKEND>, detail::unidimensional>>::type
make_dense_histogram( detail::BackendPolicy<BACKEND> backend, size_t grid,
		double lowerlimits, double upperlimits,	Iterable&& data){

	return make_dense_histogram<T>(backend,grid, lowerlimits, upperlimits,
			std::forward<Iterable>(data).begin(), std::forward<Iterable>(data).end());

}

template< typename T, hydra::detail::Backend BACKEND, typename Iterable1,typename Iterable2 >
inline typename std::enable_if< hydra::detail::is_iterable<Iterable1>::value&&
hydra::detail::is_iterable<Iterable2>::value,
DenseHistogram< T, 1,  detail::BackendPolicy<BACKEND>, detail::unidimensional>>::type
make_dense_histogram( detail::BackendPolicy<BACKEND> backend, size_t grid,
		double lowerlimits, double upperlimits, Iterable1&& data, Iterable2&& weights){

	return make_dense_histogram<T>(backend, grid, lowerlimits, upperlimits,
			std::forward<Iterable1>(data).begin(),
			std::forward<Iterable1>(data).end(),
			std::forward<Iterable2>(weights).begin());

}




}  // namespace hydra

//#endif /* DENSEHISTOGRAM_INL_ */
