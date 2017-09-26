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
 * DenseHistogram.inl
 *
 *  Created on: 23/09/2017
 *      Author: Antonio Augusto Alves Junior
 */

//#ifndef DENSEHISTOGRAM_INL_
//#define DENSEHISTOGRAM_INL_

#include <hydra/detail/external/thrust/reduce.h>
#include <hydra/detail/external/thrust/gather.h>
#include <hydra/detail/external/thrust/scatter.h>
#include <hydra/detail/functors/GetGlobalBin.h>
#include <hydra/Distance.h>
#include <hydra/detail/external/thrust/iterator/constant_iterator.h>

namespace hydra {

template<size_t N, typename T, hydra::detail::Backend  BACKEND>
template<typename Iterator>
void DenseHistogram<N, T, hydra::detail::BackendPolicy<BACKEND>, std::true_type >::Fill(Iterator begin, Iterator end)
{

	size_t data_size = hydra::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<N,T>(fGrid, fLowerLimits, fUpperLimits);

	auto keys_begin  = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(begin, key_functor );
	auto keys_end    = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(end, key_functor);

	auto reduced_values = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<T>(system_t(), data_size);
	auto reduced_keys   = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(system_t(), data_size);

    auto result = HYDRA_EXTERNAL_NS::thrust::reduce_by_key(system_t(), keys_begin, keys_end, begin,
    		reduced_keys.first, reduced_values.first);

   // HYDRA_EXTERNAL_NS::thrust::gather(system_t(), keys_begin, result.first,
    //		reduced_values.first, fContents.begin() );

    // deallocate storage with HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system_t(), reduced_values.first);
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system_t(), reduced_keys.first);

}

template<typename T, hydra::detail::Backend  BACKEND>
template<typename Iterator>
void DenseHistogram<1, T, hydra::detail::BackendPolicy<BACKEND>, std::false_type >::Fill(Iterator begin, Iterator end)
{

	size_t data_size = HYDRA_EXTERNAL_NS::thrust::distance(begin, end);

	auto key_functor = detail::GetGlobalBin<1,T>(fGrid, fLowerLimits, fUpperLimits);

	//work on local copy of data
	auto data   = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<T>(system_t(), data_size);
	hydra::copy(begin, end, data.first);
	//sort local data
	{
		auto keys_begin  = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(begin, key_functor );
		auto keys_end    = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(end, key_functor);
		auto buffer  = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(system_t(), data_size);
		HYDRA_EXTERNAL_NS::thrust::copy( keys_begin, keys_end, buffer.first);
		HYDRA_EXTERNAL_NS::thrust::sort_by_key(buffer.first, buffer.first+data_size, data.first);
		HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system_t(),buffer.first);
	}

	//bins content
	auto reduced_values  = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(system_t(), data_size);
	auto reduced_keys    = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<size_t>(system_t(), data_size);

	//reduction_by_key
	auto init_bin    = HYDRA_EXTERNAL_NS::thrust::constant_iterator<size_t>(1);
	auto keys_begin  = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(data.first, key_functor );
	auto keys_end    = HYDRA_EXTERNAL_NS::thrust::make_transform_iterator(data.first+data_size, key_functor);

	auto reduced_end = HYDRA_EXTERNAL_NS::thrust::reduce_by_key(system_t(), keys_begin, keys_end, init_bin,
    		reduced_keys.first, reduced_values.first);

	HYDRA_EXTERNAL_NS::thrust::scatter( system_t(),  reduced_values.first, reduced_end.second,
		  reduced_keys.first,fContents.begin() );

    // deallocate storage with HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system_t(), reduced_values.first);
    HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer(system_t(), reduced_keys.first);

}


}  // namespace hydra

//#endif /* DENSEHISTOGRAM_INL_ */
