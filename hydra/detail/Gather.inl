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
 * Gather.inl
 *
 *  Created on: 19/05/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GATHER_INL_
#define GATHER_INL_




#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <utility>
#include <hydra/detail/external/thrust/gather.h>
#include <hydra/Range.h>

namespace hydra {

template<typename Iterable_Source, typename Iterable_Target, typename Iterable_Map>
typename std::enable_if<hydra::detail::is_iterable<Iterable_Source>::value
					 && hydra::detail::is_iterable<Iterable_Target>::value
					 && hydra::detail::is_iterable<Iterable_Map>::value,
Range<decltype(std::declval<Iterable_Target&>().begin())>>::type
gather(Iterable_Source&& source, Iterable_Map&& map, Iterable_Target&& target){

	HYDRA_EXTERNAL_NS::thrust::gather(
			std::forward<Iterable_Source>(source).begin(),
			std::forward<Iterable_Source>(source).end(),
			std::forward<Iterable_Map>(map).begin(),
			std::forward<Iterable_Target>(target).begin() );

	return make_range(std::forward<Iterable_Target>(target).begin(),
		              std::forward<Iterable_Target>(target).end() );
}

/*
template<typename Iterable_Source, typename Iterable_Target, typename Iterator_Map>
typename std::enable_if<hydra::detail::is_iterable<Iterable_Source>::value
					 && hydra::detail::is_iterable<Iterable_Target>::value,
Range<decltype(std::declval<Iterable_Target&>().begin())>>::type
gather(Iterable_Source& source, Range<Iterator_Map>&& map, Iterable_Target& target){

	HYDRA_EXTERNAL_NS::thrust::gather( source.begin(), source.end(),
			map.begin(), target.begin() );
	return make_range(target.begin(), target.end() );
}


template<typename Iterator_Source, typename Iterable_Target, typename Iterator_Map>
typename std::enable_if<hydra::detail::is_iterable<Iterable_Target>::value,
Range<decltype(std::declval<Iterable_Target&>().begin())>>::type
gather(Range<Iterator_Source>&& source, Range<Iterator_Map>&& map, Iterable_Target& target){

	HYDRA_EXTERNAL_NS::thrust::gather( source.begin(), source.end(),
			map.begin(), target.begin() );
	return make_range(target.begin(), target.end() );
}

*/



}  // namespace hydra


#endif /* GATHER_INL_ */
