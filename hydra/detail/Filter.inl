/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2026 Antonio Augusto Alves Junior
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
 * Filter.inl
 *
 *  Created on: 15/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef FILTER_INL_
#define FILTER_INL_


namespace hydra {

template<typename Iterable, typename Functor>
requires (hydra::detail::Iterable<Iterable>)
inline hydra::Range<decltype(std::declval<Iterable>().begin())>
filter(Iterable&& container, Functor&& filter)
{

	auto new_end = hydra::thrust::partition(std::forward<Iterable>(container).begin(),
	                       std::forward<Iterable>(container).end(), filter);

     return hydra::make_range(std::forward<Iterable>(container).begin(), new_end);

}

template<typename Iterable, typename Functor>
requires (hydra::detail::Iterable<Iterable>)
inline std::pair<hydra::Range<decltype(std::declval<Iterable>().begin())>,
          hydra::Range<decltype(std::declval<Iterable>().begin())>>
segregate(Iterable&& container, Functor&& filter)
{

	auto new_end = hydra::thrust::partition(std::forward<Iterable>(container).begin(),
	                       std::forward<Iterable>(container).end(), filter);

     return std::make_pair(
    		 hydra::make_range(std::forward<Iterable>(container).begin(), new_end),
    		 hydra::make_range(new_end, std::forward<Iterable>(container).end()) );

}

}  // namespace hydra

#endif /* FILTER_INL_ */
