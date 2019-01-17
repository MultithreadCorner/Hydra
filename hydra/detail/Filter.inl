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
 * Filter.inl
 *
 *  Created on: 15/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef FILTER_INL_
#define FILTER_INL_


namespace hydra {

template<typename Iterable, typename Functor>
inline typename std::enable_if< hydra::detail::is_iterable<Iterable>::value,
hydra::Range<decltype(std::declval<Iterable>().begin())>>::type
apply_filter(Iterable&& container, Functor const& filter)
{
	auto new_end = HYDRA_EXTERNAL_NS::thrust::partition(std::forward<Iterable>(container).begin(),
			std::forward<Iterable>(container).end() , filter);
     return hydra::make_range(container.begin(), new_end);
}

}  // namespace hydra

#endif /* FILTER_INL_ */
